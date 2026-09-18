# -*- coding: utf-8 -*-
"""后端运行时状态：轮询 ``/props`` 与 ``/slots``，并接收转发时抓到的 ``timings``。

数据来源都在 buun-llama-cpp 的 llama-server 上实测确认过：

===============  ==========================================================
``/props``       ``default_generation_settings.n_ctx``（上下文长度）、
                 ``vbr{enabled,dynamic,codec,entry_type_k/v,floor_bpv,
                 capacity_floor_bpv,realized_bpv,selected_bpv,
                 vram_budget_bytes}``（当前 KV 量化状态）
``/slots``       ``n_ctx`` / ``kv_bpv``（当前每值比特数）/ ``n_prompt_tokens``
                 （当前上下文已用多少 token）/ ``is_processing``
响应体 ``timings`` ``prompt_per_second`` / ``predicted_per_second`` / ``kv_bpv``
                 —— **默认就带，不用加任何参数**（``--show-timings`` 服务端还不认）
===============  ==========================================================

注意 ``/metrics`` 默认返回 501（要 ``--metrics`` 才开），所以不用它。

这个对象里的所有方法都不抛异常：采集失败就保持上一次的值，
界面永远拿到一个能直接渲染的字典。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

POLL_INTERVAL = 1.5      # 正常轮询间隔（秒）
POLL_BACKOFF = 6.0       # 连续失败后的间隔
MAX_FAIL = 4             # 连续失败几次算「后端不在了」

# bpv → 档位名。数值是从 engine.py / schema.VBR_LADDER 同一张实测表来的：
# 本机逐档跑 0.6B 模型、读 /slots 的 kv_bpv 得到（f16=16、q8_0=8.5、
# turbo8=8.125、q4_0=4.5、turbo4=4.125、turbo3_tcq=3.25、turbo2_tcq=2.25、
# turbo1_tcq=1.25 …）。这些都是 ggml 量化块规模决定的，不是整数。
from .schema import VBR_LADDER as _LADDER   # noqa: E402  (单一事实来源)


def _exact(v: float) -> str:
    for ref, name in _LADDER:
        if abs(v - ref) <= 0.03:
            return name
    return ""


def tier_from_bpv(bpv: Any, family: str = "") -> str:
    """把 bits/value 翻译成档位名。

    * 正好等于某个档位的单价 → 直接给档位名（``turbo4``）
    * 落在两档之间（VBR 降级后是**混合**状态）→ 给「不高于它的最高档 + 实测值」，
      例如 4.25 → ``≈turbo4（4.25 bpv）``
    * 8.5 在 classic 阶梯里是 q8_0；8.125 才是 turbo8，两者不混淆
    """
    try:
        v = float(bpv)
    except (TypeError, ValueError):
        return ""
    exact = _exact(v)
    if exact:
        if exact == "q4_0" and family == "turbo":
            return "q4_0/iq4_nl"
        return exact
    for ref, name in _LADDER:          # 阶梯是从高到低排的
        if v >= ref:
            return "≈%s（%.4g bpv）" % (name, v)
    return "%.3g bpv" % v


def _get_json(port: int, path: str, timeout: float = 4.0) -> Any:
    url = "http://127.0.0.1:%d%s" % (int(port), path)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class RuntimeStatus:
    """后端（内部端口上的 llama-server）的实时状态。

    生命周期跟着统一后端走：起进程后 ``attach(port)``，停进程时 ``detach()``。
    """

    def __init__(self, log_fn: Optional[Callable[[str, str], None]] = None
                 ) -> None:
        self.log_fn = log_fn or (lambda msg, tag="": None)
        self._lock = threading.Lock()
        self._port = 0
        self._tag = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fail = 0
        self.data: Dict[str, Any] = self._blank()

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _blank() -> Dict[str, Any]:
        return {
            "attached": False, "alive": False, "port": 0,
            "n_ctx": None, "n_past": None, "used_ratio": None,
            "idle": True, "slots": 0,
            "kv_bpv": None, "kv_tier": "", "vbr": {}, "vbr_active": False,
            "budget_bytes": 0, "floor_bpv": None, "capacity_bpv": None,
            "realized_bpv": None, "codec": "",
            "gen_tps": None, "prompt_tps": None, "speed_at": 0.0,
            "cache_n": None, "last_tokens": None,
            "error": "", "updated": 0.0,
        }

    # ------------------------------------------------------------ 生命周期
    def attach(self, port: int, tag: str = "") -> None:
        """开始轮询某个内部端口（换模型会换端口，所以每次启动都要调）。

        会先把上一轮的轮询线程停掉再重开，这样换端口**立刻生效** ——
        否则旧线程可能正卡在失败退避（6 秒）里，新端口要等好几秒才开始采。
        """
        self._stop.set()
        old = self._thread
        if old is not None and old.is_alive() and old is not threading.current_thread():
            old.join(timeout=1.0)
        with self._lock:
            self._port = int(port or 0)
            self._tag = tag or ""
            self._fail = 0
            self.data = self._blank()
            self.data["attached"] = bool(self._port)
            self.data["port"] = self._port
        if not self._port:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                       name="runtime-poll", daemon=True)
        self._thread.start()

    def detach(self, note: str = "") -> None:
        self._stop.set()
        with self._lock:
            self._port = 0
            self.data = self._blank()
            if note:
                self.data["error"] = note

    # -------------------------------------------------------- 转发侧的回填
    def note_timings(self, timings: Any, usage: Any = None) -> None:
        """网关转发响应时顺手调用：把 timings / usage 记下来。

        这样速度和 kv_bpv 是「真实发生过的那次请求」的数字，
        不用为了测速再发一个请求。
        """
        if not isinstance(timings, dict):
            return
        with self._lock:
            d = self.data
            gen = timings.get("predicted_per_second")
            pmt = timings.get("prompt_per_second")
            try:
                if gen is not None:
                    d["gen_tps"] = float(gen)
                if pmt is not None:
                    d["prompt_tps"] = float(pmt)
                if timings.get("predicted_n") is not None:
                    d["last_tokens"] = int(timings["predicted_n"])
                if timings.get("cache_n") is not None:
                    d["cache_n"] = int(timings["cache_n"])
                if timings.get("kv_bpv") is not None:
                    d["kv_bpv"] = float(timings["kv_bpv"])
                    d["kv_tier"] = tier_from_bpv(d["kv_bpv"], d.get("codec", ""))
                d["speed_at"] = time.time()
            except (TypeError, ValueError):
                pass
            if isinstance(usage, dict):
                try:
                    if usage.get("prompt_tokens") is not None:
                        d["n_past"] = int(usage["prompt_tokens"]) + int(
                            usage.get("completion_tokens") or 0)
                        if d.get("n_ctx"):
                            d["used_ratio"] = min(
                                1.0, d["n_past"] / float(d["n_ctx"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

    # ---------------------------------------------------------------- 轮询
    def _loop(self) -> None:
        while not self._stop.is_set():
            port = self._port
            if not port:
                break
            ok = self._poll_once(port)
            if ok:
                self._fail = 0
                wait = POLL_INTERVAL
            else:
                self._fail += 1
                wait = POLL_BACKOFF
            if self._stop.wait(wait):
                break

    def _poll_once(self, port: int) -> bool:
        props = slots = None
        try:
            props = _get_json(port, "/props")
        except Exception:  # noqa: BLE001
            pass
        try:
            slots = _get_json(port, "/slots")
        except Exception:  # noqa: BLE001
            pass
        if props is None and slots is None:
            with self._lock:
                self.data["alive"] = False
                self.data["error"] = ("读不到后端状态（端口 %d 上的 /props 与 "
                                      "/slots 都没响应）" % port)
                if self._fail + 1 >= MAX_FAIL:
                    self.data["error"] = "后端似乎已退出"
            return False

        with self._lock:
            d = self.data
            d["alive"] = True
            d["error"] = ""
            d["updated"] = time.time()
            d["port"] = port
            d["attached"] = True
            if isinstance(props, dict):
                dgs = props.get("default_generation_settings")
                if isinstance(dgs, dict) and dgs.get("n_ctx"):
                    try:
                        d["n_ctx"] = int(dgs["n_ctx"])
                    except (TypeError, ValueError):
                        pass
                vbr = props.get("vbr")
                if isinstance(vbr, dict) and vbr:
                    d["vbr"] = vbr
                    d["vbr_active"] = bool(vbr.get("enabled"))
                    d["codec"] = str(vbr.get("codec") or "")
                    d["floor_bpv"] = vbr.get("floor_bpv")
                    d["capacity_bpv"] = vbr.get("capacity_floor_bpv")
                    d["realized_bpv"] = vbr.get("realized_bpv") or \
                        vbr.get("selected_bpv")
                    try:
                        d["budget_bytes"] = int(vbr.get("vram_budget_bytes") or 0)
                    except (TypeError, ValueError):
                        d["budget_bytes"] = 0
            if isinstance(slots, list) and slots:
                d["slots"] = len(slots)
                used = 0
                bpv = None
                busy = False
                for s in slots:
                    if not isinstance(s, dict):
                        continue
                    try:
                        used = max(used, int(s.get("n_prompt_tokens") or 0))
                    except (TypeError, ValueError):
                        pass
                    if s.get("kv_bpv") is not None:
                        try:
                            bpv = float(s["kv_bpv"])
                        except (TypeError, ValueError):
                            pass
                    busy = busy or bool(s.get("is_processing"))
                    if s.get("n_ctx") and not d.get("n_ctx"):
                        try:
                            d["n_ctx"] = int(s["n_ctx"])
                        except (TypeError, ValueError):
                            pass
                d["idle"] = not busy and used == 0
                if used:
                    d["n_past"] = used
                if bpv is not None:
                    d["kv_bpv"] = bpv
                    d["kv_tier"] = tier_from_bpv(bpv, d.get("codec", ""))
            if d.get("n_ctx") and d.get("n_past") is not None:
                try:
                    d["used_ratio"] = min(1.0, float(d["n_past"]) /
                                          float(d["n_ctx"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    d["used_ratio"] = None
        return True

    # ------------------------------------------------------------------ 读
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            d = dict(self.data)
            d["vbr"] = dict(self.data.get("vbr") or {})
        return d

    def summary(self) -> str:
        """一行摘要，给状态栏用。"""
        d = self.snapshot()
        if not d.get("attached"):
            return "运行状态：未启动"
        if not d.get("alive"):
            return "运行状态：%s" % (d.get("error") or "后端无响应")
        bits = []
        if d.get("gen_tps"):
            bits.append("生成 %.0f t/s" % d["gen_tps"])
        if d.get("prompt_tps"):
            bits.append("提示 %.0f t/s" % d["prompt_tps"])
        if d.get("kv_tier"):
            bits.append("KV %s" % d["kv_tier"])
        if d.get("n_ctx"):
            if d.get("n_past"):
                bits.append("上下文 %d/%d (%.1f%%)"
                            % (d["n_past"], d["n_ctx"],
                               (d.get("used_ratio") or 0) * 100))
            else:
                bits.append("上下文 %d（空闲）" % d["n_ctx"])
        return "运行状态：" + (" · ".join(bits) if bits else "等待第一次请求")
