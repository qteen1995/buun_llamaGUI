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

路由模式（本程序唯一跑法）的特殊性：路由器主实例的裸 ``/props`` 是占位数据
（``default_generation_settings.n_ctx`` 恒为 0、``vbr`` 为空，见引擎
``server-models.cpp`` 的 ``get_router_props``）。真正的模型数据在子模型进程上，
必须用 ``/props?model=<段名>`` / ``/slots?model=<段名>`` 去查——本模块每次轮询
就是这么做的（目标模型 id 由 UI 每帧通过 ``set_target`` 喂进来）。

⚠️ 带 ``?model=`` 的请求**必须再加 ``autoload=false``**：引擎 ``proxy_get``
（``server-models.cpp:1948``）默认取 ``--models-autoload``，为真时会
``ensure_model_ready()`` —— 也就是「按需自动加载」。不带这个参数的话，本模块的
轮询会变成一条「请求某模型」的指令，**把用户刚手动卸载的模型又自动拉回来**
（实测现象：卸载后立刻 `ensure_model: ... is not loaded, loading...`）。
带 ``autoload=false`` 时模型没在跑就直接回「model is not loaded」，既不加载、
也不会有副作用；这也与本模块「只观测、不驱动」的定位一致。

这个对象里的所有方法都不抛异常：采集失败就保持上一次的值，
界面永远拿到一个能直接渲染的字典。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
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


def _q(s: str) -> str:
    """给路由 id 做 URL 编码（段名里可能带点 / 空格）。"""
    return urllib.parse.quote(str(s or ""), safe="")


def _mq(mid: str) -> str:
    """模型级查询串：``?model=<id>&autoload=false``。

    ``autoload=false`` 是**必须的**（不是可选优化）：少了它，引擎会把这几个只读
    轮询当成「请求加载这个模型」，于是刚被手动卸载的模型会被路由又拉回来。
    """
    return "?model=" + _q(mid) + "&autoload=false"


class RuntimeStatus:
    """后端（内部端口上的 llama-server）的实时状态。

    生命周期跟着统一后端走：起进程后 ``attach(port)``，停进程时 ``detach()``。
    """

    def __init__(self, log_fn: Optional[Callable[[str, str], None]] = None,
                 router: Any = None) -> None:
        self.log_fn = log_fn or (lambda msg, tag="": None)
        self._lock = threading.Lock()
        self._port = 0
        self._tag = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fail = 0
        self._router = router          # RouterMonitor：拿已加载模型列表
        self._model = ""               # 当前目标模型（路由 id），由 UI 每帧 set_target
        self._diag_loaded = False
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

    # 「模型级」字段：一旦路由上没有已加载模型，这些必须清掉 ——
    # 否则底栏会继续显示上一个模型（已卸载）的上下文 / KV 档位，误导成「还加载着」。
    _MODEL_KEYS = ("n_ctx", "n_past", "used_ratio", "kv_bpv", "kv_tier",
                   "vbr", "vbr_active", "budget_bytes", "floor_bpv",
                   "capacity_bpv", "realized_bpv", "codec", "cache_n",
                   "last_tokens", "slots")

    @classmethod
    def _clear_model(cls, d: Dict[str, Any]) -> None:
        blank = cls._blank()
        for k in cls._MODEL_KEYS:
            d[k] = blank[k]
        d["idle"] = True

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
            self._diag_loaded = False
            self._model = ""
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

    # -------------------------------------------------- 目标模型（UI 每帧喂）
    def set_target(self, name: str) -> None:
        """UI 每帧把「目标模型」的路由 id 喂进来（路径 → 段名由 UI 翻译）。"""
        self._model = name or ""

    def _pick_model(self) -> str:
        """选这次轮询要查的子模型，**只从「已加载」里挑**。

        顺序：UI 目标 → 第一个已加载的 LLM → 第一个已加载的其它（embedding 等）
        → ``""``（一个都没加载，本轮不查任何子模型）。

        绝不返回未加载的模型：否则 ``/props?model=`` 即使带了 ``autoload=false``
        也只是白挨一个 400；而不带就会触发引擎按需加载（见模块 docstring）。
        drafter（草稿模型）永远排除——它没有自己的上下文要展示。
        """
        target = self._model or ""
        if self._router is None:
            return target              # 没有路由监控时退回旧行为（直连单实例场景）
        try:
            loaded = [str(n) for n in self._router.loaded_names()
                      if self._router.kind_of(str(n)) != "drafter"]
        except Exception:  # noqa: BLE001
            loaded = []
        if target and target in loaded:
            return target
        for n in loaded:               # 其次：优先 LLM，其次 embedding
            try:
                if self._router.kind_of(n) == "llm":
                    return n
            except Exception:  # noqa: BLE001
                pass
        if loaded:
            return loaded[0]
        return ""

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
        # 路由模式下，裸 /props（不带 ?model=）由路由器主实例返回，n_ctx 恒为 0、
        # vbr 是占位的空数据（见引擎 server-models.cpp 的 get_router_props）。
        # 真正的模型数据在子模型进程上，必须带 ?model=<id> 去查。
        # 所以这里：先用裸 /props 判活（路由器永远 200），再查子模型的真实数据。
        props = None
        try:
            props = _get_json(port, "/props")
        except Exception:  # noqa: BLE001
            props = None
        if not isinstance(props, dict):
            with self._lock:
                self.data["alive"] = False
                self.data["error"] = ("读不到后端状态（端口 %d 上的 /props 没响应）"
                                      % port)
                if self._fail + 1 >= MAX_FAIL:
                    self.data["error"] = "后端似乎已退出"
            return False

        mid = self._pick_model()
        pprops = pslots = None
        if mid:
            # 只在「该模型已经加载」时才会走到这里（_pick_model 的约定）；
            # 仍带 autoload=false 双保险，保证轮询永远不会触发引擎的按需加载。
            try:
                pprops = _get_json(port, "/props" + _mq(mid))
            except Exception:  # noqa: BLE001
                pprops = None
            try:
                pslots = _get_json(port, "/slots" + _mq(mid))
            except Exception:  # noqa: BLE001
                pslots = None

        with self._lock:
            d = self.data
            d["alive"] = True
            d["error"] = ""
            d["updated"] = time.time()
            d["port"] = port
            d["attached"] = True
            if not mid:
                # 路由上没有任何已加载模型（用户手动卸载了 / 一个都还没加载）：
                # 清掉模型级字段，别显示上一个模型的残留。注意——**不查任何子模型**，
                # 否则就变成「请求某个模型」，会被路由按需自动加载回来。
                self._clear_model(d)
                return True
            _props_bpv = None
            if isinstance(pprops, dict):
                dgs = pprops.get("default_generation_settings")
                if isinstance(dgs, dict) and dgs.get("n_ctx"):
                    try:
                        d["n_ctx"] = int(dgs["n_ctx"])
                    except (TypeError, ValueError):
                        pass
                vbr = pprops.get("vbr")
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
                    _rb = vbr.get("realized_bpv") or vbr.get("selected_bpv")
                    if _rb is not None:
                        try:
                            _props_bpv = float(_rb)
                        except (TypeError, ValueError):
                            pass
            if isinstance(pslots, list) and pslots:
                d["slots"] = len(pslots)
                used = 0
                bpv = None
                busy = False
                for s in pslots:
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
                    # 缓存命中（tok）：/slots 直接给 n_prompt_tokens_cache，
                    # 比等响应体 timings 更底层、更稳（轮询就能拿）
                    try:
                        c = int(s.get("n_prompt_tokens_cache") or 0)
                    except (TypeError, ValueError):
                        c = 0
                    if c > 0:
                        d["cache_n"] = max(d.get("cache_n") or 0, c)
                d["idle"] = not busy and used == 0
                if used:
                    d["n_past"] = used
                else:
                    d["n_past"] = 0
                    d["used_ratio"] = 0.0
                # /slots 的 kv_bpv 优先（实时值）；没有时用 /props 的回退值
                if bpv is not None:
                    d["kv_bpv"] = bpv
                    d["kv_tier"] = tier_from_bpv(bpv, d.get("codec", ""))
                elif _props_bpv is not None:
                    d["kv_bpv"] = _props_bpv
                    d["kv_tier"] = tier_from_bpv(_props_bpv, d.get("codec", ""))
            else:
                # 子模型还没数据（目标未加载 / 查不到）：保持「已就绪」形态，
                # 不清掉 n_ctx —— 等子模型有数据再填，避免来回跳。
                if _props_bpv is not None:
                    d["kv_bpv"] = _props_bpv
                    d["kv_tier"] = tier_from_bpv(_props_bpv, d.get("codec", ""))
            if d.get("n_ctx") and d.get("n_past") is not None:
                try:
                    d["used_ratio"] = min(1.0, float(d["n_past"]) /
                                          float(d["n_ctx"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    d["used_ratio"] = None
            # —— 诊断：拿到子模型真实 /props、/slots 后打一次（确认 n_ctx 现在
            # 是从子模型取的，而不是路由主实例那个 n_ctx=0 的占位）。
            try:
                if mid and not self._diag_loaded and pprops is not None:
                    self._diag_loaded = True
                    self.log_fn(
                        "[runtime-diag] 子模型 %s /props = %s"
                        % (mid, json.dumps(pprops, ensure_ascii=False)[:2000]),
                        "dbg")
                    if pslots is not None:
                        self.log_fn(
                            "[runtime-diag] 子模型 %s /slots = %s"
                            % (mid, json.dumps(pslots, ensure_ascii=False)[:2000]),
                            "dbg")
            except Exception:  # noqa: BLE001
                pass
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
