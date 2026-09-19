# -*- coding: utf-8 -*-
"""多模型路由后端：模型清单、按类限流、空闲卸载。

为什么要这样一个模块
--------------------
服务现在只有「多模型路由」一种跑法：一个 llama-server 进程带 N 个模型子进程。
模型的状态（谁在加载、谁已就绪、谁刚卸载）由引擎自己管，我们通过它的 REST
接口读和操作：

    GET    /models                  列出所有模型 + 状态
    GET    /models?reload=1         重新读一遍预置文件（改完参数后调）
    POST   /models/load   {"model": 名}
    POST   /models/unload {"model": 名}

引擎限制（读源码 + 实测确认）：

* ``--models-max`` 是**单个全局数字**，它的 LRU 驱逐**不看模型类别** ——
  只传「LLM 上限」的话 embedding 一多就会被跨类踢掉。
* 源码里那个能「钉住不许驱逐」的 ``pin``（common/arg.cpp 附近）**整段被注释掉
  了，上游没实现**。
* ``GET /models`` **不返回 last_used**。

所以「LLM / Embedding 分开记账、分开顶替、分开计时」只能在网关侧做 —— 本模块
就是干这个的。``--models-max`` 那个全局值由 App 算出来只当兜底
（= LLM 上限 + Embedding 上限 + 2），正常的驱逐都发生在这里。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

# 引擎侧的模型状态（tools/server/server-models.h 的 server_model_status_to_string）
STATUS_ZH: Dict[str, str] = {
    "downloading": "下载中",
    "downloaded": "已下载",
    "unloaded": "未加载",
    "loading": "正在加载…",
    "loaded": "已就绪",
    "sleeping": "已休眠",
}
# 「算作驻留、占名额」的状态
RESIDENT = frozenset(("loading", "loaded", "sleeping"))


class ResidencyPolicy:
    """驻留策略（服务页「驻留策略」小节的取值，由 App 每次刷新前塞进来）。"""

    def __init__(self) -> None:
        self.llm_max = 1
        self.emb_max = 2
        self.emb_uncounted = False
        self.llm_idle_min = 15
        self.emb_idle_min = 30

    def limit_for(self, kind: str) -> Optional[int]:
        """这类模型最多驻留几个；None = 不限。"""
        if kind == "llm":
            return self.llm_max if self.llm_max > 0 else None
        if self.emb_uncounted:
            return None
        return self.emb_max if self.emb_max > 0 else None

    def idle_seconds_for(self, kind: str) -> int:
        mins = self.llm_idle_min if kind == "llm" else self.emb_idle_min
        return max(0, int(mins)) * 60


class RouterMonitor:
    """轮询路由后端的模型清单，并执行按类限流 + 空闲卸载。

    * ``kind_of(name)``  → "llm" / "emb"，由 App 按模型库的分类给
    * ``policy``         → ResidencyPolicy
    * 轮询与策略都在后台线程里跑；界面线程只读 :meth:`snapshot`（有锁）
    """

    def __init__(self, kind_of: Callable[[str], str],
                 policy: ResidencyPolicy,
                 interval: float = 1.0,
                 log: Optional[Callable[[str, str], None]] = None) -> None:
        self.kind_of = kind_of
        self.policy = policy
        self.interval = float(interval)
        self._log = log
        self._lock = threading.RLock()
        self._port = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._models: List[Dict[str, Any]] = []
        self._error = ""
        self._last_used: Dict[str, float] = {}
        self._evicted: List[str] = []      # 被策略顶掉过、还没重新加载的
        self._active = False               # 路由进程是否活着
        self._waiting: Dict[str, float] = {}   # 正在排队的加载请求

    # ------------------------------------------------------------- 生命周期
    def attach(self, port: int) -> None:
        """后端（重新）起来了，绑到这个内部端口并开始轮询。"""
        with self._lock:
            self._port = int(port or 0)
            self._models = []
            self._error = ""
            self._active = bool(port)
        self.detach()
        if not port:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="router-monitor",
                                       daemon=True)
        self._thread.start()

    def detach(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._models = []
            self._active = False
            self._last_used.clear()
            self._waiting.clear()

    def mark_stopped(self) -> None:
        """后端进程已经没了（用户停服务 / 进程退出）。"""
        self.detach()

    # --------------------------------------------------------------- 读状态
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "error": self._error,
                "models": [dict(m) for m in self._models],
                "policy": self.policy,
            }

    def loaded_names(self) -> List[str]:
        with self._lock:
            return [str(m.get("id")) for m in self._models
                    if m.get("status") in RESIDENT]

    def all_names(self) -> List[str]:
        """路由当前认识的**全部**名字：id + 别名。

        网关拿它把客户端写的名字翻成正式 id。别名尤其要紧：预置文件里给每个
        模型登记了「文件名」和「去掉扩展名的文件名」（见 builder.preset_aliases），
        所以第三方软件直接用文件名请求也能落到正确的模型上。
        """
        with self._lock:
            out: List[str] = []
            for m in self._models:
                for n in [m.get("id")] + list(m.get("aliases") or []):
                    n = str(n or "")
                    if n and n not in out:
                        out.append(n)
            return out

    def touch(self, name: str) -> None:
        """网关每转发一次请求就调一下：记「这个模型最后被用是什么时候」。"""
        if not name:
            return
        with self._lock:
            self._last_used[name] = time.time()

    # --------------------------------------------------------------- 写操作
    def load(self, name: str) -> Tuple[bool, str]:
        """POST /models/load。引擎是异步的 —— 返回 success 只代表接受了请求。"""
        if not name:
            return False, "没有指定模型"
        code, body = self._post("/models/load", {"model": name})
        if code == 200:
            self.touch(name)
            return True, "已请求加载，稍等它出现在「已加载」列表里"
        msg = _err_text(body)
        if "already running" in msg:
            return True, "这个模型已经在跑了"
        return False, msg or ("HTTP %s" % code)

    def unload(self, name: str) -> Tuple[bool, str]:
        """POST /models/unload —— 子进程整个收掉，显存还回去。"""
        if not name:
            return False, "没有指定模型"
        code, body = self._post("/models/unload", {"model": name})
        if code == 200:
            with self._lock:
                self._last_used.pop(name, None)
            return True, "已请求卸载"
        msg = _err_text(body)
        if "not running" in msg:
            return True, "这个模型本来就没在跑"
        return False, msg or ("HTTP %s" % code)

    def unload_all(self) -> int:
        """把所有驻留中的模型都卸掉（路由进程留着）。返回卸载了几个。"""
        n = 0
        for name in self.loaded_names():
            ok, _msg = self.unload(name)
            if ok:
                n += 1
        return n

    def reload_presets(self) -> Tuple[bool, str]:
        """让引擎重新读一遍预置文件（改完模型参数后必须调，否则还是旧参数）。"""
        code, body = self._get("/models?reload=1")
        if code == 200:
            return True, "预置文件已重新读取"
        return False, _err_text(body) or ("HTTP %s" % code)

    def reload_model(self, name: str) -> Tuple[bool, str]:
        """重载一个模型：卸掉再装回来（会重新走一遍加载，参数按新预置生效）。"""
        if not name:
            return False, "没有指定模型"
        ok, msg = self.unload(name)
        if not ok:
            return False, msg
        # 等它真的变成 unloaded，否则 load 会被「已经在跑」挡回来
        for _ in range(60):
            if name not in self.loaded_names():
                break
            time.sleep(0.5)
        ok, msg = self.load(name)
        return ok, msg

    # ----------------------------------------------------------- HTTP 细节
    def _url(self, path: str) -> str:
        with self._lock:
            port = self._port
        return "http://127.0.0.1:%d%s" % (port, path)

    def _get(self, path: str, timeout: float = 8.0) -> Tuple[int, Any]:
        with self._lock:
            port = self._port
        if not port:
            return 0, "后端没在运行"
        try:
            with urllib.request.urlopen(self._url(path), timeout=timeout) as r:
                raw = r.read().decode("utf-8", "replace")
                return r.status, _json(raw)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            return 0, str(exc)

    def _post(self, path: str, payload: Dict[str, Any],
              timeout: float = 30.0) -> Tuple[int, Any]:
        with self._lock:
            port = self._port
        if not port:
            return 0, "后端没在运行"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path), data=data, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, _json(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            return 0, str(exc)

    # ------------------------------------------------------------- 轮询线程
    def _loop(self) -> None:
        misses = 0
        while not self._stop.is_set():
            code, body = self._get("/models")
            if code == 200 and isinstance(body, dict):
                items = []
                for m in body.get("data") or []:
                    st = (m.get("status") or {})
                    items.append({
                        "id": m.get("id", ""),
                        "status": st.get("value", ""),
                        "failed": bool(st.get("failed")),
                        "exit_code": st.get("exit_code"),
                        "aliases": m.get("aliases") or [],
                        "tags": m.get("tags") or [],
                        "modalities": (m.get("architecture") or {}).get(
                            "input_modalities") or [],
                        "source": m.get("source", ""),
                        "can_remove": bool(m.get("can_remove")),
                    })
                with self._lock:
                    self._models = items
                    self._error = ""
                    self._active = True
                misses = 0
                try:
                    self._enforce()
                except Exception as exc:  # noqa: BLE001
                    self._note("驻留策略执行出错：%r" % (exc,), "err")
            else:
                misses += 1
                with self._lock:
                    self._error = _err_text(body) or ("HTTP %s" % code)
                    if misses >= 3:
                        self._active = False
            self._stop.wait(self.interval)

    def _enforce(self) -> None:
        """按类限流 + 空闲卸载。都在轮询线程里做，不碰界面。"""
        pol = self.policy
        with self._lock:
            models = [dict(m) for m in self._models]
            last = dict(self._last_used)
        now = time.time()
        resident = [m for m in models if m.get("status") in RESIDENT]

        # 1) 空闲太久 → 卸
        for m in resident:
            name = str(m.get("id"))
            kind = self.kind_of(name)
            ttl = pol.idle_seconds_for(kind)
            if ttl <= 0:
                continue
            t0 = last.get(name, 0.0)
            if not t0:
                # 刚起来、还没人用过：以「第一次看见它」为起点，免得秒卸
                with self._lock:
                    self._last_used.setdefault(name, now)
                continue
            if now - t0 >= ttl:
                self._note("「%s」已空闲 %d 分钟，自动卸载"
                           % (name, int((now - t0) // 60)), "warn")
                self.unload(name)

        # 2) 超过该类上限 → 顶掉同类里最久没被用过的
        for kind in ("llm", "emb"):
            limit = pol.limit_for(kind)
            if limit is None:
                continue
            group = [m for m in resident if self.kind_of(str(m.get("id")))
                     == kind]
            while len(group) > limit:
                victim = min(group, key=lambda m: last.get(str(m.get("id")),
                                                            0.0) or now)
                vname = str(victim.get("id"))
                if victim.get("status") == "loading":
                    # 正在加载的那个算不上「最久没用」，换下一个
                    others = [m for m in group if m is not victim]
                    if not others:
                        break
                    victim = min(others, key=lambda m: last.get(
                        str(m.get("id")), 0.0) or now)
                    vname = str(victim.get("id"))
                self._note("「%s」类已驻留 %d 个（上限 %d），顶掉最久没用过的「%s」"
                           % ("LLM" if kind == "llm" else "Embedding",
                              len(group), limit, vname), "warn")
                self.unload(vname)
                with self._lock:
                    if vname not in self._evicted:
                        self._evicted.append(vname)
                group = [m for m in group if str(m.get("id")) != vname]

    def _note(self, text: str, tag: str = "sys") -> None:
        if self._log is not None:
            try:
                self._log(text, tag)
            except Exception:  # noqa: BLE001
                pass


def _json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _err_text(body: Any) -> str:
    """把引擎的错误体翻成一句人话。"""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
        if body.get("message"):
            return str(body["message"])
        return ""
    return str(body or "")[:200]
