# -*- coding: utf-8 -*-
"""运行实例管理：一个统一服务实例（unified）+ 一个单次运行实例（single）。

关键约束：**同一时间只有一个 llama-server 在跑，也就只驻留一个模型。**
程序只服务 LLM 大模型的推理；Embedding 角色与「两个模型同时在线」都已移除。

切换模型只有一条路径：停掉进程、换模型重启。调度（谁在等、要不要切）
由 ``unified.py`` 负责，这里只管进程的生死。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import builder as B
from . import schema as S
from .process import Runner

ROLE_KEYS = ("llm",)          # 只有一个角色：专注跑 LLM
ALL_KEYS = ROLE_KEYS + ("single", "unified")

# 展示用的名字
KEY_ZH = {"llm": "LLM", "single": "单次运行", "unified": "统一后端"}


class ManagerEvent:
    def __init__(self, kind: str, ok: bool, detail: str = "",
                 role: str = "", data: Any = None) -> None:
        self.kind = kind          # switch | note
        self.ok = ok
        self.detail = detail
        self.role = role
        self.data = data


class CommandRequest:
    __slots__ = ("name", "payload", "event", "result", "id")

    def __init__(self, rid: int, name: str, payload: Dict[str, Any]) -> None:
        self.id = rid
        self.name = name
        self.payload = payload
        self.event = threading.Event()
        self.result: Dict[str, Any] = {"ok": False,
                                       "error": "未执行（主线程未响应）"}


class CommandBus:
    """控制 API 线程 ↔ tkinter 主线程 的桥。

    HTTP 处理器把请求扔进来并等待结果；主线程每 120ms 取一次并执行。
    这样进程启停始终发生在主线程，不会跨线程碰控件。
    """

    def __init__(self) -> None:
        self.q: "queue.Queue[CommandRequest]" = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()

    def submit(self, name: str, payload: Dict[str, Any],
               timeout: float = 600.0) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            rid = self._seq
        req = CommandRequest(rid, name, payload or {})
        self.q.put(req)
        if not req.event.wait(timeout):
            return {"ok": False, "error": "等待主线程执行超时"}
        return req.result

    def post(self, name: str, payload: Dict[str, Any]) -> None:
        """提交但**不等**结果（fire-and-forget）。

        给「顺手把后端拉起来」这类动作：HTTP 线程不能为了一个后台启停
        一直挂着；调用方自己轮询端口出现即可。
        """
        with self._lock:
            self._seq += 1
            rid = self._seq
        self.q.put(CommandRequest(rid, name, payload or {}))

    def drain(self, handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
              limit: int = 8) -> None:
        for _ in range(limit):
            try:
                req = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                req.result = handler(req.name, req.payload) or {"ok": True}
            except Exception as exc:  # noqa: BLE001
                req.result = {"ok": False, "error": repr(exc)}
            finally:
                req.event.set()


class RoleState:
    """一个运行实例的元信息（谁在跑、跑的是哪个模型、什么参数）。"""

    def __init__(self, key: str) -> None:
        self.key = key
        self.runner = Runner()
        self.exe = ""
        self.argv: List[str] = []
        self.cwd = ""
        self.model = ""
        self.detached = False
        self.port = 0
        self.started_at = 0.0
        self.loaded_model = ""     # 路由模式下从 /models 读到的实际驻留模型
        self.test_running = False  # 仅自检用：不真起进程也能验证网关转发

    @property
    def running(self) -> bool:
        return self.test_running or self.runner.running

    def base_url(self, host: str = "127.0.0.1") -> str:
        return "http://%s:%d" % (host or "127.0.0.1", self.port or 8080)


class Manager:
    def __init__(self) -> None:
        self.roles: Dict[str, RoleState] = {k: RoleState(k) for k in ALL_KEYS}
        self.q: "queue.Queue[ManagerEvent]" = queue.Queue()
        self.bus = CommandBus()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 便捷
    def get(self, key: str) -> RoleState:
        return self.roles[key]

    def any_running(self) -> bool:
        return any(r.running for r in self.roles.values())

    def cleanup(self) -> None:
        for r in self.roles.values():
            try:
                r.runner.cleanup()
            except Exception:
                pass

    def stop_all(self) -> None:
        for r in self.roles.values():
            if r.running:
                r.runner.stop()

    # ------------------------------------------------------------ 启动
    def start(self, key: str, exe: str, argv: List[str], cwd: str = "",
              model: str = "", port: int = 0,
              detached: bool = False,
              health_url: str = "",
              env_extra: Optional[Dict[str, str]] = None) -> int:
        st = self.roles[key]
        if st.running:
            raise RuntimeError("%s 实例已经在运行" % key)
        st.exe, st.argv, st.cwd = exe, list(argv), cwd
        st.model, st.port = model, port
        st.detached = detached
        st.started_at = time.time()
        st.loaded_model = model
        pid = st.runner.start(exe, argv, cwd, detached=detached,
                              env_extra=env_extra)
        if health_url and not detached:
            st.runner.start_health_probe(health_url)
        return pid

    def stop(self, key: str) -> None:
        st = self.roles[key]
        st.runner.stop_health_probe()
        st.runner.stop()
        st.loaded_model = ""
        # 端口也要清掉：UnifiedBackend.port 就是读这里的（status_fn），
        # 留着旧端口会让网关以为后端还活着 —— 于是把请求转发到一个已经
        # 关掉的端口，客户端收到 502 而不是「后端没在运行」。
        st.port = 0

    # --------------------------------------------------------- 状态汇总
    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"roles": {}, "single": {}, "unified": {}}
        for key, st in self.roles.items():
            item = {
                "running": st.running,
                "pid": st.runner.pid,
                "exe": st.exe,
                "model": st.model,
                "model_name": os.path.basename(st.model) if st.model else "",
                "loaded_model": st.loaded_model,
                "port": st.port,
                "url": st.base_url() if st.running else "",
                "uptime": round(st.runner.uptime(), 1),
                "health": bool(st.runner.health_ok),
            }
            if key == "single":
                out["single"] = item
            elif key == "unified":
                out["unified"] = item
            else:
                out["roles"][key] = item
        return out


def merge_model_snapshot(store, path: str,
                         defaults: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """模型的完整参数 = 默认值 ← 该模型存的加载/对话参数。"""
    snap = defaults if defaults is not None else S.default_snapshot()
    for scope in ("load", "chat", "emb"):
        saved = store.state(path, scope)
        if not saved:
            continue
        for k, v in saved.items():
            if k in snap and isinstance(v, dict):
                snap[k] = {"on": bool(v.get("on")),
                           "value": v.get("value", "")}
    return snap
