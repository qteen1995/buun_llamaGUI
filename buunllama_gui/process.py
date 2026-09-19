# -*- coding: utf-8 -*-
"""子进程管理：启动 / 停止 / 实时日志 / 服务健康检查。

日志与状态都通过一个 :class:`queue.Queue` 交给界面主线程消费，
避免在子线程里操作 tkinter 控件。
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------- #
# 引擎 stdout 里「纯噪音」的行：不进运行日志（也不进诊断用的 out_tail，免得把
# 真正有用的报错挤出去）。
#
# 只匹配**引擎固定格式的原文**，不做宽泛匹配 —— 漏掉一条有用的行，比刷屏糟糕得多。
# 目前只有一条：多模型路由每次把请求转发给子模型都打一行（server-models.cpp 的
# `SRV_INF("proxying request to model %s on port %d\n")`），一有请求就成片刷屏：
#     1.45.595.164 I srv  proxy_reques: proxying request to model X on port 2704
# --------------------------------------------------------------------------- #
_NOISE_PATTERNS = (
    re.compile(r"proxy_reques\w*:\s*proxying request to model\b"),
    re.compile(r"\bproxying request to model .+ on port \d+"),
)


def is_noise(text: str) -> bool:
    """这行引擎输出是不是「可以不显示」的噪音。"""
    if not text:
        return False
    return any(rx.search(text) for rx in _NOISE_PATTERNS)

# Windows 进程创建标志
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NEW_CONSOLE = 0x00000010


class LogLine:
    __slots__ = ("text", "stream")

    def __init__(self, text: str, stream: str = "out") -> None:
        self.text = text
        self.stream = stream


class ExitEvent:
    __slots__ = ("code",)

    def __init__(self, code: Optional[int]) -> None:
        self.code = code


class HealthEvent:
    __slots__ = ("ok", "detail")

    def __init__(self, ok: bool, detail: str = "") -> None:
        self.ok = ok
        self.detail = detail


class StartedEvent:
    __slots__ = ("pid", "detached")

    def __init__(self, pid: int, detached: bool) -> None:
        self.pid = pid
        self.detached = detached


class Runner:
    def __init__(self) -> None:
        self.q: "queue.Queue[object]" = queue.Queue()
        self.proc: Optional[subprocess.Popen] = None
        self.detached = False
        self.started_at = 0.0
        self._threads: List[threading.Thread] = []
        self._health_thread: Optional[threading.Thread] = None
        self._health_stop = threading.Event()
        self._health_ok = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------- 状态
    @property
    def running(self) -> bool:
        p = self.proc
        return bool(p) and p.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    @property
    def health_ok(self) -> bool:
        return self._health_ok

    def uptime(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    # ------------------------------------------------------------- 启动
    def start(self, exe: str, argv: Sequence[str], cwd: Optional[str] = None,
              detached: bool = False,
              env_extra: Optional[dict] = None) -> int:
        if self.running:
            raise RuntimeError("已有进程在运行")
        self.detached = bool(detached)
        self._health_ok = False
        self.started_at = time.time()

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # 引擎专属环境（TCQ 码本路径、CUDA 运行时目录）只影响这个子进程
        for k, v in (env_extra or {}).items():
            if k and v:
                env[str(k)] = str(v)

        flags = 0
        if IS_WINDOWS:
            flags = CREATE_NEW_CONSOLE if detached else (
                CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)

        common = dict(cwd=cwd or None, env=env, creationflags=flags)

        if detached:
            self.proc = subprocess.Popen(
                [exe] + list(argv), stdin=None, stdout=None, stderr=None, **common)
            self.q.put(LogLine("已在新控制台窗口中启动（日志请看那个窗口）。", "sys"))
        else:
            self.proc = subprocess.Popen(
                [exe] + list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                **common)
            self._threads = [
                self._make_reader(self.proc.stdout, "out"),
                self._make_reader(self.proc.stderr, "err"),
                self._make_waiter(self.proc),
            ]
            for t in self._threads:
                t.start()

        self.q.put(StartedEvent(self.proc.pid, self.detached))
        return self.proc.pid

    # --------------------------------------------------------- 读取线程
    def _make_reader(self, stream, tag: str) -> threading.Thread:
        def pump() -> None:
            try:
                for line in iter(stream.readline, ""):
                    text = line.rstrip("\r\n")
                    if is_noise(text):
                        continue
                    self.q.put(LogLine(text, tag))
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
        t = threading.Thread(target=pump, name="reader-" + tag, daemon=True)
        return t

    def _make_waiter(self, proc: subprocess.Popen) -> threading.Thread:
        def wait() -> None:
            code = proc.wait()
            self._health_stop.set()
            self.q.put(ExitEvent(code))
        t = threading.Thread(target=wait, name="waiter", daemon=True)
        return t

    # ------------------------------------------------------------- 停止
    def stop(self, timeout: float = 4.0) -> None:
        p = self.proc
        if not p or p.poll() is not None:
            return
        pid = p.pid
        if IS_WINDOWS:
            # 先温和结束整棵进程树，再强杀
            subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True,
                               creationflags=CREATE_NO_WINDOW)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        else:
            p.terminate()
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()

    def force_kill(self) -> None:
        p = self.proc
        if not p or p.poll() is not None:
            return
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        else:
            p.kill()

    # --------------------------------------------------------- 健康检查
    def start_health_probe(self, url: str, interval: float = 1.0,
                           timeout: float = 2.0, max_wait: float = 900.0) -> None:
        self._health_stop.clear()
        self._health_ok = False

        def probe() -> None:
            deadline = time.time() + max_wait
            last_detail = ""
            while not self._health_stop.is_set() and time.time() < deadline:
                if not self.running:
                    return
                try:
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = resp.read(200).decode("utf-8", "replace")
                        if resp.status == 200 and '"ok"' in body.replace(" ", ""):
                            self._health_ok = True
                            self.q.put(HealthEvent(True, "服务已就绪"))
                            return
                        last_detail = "HTTP %s %s" % (resp.status, body[:80])
                except urllib.error.HTTPError as exc:
                    # 503 = 模型还在加载中
                    last_detail = "HTTP %s" % exc.code
                except Exception as exc:  # 连接被拒 = 还没起来
                    last_detail = type(exc).__name__
                self._health_stop.wait(interval)
            if not self._health_stop.is_set():
                self.q.put(HealthEvent(False, "等待超时：%s" % last_detail))
        self._health_thread = threading.Thread(target=probe, name="health",
                                               daemon=True)
        self._health_thread.start()

    def stop_health_probe(self) -> None:
        self._health_stop.set()

    # ------------------------------------------------------------- 使用
    def wait_exit(self, timeout: Optional[float] = None) -> Optional[int]:
        if not self.proc:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def cleanup(self) -> None:
        self._health_stop.set()
        if self.running:
            self.force_kill()
