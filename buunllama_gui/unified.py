# -*- coding: utf-8 -*-
"""统一后端：一个端口、一个 llama-server、同一时刻只跑一个模型。

为什么不是"两个模型一起跑"
--------------------------
llama-server 一个进程只能加载一个模型（多模态的 mmproj 不算第二个模型）。
所以"所有接口统一成一个端口，且只运行一个模型"这件事，正确做法是：
本启动器做网关，背后只维护**一个** llama-server 进程，
按请求里 ``model`` 字段决定该加载哪个模型，需要时切换。

切换与排队
----------
* 请求的目标模型 == 当前已加载的模型 → 立即转发（llama-server 自己用 -np 个槽位并发）
* 否则 → 抢到 ``lock`` 的那一个请求负责「卸载旧的 → 加载新的 → 等就绪」，
  期间其它请求**排队等这把锁**；锁释放后各自再判断一次自己的目标模型。
  于是一批混合模型的请求会按「同模型的连续处理」自动分组，而不是互相打断。

线程模型
--------
``ensure()`` 会在网关的 HTTP 线程里阻塞（可能几分钟）。它需要主线程做两件
**很短**的事：起进程、停进程 —— 这两件通过 ``CommandBus`` 提交给主线程执行，
真正的"等就绪"在网关线程里轮询 /health，不占主线程。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

READY_POLL = 1.0          # 轮询 /health 的间隔（秒）
START_TIMEOUT = 30.0      # 交给主线程「起进程」这一步的等待上限


class UnifiedBackend:
    """统一端口模式下的模型调度器。"""

    def __init__(self, mgr, bus,
                 status_fn: Callable[[], Dict[str, Any]],
                 idle_minutes_fn: Callable[[], int],
                 enabled_fn: Callable[[], bool],
                 log_fn: Optional[Callable[[str, str], None]] = None) -> None:
        self.mgr = mgr
        self.bus = bus
        self.status_fn = status_fn
        self.idle_minutes_fn = idle_minutes_fn
        self.enabled_fn = enabled_fn
        self.log_fn = log_fn or (lambda msg, tag="": None)
        self.lock = threading.Lock()
        self.model = ""            # 当前（或正在加载的）模型路径
        self.ready = False
        self.loading = False
        self.error = ""
        self.last_used = time.time()
        self.started_at = 0.0
        self.switches = 0
        self.waiters = 0           # 正在排队的请求数（仅用于展示）
        self._idle_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- 状态
    @property
    def enabled(self) -> bool:
        try:
            return bool(self.enabled_fn())
        except Exception:  # noqa: BLE001
            return False

    @property
    def port(self) -> int:
        try:
            return int(self.status_fn().get("port") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def running(self) -> bool:
        try:
            return bool(self.status_fn().get("running"))
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "model_name": self.model.rsplit("\\", 1)[-1] if self.model else "",
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "port": self.port,
            "running": self.running(),
            "idle_seconds": round(time.time() - self.last_used, 1),
            "idle_unload_minutes": self.idle_minutes_fn(),
            "switches": self.switches,
            "queued": self.waiters,
        }

    def clear(self, message: str = "") -> None:
        self.model = ""
        self.ready = False
        self.loading = False
        self.error = message
        self.started_at = 0.0

    def touch(self) -> None:
        self.last_used = time.time()

    # ------------------------------------------------------------- 唤起
    def ensure(self, path: str, timeout: float = 1800.0) -> Tuple[bool, str]:
        """确保 ``path`` 这个模型正在运行。需要时自动切换。"""
        if not path:
            return False, "没有指定模型"
        self.waiters += 1
        try:
            with self.lock:
                self.waiters = max(0, self.waiters - 1)
                self.touch()
                if self.model == path and self.running() and self.ready:
                    return True, "已在运行"
                if self.model == path and self.running() and not self.ready:
                    # 上一次加载还没完成（或中途失败），等它就绪
                    ok, msg = self._wait_ready(timeout)
                    self.ready = ok
                    return ok, msg
                # 需要切换
                if self.running():
                    self.log_fn("正在卸载 %s，准备切换到 %s"
                                % (self._short(self.model), self._short(path)),
                                "sys")
                    self._stop_backend()
                self.switches += 1
                ok, msg = self._start_backend(path)
                if not ok:
                    self.error = msg
                    self.clear(msg)
                    return False, msg
                self.model = path
                self.loading = True
                self.ready = False
                self.started_at = time.time()
                ok, msg = self._wait_ready(timeout)
                self.loading = False
                self.ready = ok
                if not ok:
                    self.error = msg
                    self._stop_backend()
                    self.clear(msg)
                    return False, msg
                self.error = ""
                self.log_fn("%s 已就绪（用时 %.1fs）"
                            % (self._short(path), time.time() - self.started_at),
                            "ok")
                return True, "已加载"
        finally:
            self.waiters = max(0, self.waiters - 1)

    def unload(self, reason: str = "") -> None:
        with self.lock:
            if self.running():
                self._stop_backend()
            self.clear("")
            self.log_fn("已卸载模型%s" % ("（%s）" % reason if reason else ""),
                        "sys")

    def mark_stopped(self) -> None:
        self.clear("")

    # --------------------------------------------------------- 内部实现
    @staticmethod
    def _short(path: str) -> str:
        return path.rsplit("\\", 1)[-1] if path else ""

    def _start_backend(self, path: str) -> Tuple[bool, str]:
        try:
            res = self.bus.submit("start_unified", {"model": path},
                                  timeout=START_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            return False, "启动请求失败：%r" % (exc,)
        if not res.get("ok"):
            return False, str(res.get("error") or res.get("note") or "启动失败")
        return True, ""

    def _stop_backend(self) -> None:
        try:
            self.bus.submit("stop_unified", {}, timeout=START_TIMEOUT)
        except Exception:  # noqa: BLE001
            pass
        # 进程退出需要一点时间，等它真的停下来再接下一个模型
        deadline = time.time() + 20
        while time.time() < deadline and self.running():
            time.sleep(0.2)

    def _wait_ready(self, timeout: float) -> Tuple[bool, str]:
        """轮询 /health 直到就绪。"""
        port = self.port
        if not port:
            return False, "内部端口未知"
        url = "http://127.0.0.1:%d/health" % port
        deadline = time.time() + max(30.0, timeout)
        last = ""
        while time.time() < deadline:
            if not self.running():
                return False, "进程已退出，请看运行日志里的报错"
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    body = resp.read().decode("utf-8", "replace")
                try:
                    data = json.loads(body)
                    if isinstance(data, dict) and \
                            str(data.get("status", "")).lower() in ("ok", "loading"):
                        if str(data.get("status")).lower() == "ok":
                            return True, ""
                except ValueError:
                    return True, ""
            except urllib.error.HTTPError as exc:
                if exc.code == 503:
                    last = "服务还在加载模型"
                else:
                    last = "HTTP %s" % exc.code
            except Exception as exc:  # noqa: BLE001
                last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(READY_POLL)
        return False, "等待模型就绪超时（%s）" % (last or "无响应")

    # --------------------------------------------------------- 空闲卸载
    def start_idle_watch(self) -> None:
        if self._idle_thread and self._idle_thread.is_alive():
            return

        def loop() -> None:
            while True:
                time.sleep(15)
                try:
                    minutes = int(self.idle_minutes_fn() or 0)
                except Exception:  # noqa: BLE001
                    minutes = 0
                if minutes <= 0 or not self.model or not self.running():
                    continue
                if self.lock.locked():
                    continue
                if time.time() - self.last_used >= minutes * 60:
                    self.log_fn("空闲超过 %d 分钟，自动卸载 %s 以释放显存"
                                % (minutes, self._short(self.model)), "warn")
                    try:
                        self.unload("空闲超时")
                    except Exception:  # noqa: BLE001
                        pass

        self._idle_thread = threading.Thread(target=loop, name="idle-unload",
                                             daemon=True)
        self._idle_thread.start()
