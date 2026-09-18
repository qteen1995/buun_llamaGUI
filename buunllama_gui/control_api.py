# -*- coding: utf-8 -*-
"""本地控制 API + 网关：用 HTTP 查询状态、列出模型、切换模型、转发推理请求。

同一时刻只有一个 llama-server 在跑（统一后端），对外只暴露一个端口
（默认 127.0.0.1:1233，可在界面上改）。

自己的端点
----------
``GET  /``         简要说明（纯文本）
``GET  /status``   统一后端 + 单次实例的实时状态
``GET  /models``   模型库里的模型，附带分类 / 参数量 / 能力 / 是否驻留
``POST /switch``   {"model":"<路径、文件名或模型名>"}  加载（必要时先切换）模型
``POST /load``     同 switch（显式叫法）
``POST /unload``   卸下当前模型、把显存还回去
``POST /stop``     停止内部 llama-server 进程
``POST /start``    按当前界面参数启动

OpenAI 兼容端点（给第三方软件用）
--------------------------------
``GET /v1/models``          **列出模型库里的全部模型**，``id`` 就是短模型名
``GET /v1/models/{id}``     单个模型
``POST /v1/chat/completions``、``/v1/completions``、``/v1/tokenize``、
``GET /health``、``/props``、``/slots``、``/metrics`` 等 —— 透明转发给内部的
llama-server；请求里的 ``model`` 决定要不要先切模型。

转发时会把请求体里的 ``model`` 从「短模型名」翻译成 llama-server 认的文件名，
所以第三方软件可以只拿到短名字、又能按需加载对的模型。

对需要几分钟的加载动作，可以带 ``?async=1`` 立刻返回，之后用 ``/status`` 轮询。

鉴权：在界面上填了 token 时，请求需要带 ``Authorization: Bearer <token>``
或 ``?token=<token>``。
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .manager import Manager

HELP = """buun-llama-cpp 启动器 · 控制 API

【自己的端点】
GET  /status
GET  /models
POST /switch   {"model":"Qwen3.8-27B-Uncensored-Heretic-v3"}
POST /load     {"model":"Qwen3-30B.gguf"}
POST /unload   卸下当前模型
POST /stop     停止内部 llama-server
POST /start    按当前界面参数启动

【OpenAI 兼容（第三方软件把 base_url 指到这里即可）】
GET  /v1/models              列出模型库里的全部模型（id = 短模型名）
POST /v1/chat/completions    转发给内部的 llama-server
GET  /health  /props  /slots  /metrics

model 字段可以填短模型名、文件名或完整路径，会自动翻译成 llama-server 认的名字。
同一时刻只驻留一个模型；请求另一个模型会自动切过去（可在界面关掉这个行为）。
加载较慢时可加 ?async=1，只用 /status 轮询进度。
"""

# 不往上游转发的请求头 / 响应头
_HOP_HEADERS = frozenset((
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    "accept-encoding", "authorization",
))

# 回给客户端时要丢掉的响应头（content-length 要保留，SSE 才靠它判断）
_RESP_SKIP = frozenset((
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
))

# 为了抓 timings 而缓冲响应体的上限（推理响应都是几 KB，SSE 不缓冲）
_CAPTURE_MAX = 4 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    server_version = "buun-llama-launcher/1.0"

    # ------------------------------------------------------------- 工具
    def log_message(self, fmt: str, *args: Any) -> None:  # 静音
        pass

    @property
    def mgr(self) -> Manager:
        return self.server.manager  # type: ignore[attr-defined]

    def _auth_ok(self, query: Dict[str, Any]) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if not token:
            return True
        got = ""
        head = self.headers.get("Authorization", "")
        if head.lower().startswith("bearer "):
            got = head[7:].strip()
        if not got:
            got = (query.get("token") or [""])[0]
        return got == token

    def _send(self, code: int, payload: Any) -> None:
        body = payload if isinstance(payload, bytes) else (
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            if not isinstance(payload, str) else payload.encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type",
                         "application/json; charset=utf-8"
                         if not isinstance(payload, str)
                         else "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _raw_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _body(self) -> Dict[str, Any]:
        raw = self._raw_body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    # ------------------------------------------------------------- 路由
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._auth_ok(query):
            return self._send(401, {"ok": False, "error": "unauthorized"})
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            return self._send(200, HELP)
        if path == "/status":
            data = self.mgr.status()
            if self.unified is not None:
                data["unified_backend"] = self.unified.status()
                data["service_mode"] = ("router" if self._router_mode()
                                        else "unified")
                if self._router_mode():
                    # 路由模式下模型由 llama-server 自己管，把它的清单也报出来
                    data["router_models"] = self._router_models()
            return self._send(200, {"ok": True, "data": data})
        if path == "/models":
            return self._send(200, self._models(query))
        # 路由模式：模型清单与单模型端点都直接透传，保证 id 和后端一致
        if self.uni_on and self._router_mode() and \
                (path == "/v1/models" or path.startswith("/v1/models/")):
            return self._proxy(path, "GET", query)
        if path == "/v1/models":
            return self._send(200, self._openai_models(query))
        if path.startswith("/v1/models/"):
            one = self._openai_model(path.rsplit("/", 1)[-1])
            code = 200 if one else 404
            return self._send(code, one or {"error": {
                "message": "model not found", "type": "invalid_request_error"}})
        if path == "/health" and self.uni_on:
            # 统一模式下由网关自己答，模型没在跑也算存活（否则前端会以为服务挂了）
            u = self.unified
            return self._send(200, {
                "status": "ok", "gateway": True,
                "model": getattr(u, "model", ""),
                "model_name": os.path.basename(getattr(u, "model", "") or ""),
                "ready": bool(getattr(u, "ready", False)),
                "loading": bool(getattr(u, "loading", False))})
        if self._is_proxy_path(path):
            return self._proxy(path, "GET", query)
        return self._send(404, {"ok": False, "error": "unknown path"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._auth_ok(query):
            return self._send(401, {"ok": False, "error": "unauthorized"})
        path = parsed.path.rstrip("/") or "/"
        raw = self._raw_body()
        if self._is_proxy_path(path):
            return self._proxy(path, "POST", query, raw)
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        body.update({k: v[0] for k, v in query.items() if k != "token"})

        # ---------------- 统一端口模式：加载 / 切换 直接驱动单模型后端
        if self.uni_on and path in ("/switch", "/load", "/models/load",
                                    "/unload", "/models/unload"):
            return self._unified_action(path, body, query)

        if path in ("/switch", "/load"):
            res = self._dispatch("switch", body, query)
        elif path == "/unload":
            res = self._dispatch("unload", body, query)
        elif path == "/stop":
            res = self._dispatch("stop", body, query)
        elif path == "/start":
            res = self._dispatch("start", body, query)
        elif path == "/models/load":
            res = self._dispatch("switch", body, query)
        elif path == "/models/unload":
            res = self._dispatch("unload", body, query)
        else:
            return self._send(404, {"ok": False, "error": "unknown path"})
        code = 200 if res.get("ok") else 400
        return self._send(code, res)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.send_header("Access-Control-Allow-Methods",
                         "GET,POST,OPTIONS")
        self.end_headers()

    # ------------------------------------------------------------- 实现
    def _models(self, query: Dict[str, Any]) -> Dict[str, Any]:
        provider = self.server.models_provider  # type: ignore[attr-defined]
        if not provider:
            return {"ok": True, "data": []}
        try:
            rows = provider()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": repr(exc)}
        cat = (query.get("category") or [""])[0]
        if cat:
            rows = [r for r in rows if r.get("category") == cat]
        brief = []
        for r in rows:
            brief.append({
                "name": r.get("name"),
                "file_name": r.get("file_name"),
                "org": r.get("org"),
                "path": r.get("path"),
                "category": r.get("category"),
                "arch": r.get("arch"),
                "params": r.get("params"),
                "params_text": r.get("params_text"),
                "quant": r.get("quant"),
                "size": r.get("size"),
                "size_text": r.get("size_text"),
                "publisher": r.get("publisher"),
                "has_think": bool(r.get("has_think")),
                "has_vision": bool(r.get("has_vision")),
                "has_tools": bool(r.get("has_tools")),
                "params_set": bool(r.get("params_set")),
                "mmproj": r.get("mmproj"),
            })
        return {"ok": True, "data": brief}

    # ------------------------------------------------- OpenAI 兼容：模型列表
    def _rows(self) -> List[Dict[str, Any]]:
        provider = self.server.models_provider  # type: ignore[attr-defined]
        if not provider:
            return []
        try:
            return list(provider())
        except Exception:  # noqa: BLE001
            return []

    def _live_ids(self) -> set:
        """当前真正驻留着的模型标识（文件名的各种写法都收进来）。"""
        out = set()
        u = self.unified
        val = str(getattr(u, "model", "") or "")
        if val:
            out.add(os.path.basename(val).lower())
            out.add(val.lower())
        return out

    @property
    def unified(self):
        return getattr(self.server, "unified", None)  # type: ignore[attr-defined]

    @property
    def uni_on(self) -> bool:
        u = self.unified
        return bool(u is not None and u.enabled)

    def _openai_entry(self, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": r.get("name"),                 # 短模型名，第三方软件直接用它
            "object": "model",
            "created": int(r.get("mtime") or 0),
            "owned_by": r.get("publisher") or r.get("org") or "local",
            # 下面几个是非标准字段，方便排障与前端展示
            "file_name": r.get("file_name"),
            "path": r.get("path"),
            "category": r.get("category"),
            "architecture": r.get("arch"),
            "quantization": r.get("quant"),
            "params": r.get("params_text"),
            "size": r.get("size"),
            "capabilities": {
                "reasoning": bool(r.get("has_think")),
                "vision": bool(r.get("has_vision")),
                "tools": bool(r.get("has_tools")),
            },
        }

    def _openai_models(self, query: Dict[str, Any]) -> Dict[str, Any]:
        cat = (query.get("category") or [""])[0]
        rows = self._rows()
        if cat:
            want = {"LLM": "LLMS", "LLMS": "LLMS", "Embedding": "Embedding",
                    "Drafter": "Drafters", "Drafters": "Drafters"}.get(
                        cat, cat)
            rows = [r for r in rows if r.get("category") == want]
        live = self._live_ids()
        u = self.unified
        uni_model = str(getattr(u, "model", "") or "")
        data = []
        for r in rows:
            item = self._openai_entry(r)
            item["loaded"] = bool(
                str(r.get("file_name", "")).lower() in live
                or str(r.get("path", "")).lower() in live)
            data.append(item)
        out = {"object": "list", "data": data}
        if u is not None:
            out["active_model"] = uni_model or None
            out["gateway"] = {"port": getattr(u, "port", 0),
                              "ready": bool(getattr(u, "ready", False)),
                              "loading": bool(getattr(u, "loading", False))}
        return out

    def _openai_model(self, ident: str) -> Optional[Dict[str, Any]]:
        r = self._resolve_row(ident)
        if not r:
            return None
        item = self._openai_entry(r)
        live = self._live_ids()
        item["loaded"] = bool(str(r.get("file_name", "")).lower() in live
                              or str(r.get("path", "")).lower() in live)
        return item

    def _resolve_row(self, ident: str) -> Optional[Dict[str, Any]]:
        """短模型名 / 文件名 / 完整路径 都认。"""
        if not ident:
            return None
        want = ident.strip().strip('"')
        low = want.lower()
        rows = self._rows()
        for r in rows:
            if str(r.get("path", "")).lower() == low:
                return r
        for r in rows:
            if str(r.get("file_name", "")).lower() == low:
                return r
        for r in rows:
            if str(r.get("name", "")).lower() == low:
                return r
        for r in rows:                       # 退一步做模糊匹配
            if low in str(r.get("name", "")).lower() \
                    or low in str(r.get("file_name", "")).lower():
                return r
        # 库里没有：如果这就是一个真实存在的 gguf，现场解析一个出来，
        # 这样「还没扫过模型库」也能直接按路径唤起
        try:
            resolver = self.server.row_resolver  # type: ignore[attr-defined]
        except AttributeError:
            resolver = None
        if resolver is not None:
            try:
                extra = resolver(want)
            except Exception:  # noqa: BLE001
                extra = None
            if extra:
                return extra
        return None

    # ------------------------------------------------- OpenAI 兼容：转发
    @staticmethod
    def _is_proxy_path(path: str) -> bool:
        if path.startswith("/v1/"):
            return True
        return path in ("/health", "/props", "/slots", "/metrics", "/models/x")

    def _proxy(self, path: str, method: str, query: Dict[str, Any],
               raw: bytes = b"") -> None:
        """所有推理请求都走同一个后端：先确保目标模型在跑，再转发。

        **多模型路由方式下这里退化成纯透传**：llama-server 自己按请求里的
        model 名驻留/路由模型，网关既不该抢锁换模型，也不该翻译模型名
        （路由的模型 id 来自预置文件 / 模型目录，跟我们库里的短名不是一套）。
        """
        u = self.unified
        if u is None:
            return self._send(503, {"error": {
                "message": "统一后端还没准备好，等界面初始化完成后再试。",
                "type": "service_unavailable"}})

        if self._router_mode():
            port = int(getattr(u, "port", 0) or 0)
            if not port:
                return self._send(503, {"error": {
                    "message": "多模型路由的后端没有在运行。"
                               "请在界面「服务」页启动，或先 POST /switch。",
                    "type": "service_unavailable"}})
            u.touch()
            if method == "POST" and raw:
                # 顶层 reasoning_effort 还是要翻译成模板变量，
                # llama-server 会静默丢掉顶层字段
                raw = self._inject_effort(raw)
            return self._forward(path, method, raw, port)

        row = None
        if method == "POST" and raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = {}
            if isinstance(body, dict) and body.get("model"):
                row = self._resolve_row(str(body["model"]))
                if row is None:
                    return self._send(404, {"error": {
                        "message": "模型库里没有「%s」。"
                                   "用 GET /v1/models 看可用模型。"
                                   % body["model"],
                        "type": "invalid_request_error"}})
        if row is None and not getattr(u, "model", ""):
            # 请求里没写 model、当前也没加载任何模型 → 用默认模型
            row = self._resolve_row(str(query.get("model", [""])[0])) \
                or self._default_row()
        if row is not None:
            want = str(row.get("path") or "")
            if not self._auto_switch():
                if getattr(u, "model", "") and \
                        str(u.model).lower() != want.lower():
                    return self._send(409, {"error": {
                        "message": "「按请求自动加载/切换」已关闭，当前只服务 "
                                   "%s。要换成 %s，请先 POST /switch。"
                                   % (getattr(u, "model_name", ""),
                                      row.get("name")),
                        "type": "conflict",
                        "loaded": getattr(u, "model", ""),
                        "requested": want}})
            ok, msg = u.ensure(want)
            if not ok:
                return self._send(503, {"error": {
                    "message": "加载 %s 失败：%s" % (row.get("name"), msg),
                    "type": "service_unavailable", "model": want}})
            u.touch()
            # 后端只认文件名/路径，把请求里的短名翻译过去
            if method == "POST" and raw:
                raw = self._rewrite_to(raw, row)
                raw = self._inject_effort(raw)
        backend_port = int(getattr(u, "port", 0) or 0)
        if not backend_port:
            return self._send(503, {"error": {
                "message": "统一后端没有在运行，先在界面启动，或 POST /switch "
                           '{"model":"<模型名>"} 让它按需加载。',
                "type": "service_unavailable"}})
        return self._forward(path, method, raw, backend_port)

    def _auto_switch(self) -> bool:
        try:
            return bool(self.server.auto_switch())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return True

    def _router_mode(self) -> bool:
        """当前服务是不是「多模型路由」（是的话 /v1/* 一律纯透传）。"""
        try:
            return bool(self.server.router_mode())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return False

    def _router_models(self) -> List[Dict[str, Any]]:
        """问后端路由要它当前认识的模型清单（失败返回空表）。"""
        port = int(getattr(self.unified, "port", 0) or 0) if self.unified else 0
        if not port:
            return []
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/v1/models" % port, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return []
        items = data.get("data") if isinstance(data, dict) else None
        out = []
        for it in (items or []):
            if isinstance(it, dict):
                out.append({"id": it.get("id"),
                            "status": (it.get("status") or {}).get("value")
                            if isinstance(it.get("status"), dict) else ""})
        return out

    def _default_row(self) -> Optional[Dict[str, Any]]:
        """请求没写 model、当前也没加载模型时，用界面上指定的默认模型。"""
        for call in (lambda: self.server.default_model(),      # 新签名
                     lambda: self.server.default_model("llm")):  # 兼容老签名
            try:
                path = str(call() or "")
            except Exception:  # noqa: BLE001
                continue
            if path:
                return self._resolve_row(path)
        return None

    @staticmethod
    def _inject_effort(raw: bytes) -> bytes:
        """请求体里的顶层 ``reasoning_effort`` —— 本 build **原生支持**，原样放行。

        ik_llama.cpp 那边这个顶层字段会被 llama-server 静默丢掉，所以那时候
        必须把它搬进 ``chat_template_kwargs``。buun-llama-cpp 的
        ``server-common.cpp`` 里有专门一段解析 ``reasoning_effort``
        （"none" 表示关掉推理，其余值塞进模板变量），所以这里什么都不做，
        免得再叠一层反而把 "none" 的语义搞乱。
        """
        return raw

    @staticmethod
    def _rewrite_to(raw: bytes, row: Dict[str, Any]) -> bytes:
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return raw
        if isinstance(body, dict) and "model" in body:
            body["model"] = row.get("file_name") or row.get("path")
            return json.dumps(body, ensure_ascii=False).encode("utf-8")
        return raw

    def _forward(self, path: str, method: str, raw: bytes,
                 port: Optional[int]) -> None:
        if not port:
            return self._send(503, {"error": {
                "message": "内部后端端口未知", "type": "service_unavailable"}})
        headers = {"Content-Type": self.headers.get("Content-Type")
                   or "application/json"}
        for key, val in self.headers.items():
            lk = key.lower()
            if lk in _HOP_HEADERS or lk in ("content-type",):
                continue
            headers[key] = val
        api_key = ""
        try:
            api_key = str(self.server.upstream_key()  # type: ignore[attr-defined]
                          or "")
        except Exception:  # noqa: BLE001
            api_key = ""
        if api_key:
            headers["Authorization"] = "Bearer " + api_key

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        try:
            conn.request(method, path, body=raw if raw else None,
                         headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            return self._send(502, {"error": {
                "message": "转发到内部后端失败：%s" % exc,
                "type": "bad_gateway"}})

        # 顺手抓一次响应里的 timings / usage 喂给运行状态面板。
        # 只对「JSON 响应体 + POST」做，SSE 流式响应绝不缓冲。
        capture = (method == "POST")
        buf: List[bytes] = []
        total = 0
        try:
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() in _RESP_SKIP:
                    continue
                self.send_header(key, val)
                if key.lower() == "content-type" and \
                        "json" not in (val or "").lower():
                    capture = False
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if capture:
                    total += len(chunk)
                    if total <= _CAPTURE_MAX:
                        buf.append(chunk)
                    else:
                        capture = False      # 太大就不解析了，避免吃内存
                        buf = []
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        if capture and buf:
            self._note_runtime(b"".join(buf))

    def _note_runtime(self, body: bytes) -> None:
        """把后端返回的 JSON 里的 timings / usage 回填给运行状态。"""
        rt = getattr(self.server, "runtime", None)
        if rt is None:
            return
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(data, dict):
            return
        try:
            rt.note_timings(data.get("timings"), data.get("usage"))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------- 统一模式的动作端点
    def _unified_action(self, path: str, body: Dict[str, Any],
                        query: Dict[str, Any]) -> None:
        """统一端口模式下：/switch /load 会真正把模型加载好再返回。"""
        u = self.unified
        async_ = (query.get("async") or body.pop("async", ""))
        async_ = str(async_).lower() in ("1", "true", "yes")

        if path in ("/unload", "/models/unload"):
            u.unload("API 请求")
            return self._send(200, {"ok": True, "note": "已卸载当前模型",
                                    "state": u.status()})

        want = str(body.get("model") or "").strip()
        row = self._resolve_row(want)
        if row is None:
            return self._send(404, {
                "ok": False,
                "error": "模型库里找不到「%s」；用 GET /v1/models 看可用模型"
                         % want})
        target = str(row.get("path") or "")

        if async_:
            threading.Thread(target=u.ensure, args=(target,),
                             name="api-load", daemon=True).start()
            return self._send(202, {
                "ok": True, "accepted": True, "model": target,
                "name": row.get("name"),
                "note": "已开始加载，用 GET /status 或 GET /health 查询进度"})

        ok, msg = u.ensure(target)
        payload = {"ok": ok, "model": target, "name": row.get("name"),
                   "note": msg, "state": u.status()}
        return self._send(200 if ok else 503, payload)

    def _dispatch(self, name: str, body: Dict[str, Any],
                  query: Dict[str, Any]) -> Dict[str, Any]:
        if (query.get("async") or [""])[0] in ("1", "true", "yes"):
            self.mgr.bus.submit(name, body, timeout=5.0)  # 不等结果
            return {"ok": True, "accepted": True,
                    "note": "已受理，请用 GET /status 轮询进展"}
        return self.mgr.bus.submit(name, body, timeout=1800.0)


class ControlAPI:
    def __init__(self, manager: Manager, token: str = "",
                 models_provider=None, upstream_key=None, unified=None,
                 auto_switch=None, default_model=None,
                 row_resolver=None, runtime=None) -> None:
        self.manager = manager
        self.token = token
        self.models_provider = models_provider
        self.upstream_key = upstream_key or (lambda: "")
        self.unified = unified
        self.auto_switch = auto_switch or (lambda: True)
        self.default_model = default_model or (lambda: "")
        self.row_resolver = row_resolver
        # 运行时状态采集（可选）：转发时顺手把 timings 喂给它
        self.runtime = runtime
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.host = ""
        self.port = 0

    @property
    def running(self) -> bool:
        return self.httpd is not None

    def start(self, host: str = "127.0.0.1", port: int = 8090) -> Tuple[bool, str]:
        self.stop()
        host = (host or "127.0.0.1").strip()
        if host in ("0.0.0.0", "::"):
            bind = ""          # 绑定全部网卡
        else:
            bind = host
        try:
            httpd = ThreadingHTTPServer((bind, int(port)), _Handler)
        except OSError as exc:
            return False, "无法监听 %s:%s（%s）" % (host, port, exc)
        httpd.daemon_threads = True
        httpd.manager = self.manager                     # type: ignore[attr-defined]
        httpd.token = self.token                         # type: ignore[attr-defined]
        httpd.models_provider = self.models_provider     # type: ignore[attr-defined]
        httpd.upstream_key = self.upstream_key           # type: ignore[attr-defined]
        httpd.unified = self.unified                     # type: ignore[attr-defined]
        httpd.auto_switch = self.auto_switch             # type: ignore[attr-defined]
        httpd.default_model = self.default_model         # type: ignore[attr-defined]
        httpd.row_resolver = self.row_resolver           # type: ignore[attr-defined]
        httpd.runtime = self.runtime                     # type: ignore[attr-defined]
        try:
            bound = int(httpd.server_address[1])
        except (IndexError, TypeError, ValueError):
            bound = int(port)
        self.httpd, self.host, self.port = httpd, host, bound
        self.thread = threading.Thread(target=httpd.serve_forever,
                                       name="control-api", daemon=True)
        self.thread.start()
        return True, ""

    def stop(self) -> None:
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        self.httpd = None
        self.thread = None
