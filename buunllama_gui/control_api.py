# -*- coding: utf-8 -*-
"""本地控制 API + 网关：用 HTTP 查询状态、列出模型、装卸模型、转发推理请求。

**服务只有「多模型路由」一种跑法**：一个 llama-server 进程（绑内部端口）
下面挂 N 个模型子进程，各自独立加载 / 卸载；对外只暴露一个端口
（默认 127.0.0.1:1233，可在界面上改）。所以本网关对 ``/v1/*`` 基本是**纯透传**
（模型清单与装卸除外）。

自己的端点
----------
``GET  /``                简要说明（纯文本）
``GET  /status``          后端 + 路由模型清单的实时状态
``GET  /models``          模型库里的模型（分类 / 参数量 / 能力 / 是否驻留）
``GET  /v1/models``       同上，OpenAI 格式；``id`` = 路由里的段名（= 显示名），
                          额外给 ``aliases``（文件名等写法，写哪个都能调用）
``POST /models/load``     {"model":"<段名 / 文件名 / 路径>"} 加载一个模型
``POST /models/unload``   同上；不带 model 就卸载全部
``POST /switch`` ``/load`` ``/unload``   老名字，等价于上面两个
``POST /stop`` ``/start`` 停止 / 启动内部 llama-server 进程

OpenAI 兼容端点（给第三方软件用）
--------------------------------
``POST /v1/chat/completions``、``/v1/completions``、``/v1/embeddings``、
``/v1/tokenize``、``GET /health``、``/props``、``/slots``、``/metrics`` 等 ——
直接转发给内部的 llama-server 路由；请求里的 ``model`` 就是它的路由 key。

模型名会被**归一化**成路由里的正式 id：段名、文件名（带/不带 .gguf）都能用。
认不出的名字由网关回 404 并附上候选名单（引擎只会回一句 not found）。

鉴权：在界面上填了 token 时，请求需要带 ``Authorization: Bearer <token>``
或 ``?token=<token>``。
"""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .manager import Manager

HELP = """buun-llama-cpp 启动器 · 控制 API

【服务方式】一个 llama-server 进程下挂多个模型，各自加载 / 卸载

【自己的端点】
GET  /status                 后端 + 路由模型清单
GET  /models                 模型库里的模型
POST /models/load   {"model":"Qwen3.8-27B-Uncensored-Heretic-v3"}
POST /models/unload {"model":"..."}   不带 model 就卸载全部
POST /stop /start            启停内部 llama-server

【OpenAI 兼容（第三方软件把 base_url 指到这里即可）】
GET  /v1/models              模型列表（id = 段名，另附 aliases）
POST /v1/chat/completions    推理（model 写段名或文件名都行）
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

    # 「请求来了但路由没起来」时自动拉起后端的节流时间戳。
    # 必须放类属性：每个请求都会新建一个 handler 实例，实例属性拦不住重复尝试。
    _start_try_at = 0.0

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
        # ⚠️ /v1/models 与 /v1/models/<名字> 一律回**本程序**这份，别透传：
        #    · 我们这份带 file_name / 参数量 / 能力 / 是否驻留，信息更全，
        #      而且 id 就是路由里的段名；
        #    · 实测引擎在路由模式下**根本没有** GET /v1/models/<id> ——
        #      透传过去只会拿到一句 404 File Not Found（比我们自己按
        #      文件名/显示名/路径模糊匹配差得多）。
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

        # ---------------- 加载 / 卸载：转成路由的 /models/load|unload
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

    def _resident_ids(self) -> set:
        """路由里**真的驻留着**的模型 id。

        路由模式下 ``/v1/models`` 的 ``loaded`` 必须看这里 —— 单个的
        ``unified.model`` 只是「目标模型」，它可能压根没加载。
        """
        out = set()
        for m in self._router_snapshot_models():
            if str(m.get("status") or "") in ("loading", "loaded", "sleeping"):
                out.add(str(m.get("id") or ""))
        return out

    @property
    def unified(self):
        return getattr(self.server, "unified", None)  # type: ignore[attr-defined]

    @property
    def uni_on(self) -> bool:
        u = self.unified
        return bool(u is not None and u.enabled)

    def _openai_entry(self, r: Dict[str, Any]) -> Dict[str, Any]:
        fname = str(r.get("file_name") or "")
        stem = os.path.splitext(fname)[0]
        name = str(r.get("name") or "")
        aliases = [a for a in (stem, fname)
                   if a and a != name and "," not in a]
        return {
            "id": name,                          # 段名 = 模型库里的显示名
            "object": "model",
            "created": int(r.get("mtime") or 0),
            "owned_by": r.get("publisher") or r.get("org") or "local",
            # 别名：请求里写这些名字同样有效（见 builder.preset_aliases）
            "aliases": aliases,
            # 下面几个是非标准字段，方便排障与前端展示
            "file_name": r.get("file_name"),
            "path": r.get("path"),
            "category": r.get("category"),
            "architecture": r.get("arch"),
            "quantization": r.get("quant"),
            "params": r.get("params_text"),
            "size": r.get("size"),
            "capabilities": {
                # 界面「能力」列只看这两项（用户要求）
                "vision": bool(r.get("has_vision")),
                "mtp": bool(r.get("has_mtp")),
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
        resident = self._resident_ids()
        u = self.unified
        uni_model = str(getattr(u, "model", "") or "")
        data = []
        for r in rows:
            item = self._openai_entry(r)
            item["loaded"] = bool(
                str(r.get("name") or "") in resident
                or str(r.get("file_name", "")).lower() in live
                or str(r.get("path", "")).lower() in live)
            data.append(item)
        # 路由里那些**不在模型库**的模型（HF 缓存自动发现的）也报出来 ——
        # 否则客户端看得见它、却因为列表里没有而不知道怎么调。
        seen = {str(m.get("id")) for m in data}
        for m in self._router_snapshot_models():
            mid = str(m.get("id") or "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            data.append({
                "id": mid, "object": "model", "created": 0,
                "owned_by": "router",
                "aliases": list(m.get("aliases") or []),
                "source": m.get("source") or "",
                "file_name": None, "path": None,
                "loaded": str(m.get("status") or "") in
                          ("loading", "loaded", "sleeping"),
            })
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
        item["loaded"] = bool(
            str(r.get("name") or "") in self._resident_ids()
            or str(r.get("file_name", "")).lower() in live
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

    def _router(self) -> Any:
        """路由监控（没接线时返回 None）。"""
        return getattr(self.server, "router", None)

    def _touch_router_model(self, name: str) -> None:
        """记下「这个模型刚刚被用过」，供 RouterMonitor 的空闲卸载计时。

        ⚠️ 引擎的 ``GET /models`` **不返回 last_used**，唯一知道「谁刚被用」的
        地方就是这里 —— 每转发一次就记一笔。
        """
        if not name:
            return
        rt = self._router()
        if rt is None:
            return
        try:
            rt.touch(name)
        except Exception:  # noqa: BLE001
            pass

    def _known_model_names(self) -> List[str]:
        """路由当前认识的模型名（id + 别名）。"""
        rt = self._router()
        if rt is None:
            return []
        try:
            return list(rt.all_names() or [])
        except Exception:  # noqa: BLE001
            return []

    def _router_id_for(self, want: str) -> str:
        """客户端写的模型名 → 路由里的正式 id（认不出来返回空串）。

        客户端手里的名字可能来自好几处：本程序 ``GET /v1/models`` 给的是**段名**
        （= 模型库里的显示名），第三方软件常常直接用**文件名**，用户手抄也会带
        ``.gguf``，还有的客户端自己截断。这三种写法这里都认：

          1. 路由已经认识的 id / 别名 —— 原样放行（引擎自己就能解析，
             包括 HF 缓存里那些 ``repo/名字:量化`` 形式）；
          2. 模型库里的显示名 / 文件名 / 去掉扩展名的文件名 —— 翻成段名；
          3. 认不出来 → 空串，调用方回 404 并告诉用户可选名字。
        """
        nm = str(want or "").strip()
        if not nm:
            return ""
        # 别名 → 正式 id 的反查表（HF 缓存那种 repo/名字:量化 的别名也在里面）
        alias2id: Dict[str, str] = {}
        ids: List[str] = []
        for m in self._router_snapshot_models():
            mid = str(m.get("id") or "")
            if not mid:
                continue
            ids.append(mid)
            alias2id.setdefault(mid.lower(), mid)
            for al in (m.get("aliases") or []):
                if al:
                    alias2id.setdefault(str(al).lower(), mid)
        low_ids = {i.lower(): i for i in ids}

        # ① 先按模型库认（显示名 / 文件名 / 去掉扩展名的文件名）
        #
        # ⚠️ 顺序很要紧：**必须**先翻成正式 id 再转发。引擎虽然也能按别名解析，
        #    但我们自己的记账（last_used / 空闲卸载 / 按类限流）是按名字做的 ——
        #    客户端这次写文件名、下次写段名的话，同一模型会被记成两个「模型」，
        #    于是「刚被用过」的那个还可能被当成空闲卸掉。
        base = os.path.basename(nm.replace("\\", "/"))
        stem = os.path.splitext(base)[0]
        cand = {nm.lower(), base.lower(), stem.lower()}
        try:
            rows = self.server.models_provider() or []   # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            rows = []
        for r in rows:
            names = {str(r.get("name") or "").lower(),
                     str(r.get("file_name") or "").lower(),
                     os.path.splitext(str(r.get("file_name") or ""))[0].lower()}
            if not (cand & names):
                continue
            rid = alias2id.get(str(r.get("name") or "").lower()) \
                or low_ids.get(str(r.get("name") or "").lower()) \
                or str(r.get("name") or "")
            return rid
        # ② 路由自己认识的 id / 别名（模型库里没有的，比如 HF 缓存）
        return alias2id.get(nm.lower(), "")

    def _router_snapshot_models(self) -> List[Dict[str, Any]]:
        """路由当前认识的模型（含 HF 缓存里那些）。拿不到就给空表。"""
        rt = self._router()
        if rt is None:
            return []
        try:
            return list(rt.snapshot().get("models") or [])
        except Exception:  # noqa: BLE001
            return []

    def _router_rewrite(self, path: str, method: str, query: Dict[str, Any],
                        raw: bytes) -> Tuple[str, bytes, Optional[str]]:
        """把请求里的模型名换成路由的正式 id。

        返回 ``(path, raw, rid)``。``rid is None`` = 用户确实写了模型名，但
        库里和路由里都没有 → 调用方直接回 404（引擎只会回一句
        ``model 'xxx' not found``，连可选名字都不给，很难查）。
        """
        nm = str((query.get("model") or [""])[0] or "")
        if method == "POST" and raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = None
            if isinstance(body, dict) and not body.get("model"):
                # 客户端没写 model：老版本单模型后端可以省略，这里补上界面上
                # 的「目标模型」，省得用户被引擎一句 "model is required" 卡住。
                drow = self._default_row()
                did = self._router_id_for(str((drow or {}).get("name") or ""))
                if did:
                    body["model"] = did
                    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    return path, raw, did
            if isinstance(body, dict) and body.get("model"):
                nm = str(body["model"])
                rid = self._router_id_for(nm)
                if not rid:
                    return path, raw, None
                if rid != nm:
                    body["model"] = rid
                    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                return path, raw, rid
        if path.startswith("/v1/models/"):
            tail = path[len("/v1/models/"):]
            rid = self._router_id_for(tail)
            return "/v1/models/" + (rid or tail), raw, rid or ""
        return path, raw, ""

    def _unknown_model_payload(self, want: str) -> Dict[str, Any]:
        """认不出的模型名 → 一份带「可选名字」的 404。"""
        known = self._known_model_names()
        toks = [t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", want.lower())
                if len(t) > 2]
        near = [k for k in known
                if any(t in k.lower() for t in toks)][:6] or known[:6]
        tip = ("用 GET /v1/models 看可用名字。名字可以是路由里的段名"
               "（= 模型库里的显示名），也可以是文件名或去掉扩展名的文件名。")
        return {"error": {"message": "没有叫「%s」的模型。%s\n可选：%s"
                                     % (want, tip, "、".join(near) or "（一个都没有，"
                                        "先到界面「模型库」扫描目录）"),
                          "type": "invalid_request_error"},
                "model": want}

    def _path_for_id(self, rid: str) -> str:
        """路由 id → 库里的模型文件路径（认不出来给空串）。"""
        want = str(rid or "").strip().lower()
        if not want:
            return ""
        try:
            rows = self.server.models_provider() or []   # type: ignore
        except Exception:  # noqa: BLE001
            return ""
        for r in rows:
            if str(r.get("name") or "").strip().lower() == want:
                return str(r.get("path") or "")
        return ""

    def _start_router(self, prefer: str = "") -> bool:
        """请求来了但路由进程没在跑 → 让主线程把它拉起来。

        走 CommandBus（进程启停只能在主线程做）。起来的是「空路由」：预置文件
        里声明了库里全部模型，随后引擎的 ``--models-autoload`` 会按请求把目标
        模型装进来 —— 所以调用方不需要先手动加载。
        失败后在几秒内不再重试，免得每个请求都去撞一次。
        """
        # 自检里不真起进程：run_selftest 跑在主线程上、没人 drain CommandBus，
        # 一提交就会把 HTTP 请求挂住（客户端 60s 超时）。
        if os.environ.get("BUUN_SELFTEST"):
            return False
        now = time.time()
        if now - type(self)._start_try_at < 8.0:
            return False
        type(self)._start_try_at = now
        # 优先用「这次请求要的那个模型」当目标 —— 起完路由引擎的
        # --models-autoload 会顺手把它装进来，调用方就能直接拿到结果。
        path = str(prefer or "")
        if not path or not os.path.isfile(path):
            try:
                path = str(self.server.default_model() or "")  # type: ignore
            except Exception:  # noqa: BLE001
                path = ""
        if not path or not os.path.isfile(path):
            # 退而求其次：模型库里的第一个对话模型（embedding 不能当目标）
            try:
                rows = self.server.models_provider() or []  # type: ignore
            except Exception:  # noqa: BLE001
                rows = []
            cand = [r for r in rows
                    if str(r.get("category") or "") != "Embedding"]
            path = str((cand or rows or [{}])[0].get("path") or "")
        if not path:
            return False
        # 只管提交，不等结果 —— 进程启停在主线程做，HTTP 线程不能一直挂着。
        # 起完是「空路由」，随后 --models-autoload 会按这次请求把模型装进来；
        # 所以只要端口出现就够了，不用等模型就绪（大模型要十几秒）。
        try:
            self.mgr.bus.post("start_unified", {"model": path})
        except Exception:  # noqa: BLE001
            return False
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if int(getattr(self.unified, "port", 0) or 0):
                return True
            time.sleep(0.25)
        return False

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
                # 请求来了但路由进程没在跑 → 先让主线程把它拉起来。
                # 起来的是「空路由」：预置文件里声明了模型库里全部模型，
                # 引擎的 --models-autoload 会按这次请求把目标模型装进来。
                _want = str((query.get("model") or [""])[0] or "")
                if not _want:
                    _want = _body_model(raw)
                self._start_router(self._path_for_id(self._router_id_for(_want)))
                port = int(getattr(u, "port", 0) or 0)
            if not port:
                return self._send(503, {"error": {
                    "message": "多模型路由的后端没有在运行。\n"
                               "· 界面：到「模型库」选中模型点「加载」；\n"
                               "· 或者 POST /models/load {\"model\": \"<模型名>\"}。",
                    "type": "service_unavailable"}})
            # 把客户端写的模型名翻成路由的正式 id（详见 _router_rewrite）
            path, raw, rid = self._router_rewrite(path, method, query, raw)
            if rid is None:
                return self._send(404, self._unknown_model_payload(
                    str((query.get("model") or [""])[0] or _body_model(raw))))
            u.touch()
            if method == "POST" and raw:
                # 顶层 reasoning_effort 由引擎原生解析（见 _inject_effort）
                raw = self._inject_effort(raw)
            self._touch_router_model(rid)
            return self._forward(path, method, raw, port)

        # ⚠️ 这里曾经有「不是路由模式就抢锁换模型」的整段老逻辑
        #    （_resolve_row → u.ensure → 409 → 短名翻译）。路由是唯一的服务
        #    方式之后它永远走不到，已整段删除 —— 留着只会让人以为还有两条路。
        return self._send(503, {"error": {
            "message": "统一后端没有在运行。\n"
                       "· 界面：到「模型库」选中模型点「加载」；\n"
                       "· 或者 POST /models/load {\"model\": \"<模型名>\"}。",
            "type": "service_unavailable"}})

    def _router_mode(self) -> bool:
        """网关是不是「只做透传」。

        ⚠️ 这里原来写的是 ``self.server.router_mode()``，但 ``router_mode``
        从来没人挂到 httpd 上 → 每次都 AttributeError → 被 except 吞掉 →
        **恒为 False**。后果是每个 /v1/* 请求都掉进老的单模型 ensure 分支：

          * 网关不把请求转发给路由，而是自己再起一个 llama-server（抢显存）；
          * 转发前还拿模型参数做一遍校验，把「head_dim 不是 128 的倍数」
            「推测方式 DSpark 需要指定草稿模型」这类**跟调用者毫不相干**的
            错误当成 503 抛给 API 客户端 —— 这就是「列表拿得到、调用全报错」
            的真正原因。

        现在服务只剩多模型路由一种跑法，所以恒真。
        """
        return True

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
        """请求没写 model、当前也没加载模型时，用界面上选中的「目标模型」兜底。

        （原来这里兜的是「默认模型」；该功能已按用户要求删除，现在兜的是
        模型库里选中的那一行。）
        """
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

        # 抓响应里的 timings / usage 喂给运行状态面板。
        # - 非流式（JSON 响应体 + POST）：缓冲整个响应体解析。
        # - 流式（SSE，text/event-stream）：只保留尾部若干 KB，取最后一个带
        #   timings/usage 的 data 事件（llama-server 把统计放在末包）。默认对话是
        #   流式的，不处理这路就拿不到生成/提示速度。
        capture = (method == "POST")
        buf: List[bytes] = []
        total = 0
        is_sse = False
        try:
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() in _RESP_SKIP:
                    continue
                self.send_header(key, val)
                if key.lower() == "content-type":
                    v = (val or "").lower()
                    if "event-stream" in v:
                        is_sse = True
                        capture = False
                    elif "json" not in v:
                        capture = False
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            tail = b""
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (OSError, ValueError):
                    break
                if capture:
                    total += len(chunk)
                    if total <= _CAPTURE_MAX:
                        buf.append(chunk)
                    else:
                        capture = False      # 太大就不解析了，避免吃内存
                        buf = []
                elif is_sse:
                    tail = (tail + chunk)[-16384:]
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        if capture and buf:
            self._note_runtime(b"".join(buf))
        elif is_sse and tail:
            self._note_sse(tail)

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

    def _note_sse(self, body: bytes) -> None:
        """流式响应：取最后一个带 timings / usage 的 ``data:`` 事件回填运行状态。

        只缓冲了尾部约 16KB，这里从这截尾里解析。llama-server 把统计放在
        最后一个 SSE 事件里（``usage`` 与 ``timings``）。
        """
        rt = getattr(self.server, "runtime", None)
        if rt is None:
            return
        try:
            text = body.decode("utf-8", "replace")
        except (ValueError, UnicodeDecodeError):
            return
        last = None
        for block in text.split("\n\n"):
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj, dict) and (obj.get("timings")
                                             or obj.get("usage")):
                    last = obj
        if last:
            try:
                rt.note_timings(last.get("timings"), last.get("usage"))
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------- 统一模式的动作端点
    def _unified_action(self, path: str, body: Dict[str, Any],
                        query: Dict[str, Any]) -> None:
        """``/switch`` ``/load`` ``/models/load`` ``/models/unload`` 的实现。

        ⚠️ 这里以前调 ``unified.ensure()``（单模型网关时代的老路）。它会
        **自己再起一个 llama-server**，跟路由进程抢显存/端口；而且先拿模型
        参数做一遍校验，把「head_dim 不是 128 的倍数」这类跟调用者毫不相干的
        错误当成 503 抛出去。现在只跟路由打交道：

          * 装：``POST /models/load`` —— 路由起来时顺手把它拉起来
          * 卸：``POST /models/unload`` —— 带 model 就卸那一个，不带就全卸
        """
        rt = self._router()
        if rt is None:                       # 没接线（老调用点 / 单测）→ 退回老路
            return self._unified_action_legacy(path, body, query)

        want = str(body.get("model") or "").strip()
        rid = self._router_id_for(want) if want else ""

        if path in ("/unload", "/models/unload"):
            if not want:
                n = rt.unload_all()
                return self._send(200, {"ok": True,
                                        "note": "已卸载全部 %d 个模型" % n})
            if not rid:
                return self._send(404, self._unknown_model_payload(want))
            ok, msg = rt.unload(rid)
            return self._send(200 if ok else 503,
                              {"ok": ok, "model": rid, "note": msg})

        if not want:
            return self._send(400, {"ok": False,
                                    "error": "请求里要带 model 字段"})
        if not rid:
            return self._send(404, self._unknown_model_payload(want))
        if not int(getattr(self.unified, "port", 0) or 0):
            self._start_router()
        if not int(getattr(self.unified, "port", 0) or 0):
            return self._send(503, {"ok": False,
                                    "error": "路由后端没起来（看界面日志）"})
        ok, msg = rt.load(rid)
        return self._send(200 if ok else 503,
                          {"ok": ok, "model": rid, "note": msg,
                           "state": self.unified.status()})

    def _unified_action_legacy(self, path: str, body: Dict[str, Any],
                               query: Dict[str, Any]) -> None:
        """没有路由接线时的兜底（自检里构造的老式 ControlAPI 会走到）。"""
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


def _body_model(raw: bytes) -> str:
    """从请求体里抠出 model 字段（认不出来给空串）。"""
    if not raw:
        return ""
    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if isinstance(body, dict):
        return str(body.get("model") or "")
    return ""


class ControlAPI:
    def __init__(self, manager: Manager, token: str = "",
                 models_provider=None, upstream_key=None, unified=None,
                 default_model=None,
                 row_resolver=None, runtime=None, router=None) -> None:
        self.manager = manager
        self.token = token
        self.models_provider = models_provider
        self.upstream_key = upstream_key or (lambda: "")
        self.unified = unified
        self.default_model = default_model or (lambda: "")
        self.row_resolver = row_resolver
        # 运行时状态采集（可选）：转发时顺手把 timings 喂给它
        self.runtime = runtime
        # 多模型路由（可选）：转发时记「这个模型最后一次被用」，供空闲卸载用。
        # ⚠️ 引擎的 GET /models **不返回 last_used**，所以这是唯一的来源。
        self.router = router
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
        httpd.default_model = self.default_model         # type: ignore[attr-defined]
        httpd.row_resolver = self.row_resolver           # type: ignore[attr-defined]
        httpd.runtime = self.runtime                     # type: ignore[attr-defined]
        # 路由后端。⚠️ 以前只挂了 runtime 没挂 router，而 _Handler 里的
        # _touch_router_model 又用 self.router 取它 —— 那个 AttributeError
        # 会在每个推理请求上都抛一次（HTTP 线程里没人接，直接 500）。
        httpd.router = self.router                       # type: ignore[attr-defined]
        # 网关是否「只做透传」。现在服务只剩多模型路由一种跑法，恒真；
        # 留成属性是为了以后要加别的服务方式时只改这一处。
        httpd.router_only = True                         # type: ignore[attr-defined]
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
