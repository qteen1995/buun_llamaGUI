# -*- coding: utf-8 -*-
"""程序入口。

* 正常启动：创建窗口，进入事件循环。
* ``--selftest``：不开事件循环，构建界面并跑一遍核心逻辑（导航、五个运行模式、
  模型扫描与分类、参数探测、预设读写、每模型 ini、控制 API），结果写到
  ``config/selftest.log``，用于快速验证改动没把东西搞坏。
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import time
import tkinter as tk
import traceback
import urllib.parse
from typing import Any, Dict, List, Tuple

from . import builder as B
from . import engine as E
from . import gguf as G
from . import manager as MG
from . import probe as PR
from . import scan as SC
from . import schema as S
from . import theme as T
from .store import model_key
from . import control_api as C
from .unified import UnifiedBackend
from .ui import APP_TITLE, App


def _writable(d: str) -> bool:
    """目录能不能写（能不能把配置放在这里）。"""
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("")
        os.remove(probe)
        return True
    except OSError:
        return False


# 便携模式（打包成 exe）下程序目录不可写时，退到别处并记下原因
PORTABLE_NOTE = ""


def _force_utf8_io() -> None:
    """把标准输出/错误切成 UTF-8。

    中文 Windows 的控制台默认是 GBK，而自检输出里带着 ``◉ ▸ ⚙`` 这类符号，
    直接 print 会抛 ``UnicodeEncodeError: 'gbk' codec can't encode character``。
    源码运行时管道往往是 UTF-8 所以看不出来，打包成 exe 后双击/被重定向就必踩。
    """
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass                    # 没有 stdout（无控制台打包）就跳过


def _base_dir() -> str:
    """数据目录 —— **便携**：设置跟程序放一起。

    * 源码运行：项目根目录
    * 打包成 exe：**exe 所在目录**（配置写在 ``<exe目录>/config/``：
      ``app.json`` 放软件级设置，``models/<模型名>.json`` 每个模型一个，
      ``cache/scan.json`` 是模型库扫描缓存）
    只有程序目录真的不可写（例如放进 Program Files）才退回 %LOCALAPPDATA%，
    并在日志里说明 —— 否则「保存」会静默失败，很难查。
    """
    global PORTABLE_NOTE
    if getattr(sys, "frozen", False):
        here = os.path.dirname(os.path.abspath(sys.executable))
    else:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _writable(here):
        return here
    alt = os.path.join(os.environ.get("LOCALAPPDATA")
                       or os.path.expanduser("~"), "buun_llama_gui")
    try:
        os.makedirs(alt, exist_ok=True)
    except OSError:
        pass
    PORTABLE_NOTE = ("程序目录不可写（%s），配置改存到 %s" % (here, alt))
    return alt


def _log_path(base_dir: str, name: str) -> str:
    d = os.path.join(base_dir, "config")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.join(base_dir, name)
    return os.path.join(d, name)


# --------------------------------------------------------------------------- #
# 造几个假的 GGUF，用来验证扫描 / 分类 / 能力图标
# --------------------------------------------------------------------------- #

def _fake_gguf(path: str, meta: Dict[str, Tuple[int, Any]]) -> None:
    def s(v: str) -> bytes:
        b = v.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    body = b""
    for key, (vtype, val) in meta.items():
        if vtype == 8:
            body += s(key) + struct.pack("<I", 8) + s(str(val))
        else:
            body += s(key) + struct.pack("<I", 4) + struct.pack("<I",
                                                                int(val))
    head = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
            + struct.pack("<Q", len(meta)))
    with open(path, "wb") as fh:
        fh.write(head + body)


def _make_fake_lib(root: str) -> List[str]:
    os.makedirs(root, exist_ok=True)
    files = []

    llm = os.path.join(root, "Qwen3-30B-A3B-Instruct-IQ4_KSS.gguf")
    _fake_gguf(llm, {
        "general.architecture": (8, "qwen3moe"),
        "general.name": (8, "Qwen3 30B A3B Instruct"),
        "tokenizer.chat.ggml.chat_template": (
            8, "{% if tools %}<|tool|>{% endif %} thinking"),
        "general.file_type": (4, 39),
        "qwen3moe.block_count": (4, 48),
        "qwen3moe.context_length": (4, 40960),
        "qwen3moe.expert_count": (4, 128),
        # head_dim 是 128 的倍数 → turbo / TCQ / VBR 可用
        "qwen3moe.attention.key_length": (4, 128),
        "qwen3moe.attention.value_length": (4, 128),
    })
    files.append(llm)

    emb = os.path.join(root, "bge-m3-F16.gguf")
    _fake_gguf(emb, {
        "general.architecture": (8, "bert"),
        "general.name": (8, "bge-m3"),
        "bert.block_count": (4, 24),
        "bert.context_length": (4, 8192),
        # bert 系 head_dim=64 → not a multiple of 128，turbo 档位会被引擎拒绝
        "bert.attention.key_length": (4, 64),
    })
    files.append(emb)

    draft = os.path.join(root, "Qwen3-0.6B-DFlash.gguf")
    _fake_gguf(draft, {
        "general.architecture": (8, "qwen3"),
        "general.name": (8, "Qwen3 0.6B DFlash drafter"),
        "qwen3.block_count": (4, 28),
        "qwen3.context_length": (4, 32768),
        "qwen3.attention.key_length": (4, 128),
    })
    files.append(draft)

    mm = os.path.join(root, "mmproj-Qwen3-f16.gguf")
    _fake_gguf(mm, {"general.architecture": (8, "clip")})

    # HuggingFace 风格的目录结构：…\发布者\模型名\文件.gguf
    repo = os.path.join(root, "huihui-ai",
                        "Huihui-Qwen3.8-27B-abliterated-GGUF")
    os.makedirs(repo, exist_ok=True)
    hf = os.path.join(
        repo, "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf")
    _fake_gguf(hf, {
        "general.architecture": (8, "qwen3"),
        "general.name": (8, "Huihui Qwen3.8 27B abliterated"),
        "tokenizer.chat.ggml.chat_template": (8, "think"),
        "qwen3.block_count": (4, 48),
        "qwen3.context_length": (4, 131072),
    })
    files.append(hf)
    return files


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def run_selftest(app: App, root: tk.Tk) -> List[str]:
    out: List[str] = []
    root.update_idletasks()
    root.update()
    out.append("配置目录：%s（临时，不动真实 config.json）"
               % app.store.config_dir)
    out.append("窗口：%s  屏幕 %dx%d" % (root.winfo_geometry(),
                                      root.winfo_screenwidth(),
                                      root.winfo_screenheight()))
    out.append("高度分布：顶栏 %d / 中部 %d / 底部 %d / 底栏 %d"
               % (app.topbar.winfo_height(), app.middle.winfo_height(),
                  app.bottom_host.winfo_height(), app.footer.winfo_height()))
    out.append("侧边栏项：%s" % " ".join(
        "%s%s" % ("*" if p == app.sidebar.value else "", p)
        for p in app.sidebar.order))
    out.append("ttk 主题：%s" % root.tk.call("ttk::style", "theme", "use"))

    # ---------------- 1. 导航：每个页面都能打开且参数行齐全
    total_rows = 0
    for page in S.pages_with_fields():
        app.show(page)
        root.update()
        n = len([1 for r in app.rows.values() if r.f.page == page])
        total_rows += n
        mapped = app.pages[page].winfo_ismapped()
        out.append("页面 %-8s 参数 %2d 项  已显示=%s" % (page, n, mapped))
    custom = [f.key for f in S.FIELDS
              if (f.page, f.section) in
              {(p_, s_) for p_, secs in S.CUSTOM_SECTIONS.items()
               for s_ in secs}]
    out.append("参数总项数：%d 行 + %d 项自定义小节（schema 定义 %d）= %s"
               % (total_rows, len(custom), len(S.FIELDS),
                  total_rows + len(custom) == len(S.FIELDS)))

    # ---------------- 2. 三种运行模式的命令行
    fake_root = os.path.join(tempfile.gettempdir(), "buun_selftest")
    shutil.rmtree(fake_root, ignore_errors=True)
    lib = _make_fake_lib(fake_root)
    engine = os.path.join(fake_root, "bin", "Release")
    os.makedirs(engine, exist_ok=True)
    for name in ("llama-server", "llama-cli", "llama-completion"):
        with open(os.path.join(engine, name + ".exe"), "wb") as fh:
            fh.write(b"MZ")
    app.engine_var.set(engine)
    app.commit_engine()

    # 真实引擎（有的话）：用来把参数表跟 exe 的 --help 对撞
    engine_probe = ""
    for cand in E.default_engine_candidates():
        if all(B.resolve_exe(cand, m) for m in S.MODE_ORDER):
            engine_probe = cand
            break
    out.append("真实引擎（用于参数对齐）：%s" % (engine_probe or "未找到"))

    # ---------------- 3. 扫描 + 分类 + 能力
    paths = SC.discover([fake_root], recursive=True, depth=3)
    mm = SC.index_mmproj([fake_root], recursive=True, depth=3)
    rows = SC.scan(paths, mm, [], cache={}, roots=[fake_root])
    out.append("扫描到 %d 个模型（mmproj 已排除）：%d 个投影文件"
               % (len(rows), len(mm)))
    for r in rows:
        out.append("  %-42s %-10s %-8s %-7s %s 参数=%s"
                   % (r["name"][:42], r["category"], r["arch"],
                      r["params_text"] or "-", r["pip"],
                      r["params_set"] or "-"))
    cats = sorted({r["category"] for r in rows})
    out.append("分类覆盖：%s" % ",".join(cats))
    out.append("排序自检（按参数量倒序）：%s"
               % " > ".join(r["name"][:18] for r in
                            SC.sort_rows(list(rows), "params", True)))
    out.append("筛选自检（Embedding）：%s"
               % [r["name"] for r in SC.filter_rows(rows, category="Embedding")])
    out.append("能力筛选自检（多模态）：%s"
               % [r["name"] for r in SC.filter_rows(rows, cap="vision")])

    # 目录结构取发布者 / 模型名
    demo_path = (r"E:\LM_models\huihui-ai"
                 r"\Huihui-Qwen3.8-27B-abliterated-GGUF"
                 r"\Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf")
    org, mname, ok = G.split_repo(demo_path, 2)
    out.append("目录结构解析（根\\发布者\\模型名\\文件 → depth=2）：发布者=%s "
               "模型名=%s 取到=%s" % (org, mname, ok))
    out.append("一层目录（根\\模型名\\文件 → depth=1）：发布者=%r 模型名=%s 取到=%s"
               % G.split_repo(demo_path, 1))
    out.append("零层（文件直接在根目录 depth=0）：取到=%s（应为 False）"
               % G.split_repo(demo_path, 0)[2])
    flat = os.path.join(fake_root, "Qwen3-30B-A3B-Instruct-IQ4_KSS.gguf")
    out.append("平铺目录回退：%s → depth=%d → 名称用文件名去后缀=%s"
               % (os.path.basename(flat),
                  G.depth_from_roots(flat, [fake_root]),
                  G.split_repo(flat, G.depth_from_roots(flat, [fake_root]))[2]))
    hf_row = next((r for r in rows if r.get("org") == "huihui-ai"), None)
    out.append("扫描结果中的 HF 模型：名称=%s 发布者=%s 文件=%s"
               % (hf_row["name"], hf_row["publisher"], hf_row["file_name"])
               if hf_row else "扫描结果里没有 HF 目录结构的模型")
    flat_row = next((r for r in rows
                     if os.path.basename(r["path"])
                     == "Qwen3-30B-A3B-Instruct-IQ4_KSS.gguf"), None)
    if flat_row:
        out.append("平铺目录那行的显示：名称=%s 发布者=%r（不应是目录名）"
                   % (flat_row["name"], flat_row["publisher"]))

    # 让模型库真的加载一遍
    app.lib_rows = rows
    app.lib.refresh_table()
    root.update()
    out.append("表格行数：%d，表头：%s"
               % (len(app.lib.tree.get_children()),
                  " | ".join(str(app.lib.tree.heading(c, "text"))
                             for c in SC.COLUMN_IDS[:5])))
    if app.lib.tree.get_children():
        iid = app.lib.tree.get_children()[0]
        app.lib.tree.selection_set(iid)
        root.update()
        app.lib._on_select()
        out.append("选中首行 → %s" % os.path.basename(app.selected_model))
        out.append("详情首行：%s" % app.lib.detail_lbl.cget("text").split("\n")[0])

    llm_row = next(r for r in rows if r["category"] == "LLMS")
    emb_row = next(r for r in rows if r["category"] == "Embedding")
    dr_row = next(r for r in rows if r["category"] == "Drafters")
    app.selected_model = llm_row["path"]
    app.store.ensure_model(llm_row["path"])

    for mode in S.MODE_ORDER:
        snap = app.effective_snapshot(mode, "llm", llm_row["path"])
        argv = B.build_argv(snap, mode, model=llm_row["path"],
                            role="llm")
        out.append("[%s] %s ← %s" % (mode, S.MODES[mode]["exe"],
                                     " ".join(argv) or "(空)"))

    # ---------------- 4. 服务模式命令行（只有一个角色，一套服务参数）
    app.assign_role("llm", llm_row["path"], start=False)
    role = "llm"
    snap = app.effective_snapshot("server", role)
    argv = B.build_argv(snap, "server",
                        model=app.store.role_model(role), role=role)
    out.append("[服务 %s] %s" % (role, " ".join(argv) or "(空)"))
    # 换一个模型时，命令行里不该再带上 Embedding 专属参数
    out.append("服务命令行不含 embedding 专属 flag=%s"
               % (not any(a in argv for a in
                          ("--embedding", "--pooling", "--embd-normalize",
                           "--reranking"))))

    # ---------------- 5. 推测解码
    # buun 的 --spec-type 收的是逗号分隔的**类型名列表**，细项都是独立 flag；
    # ik 那套 `--spec-type dflash:n_max=4` 的内联写法在本 build 是非法参数。
    def _spec_argv(spec_type="", draft=True, extra=None):
        for key in ("spec_enable", "spec_type", "model_draft"):
            row = app.rows.get(key)
            if row is not None:
                row.on_var.set(False)
        app.rows["spec_enable"].on_var.set(True)
        if spec_type:
            app.rows["spec_type"].on_var.set(True)
            app.rows["spec_type"].var.set(spec_type)
        if draft:
            app.rows["model_draft"].on_var.set(True)
            app.rows["model_draft"].var.set(dr_row["path"])
        for key, value in (extra or {}).items():
            row = app.rows.get(key)
            if row is not None:
                row.on_var.set(True)
                row.var.set(value)
        for row in app.rows.values():
            row._apply_state()
            row._update_flag_label()
        app.collect("spec")
        s = app.effective_snapshot("server", "llm", llm_row["path"])
        return s, B.build_argv(s, "server", model=llm_row["path"], role="llm")

    app.show("spec")
    root.update()
    snap, argv = _spec_argv("draft-dflash",
                            extra={"spec_draft_n_max": "4",
                                   "spec_draft_p_min": "0.6"})
    out.append("DFlash 命令行：%s" % " ".join(argv))
    st = str(argv[argv.index("--spec-type") + 1]).strip() if \
        "--spec-type" in argv else ""
    out.append("spec-type 是纯类型名（不含冒号/等号）=%s，值=%r"
               % ((":" not in st and "=" not in st), st))
    out.append("草稿细项走独立 flag=%s"
               % all(f in argv for f in ("--spec-draft-n-max", "--spec-draft-p-min")))

    _, argv = _spec_argv("draft-mtp", draft=False)
    out.append("MTP 命令行：%s" % " ".join(argv))
    out.append("草稿模型校验（缺草稿 + draft-mtp，应通过）：%s"
               % (B.validate(app.effective_snapshot("server", "llm",
                                                    llm_row["path"]),
                             "server", B.resolve_exe(engine, "server"),
                             model=llm_row["path"], role="llm") or "通过"))

    _, argv = _spec_argv("draft-dflash", draft=False)
    out.append("草稿模型校验（缺草稿 + draft-dflash，应报错）：%s"
               % [m for _k, m in B.validate(
                   app.effective_snapshot("server", "llm", llm_row["path"]),
                   "server", B.resolve_exe(engine, "server"),
                   model=llm_row["path"], role="llm")])

    _, argv = _spec_argv("ngram-simple,suffix", draft=False,
                         extra={"spec_ngram_simple_size_n": "16"})
    st2 = str(argv[argv.index("--spec-type") + 1]) if "--spec-type" in argv else ""
    out.append("逗号列表原样透传=%s（%r）" % (st2 == "ngram-simple,suffix", st2))

    _, argv = _spec_argv("dflash:foo=1")
    out.append("不认识的内联写法被抓出来=%s"
               % [m for _k, m in B.validate(
                   app.effective_snapshot("server", "llm", llm_row["path"]),
                   "server", B.resolve_exe(engine, "server"),
                   model=llm_row["path"], role="llm")])

    app.rows["spec_enable"].on_var.set(False)
    for key in ("spec_type", "model_draft"):
        app.rows[key].on_var.set(False)
    app.collect("spec")

    # ---------------- 5b. turbo / TCQ / VBR 的 128-block 兼容性
    # bert 系模型的 head_dim=64，用 turbo 档位会被引擎直接拒绝。
    _narrow = next((r for r in rows if (r.get("head_dim") or 0) % 128 != 0), None)
    _wide = next((r for r in rows if (r.get("head_dim") or 0) % 128 == 0), None)
    out.append("从 GGUF 读到 head_dim：%s"
               % [(r["name"][:26], r.get("head_dim"))
                  for r in rows if r.get("head_dim")])
    for _tag, _row, _ct, _want in (
            ("窄 head_dim + vbr（显式）", _narrow, "vbr", True),
            ("窄 head_dim + q8_0（两侧）", _narrow, "q8_0", False),
            ("宽 head_dim + turbo3_tcq", _wide, "turbo3_tcq", False)):
        if _row is None:
            continue
        _snap = S.default_snapshot()
        _snap["ct"] = {"on": True, "value": _ct}
        _errs = [k for k, _m in B.kv_compat_errors(_snap, _row["path"])]
        out.append("%s → 报错=%s（应=%s）%s"
                   % (_tag, bool(_errs), _want,
                      "" if bool(_errs) == _want else " <<< 不符"))
    if _narrow is not None:
        # 最容易踩的一种：什么都不设，本 build 默认的 vbr 也会走 turbo 阶梯
        _snap = S.default_snapshot()
        out.append("窄 head_dim + 完全不设 KV 档位（默认 vbr）→ 报错=%s（应=True）"
                   % bool(B.kv_compat_errors(_snap, _narrow["path"])))
        out.append("  实际生效档位：K=%s V=%s"
                   % B.effective_kv_types(_snap))
        # 只设 K 一侧，V 侧仍是默认 vbr → 照样起不来
        _snap = S.default_snapshot()
        _snap["ctk"] = {"on": True, "value": "q8_0"}
        out.append("窄 head_dim + 只设 K=q8_0（V 仍默认）→ 报错=%s（应=True）"
                   % bool(B.kv_compat_errors(_snap, _narrow["path"])))
        _snap["ct"] = {"on": True, "value": "q8_0"}
        out.append("窄 head_dim + 统一档位 q8_0 → 报错=%s（应=False）"
                   % bool(B.kv_compat_errors(_snap, _narrow["path"])))
    _snap = S.default_snapshot()
    _snap["ctx"] = {"on": False, "value": ""}
    out.append("vbr 且 -c 未设的提示：%s"
               % ("有" if B.kv_compat_notes(
                   _snap, _wide["path"] if _wide else "") else "（无）"))
    _snap["ctx"] = {"on": True, "value": "65536"}
    out.append("vbr 且 -c 有值时的提示：%s"
               % ("有" if any("固定" in n for n in B.kv_compat_notes(
                   _snap, _wide["path"] if _wide else "")) else "（无）"))

    # ---------------- 5d. 运行时状态采集
    from . import runtime as _RT
    out.append("bpv→档位反查（表值取自实测）：%s"
               % {v: _RT.tier_from_bpv(v) for v in
                  (16.0, 8.5, 8.125, 6.0, 5.5, 5.0, 4.5, 4.125, 3.5, 3.25,
                   2.25, 1.25)})
    out.append("降级后的混合值 4.25 → %s（应 ≈turbo4）"
               % _RT.tier_from_bpv(4.25))
    _rt = _RT.RuntimeStatus()
    out.append("未 attach 时 summary：%s" % _rt.summary())
    _rt.attach(0)
    out.append("attach 端口 0 时不轮询、摘要=%s" % _rt.summary())
    _rt.detach()
    _rt.attach(1)          # 故意指向一个不存在的端口，测降级不崩
    _rt.note_timings({"predicted_per_second": 261.9, "prompt_per_second": 248.8,
                      "predicted_n": 24, "cache_n": 0, "kv_bpv": 16.0},
                     {"prompt_tokens": 11, "completion_tokens": 24})
    _s = _rt.snapshot()
    out.append("回填 timings/usage → 生成速度=%.1f 提示速度=%.1f KV=%s 已用=%s"
               % (_s.get("gen_tps") or 0, _s.get("prompt_tps") or 0,
                  _s.get("kv_tier"), _s.get("n_past")))
    out.append("  摘要：%s" % _rt.summary())
    _rt.note_timings(None, None)          # 空数据不许抛
    _rt.note_timings({"kv_bpv": 1.25}, {})
    out.append("降级后 KV 档位=%s（应 turbo1_tcq）" % _rt.snapshot().get("kv_tier"))
    _rt.detach()
    out.append("detach 后 attached=%s（应 False）" % _rt.snapshot().get("attached"))
    out.append("状态栏已建=%s" % (app.rt_lbl is not None))

    # ---------------- 5d2. 多模态投影三项
    _s = S.default_snapshot()
    _s["mmproj"] = {"on": True, "value": "E:/x/mmproj.gguf"}
    _s["mmproj_gpu_swap"] = {"on": True, "value": ""}
    _s["mmproj_offload"] = {"on": True, "value": "强制关闭"}
    _s["mmproj_device"] = {"on": True, "value": "CUDA0"}
    _a = B.build_argv(_s, "server", model=llm_row["path"], role="llm")
    out.append("多模态投影三项都在命令行：%s"
               % all(x in _a for x in ("--mmproj", "--mmproj-gpu-swap",
                                       "--no-mmproj-offload", "-mmdev")))
    out.append("  实际片段：%s"
               % " ".join(a for a in _a
                          if "mmproj" in a or a == "-mmdev" or a == "CUDA0"))
    _ag = B.build_argv(_s, "gen", model=llm_row["path"], role="llm")
    out.append("生成模式不带 server 专属的 --mmproj-gpu-swap=%s（应 True）"
               % ("--mmproj-gpu-swap" not in _ag))
    _ac = B.build_argv(_s, "cli", model=llm_row["path"], role="llm")
    out.append("对话模式带 --mmproj 但不带 gpu-swap（cli 没有这个参数）=%s"
               % ("--mmproj" in _ac and "--mmproj-gpu-swap" not in _ac))

    # ---------------- 5e. VBR 下限挡位 + 上下文自动/指定
    _snap = S.default_snapshot()
    _snap["ct"] = {"on": True, "value": "vbr"}
    _snap["fa"] = {"on": True, "value": "关闭"}
    out.append("vbr 档位 + FA 关闭 → 报错=%s（应=True）"
               % bool(B.kv_compat_errors(_snap, _wide["path"] if _wide else "")))
    _snap["fa"] = {"on": True, "value": "开启"}
    out.append("VBR 下限挡位=%s\n  默认=%s，各档 bpv=%s"
               % (list(S.VBR_FLOOR_CHOICES),
                  S.FIELD_BY_KEY["vbr_floor"].default_value(),
                  {k: S.VBR_BPV.get(v) for k, v in S.VBR_FLOOR_GEARS}))
    for _g in S.VBR_FLOOR_CHOICES:
        _s = S.default_snapshot()
        _s["vbr_floor"] = {"on": True, "value": _g}
        _a = B.build_argv(_s, "server", model=llm_row["path"], role="llm")
        _got = (_a[_a.index("--vbr-floor") + 1] if "--vbr-floor" in _a else "（不传）")
        out.append("  下限挡位「%s」→ %s" % (_g, _got))
    # floor 高于 entry → 必须报错；classic 阶梯配 turbo 档 → 必须报错
    _s = S.default_snapshot()
    _s["vbr_floor"] = {"on": True, "value": "turbo4"}
    _s["vbr_entry"] = {"on": True, "value": "t2"}
    out.append("下限 turbo4(%.4g) 高于起始档位 t2(%.4g) → 报错=%s（应=True）"
               % (S.VBR_BPV["turbo4"], S.VBR_BPV["t2"],
                  bool(B.vbr_floor_errors(_s))))
    _s["vbr_entry"] = {"on": True, "value": "f16"}
    out.append("下限 turbo4 低于起始档位 f16 → 报错=%s（应=False）"
               % bool(B.vbr_floor_errors(_s)))
    _s["vbr_codec"] = {"on": True, "value": "classic"}
    out.append("classic 阶梯 + turbo4 下限 → 报错=%s（应=True）"
               % bool(B.vbr_floor_errors(_s)))
    _s["vbr_floor"] = {"on": True, "value": "跟随引擎"}
    out.append("classic 阶梯 + 跟随引擎 → 报错=%s（应=False）"
               % bool(B.vbr_floor_errors(_s)))
    # 上下文：留空 = 不传（自动），填值 = -c N
    for _tag, _on, _val, _want in (("留空（自动）", True, "", None),
                                   ("指定 65536", True, "65536", "65536"),
                                   ("不勾", False, "65536", None)):
        _s = S.default_snapshot()
        _s["ctx"] = {"on": _on, "value": _val}
        _a = B.build_argv(_s, "server", model=llm_row["path"], role="llm")
        _got = _a[_a.index("-c") + 1] if "-c" in _a else None
        out.append("上下文「%s」→ -c %s（应 %s）"
                   % (_tag, _got, _want))
    _s = S.default_snapshot()
    _s["fit_ctx"] = {"on": True, "value": "8192"}
    _s["fit_target"] = {"on": True, "value": "512"}
    _a = B.build_argv(_s, "server", model=llm_row["path"], role="llm")
    out.append("fit 参数进命令行=%s（--fit-ctx %s，--fit-target %s）"
               % ("--fit-ctx" in _a and "--fit-target" in _a,
                  _a[_a.index("--fit-ctx") + 1] if "--fit-ctx" in _a else "-",
                  _a[_a.index("--fit-target") + 1] if "--fit-target" in _a else "-"))
    # VBR 那一组只在 KV 确实是 vbr 时才输出。
    # 引擎对「非 vbr 的 KV + --vbr-*」是硬报错：
    #   --vbr-* flags need a VBR cache side
    # 而 --vbr-floor 默认就是开启的，所以这道闸必须守住。
    for _tag, _ct, _want in (("默认（引擎默认=vbr）", "", True),
                             ("显式 vbr", "vbr", True),
                             ("q8_0", "q8_0", False),
                             ("turbo3_tcq", "turbo3_tcq", False),
                             ("f16", "f16", False)):
        _s = S.default_snapshot()
        if _ct:
            _s["ct"] = {"on": True, "value": _ct}
        _a = B.build_argv(_s, "server", model=llm_row["path"], role="llm")
        _has = [x for x in _a if x.startswith("--vbr-")]
        out.append("KV=%-20s → VBR 参数 %d 个（应 %s）%s"
                   % (_tag, len(_has), "有" if _want else "无",
                      "" if bool(_has) == _want else
                      " <<< 不符：%s" % " ".join(_has)))
    _s = S.default_snapshot()
    _s["ct"] = {"on": True, "value": "q8_0"}
    _s["vbr_floor"] = {"on": True, "value": "turbo4"}
    out.append("非 vbr 时提示已忽略 VBR 参数=%s"
               % any("不会传给引擎" in n
                     for n in B.kv_compat_notes(_s, llm_row["path"])))

    # ---------------- 6. 挡位取值
    row = app.rows["reasoning"]
    row.seg.set_value("关")
    row.var.set("关")
    row.on_var.set(True)
    app.collect("chat")

    # 推理强度 = 原生 --reasoning-effort（不是 token 上限，也不走模板变量 hack）
    def effort_argv(val: str) -> List[str]:
        s2 = app.effective_snapshot("server", "llm", llm_row["path"])
        s2["reasoning_effort"] = {"on": True, "value": val}
        return B.build_argv(s2, "server", model=llm_row["path"], role="llm")

    for val in ("minimal", "low", "medium", "high", "xhigh", "max", "不设置"):
        got = [a for a in effort_argv(val) if "reasoning-effort" in a]
        out.append("推理强度「%s」→ %s" % (val, got[0] if got else "（不传参数）"))
    _ea = effort_argv("high")
    out.append("推理强度用原生 --reasoning-effort（不经模板变量）=%s"
               % ("--reasoning-effort" in _ea
                  and _ea[_ea.index("--reasoning-effort") + 1] == "high"
                  and "--chat-template-kwargs" not in _ea))
    # 模板附加参数仍然是独立的 --chat-template-kwargs（不合并、不冲突）
    s3 = app.effective_snapshot("server", "llm", llm_row["path"])
    s3["reasoning_effort"] = {"on": True, "value": "xhigh"}
    s3["chat_template_kwargs"] = {"on": True, "value": '{"top_k": 3}'}
    argv3 = B.build_argv(s3, "server", model=llm_row["path"], role="llm")
    out.append("两处输出各自独立：--reasoning-effort=%s，--chat-template-kwargs=%s"
               % (argv3[argv3.index("--reasoning-effort") + 1],
                  argv3[argv3.index("--chat-template-kwargs") + 1]))
    # 挡位改成数字框后，存档里的旧挡位名必须被丢掉（不能变成 --reasoning-budget 不限）
    s4 = app.effective_snapshot("server", "llm", llm_row["path"])
    s4["reasoning_budget"] = {"on": True, "value": "不限"}
    argv4 = B.build_argv(s4, "server", model=llm_row["path"], role="llm")
    out.append("旧挡位值「不限」喂给数字项 → 产出=%s（应为空）"
               % [a for a in argv4 if "reasoning-budget" in a])
    s5 = app.effective_snapshot("server", "llm", llm_row["path"])
    s5["reasoning_budget"] = {"on": True, "value": "4096"}
    argv5 = B.build_argv(s5, "server", model=llm_row["path"], role="llm")
    out.append("思考 token 上限=4096 → %s"
               % " ".join(argv5[argv5.index("--reasoning-budget"):
                                 argv5.index("--reasoning-budget") + 2]))
    # 请求体里的顶层 reasoning_effort：本 build 原生支持，必须原样放行
    _h = C._Handler  # noqa: SLF001
    _body = b'{"model":"m","reasoning_effort":"high"}'
    out.append("请求体 reasoning_effort=high 原样透传=%s"
               % (_h._inject_effort(_body) == _body))  # noqa: SLF001
    _body2 = b'{"reasoning_effort":"none"}'
    out.append("请求体 reasoning_effort=none 原样透传=%s"
               % (_h._inject_effort(_body2) == _body2))  # noqa: SLF001

    snap = app.effective_snapshot("server", "llm", llm_row["path"])
    out.append("推理开关展开：%s" % [a for a in
                                 B.build_argv(snap, "server",
                                              model=llm_row["path"],
                                              role="llm")
                                 if a in ("-rea", "off", "on", "auto")])

    # ---------------- 6b2. 参数表里不能留本 build 不认识的参数
    # 真实事故：界面上勾了 -kvu / -a，而用户那份 build 的 --help 里根本没有，
    # llama-server 直接 error: unknown argument 退出，服务永远起不来。
    # 现在这些项已经从参数表里删掉了，这里守着「不许再回来」。
    gone = (
        # 上游早就删掉、两套 build 都没有的
        "alias", "router_enable", "cache_idle_slots", "reasoning_preserve",
        "spec_legacy", "draft_max", "draft_min", "custom_q", "imatrix_file",
        "output_tensor_type", "token_embedding_type", "fudge_factor",
        "slow_iq2ks", "bench_tgb", "lsim",
        # ik_llama.cpp 专属、buun 这套 exe 一个都不认的
        "cache_ram_similarity", "chunks", "amb", "fit_margin", "rtr",
        "khad", "vhad", "dump_kv", "mla", "fmoe", "gr", "sas", "mqkv",
        "ser", "vq", "minilog", "parallel_tool_calls", "spec_autotune",
        "spec_ckpt_mode", "mtp_requant", "draft_params",
        "spec_n_max", "spec_n_min", "spec_p_min", "spec_heads",
        "spec_ngram_n", "spec_ngram_m", "spec_ngram_min_hits",
        "spec_suffix_min_match_len", "spec_chunk_size",
    )
    still = [k for k in gone if k in S.FIELD_BY_KEY]
    out.append("已删除的不支持参数项=%d，残留=%s"
               % (len(gone), still or "无"))

    # ---------------- 6b3. 每一项的 flag 必须真的存在于 buun 对应 exe 里
    # 这是最强的一条：把参数表里声明的 modes 拿去跟真实 exe 的 --help 对撞。
    # 找不到 buun 引擎时跳过（不影响其它断言）。
    live: Dict[str, set] = {}
    _probe_help: Dict[str, str] = {}
    for mode in S.MODE_ORDER:
        p = B.resolve_exe(engine_probe, mode)
        if not p:
            continue
        info = PR.probe(p)
        if info.get("ok"):
            live[mode] = set(info["flags"])
            _probe_help[mode] = str(info.get("text") or "")
    if len(live) < len(S.MODE_ORDER):
        out.append("跳过「参数 flag 与真实 exe 对齐」：没找到 buun 引擎（%s）"
                   % (engine_probe or "未指定"))
    else:
        # cli / gen 两页各有一份自己的 -p / -sys / -st，是有意分开的
        split_ok = {"prompt", "sys_prompt", "single_turn",
                    "g_prompt", "g_sys_prompt", "g_single_turn"}
        bad_flags = []
        for f in S.FIELDS:
            cands = list(PR.candidates(f))
            if not cands or f.page not in ("load", "chat", "spec", "cli", "gen"):
                continue
            for mode in S.MODE_ORDER:
                want = mode in f.modes
                got = any(c in live[mode] for c in cands)
                if want and not got:
                    bad_flags.append("%s(%s) 声明支持 %s，但该 exe 没有"
                                     % (f.key, "/".join(cands), mode))
                elif got and not want and f.key not in split_ok:
                    bad_flags.append("%s(%s) 该 exe 支持 %s，但声明里漏了"
                                     % (f.key, "/".join(cands), mode))
        out.append("参数 flag 与真实 exe 全量对齐=%s%s"
                   % (not bad_flags,
                      "" if not bad_flags
                      else " → " + "；".join(bad_flags[:6])))

        # 光「有这个 flag」还不够 —— -dr 在 ik 里是 --dry-run，
        # 在 buun 里却是 --docker-repo（还要跟一个值）。这种同名不同义的坑
        # 只看存在性会漏掉，所以再把每个短参数的长名对一遍。
        import re as _re
        _lead = _re.compile(
            r"^\s*(?P<f>(?:-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)"
            r"(?:\s*,\s*-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)*)")
        sem_bad = []
        for mode, _text in _probe_help.items():
            defmap: Dict[str, List[str]] = {}
            for line in _text.splitlines():
                m = _lead.match(line)
                if not m:
                    continue
                rest = line[m.end():]
                if rest and not rest[:1].isspace() and rest[0] != "(":
                    continue
                toks = [t.strip() for t in m.group("f").split(",")]
                for t in toks:
                    defmap[t] = toks
            for f in S.FIELDS:
                if not f.flag or f.flag.startswith("--") or mode not in f.modes:
                    continue
                toks = defmap.get(f.flag)
                if not toks:
                    continue
                longs = [t for t in toks if t.startswith("--")]
                if f.long and ("--" + f.long) in longs:
                    continue
                if not f.long and not longs:
                    continue
                sem_bad.append("%s(%s) 在 %s 里其实是 %s"
                               % (f.key, f.flag, mode, " ".join(longs) or "?"))
        out.append("短参数语义与 exe 一致=%s%s"
                   % (not sem_bad,
                      "" if not sem_bad else " → " + "；".join(sem_bad[:6])))

    # ---------------- 6b4. 多模型路由
    _rsnap = S.default_snapshot()
    _rsnap["run_kind"] = {"on": True, "value": S.RUN_ROUTER}
    _rsnap["models_dir"] = {"on": True, "value": r"E:\LM_models"}
    _rsnap["models_max"] = {"on": True, "value": "3"}
    _rargv = B.build_argv(_rsnap, "server", model=llm_row["path"], role="llm")
    out.append("路由模式命令行：%s" % " ".join(_rargv))
    out.append("路由模式不塞 -m=%s / 带 --models-dir=%s / 带 --models-max=%s"
               % ("-m" not in _rargv, "--models-dir" in _rargv,
                  "--models-max" in _rargv))
    _gsnap = S.default_snapshot()
    _gargv = B.build_argv(_gsnap, "server", model=llm_row["path"], role="llm")
    out.append("单模型网关不输出路由参数=%s"
               % (not any(a in _gargv for a in
                          ("--models-dir", "--models-max", "--models-preset"))))
    _esnap = S.default_snapshot()
    _esnap["run_kind"] = {"on": True, "value": S.RUN_ROUTER}
    out.append("路由模式但没有模型来源 → 报错=%s（应=True）"
               % bool(B.validate(_esnap, "server",
                                 B.resolve_exe(engine, "server"),
                                 model=llm_row["path"], role="llm")))
    _ini_snap = S.default_snapshot()
    _ini_snap["ngl"] = {"on": True, "value": "999"}
    _ini_snap["ct"] = {"on": True, "value": "q8_0"}
    _ini_snap["mlock"] = {"on": True, "value": ""}
    _ini = B.preset_ini([("模型甲", r"E:\a\x.gguf", _ini_snap),
                         ("模型乙", r"E:\b\y.gguf", S.default_snapshot())],
                        S.default_snapshot())
    _secs = [ln for ln in _ini.splitlines() if ln.startswith("[")]
    out.append("preset INI 段=%s（应 [*] + 两个模型）" % _secs)
    out.append("preset INI 每段都有 model= =%s"
               % (_ini.count("model = ") == 2))
    out.append("preset INI 内部键：%s"
               % [ln.split("=")[0].strip() for ln in _ini.splitlines()
                  if "=" in ln][:8])
    if _probe_help:
        # 最强的守门：INI 里的每个键都必须是本 build 真有的长参数
        # （否则引擎会抛 "option 'xxx' not recognized in preset" 整个起不来）
        _longs = set()
        for _t in _probe_help.values():
            _longs.update(PR.parse_flags(_t))
        _badkeys = []
        for ln in _ini.splitlines():
            if ln.startswith(("[", ";")) or "=" not in ln:
                continue
            _k = ln.split("=")[0].strip()
            if ("--" + _k) not in _longs:
                _badkeys.append(_k)
        out.append("preset INI 的键都是真实长参数=%s%s"
                   % (not _badkeys,
                      "" if not _badkeys else " → " + ", ".join(_badkeys)))

    # 兜底过滤仍在：合成一份「本 build 只认识这几个 flag」的探测结果，
    # 界面上任何不认识当前 build 的项都必须被剔除、且不误伤核心参数。
    fake_flags = {"-m", "--host", "--port", "-c", "-ngl", "-t", "-b", "-ub"}
    snap_flt = app.effective_snapshot("server", "llm", llm_row["path"])
    # 上下文默认留空=自动（不传 -c），这里显式给一个值，才能验证 -c 不被误伤
    snap_flt["ctx"] = {"on": True, "value": "32768"}
    snap_flt["cram"] = {"on": True, "value": "4096"}
    snap_flt["cache_idle_slots"] = {"on": True, "value": "4"}
    argv_flt = B.build_argv(snap_flt, "server", model=llm_row["path"],
                            role="llm", flags=fake_flags)
    left = [a for a in argv_flt
            if a.startswith("-") and a not in fake_flags]
    out.append("不支持的 flag 被剔除：剩余=%s（应为空）" % (left or "无"))
    out.append("过滤后仍保留核心参数（-c/-ngl/-t 都还在）=%s"
               % all(x in argv_flt for x in ("-c", "-ngl", "-t")))
    # 守着「真实启动路径必须走过滤」—— 这条断言是有来历的：
    # 过滤函数早就写好了，但 start_unified_backend 里漏传了 flags，
    # 结果界面上勾了本 build 没有的参数就直接喂给 llama-server（起不来）。
    import inspect
    _srcs = []
    try:
        _srcs = [(inspect.getsource(App.start_unified_backend), "统一后端启动"),
                 (inspect.getsource(App.start_single), "对话模式启动")]
    except (OSError, TypeError):
        # 打包成 exe 之后没有源码可读，跳过这条（不影响其它断言）
        out.append("打包运行，跳过「启动路径必须过滤参数」的源码断言")
    for _text, _name in _srcs:
        out.append("%s走参数过滤=%s"
                   % (_name, "build_argv_checked" in _text))
    # 提示语必须点明是哪个 exe 没有这个参数（不是笼统的「build 不支持」）
    out.append("跳过提示按 exe 去重=%s"
               % hasattr(app, "_skip_warned"))

    # ---------------- 6c. LoRA 适配器（手动添加路径 + 比例）
    out.append("LoRA 面板已建=%s 所在自定义小节=%s"
               % (app.lora_panel is not None,
                  S.CUSTOM_SECTIONS.get("load")))
    _lora_val = json.dumps([
        {"path": r"E:\lora\style.safetensors", "scale": 1.0},
        {"path": r"E:\lora\tone.gguf", "scale": 0.75},
        {"path": r"E:\lora\neg.gguf", "scale": -0.4}], ensure_ascii=False)
    snap_lora = app.effective_snapshot("server", "llm", llm_row["path"])
    snap_lora["lora"] = {"on": True, "value": _lora_val}
    argv_lora = B.build_argv(snap_lora, "server", model=llm_row["path"],
                             role="llm")
    out.append("LoRA 命令行：%s"
               % " ".join(argv_lora[argv_lora.index("--lora"):]
                          [:8]))
    out.append("LoRA 展开正确（1 个 --lora + 2 个 --lora-scaled + 比例）=%s"
               % (argv_lora.count("--lora") == 1
                  and argv_lora.count("--lora-scaled") == 2
                  and "0.75" in argv_lora and "-0.4" in argv_lora))
    snap_off = app.effective_snapshot("server", "llm", llm_row["path"])
    snap_off["lora"] = {"on": False, "value": _lora_val}
    out.append("未勾选 LoRA 时完全不传=%s"
               % (not any("lora" in a for a in
                          B.build_argv(snap_off, "server",
                                       model=llm_row["path"], role="llm"))))
    # 纯文本写法（每行一个路径，行尾可用 | 或 Tab 跟比例）也要认
    out.append("LoRA 文本写法兼容=%s"
               % (B.lora_args("E:/a.gguf\nE:/b.gguf|0.3\nE:/c.gguf\t2")
                  == ["--lora", "E:/a.gguf",
                      "--lora-scaled", "E:/b.gguf", "0.3",
                      "--lora-scaled", "E:/c.gguf", "2"]))
    out.append("空 LoRA 列表不产参数=%s" % (B.lora_args("") == []))
    # 面板列表跟着模型走：写入 → 读回 → 换个模型应该读不到
    app.selected_model = llm_row["path"]
    app.store.snapshot_ref(llm_row["path"], "load")["lora"] = {
        "on": True, "value": _lora_val}
    app.lora_panel.reload()
    root.update()
    rows_lora = len(app.lora_panel.tree.get_children())
    app.selected_model = dr_row["path"]          # 另一个模型
    app.lora_panel.reload()
    root.update()
    other_lora = len(app.lora_panel.tree.get_children())
    app.selected_model = llm_row["path"]
    app.lora_panel.reload()
    root.update()
    out.append("LoRA 面板读回 %d 行；换到别的模型后 %d 行（按模型保存）=%s"
               % (rows_lora, other_lora, rows_lora == 3 and other_lora == 0))
    out.append("面板提示：%s"
               % app.lora_panel.argv_lbl.cget("text").splitlines()[0][:60])

    # ---------------- 6b3. 进程非 0 退出的定向诊断
    # 真实事故：用户的 build 不认 gemma-embedding 架构，日志里明明写了
    # unknown model architecture，但启动器只甩了一句笼统的「排查提示」。
    app.out_tail["unified"] = [
        "llama_model_load: error loading model: error loading model "
        "architecture: unknown model architecture: 'gemma-embedding'",
        "llama_model_load_from_file: failed to load model",
        " ERR [ load_model] unable to load model",
    ]
    mark = len(app.log_view.get_text())
    app.diagnose_exit("unified", 1)
    diag = app.log_view.get_text()[mark:]
    out.append("退出诊断（未知架构）：识别=%s 点名架构=%s 给了办法=%s"
               % ("模型架构不被支持" in diag,
                  "gemma-embedding" in diag,
                  "更新" in diag and "buun-llama-cpp" in diag))
    app.out_tail.pop("unified", None)
    app.out_tail["unified"] = ["ggml_cuda_init: found 1 CUDA devices",
                               "CUDA out of memory on device 0"]
    mark = len(app.log_view.get_text())
    app.diagnose_exit("unified", 1)
    out.append("退出诊断（显存不足）：识别=%s"
               % ("显存不足" in app.log_view.get_text()[mark:]))
    app.out_tail.pop("unified", None)
    app.learn_bad_arch(llm_row["path"], "gemma-embedding")
    _bk = app._bad_arch_key(llm_row["path"])
    _got = ((app.store.get("learned") or {}).get("bad_arch") or {}).get(_bk)
    out.append("失败过的架构能记住并回读=%s" % ("gemma-embedding" in (_got or [])))

    # ---------------- 6c. 纯界面设置绝不能漏进命令行
    snap_chk = app.effective_snapshot("server", "llm", llm_row["path"])
    snap_chk["unified_mode"] = {"on": True, "value": "统一端口（单模型）"}
    snap_chk["auto_switch"] = {"on": True, "value": ""}
    snap_chk["idle_unload"] = {"on": True, "value": "15"}
    argv_chk = B.build_argv(snap_chk, "server", model=llm_row["path"],
                            role="llm")
    ui_only = [f.key for f in S.FIELDS
               if not f.flag and not f.argmap and not f.positional
               and f.kind not in (S.K_MULTI,)]
    leaked = []
    for f in S.FIELDS:
        if f.key not in ui_only:
            continue
        val = str((snap_chk.get(f.key) or {}).get("value") or "").strip() or "x"
        if B._field_args(f, True, val):
            leaked.append(f.key)
    out.append("纯界面设置 %d 项（%s…）产出参数=%s"
               % (len(ui_only), ",".join(ui_only[:4]), leaked or "无"))
    out.append("命令行里没有中文标签=%s"
               % (not [a for a in argv_chk
                       if a and max(ord(c) for c in a) > 0x2E80]))

    # ---------------- 7. 每模型预设 + ini
    app.store.save_preset(llm_row["path"], "load", "长上下文",
                          {"ctx": {"on": True, "value": "131072"},
                           "ngl": {"on": True, "value": "999"}})
    app.store.save_preset(llm_row["path"], "chat", "严谨",
                          {"temp": {"on": True, "value": "0.2"}})
    out.append("预设列表 load=%s chat=%s"
               % (app.store.list_presets(llm_row["path"], "load"),
                  app.store.list_presets(llm_row["path"], "chat")))

    # ---------------- 7b. 配置拆分：app.json + models/<显示名>.json + cache/scan.json
    from .store import Store as _Store, safe_file_name as _safe
    _cfg = os.path.join(tempfile.gettempdir(), "buun_cfg_split")
    shutil.rmtree(_cfg, ignore_errors=True)
    os.makedirs(_cfg, exist_ok=True)
    _st = _Store(_cfg)
    _st.set("engine_path", r"E:\AID\buun-llama-cpp")
    _st.set_pref("show_english", True)
    _a = os.path.join(_cfg, "config", "models", "a.gguf")
    _b = os.path.join(_cfg, "config", "models", "b.gguf")
    _st.ensure_model(_a, name="Qwen3.8-27B: 中文/测试")
    _st.ensure_model(_b, name="b")
    _st.snapshot_ref(_a, "load")["ctx"] = {"on": True, "value": "4096"}
    _st.save()
    _cdir = os.path.join(_cfg, "config")
    _files = sorted(os.listdir(os.path.join(_cdir, "models")))
    out.append("配置目录内容：%s"
               % sorted(os.listdir(_cdir)))
    out.append("models/ 下的文件：%s" % _files)
    out.append("app.json 已生成=%s，老的单文件 config.json 不再生成=%s"
               % (os.path.isfile(os.path.join(_cdir, "app.json")),
                  not os.path.isfile(os.path.join(_cdir, "config.json"))))
    out.append("文件名做了非法字符替换=%s（%s）"
               % (all(_safe(x) == x for x in _files),
                  [x for x in _files]))
    # 没改动就不该重写（看 mtime）
    _p1 = os.path.join(_cdir, "models", [x for x in _files if "Qwen" in x][0])
    _m1 = os.stat(_p1).st_mtime
    _ap = os.path.join(_cdir, "app.json")
    _m1a = os.stat(_ap).st_mtime
    time.sleep(1.1)
    _st.save()
    out.append("无改动时 save() 不重写文件=%s"
               % (os.stat(_p1).st_mtime == _m1
                  and os.stat(_ap).st_mtime == _m1a))
    # 改了才写
    time.sleep(0.05)
    _st.snapshot_ref(_a, "chat")["temp"] = {"on": True, "value": "0.3"}
    _st.save()
    out.append("改过之后确实重写了=%s" % (os.stat(_p1).st_mtime != _m1))
    # 读回来
    _st2 = _Store(_cfg)
    _e = _st2.ensure_model(_a, create=False) or {}
    out.append("重载后：模型数=%d，加载参数=%s，对话参数=%s"
               % (len(_st2.data["models"]),
                  bool(_e.get("load")), bool(_e.get("chat"))))
    # 显示名改了 → 文件跟着改名、老文件删掉
    _st2.set_model_name(_a, "改名之后")
    _st2.save()
    _files2 = sorted(os.listdir(os.path.join(_cdir, "models")))
    out.append("改名后 models/ 下的文件：%s" % _files2)
    out.append("改名只留一个新文件=%s（老文件已删）"
               % (len(_files2) == 2
                  and not any("Qwen" in x for x in _files2)))
    # 扫描缓存单独一个文件
    _st2.cache()["x.gguf"] = {"stamp": "1:2", "info": {}}
    _st2.save()
    out.append("扫描缓存写在 cache/scan.json=%s"
               % os.path.isfile(os.path.join(_cdir, "cache", "scan.json")))
    out.append("app.json 里不含 models / model_cache=%s"
               % (lambda d: "models" not in d and "model_cache" not in d)(
                   json.load(open(os.path.join(_cdir, "app.json"),
                                  encoding="utf-8"))))
    # 老配置迁移
    _cfg2 = os.path.join(tempfile.gettempdir(), "buun_cfg_migrate")
    shutil.rmtree(_cfg2, ignore_errors=True)
    os.makedirs(os.path.join(_cfg2, "config"), exist_ok=True)
    with open(os.path.join(_cfg2, "config", "config.json"), "w",
              encoding="utf-8") as _fh:
        json.dump({"version": 3, "engine_path": "E:/old",
                   "prefs": {"show_english": False},
                   "models": {os.path.normcase(_a): {"path": _a,
                                                     "name": "老模型",
                                                     "load": {}, "chat": {},
                                                     "load_presets": {},
                                                     "chat_presets": {}}},
                   "model_cache": {"y.gguf": {"stamp": "9:9"}}}, _fh)
    _st3 = _Store(_cfg2)
    _cd2 = os.path.join(_cfg2, "config")
    out.append("迁移：%s" % (_st3.migration_note or "（没触发）"))
    out.append("迁移后文件：app.json=%s models=%s cache=%s legacy=%s"
               % (os.path.isfile(os.path.join(_cd2, "app.json")),
                  sorted(os.listdir(os.path.join(_cd2, "models"))),
                  os.path.isfile(os.path.join(_cd2, "cache", "scan.json")),
                  os.path.isfile(os.path.join(_cd2, "config.legacy.json"))))
    out.append("迁移后设置读回来了：engine_path=%s，缓存=%s"
               % (_st3.get("engine_path"), list(_st3.cache())))

    # ---------------- 8. 参数探测（用合成 help 文本验证解析）
    fake_help = ("-h, --help  print usage\n-c, --ctx-size N  size\n"
                 "-fa, --flash-attn  enable\n--spec-type TYPE  spec\n"
                 "-rea, --reasoning [on|off|auto]\n"
                 "--reasoning-budget N\n--chat-template-kwargs JSON\n")
    flags = PR.parse_flags(fake_help)
    out.append("help 解析出 %d 个 flag" % len(flags))
    sup = {f.key: PR.supports(flags, f) for f in S.FIELDS
           if f.key in ("ctx", "fa", "model_draft", "host",
                        "reasoning_budget")}
    out.append("抽样支持判定：%s" % sup)

    # ---------------- 9. 控制 API 端到端
    # 只有一种运行方式：统一端口（一个端口 + 单模型 + 按需切换）
    out.append("空闲卸载=%s 分钟（统一端口是唯一运行方式）"
               % app.idle_unload_minutes())
    app.store.control().update({"enabled": True, "host": "127.0.0.1",
                                "port": 0, "token": "selftest-token"})
    app.store.role_state("llm")["port"] = {"on": True, "value": "0"}
    app.store.role_state("llm")["host"] = {"on": True, "value": "127.0.0.1"}
    app.store.role_state("llm")["idle_unload"] = {"on": True, "value": "15"}
    app._sync_gateway(force=True)
    ok, err = app.api.start("127.0.0.1", 0)
    out.append("控制 API 启动：%s %s（端口 %d）" % (ok, err, app.api.port))
    if ok:
        import http.server as _hs
        import threading

        base = "http://127.0.0.1:%d" % app.api.port

        def call(path: str, payload: Any = None) -> Any:
            req = urllib.request.Request(
                base + path,
                data=(json.dumps(payload).encode() if payload is not None
                      else None),
                method="POST" if payload is not None else "GET")
            req.add_header("Authorization", "Bearer selftest-token")
            if payload is not None:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def err_body(exc: Any) -> str:
            try:
                return exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001
                return str(exc)

        def call_bg(path: str, payload: Any, store: Dict[str, Any]) -> None:
            """网关可能要等主线程起进程，所以放到后台线程里发请求。"""
            def run() -> None:
                try:
                    store.update(call(path, payload))
                except Exception as exc:  # noqa: BLE001
                    store["error"] = "%s: %s" % (type(exc).__name__,
                                                 err_body(exc))
            t_ = threading.Thread(target=run, daemon=True)
            t_.start()
            deadline = time.time() + 40
            while t_.is_alive() and time.time() < deadline:
                app.manager.bus.drain(app.handle_command)
                root.update()
                time.sleep(0.05)
            t_.join(timeout=3)

        st = call("/status")
        out.append("GET /status → 统一后端=%s"
                   % {k: st["data"]["unified_backend"].get(k)
                      for k in ("model", "ready", "running")})
        models = call("/models")
        out.append("GET /models → %d 条，首条 %s"
                   % (len(models["data"]), models["data"][0]["name"]
                      if models["data"] else "-"))

        # ---- OpenAI 兼容：/v1/models 用短模型名
        v1 = call("/v1/models")
        ids = [m["id"] for m in v1["data"]]
        out.append("GET /v1/models → %d 个：%s" % (len(ids), ids))
        longest = max((len(i) for i in ids), default=0)
        out.append("id 最长 %d 字符（短模型名，不含 .gguf / 完整路径）"
                   % longest)
        if ids:
            one = call("/v1/models/%s" % urllib.parse.quote(ids[0]))
            out.append("GET /v1/models/%s → file_name=%s quant=%s"
                       % (ids[0], one.get("file_name"), one.get("quantization")))

        # ---- 假的上游 llama-server，用来验证转发与 model 名翻译
        class _Stub(_hs.BaseHTTPRequestHandler):
            seen: List[Any] = []

            def log_message(self, *a: Any) -> None:  # noqa: A003
                pass

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                _Stub.seen.append((self.path, body.get("model")))
                # 带上 timings / usage —— 和真引擎一样（默认就带，不用加参数），
                # 用来验证网关转发时能不能把运行速度抓回来
                payload = json.dumps({
                    "ok": True, "model": body.get("model"), "streamed": True,
                    "choices": [{"message": {"role": "assistant",
                                             "content": "stub"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 24,
                              "total_tokens": 35},
                    "timings": {"prompt_per_second": 248.8,
                                "predicted_per_second": 261.9,
                                "predicted_n": 24, "cache_n": 0,
                                "kv_bpv": 4.125}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                _Stub.seen.append((self.path, None))
                path = self.path.split("?")[0]
                if path == "/props":
                    body = {"default_generation_settings": {"n_ctx": 156672},
                            "total_slots": 1, "is_sleeping": False,
                            "vbr": {"enabled": True, "dynamic": True,
                                    "codec": "turbo", "entry_type_k": "f16",
                                    "entry_type_v": "f16", "floor_bpv": 4.125,
                                    "capacity_floor_bpv": 4.125,
                                    "realized_bpv": None, "selected_bpv": None,
                                    "vram_budget_bytes": 5111517184}}
                    payload = json.dumps(body).encode()
                elif path == "/slots":
                    payload = json.dumps([{
                        "id": 0, "n_ctx": 156672, "kv_bpv": 16.0,
                        "n_prompt_tokens": 1234, "is_processing": False}]).encode()
                else:
                    payload = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        stub2 = _hs.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        stub2.daemon_threads = True
        threading.Thread(target=stub2.serve_forever, daemon=True).start()
        stub_port2 = int(stub2.server_address[1])

        # ---- 真实唤起：让 API 真的去起一个进程（用 cmd.exe 冒充 llama-server，
        #      它会立刻报错退出，正好验证「起了进程 + 把失败原因回给调用者」）
        wake_row = next(r for r in rows if r["path"] != llm_row["path"])
        eng2 = os.path.join(fake_root, "engine_cmd")
        os.makedirs(eng2, exist_ok=True)
        src = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                           "System32", "cmd.exe")
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(eng2, "llama-server.exe"))
            # 这个模型可能是 bert 系（head_dim=64），默认的 vbr 档位会被
            # 兼容性校验拦住 —— 那样就测不到「真起进程」这一步了。
            # 显式把 KV 档位设成常规量化，让它走到真正的启动逻辑。
            for _k in ("ct", "ctk", "ctv"):
                app.store.snapshot_ref(wake_row["path"], "load")[_k] = {
                    "on": True, "value": "q8_0"}
            app.store.snapshot_ref(wake_row["path"], "load")["fa"] = {
                "on": True, "value": "开启"}
            old_engine = app.engine_var.get()
            app.engine_var.set(eng2)
            res3: Dict[str, Any] = {}
            call_bg("/switch", {"model": wake_row["name"]}, res3)
            out.append("真实唤起（拿 cmd.exe 冒充 llama-server）→ %s"
                       % {k: str(res3.get(k))[:90]
                          for k in ("ok", "error", "note")})
            out.append("确实起了进程且把原因回给了调用者=%s"
                       % any(w in str(res3.get("error"))
                             for w in ("进程已退出", "无法启动进程", "超时")))
            # 顺便验证兼容性校验本身也会把原因回给调用者
            for _k in ("ct", "ctk", "ctv"):
                app.store.snapshot_ref(wake_row["path"], "load").pop(_k, None)
            res4: Dict[str, Any] = {}
            call_bg("/switch", {"model": wake_row["name"]}, res4)
            _r4 = str(res4.get("error"))
            out.append("窄 head_dim 模型被兼容性校验拦住并说明原因=%s"
                       % (("head_dim" in _r4) or ("128" in _r4)))
            app.engine_var.set(old_engine)
            app.unified.clear("")
            app.manager.get("unified").test_running = False

        # ---- 之后用假的启动函数，验证调度/转发/排队/卸载
        started: List[Any] = []

        def fake_start(path: str, role: str = "") -> Dict[str, Any]:
            started.append(path)
            st_ = app.manager.get("unified")
            st_.test_running = True
            st_.port = stub_port2
            st_.model = path
            return {"ok": True, "port": stub_port2, "model": path}

        def fake_stop() -> None:
            st_ = app.manager.get("unified")
            st_.test_running = False
            st_.model = ""
            app.unified.mark_stopped()

        app.start_unified_backend = fake_start   # type: ignore[assignment]
        app.stop_unified_backend = fake_stop     # type: ignore[assignment]

        h = call("/health")
        out.append("GET /health（模型没跑也应为 ok）→ %s"
                   % {k: h.get(k) for k in ("status", "gateway", "ready")})
        vm = call("/v1/models")
        out.append("统一端口 GET /v1/models → %d 个，active=%s"
                   % (len(vm["data"]), vm.get("active_model")))

        # 请求一个「当前没加载」的模型 → 应自动切换后再转发
        target_row = next(r for r in rows if r["path"] != llm_row["path"])
        res: Dict[str, Any] = {}
        call_bg("/v1/chat/completions",
                {"model": target_row["name"], "messages": []}, res)
        out.append("转发「%s」→ 实际启动=%s 上游收到 model=%s"
                   % (target_row["name"],
                      [os.path.basename(p) for p in started],
                      _Stub.seen[-1][1] if _Stub.seen else None))
        out.append("返回=%s 后端端口=%s 当前驻留=%s"
                   % ({k: res.get(k) for k in ("ok", "model", "error")},
                      app.unified.port,
                      os.path.basename(app.unified.model or "")))
        out.append("切换并转发成功=%s（短名已翻译成文件名=%s）"
                   % (bool(started) and started[-1] == target_row["path"]
                      and res.get("ok"),
                      bool(_Stub.seen)
                      and _Stub.seen[-1][1] == target_row["file_name"]))

        # ---- 运行时状态：轮询桩后端的 /props + /slots，并接收转发回来的 timings
        app.runtime.attach(stub_port2, "stub")
        _wake = time.time() + 9.0
        while time.time() < _wake:
            if app.runtime.snapshot().get("n_ctx"):
                break
            time.sleep(0.1)
        _rs = app.runtime.snapshot()
        out.append("运行时轮询：alive=%s slots=%s n_ctx=%s kv_bpv=%s 档位=%s 已用=%s"
                   % (_rs["alive"], _rs["slots"], _rs["n_ctx"], _rs["kv_bpv"],
                      _rs["kv_tier"], _rs["n_past"]))
        out.append("VBR 状态解析：codec=%s floor=%s capacity=%s realized=%s "
                   "预算=%.2f GiB"
                   % (_rs["codec"], _rs["floor_bpv"], _rs["capacity_bpv"],
                      _rs["realized_bpv"],
                      (_rs["budget_bytes"] or 0) / 1073741824.0))
        out.append("已用比例=%.4f（1234/156672）"
                   % (_rs["used_ratio"] or 0))
        # 网关转发时抓 timings 的那条路：再发一次，然后看有没有回填
        app.runtime.note_timings(None, None)          # 空数据不许抛异常
        res2: Dict[str, Any] = {}
        call_bg("/v1/chat/completions",
                {"model": target_row["name"], "messages": []}, res2)
        _rs2 = app.runtime.snapshot()
        out.append("转发一次后抓到速度：生成=%.1f 提示=%.1f t/s（应 261.9/248.8）"
                   % (_rs2.get("gen_tps") or 0, _rs2.get("prompt_tps") or 0))
        out.append("转发后 KV 档位=%s（应 turbo4）" % _rs2.get("kv_tier"))
        out.append("状态条摘要：%s" % app.runtime.summary())
        app.runtime.detach()
        out.append("detach 后 attached=%s（应 False）"
                   % app.runtime.snapshot().get("attached"))

        st_now = call("/status")["data"]["unified_backend"]
        out.append("统一后端状态：model=%s ready=%s 切换次数=%s 空闲卸载=%s 分钟"
                   % (os.path.basename(st_now.get("model") or ""),
                      st_now.get("ready"), st_now.get("switches"),
                      st_now.get("idle_unload_minutes")))

        # 关掉自动切换 → 请求别的模型应 409
        app.store.role_state("llm")["auto_switch"] = {"on": False, "value": ""}
        try:
            call("/v1/chat/completions",
                 {"model": llm_row["name"], "messages": []})
            out.append("关闭自动切换后请求别的模型 → 竟然成功了（应 409）")
        except Exception as exc:  # noqa: BLE001
            out.append("关闭自动切换后请求别的模型 → 正确拒绝（HTTP %s）"
                       % getattr(exc, "code", "?"))
        app.store.role_state("llm")["auto_switch"] = {"on": True, "value": ""}

        # 排队：两个并发请求、目标模型不同 → 必须串行切换，不能互相插队
        seq: List[Any] = []

        def slow_start(path: str) -> Tuple[bool, str]:
            seq.append(("start", os.path.basename(path)))
            time.sleep(0.4)
            st_ = app.manager.get("unified")
            st_.test_running = True
            st_.port = stub_port2
            st_.model = path
            return True, ""

        def fast_stop() -> None:
            seq.append(("stop", ""))
            st_ = app.manager.get("unified")
            st_.test_running = False
            st_.model = ""

        app.unified._start_backend = slow_start    # type: ignore[assignment]
        app.unified._stop_backend = fast_stop      # type: ignore[assignment]
        app.unified.clear("")
        r1: Dict[str, Any] = {}
        r2: Dict[str, Any] = {}
        a_row, b_row = rows[2], rows[3]

        def hit(path: str, store: Dict[str, Any]) -> None:
            ok_, msg_ = app.unified.ensure(path, timeout=10)
            store.update({"ok": ok_, "msg": msg_})

        t1 = threading.Thread(target=hit, args=(a_row["path"], r1))
        t2 = threading.Thread(target=hit, args=(b_row["path"], r2))
        t0 = time.time()
        t1.start()
        time.sleep(0.1)
        t2.start()
        t1.join(timeout=20)
        t2.join(timeout=20)
        spent = time.time() - t0
        out.append("并发两个不同模型：用时 %.1fs（串行应 ≥0.8s），切换序列=%s"
                   % (spent, seq))
        out.append("两个请求都成功=%s，最终驻留=%s"
                   % (bool(r1.get("ok")) and bool(r2.get("ok")),
                      os.path.basename(app.unified.model or "")))
        app.unified._start_backend = UnifiedBackend._start_backend  # type: ignore[assignment]
        app.unified._stop_backend = UnifiedBackend._stop_backend    # type: ignore[assignment]
        app.unified.clear("")
        app.manager.get("unified").test_running = False

        # /unload 应把统一后端停掉
        res2: Dict[str, Any] = {}
        call_bg("/unload", {}, res2)
        out.append("POST /unload → ok=%s，后端还在跑=%s"
                   % (res2.get("ok"), app.manager.get("unified").running))
        stub2.shutdown()
        stub2.server_close()

        try:
            req = urllib.request.Request(base + "/status")
            with urllib.request.urlopen(req, timeout=10):
                pass
            out.append("无 token 访问 → 竟然成功了（鉴权失效）")
        except Exception:
            out.append("无 token 访问 → 正确拒绝（HTTPError）")
        app.api.stop()
        out.append("控制 API 已停止")

    # ---------------- 10. 校验逻辑
    app.store.set_role_model("llm", "")
    snap = app.effective_snapshot("server", "llm", "")
    out.append("无模型时校验：%s"
               % [m for _k, m in B.validate(
                   snap, "server", B.resolve_exe(engine, "server"),
                   model="", role="llm")])
    out.append("无引擎时校验：%s"
               % [m for _k, m in B.validate(snap, "server", None, model="x",
                                            role="llm")][:1])

    # ---------------- 11. 界面开关同步
    app.show("server")
    root.update()
    app.role = "llm"
    app._sync_role_fields()
    root.update()
    llm_visible = [k for k, r in app.rows.items()
                   if r.f.page == "server" and r.chk.winfo_ismapped()]
    out.append("服务页可见项（只有一个角色，不该再出现双份）：%s"
               % llm_visible)
    out.append("服务页没有 embedding 专属项=%s"
               % (not any(k in llm_visible for k in
                          ("embeddings", "pooling", "embd_normalize",
                           "reranking"))))

    app.show("lib")
    root.update()
    out.append("面板尺寸：侧边栏 %dx%d 内容区 %dx%d"
               % (app.sidebar.winfo_width(), app.sidebar.winfo_height(),
                  app.content.winfo_width(), app.content.winfo_height()))
    app._show_help() if hasattr(app, "_show_help") else app.show_help()
    root.update()
    tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    # 别把托盘的隐藏 owner 窗口也一起销毁了（它是 App 长期持有的 Toplevel）
    _tray_owner = getattr(app.tray, "_owner", None) if app.tray else None
    for w in tops:
        if w is _tray_owner:
            continue
        w.destroy()
    out.append("参数对照表窗口：OK（%d 个）"
               % len([w for w in tops if w is not _tray_owner]))

    shutil.rmtree(fake_root, ignore_errors=True)

    # ---------------- 8. 托盘 / 开机自启 / 单实例（零依赖，纯 ctypes）
    from . import tray as _TRAY
    out.append("托盘模块导入 OK，图标路径=%s" % _TRAY._icon_path())
    out.append("图标文件存在=%s" % os.path.isfile(_TRAY._icon_path()))
    # App.__init__ 里已经调过 _setup_tray()
    out.append("App.tray 已建立=%s" % (app.tray is not None))
    if app.tray is not None:
        try:
            app.tray.show()
            app.tray.hide()
            out.append("托盘 show/hide 不抛异常=OK")
        except Exception as exc:
            out.append("托盘 show/hide 抛异常=<<< %r" % exc)
    # 单实例：首个实例拿到句柄；同名二次（首个仍在）应判为「已存在」
    h = _TRAY.acquire_mutex("buun_selftest_instance_x")
    out.append("首次 acquire_mutex 拿到句柄=%s" % bool(h))
    h2 = _TRAY.acquire_mutex("buun_selftest_instance_x")
    out.append("同名二次 acquire_mutex 判为已存在=%s（应 True）" % (h2 == 0))
    if h:
        _TRAY.kernel32.CloseHandle(h)
    if h2:
        _TRAY.kernel32.CloseHandle(h2)
    # 开机自启：selftest 走临时文件，不碰真实注册表
    _TRAY.set_autostart(True)
    out.append("set_autostart(True) → is_autostart=%s（应 True）"
               % _TRAY.is_autostart())
    _TRAY.set_autostart(False)
    out.append("set_autostart(False) → is_autostart=%s（应 False）"
               % _TRAY.is_autostart())

    out.append("OK")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: List[str] = None) -> int:
    _force_utf8_io()
    args = list(sys.argv[1:] if argv is None else argv)
    base_dir = _base_dir()
    selftest = "--selftest" in args
    start_minimized = "--minimized" in args

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        msg = ("无法创建窗口：%s\n\n请确认使用的是带 tkinter 的 Python "
               "（本机建议用 E:\\Python\\Python311\\pythonw.exe）。" % exc)
        print(msg)
        return 1

    T.apply(root)
    root.title(APP_TITLE)

    # 自检用独立的临时配置目录，绝不改动用户的 config.json
    if selftest:
        os.environ["BUUN_SELFTEST"] = "1"
        work_dir = os.path.join(tempfile.gettempdir(), "buun_selftest_cfg")
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = base_dir

    # 单实例：已有实例则把它的窗口恢复出来，自己直接退出
    # （自检时跳过，否则开着 GUI 跑 --selftest 会被误判成「已有实例」）
    from . import tray as TRAY
    if not selftest and not TRAY.single_instance_guard(APP_TITLE):
        try:
            root.destroy()
        except Exception:
            pass
        return 0

    try:
        app = App(root, work_dir, start_minimized=start_minimized)
    except Exception:
        detail = traceback.format_exc()
        with open(_log_path(base_dir, "last_error.log"), "w",
                  encoding="utf-8") as fh:
            fh.write(detail)
        try:
            from tkinter import messagebox
            messagebox.showerror(APP_TITLE, "启动失败：\n\n" + detail[-1800:])
        except Exception:
            pass
        print(detail)
        return 1

    if PORTABLE_NOTE:
        app.log_append(PORTABLE_NOTE, "warn")
    if getattr(app.store, "migration_note", ""):
        app.log_append(app.store.migration_note, "warn")

    if selftest:
        try:
            lines = run_selftest(app, root)
        except Exception:
            lines = ["自检异常：", traceback.format_exc()]
        text = "\n".join(lines)
        with open(_log_path(base_dir, "selftest.log"), "w",
                  encoding="utf-8") as fh:
            fh.write(text + "\n")
        try:
            print(text)
        except Exception:  # noqa: BLE001
            pass                    # 控制台编码千奇百怪，打不出来就算了（都写进日志了）
        try:
            app._closing = True
            app.on_close()
        except Exception:
            try:
                root.destroy()
            except Exception:
                pass
        return 0 if lines and lines[-1] == "OK" else 2

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
