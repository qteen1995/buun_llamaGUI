# -*- coding: utf-8 -*-
"""主窗口。

导航结构（替代原来的「模式按钮栏 + 参数页签栏」两排横向栏）：

    左侧一条竖向导航
      ├ 模型        模型库      —— 扫描 / 表格排序 / 分类 / 能力图标 / 加载与切换
      ├ 运行模式    服务 对话
      └ 参数        加载参数 对话参数 推测解码   （跟随模型库中选中的模型）

底部常驻：等价命令行 / 运行日志；顶栏一张卡片显示当前模型与启停。
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import builder as B
from . import engine as E
from . import gguf as GG
from . import manager as MG
from . import process as P
from . import probe as PR
from . import router as ROU
from . import runtime as RT
from . import scan as SC
from . import schema as S
from . import theme as T
from . import tray as TRAY
from .control_api import ControlAPI
from .store import Store, model_key
from .theme import C
from .unified import UnifiedBackend
from .widgets import (LogView, ScrollableFrame, Segmented, Sidebar, Tooltip,
                      classify)

APP_TITLE = "buun-llama 启动器"

# 进程非 0 退出时的定向诊断：(正则, 标题, 建议)
# 扫的是这个进程最后几百行输出 —— 比让用户自己往上翻靠谱
_EXIT_HINTS: Tuple[Tuple["re.Pattern[str]", str, str], ...] = (
    (re.compile(r"unknown model architecture:\s*'([^']+)'", re.I),
     "模型架构不被支持",
     "这不是参数问题：llama-server 里没有编译进这个架构。"
     "①把 buun-llama-cpp 更新到支持它的版本，再在「引擎目录」指过去；"
     "②或者换一个模型。llama.cpp 上游迭代很快，新架构往往要先更新才能用。"),
    (re.compile(r"unknown argument", re.I),
     "参数不被支持",
     "启动器没把这个参数过滤掉 —— 点一次「探测参数支持」，"
     "重启后再试；还这样就翻日志看看是哪个参数。"),
    (re.compile(r"out of memory|failed to allocate|CUDA out of memory"
                r"|ggml_cuda.*out of memory", re.I),
     "显存不足",
     "减小「GPU 卸载层数 -ngl」或「上下文 -c」，"
     "或者换量化更狠的模型（IQ4_XS / Q4_K_M → Q3_K_S）。"),
    (re.compile(r"invalid magic|unsupported (model )?file version"
                r"|tensor .* not found|file is (corrupt|truncated)", re.I),
     "模型文件损坏或不完整",
     "GGUF 分片没下全或写坏了，重新下载这个模型。"),
    (re.compile(r"(no such file|unable to open|failed to open)", re.I),
     "文件路径不对",
     "检查模型路径，以及分片模型是不是少了几片。"),
    (re.compile(r"CUDA error|no kernel image|driver", re.I),
     "CUDA / 驱动问题",
     "检查显卡驱动与 CUDA 运行时是否匹配这份编译产物。"),
    (re.compile(r"failed.*speculative.*model.*context"
                r"|failed to create llama_context from model"
                r"|speculative.*failed to measure", re.I),
     "推测解码草稿模型无法加载",
     "草稿模型（-md）创建 llama_context 失败。常见原因："
     "①草稿模型的架构与主模型不兼容（需要同一架构族）；"
     "②草稿模型的上下文长度或头维度与主模型不匹配；"
     "③显存不够同时装两个模型，先关掉推测解码单独加载主模型试试。"),
)

# 侧边栏图标：**故意留空**。
# 用户要求把「模型库 / 服务 / 对话 / 生成 / 加载参数 / 对话参数 / 推测解码」
# 这些标题前面的符号全去掉 —— 导航就是一列纯文字，干净。
# widgets.Sidebar 里已经有 `if icon:` 分支，空串就不渲染前缀。
# 要恢复图标的话，在这里填 {页面 id: 字符} 就行。
PAGE_ICONS: Dict[str, str] = {}

SIDEBAR_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str, str], ...]], ...] = tuple(
    (zh, tuple((pid, S.PAGE_ZH.get(pid, pid), PAGE_ICONS.get(pid, ""))
               for pid in ids))
    for zh, _en, ids in S.NAV_GROUPS)

CATEGORY_TINT = {
    "LLMS": "#ffffff",
    "Embedding": "#f2f7ff",
    "Drafters": "#f7f3ff",
}


# --------------------------------------------------------------------------- #
# 一行参数
# --------------------------------------------------------------------------- #

class FieldRow:
    def __init__(self, app: "App", parent: ttk.Frame, f: S.F, rowno: int,
                 state: Optional[Dict[str, Any]] = None) -> None:
        self.app = app
        self.f = f
        st = state or {"on": bool(f.on), "value": f.default_value()}
        self.on_var = tk.BooleanVar(value=bool(st.get("on")))
        self.var = tk.StringVar(value=str(st.get("value", "") or ""))
        self.text: Optional[tk.Text] = None
        self.ctrl: Optional[tk.Widget] = None
        self.seg: Optional[Segmented] = None
        self.browse: Optional[ttk.Button] = None
        self.holder: Optional[ttk.Frame] = None
        self.unsupported = False
        self.hidden_by_probe = False
        self.visible = True
        self.ctrl_state = "readonly" if f.kind == S.K_CHOICE else "normal"

        if f.kind == S.K_BOOL:
            self.chk = ttk.Checkbutton(
                parent, text=f.zh, variable=self.on_var,
                style="Card.TCheckbutton", command=self._on_toggle)
            self.chk.grid(row=rowno, column=0, columnspan=3, sticky="w", pady=4)
            self.lbl: Any = self.chk
            self.flag_lbl = ttk.Label(parent, text="", style="MutedCard.TLabel")
            self.flag_lbl.grid(row=rowno, column=3, sticky="w", padx=(10, 4))
            self.tip_targets = [self.chk]
        else:
            self.chk = ttk.Checkbutton(parent, variable=self.on_var,
                                       style="Card.TCheckbutton",
                                       command=self._on_toggle)
            self.chk.grid(row=rowno, column=0, sticky="w", padx=(2, 4), pady=4)
            self.lbl = ttk.Label(parent, text=f.zh, style="Card.TLabel")
            self.lbl.grid(row=rowno, column=1, sticky="w", padx=(0, 12), pady=4)

            holder = ttk.Frame(parent, style="Card.TFrame")
            holder.grid(row=rowno, column=2, sticky="ew", pady=4)
            holder.columnconfigure(0, weight=1)
            self.holder = holder
            self._build_control(holder)

            self.flag_lbl = ttk.Label(parent, text="", style="MutedCard.TLabel")
            self.flag_lbl.grid(row=rowno, column=3, sticky="nw",
                               padx=(10, 4),
                               pady=(9 if f.kind == S.K_MULTI else 6, 4))
            self.tip_targets = [self.chk, self.lbl, self.flag_lbl]
            for extra in (self.ctrl, self.browse, self.seg):
                if extra is not None:
                    self.tip_targets.append(extra)

        tip = self._tooltip_text()
        for w in self.tip_targets:
            Tooltip(w, tip)
        self._apply_state()
        self._update_flag_label()

    # ------------------------------------------------------ 控件构建
    def _build_control(self, holder: ttk.Frame) -> None:
        f = self.f
        width = f.width

        if f.kind == S.K_GEAR:
            self.seg = Segmented(holder, list(f.choices),
                                 on_change=self._on_gear)
            self.seg.grid(row=0, column=0, sticky="w")
            if self.var.get() not in f.choices and f.choices:
                self.var.set(f.choices[0])
            self.seg.set_value(self.var.get())
            return

        if f.kind == S.K_CHOICE:
            self.ctrl = ttk.Combobox(holder, textvariable=self.var,
                                     values=list(f.choices), state="readonly",
                                     width=width)
        elif f.kind == S.K_CHOICE_EDIT:
            self.ctrl = ttk.Combobox(holder, textvariable=self.var,
                                     values=list(f.choices), width=width)
        elif f.kind == S.K_MULTI:
            box = tk.Frame(holder, background=C["border_strong"])
            box.grid(row=0, column=0, sticky="ew")
            box.columnconfigure(0, weight=1)
            self.text = tk.Text(box, height=4, width=44, wrap="none",
                                font=T.FONT_MONO, relief="flat", borderwidth=0,
                                background=C["panel"], foreground=C["text"],
                                insertbackground=C["text"], padx=6, pady=4)
            vbar = ttk.Scrollbar(box, orient="vertical",
                                 command=self.text.yview)
            self.text.configure(yscrollcommand=vbar.set)
            self.text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
            vbar.grid(row=0, column=1, sticky="ns", pady=1, padx=(0, 1))
            if self.var.get():
                self.text.insert("1.0", self.var.get())
            self.text.bind("<KeyRelease>", self._on_text_change)
            self.text.bind("<<Paste>>", self._on_text_change)
            self.text.bind("<FocusOut>", self._on_text_change)
            return
        else:
            self.ctrl = ttk.Entry(holder, textvariable=self.var, width=width)

        self.ctrl.grid(row=0, column=0, sticky="ew")
        if f.kind in S.FILE_KINDS:
            label = {S.K_OPEN: "浏览…", S.K_SAVE: "另存为…",
                     S.K_DIR: "选择…"}[f.kind]
            self.browse = ttk.Button(holder, text=label, style="Mini.TButton",
                                     command=self._browse)
            self.browse.grid(row=0, column=1, sticky="w", padx=(6, 0))
        if isinstance(self.ctrl, ttk.Combobox):
            self.ctrl.bind("<<ComboboxSelected>>", self._on_text_change)

    # ------------------------------------------------------------ 提示
    def _tooltip_text(self) -> str:
        f = self.f
        lines = [f.en]
        if f.flag:
            lines.append("命令行参数：%s" % f.flag)
        if f.kind == S.K_BOOL:
            lines.append("勾选 = 附加该开关；不勾选 = 完全不传。")
        elif f.kind == S.K_GEAR and f.argmap:
            lines.append("挡位对应：")
            for k, v in f.argmap.items():
                lines.append("  %s → %s" % (k, " ".join(v) if v else "（不传）"))
        else:
            lines.append("勾选后才会写入命令行；不勾选 = 使用引擎默认值。")
        if f.default_value():
            lines.append("默认：%s" % f.default_value())
        if f.ini_key:
            lines.append("每模型参数文件里的键名：%s" % f.ini_key)
        lines.append("")
        lines.append(f.hint)
        return "\n".join(lines)

    # ------------------------------------------------------------ 状态
    def _on_gear(self, value: str) -> None:
        self.var.set(value)
        if not self.on_var.get():
            self.on_var.set(True)
        self._apply_state()
        self._update_flag_label()
        self.app.on_value_change()

    def _on_toggle(self) -> None:
        self._apply_state()
        self._update_flag_label()
        self.app.on_value_change()

    def _apply_state(self) -> None:
        enabled = self.on_var.get()
        if self.f.kind == S.K_BOOL:
            return
        if self.seg is not None:
            self.seg.set_enabled(enabled)
        widgets: List[tk.Widget] = []
        if self.ctrl is not None:
            widgets.append(self.ctrl)
        if self.browse is not None:
            widgets.append(self.browse)
        for w in widgets:
            try:
                if isinstance(w, ttk.Combobox):
                    w.configure(state=(self.ctrl_state if enabled
                                       else "disabled"))
                else:
                    w.configure(state=("normal" if enabled else "disabled"))
            except tk.TclError:
                pass
        if self.text is not None:
            self.text.configure(state=("normal" if enabled else "disabled"))
            self.text.configure(background=C["panel"] if enabled
                                else "#f0f2f5")
        colour = C["text"] if enabled else C["faint"]
        try:
            self.lbl.configure(foreground=colour)
        except tk.TclError:
            pass

    def _update_flag_label(self) -> None:
        f = self.f
        try:
            if f.kind in (S.K_GEAR, S.K_CHOICE) and f.argmap:
                argv = f.argmap.get(self.var.get())
                text = " ".join(argv) if argv else "（不传参数）"
            elif f.kind == S.K_BOOL:
                text = f.flag or "（界面设置）"
            elif f.kind == S.K_GEAR:
                text = "（界面设置）"
            else:
                text = f.flag or ("（位置参数）" if f.positional
                                  else "（界面设置）")
            if self.unsupported:
                text = "%s ⚠ 本版本无此参数" % text
            self.flag_lbl.configure(
                text=text,
                style="ErrCard.TLabel" if self.unsupported
                else "MutedCard.TLabel")
        except tk.TclError:
            pass

    def mark_unsupported(self, bad: bool) -> None:
        self.unsupported = bool(bad)
        if bad and self.on_var.get():
            self.on_var.set(False)
            self._apply_state()
        self._update_flag_label()

    def apply_visibility(self, show: bool) -> None:
        """整行显示/隐藏（含承载控件的 holder，否则会留下空白行）。"""
        self.visible = bool(show)
        for w in self.app._row_widgets(self):
            try:
                w.grid() if show else w.grid_remove()
            except tk.TclError:
                pass

    def mark_error(self, is_error: bool) -> None:
        try:
            if self.f.kind == S.K_BOOL:
                self.chk.configure(
                    foreground=C["err"] if is_error else C["text"])
            else:
                self.lbl.configure(
                    foreground=C["err"] if is_error else
                    (C["text"] if self.on_var.get() else C["faint"]))
        except tk.TclError:
            pass

    def _on_text_change(self, _evt=None) -> None:
        self._update_flag_label()
        self.app.schedule_refresh()

    def _browse(self) -> None:
        f = self.f
        cur = self.var.get().strip().strip('"')
        if cur and os.path.isfile(cur):
            init = os.path.dirname(cur)
        elif cur and os.path.isdir(cur):
            init = cur
        else:
            init = self.app.default_dir()
        if f.kind == S.K_DIR:
            path = filedialog.askdirectory(initialdir=init,
                                           parent=self.app.root)
        elif f.kind == S.K_SAVE:
            path = filedialog.asksaveasfilename(
                initialdir=init, parent=self.app.root,
                initialfile=os.path.basename(cur) or "output.gguf",
                defaultextension=".gguf",
                filetypes=[("GGUF 模型", "*.gguf"),
                           ("全部文件", "*.*")])
        else:
            path = filedialog.askopenfilename(
                initialdir=init, parent=self.app.root,
                filetypes=[("GGUF 模型", "*.gguf"),
                           ("Jinja 模板", "*.jinja *.txt"),
                           ("文本文件", "*.txt *.dat"),
                           ("可执行文件", "*.exe"), ("全部文件", "*.*")])
        if path:
            self.var.set(os.path.normpath(path))
            if not self.on_var.get():
                self.on_var.set(True)
            self._apply_state()
            self._update_flag_label()
            self.app.on_value_change()

    # ------------------------------------------------------------ 取值
    def get(self) -> Dict[str, Any]:
        value = self.var.get()
        if self.text is not None:
            value = self.text.get("1.0", "end-1c")
        return {"on": bool(self.on_var.get()), "value": value}

    def set(self, st: Optional[Dict[str, Any]]) -> None:
        st = st or {}
        self.on_var.set(bool(st.get("on")))
        val = st.get("value", "")
        val = "" if val is None else str(val)
        # 挡位改名 / 挡位改成数字框之后，存档里的老值（如「high」）要在这里纠正，
        # 否则挡位控件会显示成一个不存在的选项、数字框里会塞着中文
        if self.f.kind in S.CHOICE_KINDS:
            val = B.valid_choice(self.f, val)
        elif (self.f.kind in (S.K_INT, S.K_FLOAT) and val
              and not B.numeric_ok(val, self.f.kind)):
            val = str(self.f.default_value() or "")
        self.var.set(val)
        if self.seg is not None:
            self.seg.set_value(self.var.get())
        if self.text is not None:
            self.text.delete("1.0", "end")
            if val:
                self.text.insert("1.0", str(val))
        self._apply_state()
        self._update_flag_label()

    def set_section_lang(self, english: bool) -> None:
        """小节标题也跟着中英切换。"""
        for lbl, zh, en in self.section_lbls:
            try:
                lbl.configure(text="── %s " % (en if english else zh))
            except tk.TclError:
                pass

    def set_label_lang(self, english: bool) -> None:
        text = self.f.en.split(" (")[0] if english else self.f.zh
        try:
            if self.f.kind == S.K_BOOL:
                self.chk.configure(text=text)
            else:
                self.lbl.configure(text=text)
        except tk.TclError:
            pass


# --------------------------------------------------------------------------- #
# 参数页
# --------------------------------------------------------------------------- #

class ParamPage(ttk.Frame):
    """一组参数（可带小节标题）。"""

    def __init__(self, app: "App", page: str) -> None:
        super().__init__(app.content, style="TFrame")
        self.app = app
        self.page = page
        self.body = ScrollableFrame(self)
        self.body.pack(fill="both", expand=True)
        inner = self.body.body
        inner.columnconfigure(2, weight=1, minsize=190)
        inner.columnconfigure(3, minsize=150)

        groups: List[Tuple[Optional[str], List[S.F]]] = []
        # (标题控件, 中文, 英文)：小节标题也要跟着中英切换
        self.section_lbls: List[Tuple[ttk.Label, str, str]] = []
        self._en = bool(app.store.pref("show_english"))
        sections = S.sections_for(page)
        if sections:
            # 声明了小节的页面：没有 section 的字段先作为「无标题组」排在最前
            # （服务页就是这样：服务/端口那一批 + 驻留策略 + 路由预置）
            loose = list(S.fields_in(page, ""))
            if loose:
                groups.append((None, loose))
            for sid, title in sections:
                fields = list(S.fields_in(page, sid))
                if fields:
                    groups.append((sid, fields))
        else:
            fields = list(S.fields_for(page))
            if fields:
                groups.append((None, fields))

        rowno = 0
        if page == "server":
            # 角色选择器：服务页的参数是「按角色」保存的，所以先明确在配谁
            head = ttk.Frame(inner, style="Alt.TFrame", padding=(10, 6))
            head.grid(row=0, column=0, columnspan=4, sticky="ew",
                      pady=(2, 8))
            ttk.Label(head, text="正在配置角色", style="Alt.TLabel").pack(
                side="left")
            ttk.Label(head,
                      text="服务跑的是多模型路由：一个进程下挂多个模型，"
                           "下面的「已加载模型」列表可以分别卸载",
                      style="Alt.TLabel").pack(side="left")
            ttk.Button(head, text="导出路由预置 INI", style="Mini.TButton",
                       command=app.export_router_preset).pack(side="right")
            rowno = 1
        elif page in ("load", "chat", "spec"):
            # 这三个页面的参数是「按模型」保存的，先说明在编辑哪个模型
            head = ttk.Frame(inner, style="Alt.TFrame", padding=(10, 6))
            head.grid(row=0, column=0, columnspan=4, sticky="ew",
                      pady=(2, 8))
            ttk.Label(head, text="正在编辑模型", style="Alt.TLabel").pack(
                side="left")
            lbl = ttk.Label(head, text="", style="Alt.TLabel",
                            font=T.FONT_UI_BOLD)
            lbl.pack(side="left", padx=(8, 10))
            ttk.Label(head,
                      text="加载/对话参数按模型分别保存，切到别的模型会自动带出它自己的",
                      style="Alt.TLabel").pack(side="left")
            app.model_labels.append(lbl)
            rowno = 1
        for title, fields in groups:
            if title:
                zh = S.SECTION_ZH.get((page, title), title)
                en = S.SECTION_EN.get((page, title), title)
                cap = ttk.Frame(inner, style="Card.TFrame")
                cap.grid(row=rowno, column=0, columnspan=4, sticky="ew",
                         pady=(14 if rowno else 2, 2))
                cap_lbl = ttk.Label(cap, text="── %s " % (en if self._en
                                                          else zh),
                                    style="MutedCard.TLabel")
                cap_lbl.pack(side="left")
                self.section_lbls.append((cap_lbl, zh, en))
                ttk.Separator(cap, orient="horizontal").pack(
                    side="left", fill="x", expand=True, padx=(8, 0))
                rowno += 1
            for f in fields:
                if f.hidden:
                    continue           # 纯计算项（如 --models-max），不占界面
                row = FieldRow(app, inner, f, rowno, app.initial_state(f))
                app.rows[f.key] = row
                rowno += 1
        ttk.Frame(inner, style="Card.TFrame", height=16).grid(
            row=rowno, column=0, columnspan=4, sticky="ew")


# --------------------------------------------------------------------------- #
# LoRA 适配器面板（加载参数页的一个自定义小节）
# --------------------------------------------------------------------------- #

LORA_FILETYPES = [("LoRA 适配器", "*.gguf *.safetensors *.bin"),
                  ("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors"),
                  ("全部文件", "*.*")]


class LoRAPanel(ttk.Frame):
    """LoRA 适配器列表：手动添加路径、设比例、排序、删除。

    数据存在「这个模型的加载参数」里的 ``lora`` 项：value 是一个 JSON 字符串，
    所以它跟别的加载参数一样跟着模型走、也走同一套命令行拼装。
    ⚠️ buun 的 LoRA 参数是**单值逗号分隔**形式（跟上游 llama.cpp 不一样）：
        --lora a.gguf,b.gguf
        --lora-scaled p.gguf:0.8,q.gguf:0.5
    这也是它能写进路由预置 INI 的原因（两值写法会被引擎的 preset 层拒绝）。
    """

    def __init__(self, app: "App", parent: tk.Widget) -> None:
        super().__init__(parent, style="Card.TFrame", padding=(2, 4))
        self.app = app
        self.columnconfigure(0, weight=1)

        ttk.Label(
            self,
            text=("给当前模型挂 LoRA 适配器。路径手动添加（也可以选文件）；"
                  "一个模型可以挂多个，按列表顺序依次应用。\n"
                  "比例 = 1.0 用 --lora，其它值用 --lora-scaled 路径:比例"
                  "（负数 = 反向效果）。列表按模型保存，改完要重新加载模型才生效。"),
            style="MutedCard.TLabel", justify="left").grid(
                row=0, column=0, sticky="w", pady=(0, 4))

        box = tk.Frame(self, background=C["border_strong"])
        box.grid(row=1, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(box, columns=("path", "scale"), height=4,
                                 show="headings", selectmode="extended")
        self.tree.heading("path", text="LoRA 路径")
        self.tree.heading("scale", text="比例")
        self.tree.column("path", width=560, anchor="w")
        self.tree.column("scale", width=70, anchor="center", stretch=False)
        vbar = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        vbar.grid(row=0, column=1, sticky="ns", padx=(0, 1), pady=1)
        self.tree.tag_configure("missing", foreground=C["err"])
        self.tree.bind("<Double-1>", lambda _e: self.edit_scale())

        btns = ttk.Frame(self, style="Card.TFrame")
        btns.grid(row=2, column=0, sticky="w", pady=(6, 2))
        for text, cmd in (("添加文件…", self.add_files),
                          ("手动输入路径…", self.add_manual),
                          ("改比例…", self.edit_scale),
                          ("移除", self.remove_selected),
                          ("上移", lambda: self.move(-1)),
                          ("下移", lambda: self.move(1)),
                          ("清空", self.clear_all)):
            ttk.Button(btns, text=text, style="Mini.TButton",
                       command=cmd).pack(side="left", padx=(0, 6))

        self.argv_lbl = ttk.Label(self, text="", style="MutedCard.TLabel",
                                  justify="left", font=T.FONT_MONO_SMALL)
        self.argv_lbl.grid(row=3, column=0, sticky="w", pady=(2, 0))

    # ----------------------------------------------------------- 数据读写
    def _model(self) -> str:
        return self.app.selected_model or ""

    def entries(self) -> List[Dict[str, Any]]:
        path = self._model()
        if not path:
            return []
        st = (self.app.store.snapshot_ref(path, "load") or {}).get("lora") or {}
        return [{"path": p, "scale": (1.0 if s is None else s)}
                for p, s in B.parse_lora(st.get("value", ""))]

    def _store(self, items: List[Dict[str, Any]]) -> None:
        path = self._model()
        if not path:
            return
        snap = self.app.store.snapshot_ref(path, "load")
        snap["lora"] = {"on": True, "value": json.dumps(items,
                                                        ensure_ascii=False)}
        self.app.store.save()
        self.reload()
        self.app.schedule_refresh()
        self.app.log_append("LoRA 列表已更新（%d 个）" % len(items), "sys")

    # ------------------------------------------------------------- 渲染
    def reload(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        path = self._model()
        if not path:
            self.argv_lbl.configure(text="先在「模型库」里选一个模型。")
            return
        items = self.entries()
        for i, it in enumerate(items):
            missing = not os.path.isfile(it["path"].strip('"'))
            self.tree.insert(
                "", "end", iid=str(i),
                values=(it["path"], "%g" % it["scale"]),
                tags=("missing",) if missing else ())
        args = B.lora_args(json.dumps(items, ensure_ascii=False))
        if args:
            self.argv_lbl.configure(
                text="命令行：%s" % " ".join(args),
                foreground=C["text"])
        else:
            self.argv_lbl.configure(text="当前没有挂 LoRA（不传参数）",
                                    foreground=C["muted"])
        bad = [it["path"] for it in items
               if not os.path.isfile(it["path"].strip('"'))]
        if bad:
            self.argv_lbl.configure(
                text=self.argv_lbl.cget("text")
                + "\n⚠ 找不到文件：%s" % "、".join(os.path.basename(b)
                                                   for b in bad),
                foreground=C["err"])

    # ------------------------------------------------------------- 操作
    def _require_model(self) -> bool:
        if self._model():
            return True
        messagebox.showinfo("还没有选模型",
                            "LoRA 列表是按模型保存的，请先在「模型库」里"
                            "选一个模型。", parent=self.app.root)
        return False

    def add_files(self) -> None:
        if not self._require_model():
            return
        init = os.path.dirname(self._model()) or self.app.base_dir
        picked = filedialog.askopenfilenames(
            parent=self.app.root, initialdir=init, title="选择 LoRA 适配器",
            filetypes=LORA_FILETYPES)
        if not picked:
            return
        items = self.entries()
        have = {os.path.normcase(it["path"]) for it in items}
        added = 0
        for p in picked:
            full = os.path.normcase(os.path.normpath(p))
            if full in have:
                continue
            items.append({"path": os.path.normpath(p), "scale": 1.0})
            have.add(full)
            added += 1
            self.app.store.push_unique("recent_loras", os.path.normpath(p))
        if added:
            self._store(items)
        else:
            self.app.log_append("这些 LoRA 已经在列表里了", "warn")

    def add_manual(self) -> None:
        if not self._require_model():
            return
        recent = list(self.app.store.get("recent_loras") or [])
        tip = recent[0] if recent else ""
        path = simpledialog.askstring(
            "手动添加 LoRA 路径",
            "填 LoRA 适配器的完整路径（.gguf / .safetensors）：\n"
            "多个可以用 ; 或换行分隔。",
            initialvalue=tip, parent=self.app.root)
        if not path:
            return
        raw = path.replace("\n", ";").replace("\r", ";")
        parts = [p.strip().strip('"') for p in raw.split(";")]
        items = self.entries()
        have = {os.path.normcase(it["path"]) for it in items}
        added = 0
        for p in parts:
            if not p:
                continue
            full = os.path.normcase(os.path.normpath(p))
            if full in have:
                continue
            items.append({"path": os.path.normpath(p), "scale": 1.0})
            have.add(full)
            added += 1
            self.app.store.push_unique("recent_loras", os.path.normpath(p))
        if not added:
            self.app.log_append("没有新增 LoRA（路径为空或已在列表里）", "warn")
            return
        for p in [it["path"] for it in items if it["path"] in parts
                  or os.path.normcase(it["path"]) in
                  {os.path.normcase(os.path.normpath(x)) for x in parts}]:
            if not os.path.isfile(p):
                self.app.log_append("注意：%s 不存在" % p, "warn")
        self._store(items)

    def _selected(self) -> List[int]:
        return sorted(int(i) for i in self.tree.selection())

    def edit_scale(self) -> None:
        sel = self._selected()
        if not sel:
            messagebox.showinfo("先选一个", "先在列表里点一个 LoRA。",
                                parent=self.app.root)
            return
        items = self.entries()
        cur = items[sel[0]]["scale"] if sel[0] < len(items) else 1.0
        val = simpledialog.askfloat(
            "改比例", "缩放比例（1.0 = 原样；0.5 = 减半；-1.0 = 反向）：",
            initialvalue=cur, minvalue=-100.0, maxvalue=100.0,
            parent=self.app.root)
        if val is None:
            return
        for i in sel:
            if i < len(items):
                items[i]["scale"] = float(val)
        self._store(items)

    def remove_selected(self) -> None:
        sel = self._selected()
        if not sel:
            messagebox.showinfo("先选一个", "先在列表里点要移除的 LoRA。",
                                parent=self.app.root)
            return
        items = [it for i, it in enumerate(self.entries()) if i not in sel]
        self._store(items)

    def move(self, delta: int) -> None:
        sel = self._selected()
        if len(sel) != 1:
            messagebox.showinfo("只支持单个", "一次只移动一个，先选一个。",
                                parent=self.app.root)
            return
        items = self.entries()
        i = sel[0]
        j = i + delta
        if not (0 <= i < len(items) and 0 <= j < len(items)):
            return
        items[i], items[j] = items[j], items[i]
        self._store(items)
        self.tree.selection_set(str(j))

    def clear_all(self) -> None:
        if not self.entries():
            return
        if not messagebox.askyesno("清空 LoRA",
                                   "把这个模型的 LoRA 列表清空？",
                                   parent=self.app.root):
            return
        self._store([])


# --------------------------------------------------------------------------- #
# 模型库页
# --------------------------------------------------------------------------- #

class LoRAPage(ttk.Frame):
    """LoRA 适配器（独立导航页）。

    以前它是「加载参数」页里的一个自定义小节，现在按用户要求挪到左侧导航里，
    跟 模型库 / 服务 / 对话 / 生成 / 加载参数 / 对话参数 / 推测解码 并排。
    数据仍在 load 作用域（跟模型走），`--lora` / `--lora-scaled` 照旧写进
    这个模型的路由预置段 —— 换句话说，搬的只是「编辑入口」。
    """

    def __init__(self, app: "App") -> None:
        super().__init__(app.content, style="TFrame")
        self.app = app
        body = ScrollableFrame(self)
        body.pack(fill="both", expand=True)
        inner = body.body
        inner.columnconfigure(0, weight=1)

        head = ttk.Frame(inner, style="Alt.TFrame", padding=(10, 6))
        head.grid(row=0, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(head, text="正在编辑模型", style="Alt.TLabel").pack(side="left")
        lbl = ttk.Label(head, text="", style="Alt.TLabel",
                        font=T.FONT_UI_BOLD)
        lbl.pack(side="left", padx=(8, 10))
        ttk.Label(head, text="LoRA 列表按模型分别保存，切到别的模型会自动带出它自己的",
                  style="Alt.TLabel").pack(side="left")
        app.model_labels.append(lbl)

        self.panel = LoRAPanel(app, inner)
        self.panel.grid(row=1, column=0, sticky="ew")
        ttk.Frame(inner, style="Card.TFrame", height=16).grid(
            row=2, column=0, sticky="ew")

    def reload(self) -> None:
        self.panel.reload()


class EmbDialog(tk.Toplevel):
    """embedding / reranker 的专属设置窗口（需求 5）。

    embedding 模型没有对话模板、采样、推理强度这些东西 —— 把「加载参数」和
    「对话参数」两大页塞给它，90% 的项都是无意义的。所以模型库选中 embedding
    类模型时，那个「编辑 xxx 参数」按钮会换成一个「embedding 设置」，
    弹出来的就是这个小窗口，里面只有引擎真正认的 5 个 embedding 参数：

        --embedding / --rerank / --pooling / --embd-normalize
        --embd-gemma-default

    （KV 档位之类由 builder.emb_guard 按模型几何自动处理，不用用户管。）
    取值存在这个模型的 ``emb`` 作用域里，跟别的模型参数一样走 config\models\<名>.json。
    """

    def __init__(self, app: "App", path: str) -> None:
        super().__init__(app.root)
        self.app = app
        self.path = path
        self.rows: Dict[str, FieldRow] = {}
        self.configure(background=C["panel"])
        self.title("Embedding 设置 · %s" % os.path.basename(path)[:44])
        self.transient(app.root)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(self, style="Card.TFrame", padding=(16, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(2, weight=1, minsize=210)
        body.columnconfigure(3, minsize=96)
        ttk.Label(body, text="Embedding 专属设置", style="Card.TLabel",
                  font=T.FONT_UI_BOLD).grid(row=0, column=0, columnspan=4,
                                            sticky="w")
        ttk.Label(
            body,
            text=("只放引擎真正认的 embedding / reranker 参数，不加任何"
                  "对话、采样、推理链相关的项。\n"
                  "KV 档位会被自动钉成合适的值（bert 系必须 f16），不用管。"),
            style="MutedCard.TLabel", justify="left").grid(
                row=1, column=0, columnspan=4, sticky="w", pady=(2, 10))

        prev = app.selected_model
        app.selected_model = path
        try:
            for i, f in enumerate(S.fields_for("emb")):
                if f.hidden:
                    continue
                row = FieldRow(app, body, f, 2 + i, app.initial_state(f))
                app.rows[f.key] = row
                self.rows[f.key] = row
        finally:
            app.selected_model = prev

        bar = ttk.Frame(body, style="Card.TFrame")
        bar.grid(row=2 + len(self.rows) + 1, column=0, columnspan=4,
                 sticky="ew", pady=(14, 0))
        ttk.Button(bar, text="保存", style="Accent.TButton",
                   command=self.save).pack(side="right")
        ttk.Button(bar, text="取消", style="TButton",
                   command=self.close).pack(side="right", padx=(0, 6))
        self.bind("<Escape>", lambda _e: self.close())
        self.update_idletasks()
        # 摆在主窗口中间
        try:
            x = app.root.winfo_rootx() + (app.root.winfo_width()
                                         - self.winfo_width()) // 2
            y = app.root.winfo_rooty() + 120
            self.geometry("+%d+%d" % (max(0, x), max(0, y)))
        except Exception:  # noqa: BLE001
            pass
        self.grab_set()

    def save(self) -> None:
        prev = self.app.selected_model
        self.app.selected_model = self.path
        try:
            self.app.collect("emb")
        finally:
            self.app.selected_model = prev
        self.app.store.save()
        self.app.log_append("已保存 %s 的 embedding 设置"
                            % os.path.basename(self.path), "ok")
        if self.app.lib is not None:
            self.app.lib.refresh_table()
        self.close()

    def close(self) -> None:
        for k in list(self.rows):
            self.app.rows.pop(k, None)
        try:
            self.grab_release()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()


class LibraryPage(ttk.Frame):
    def __init__(self, app: "App") -> None:
        super().__init__(app.content, style="TFrame")
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ---------------- 工具条
        bar = ttk.Frame(self, style="Card.TFrame", padding=(12, 8))
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="扫描目录", style="Card.TLabel").grid(
            row=0, column=0, sticky="w")
        self.dir_var = tk.StringVar(value=str(app.store.get("models_dir")
                                              or ""))
        ent = ttk.Entry(bar, textvariable=self.dir_var)
        ent.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Button(bar, text="浏览…", style="Mini.TButton",
                   command=self.pick_dir).grid(row=0, column=2)
        ttk.Button(bar, text="↻ 扫描", style="Accent.TButton",
                   command=self.scan_now).grid(row=0, column=3, padx=(6, 0))
        Tooltip(ent, "填入模型所在目录后点「扫描」，会把这个目录（可含子目录）下"
                     "所有 .gguf 读一遍头部，列出架构、参数量、发布者、量化规格等。\n"
                     "mmproj 与分片的第 2 片之后会自动跳过。")
        self.recursive_var = tk.BooleanVar(
            value=bool(app.store.get("models_recursive", True)))
        ttk.Checkbutton(bar, text="含子目录", variable=self.recursive_var,
                        style="Card.TCheckbutton",
                        command=self._save_prefs).grid(row=0, column=4,
                                                       padx=(12, 0))
        ttk.Label(bar, text="深度", style="Card.TLabel").grid(row=0, column=5,
                                                              padx=(8, 2))
        self.depth_var = tk.StringVar(
            value=str(app.store.get("models_depth", 3)))
        ttk.Spinbox(bar, from_=1, to=12, width=3, textvariable=self.depth_var,
                    command=self._save_prefs).grid(row=0, column=6)

        # ---------------- 过滤条
        flt = ttk.Frame(self, style="Card.TFrame", padding=(12, 6))
        flt.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 0))
        flt.columnconfigure(6, weight=1)
        ttk.Label(flt, text="分类", style="Card.TLabel").grid(row=0, column=0)
        self.cat_seg = Segmented(flt, ("全部", "LLMS", "Embedding", "Drafters"),
                                 on_change=self._on_cat)
        self.cat_seg.grid(row=0, column=1, padx=(8, 14))
        self.cat_seg.set_value(str(app.store.pref("category", "全部")))

        caps = (("全部能力", ""), ("能读图", "vision"), ("MTP", "mtp"))
        self._cap_map = {c[0]: c[1] for c in caps}
        self.cap_seg = Segmented(flt, tuple(c[0] for c in caps),
                                 on_change=self._on_cap)
        self.cap_seg.grid(row=0, column=2, padx=(0, 14))
        cur_cap = str(app.store.pref("cap", ""))
        for label, code in self._cap_map.items():
            if code == cur_cap:
                self.cap_seg.set_value(label)

        self.only_set_var = tk.BooleanVar(
            value=bool(app.store.pref("only_set", False)))
        ttk.Checkbutton(flt, text="只看已调参", variable=self.only_set_var,
                        style="Card.TCheckbutton",
                        command=self._on_filter).grid(row=0, column=3,
                                                      padx=(0, 14))
        ttk.Label(flt, text="筛选", style="Card.TLabel").grid(row=0, column=4)
        self.q_var = tk.StringVar()
        qe = ttk.Entry(flt, textvariable=self.q_var, width=18)
        qe.grid(row=0, column=5, padx=(6, 10))
        qe.bind("<KeyRelease>", lambda _e: self.refresh_table())
        Tooltip(qe, "按模型名 / 文件名 / 架构 / 发布者 / 量化规格模糊匹配。")
        ttk.Label(
            flt, text="能力列：" + SC.LEGEND
            + "      模型名与发布者取自目录结构（…\\发布者\\模型名\\xxx.gguf）",
            style="MutedCard.TLabel").grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))
        self.count_lbl = ttk.Label(flt, text="", style="MutedCard.TLabel")
        self.count_lbl.grid(row=1, column=5, columnspan=2, sticky="e",
                            pady=(4, 0))

        # ---------------- 表格
        wrap = ttk.Frame(self, style="TFrame")
        wrap.grid(row=2, column=0, sticky="nsew", padx=10, pady=(6, 0))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(wrap, columns=[c[0] for c in SC.COLUMNS],
                                 show="headings", selectmode="browse")
        for cid, zh, _en, width, anchor in SC.COLUMNS:
            self.tree.heading(cid, text=zh,
                              command=lambda c=cid: self.sort_by(c))
            self.tree.column(cid, width=width, anchor=anchor, stretch=False)
        self.tree.column("name", stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        vb.grid(row=0, column=1, sticky="ns")
        hb = ttk.Scrollbar(wrap, orient="horizontal", command=self.tree.xview)
        hb.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vb.set, xscrollcommand=hb.set)
        for cat, tint in CATEGORY_TINT.items():
            self.tree.tag_configure(cat, background=tint)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        self.tree.bind("<Double-1>", lambda _e: self.edit_load())

        # ---------------- 详情
        det = ttk.Frame(self, style="Alt.TFrame", padding=(12, 8))
        det.grid(row=3, column=0, sticky="ew", padx=10, pady=(6, 8))
        det.columnconfigure(0, weight=1)
        self.detail_lbl = ttk.Label(det, text="在表格里选一个模型",
                                    style="Alt.TLabel", justify="left")
        self.detail_lbl.grid(row=0, column=0, sticky="w")
        btns = ttk.Frame(det, style="Alt.TFrame")
        btns.grid(row=1, column=0, sticky="w", pady=(6, 0))
        # 用户要求：删掉「设为当前模型并启动」，保留「卸载」。
        # 现在「选中哪一行」就等于「目标模型是谁」，所以只需要一个「加载」。
        self.btn_load = ttk.Button(btns, text="加载", style="Accent.TButton",
                                   command=self.load_selected)
        self.btn_load.pack(side="left")
        Tooltip(self.btn_load,
                "把这个模型装进路由（同时可以装多个，各自独立）。\n"
                "路由没在跑的话会顺便把它启动起来。\n"
                "装进来的模型会出现在顶栏「已加载模型」里，可以单独卸载。")
        # 「编辑加载参数 / 编辑对话参数」和「embedding 设置」互斥地放在这里的
        # 小容器里 —— 选中 embedding 类模型时换成后者（见 _paint_detail）
        self.edit_holder = ttk.Frame(btns, style="Alt.TFrame")
        self.edit_holder.pack(side="left", padx=(8, 0))
        self.btn_edit_load = ttk.Button(self.edit_holder, text="编辑加载参数",
                                       style="Mini.TButton",
                                       command=self.edit_load)
        self.btn_edit_load.pack(side="left")
        self.btn_edit_chat = ttk.Button(self.edit_holder, text="编辑对话参数",
                                       style="Mini.TButton",
                                       command=self.edit_chat)
        self.btn_edit_chat.pack(side="left", padx=(4, 0))
        self.btn_emb = ttk.Button(self.edit_holder, text="embedding 设置",
                                  style="Mini.TButton",
                                  command=self.edit_emb)
        Tooltip(self.btn_emb,
                "embedding / reranker 模型没有对话模板、采样、推理链，\n"
                "所以这里只弹一个只含 embedding 专属参数的小窗口。")
        ttk.Button(btns, text="用作推测草稿", style="Mini.TButton",
                   command=self.use_as_draft).pack(side="left", padx=(8, 0))
        self.btn_unload = ttk.Button(btns, text="卸载", style="Danger.TButton",
                                     command=self.unload_selected)
        self.btn_unload.pack(side="left", padx=(8, 0))
        self.btn_unload.state(["disabled"])
        Tooltip(self.btn_unload,
                "把这个模型从路由里卸掉（子进程收掉、显存还回去）。\n"
                "其他已加载的模型不受影响。")
        ttk.Button(btns, text="打开目录", style="Mini.TButton",
                   command=self.open_dir).pack(side="left", padx=(8, 0))

    # ------------------------------------------------------------ 扫描
    def _save_prefs(self) -> None:
        self.app.store.set("models_dir", self.dir_var.get().strip())
        self.app.store.set("models_recursive", bool(self.recursive_var.get()))
        try:
            self.app.store.set("models_depth", int(self.depth_var.get()))
        except ValueError:
            pass
        self.app.store.save()

    def _on_cat(self, value: str) -> None:
        self.app.store.set_pref("category", value)
        self._on_filter()

    def _on_cap(self, value: str) -> None:
        self.app.store.set_pref("cap", self._cap_map.get(value, ""))
        self._on_filter()

    def _on_filter(self) -> None:
        self.app.store.set_pref("only_set", bool(self.only_set_var.get()))
        self.app.store.save()
        self.refresh_table()

    def pick_dir(self) -> None:
        init = self.dir_var.get().strip() or self.app.default_dir()
        path = filedialog.askdirectory(parent=self.app.root, initialdir=init,
                                       title="选择模型目录")
        if path:
            self.dir_var.set(os.path.normpath(path))
            self._save_prefs()
            self.scan_now()

    def _depth(self) -> int:
        try:
            return max(1, int(self.depth_var.get()))
        except ValueError:
            return 3

    def scan_now(self) -> None:
        d = self.dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showwarning("目录无效", "请先选择一个存在的模型目录。",
                                   parent=self.app.root)
            return
        self._save_prefs()
        self.count_lbl.configure(text="正在扫描…")
        self.app.scan_async([d], bool(self.recursive_var.get()), self._depth(),
                            on_done=self._scan_done)

    def _scan_done(self, rows: List[Dict[str, Any]], error: str) -> None:
        if error:
            self.count_lbl.configure(text="扫描失败：%s" % error)
            return
        self.app.lib_rows = rows
        self.refresh_table()

    # ------------------------------------------------------------ 表格
    def refresh_table(self) -> None:
        rows = SC.filter_rows(self.app.lib_rows,
                             category=self.cat_seg.value,
                             text=self.q_var.get(),
                             only_set=bool(self.only_set_var.get()),
                             cap=str(self.app.store.pref("cap", "")))
        SC.sort_rows(rows, self.app.sort_col, self.app.sort_rev)
        total = len(self.app.lib_rows)
        self.count_lbl.configure(
            text="显示 %d / 共 %d 个%s" % (len(rows), total,
                                        "" if total else "（点「扫描」开始）"))
        for cid, zh, _en, _w, _a in SC.COLUMNS:
            arrow = ""
            if cid == self.app.sort_col:
                arrow = "  \u25bc" if self.app.sort_rev else "  \u25b2"
            self.tree.heading(cid, text=zh + arrow)
        keep = self.app.selected_model
        self.tree.delete(*self.tree.get_children())
        self.app.lib_index = {}
        configured = {model_key(p) for p in self.app.store.configured_models()}
        for r in rows:
            # ★ 表示这个模型已经有自己的加载/对话参数（实时算，改完立刻反映）
            r["params_set"] = (SC.ICO_SET if model_key(r["path"]) in configured
                               else "")
            iid = self.tree.insert("", "end", values=(
                r["name"], SC.CATEGORY_ZH.get(r["category"], r["category"]),
                r.get("arch", ""), r.get("params_text", ""),
                r.get("publisher", ""), r.get("pip", ""), r.get("quant", ""),
                r.get("size_text", ""), r.get("mtime_text", ""),
                r.get("params_set", "")), tags=(r["category"],))
            self.app.lib_index[iid] = r
            if keep and model_key(keep) == model_key(r["path"]):
                self.tree.selection_set(iid)
                self.tree.see(iid)

    def sort_by(self, column: str) -> None:
        if self.app.sort_col == column:
            self.app.sort_rev = not self.app.sort_rev
        else:
            self.app.sort_col, self.app.sort_rev = column, False
        self.app.store.set_pref("table_sort",
                                [self.app.sort_col, self.app.sort_rev])
        self.app.store.save()
        self.refresh_table()

    def _on_select(self) -> None:
        row = self.selected_row()
        if not row:
            return
        self.app.selected_model = row["path"]
        self.app.store.ensure_model(row["path"])
        self.app.autofill_mmproj(row)
        self.app.update_target_caption()
        self._paint_detail(row)

    def _paint_detail(self, row: Dict[str, Any]) -> None:
        caps = "能读图 %s    带 MTP 预测头 %s" % (
            "是" if row.get("has_vision") else "否",
            "是" if row.get("has_mtp") else "否")
        bits = [
            row["name"],
            "%s · %s · %s" % (row.get("arch") or "架构未知",
                              row.get("params_text") or "参数量未知",
                              "%s 层" % row["layers"] if row.get("layers")
                              else "层数未知"),
            "上下文 %s · 量化 %s · %s" % (
                format(row["ctx_train"], ",") if row.get("ctx_train")
                else "未知", row.get("quant") or "未知",
                row.get("size_text") or "未知"),
            "发布者 %s%s     能力：%s" % (
                row.get("publisher") or "未知",
                "（取自目录 %s）" % row["org"]
                if row.get("org_display") else
                ("（上传者 %s）" % row["uploader"] if row.get("uploader")
                 else ""),
                caps),
            "文件：%s" % (row.get("file_name") or row.get("name", "")),
            "多模态投影：%s" % (os.path.basename(row["mmproj"])
                                if row.get("mmproj") else "无"),
        ]
        if row.get("has_mtp"):
            bits.append("带 MTP 预测头（GGUF 里 nextn_predict_layers=%d）："
                        "推测解码里选 MTP 方式即可，不需要另配草稿模型"
                        % int(row.get("nextn_layers") or 0))
        elif row.get("mtp_name_hint"):
            bits.append("⚠ 名字里带 MTP/NextN，但 GGUF 里**没有** MTP 层"
                        "（nextn_predict_layers 为空）。选 MTP 方式不会生效，"
                        "只能配草稿模型走 DFlash / DSpark。")
        exe_now = B.resolve_exe(self.app.engine_var.get().strip(), "server")
        bad = self.app.known_bad_arch(exe_now, row["path"])
        if bad:
            bits.append("⚠ 当前的 llama-server 上一次加载「%s」架构就失败了"
                        "（unknown model architecture）。换模型，"
                        "或把 buun-llama-cpp 更新到支持它的版本。" % bad)
        # 是否已驻留：问路由（多模型了，不再是「某个角色正在跑」）
        name = self.app.router_name_for(row["path"])
        resident = bool(name) and name in (self.app.router.loaded_names() or [])
        if resident:
            state = "已驻留"
            for m in (self.app.router.snapshot().get("models") or []):
                if m.get("id") == name:
                    state = ROU.STATUS_ZH.get(str(m.get("status")), state)
                    break
            bits.append("已加载（%s）—— 顶栏「已加载模型」里可以单独卸载"
                        % state)
        # embedding 类模型：两个「编辑 xxx 参数」换成单个「embedding 设置」
        emb = row.get("category") == "Embedding"
        if emb:
            self.btn_edit_load.pack_forget()
            self.btn_edit_chat.pack_forget()
            if not self.btn_emb.winfo_manager():
                self.btn_emb.pack(side="left")
        else:
            self.btn_emb.pack_forget()
            if not self.btn_edit_load.winfo_manager():
                self.btn_edit_load.pack(side="left")
            if not self.btn_edit_chat.winfo_manager():
                self.btn_edit_chat.pack(side="left", padx=(4, 0))
        self.detail_lbl.configure(text="\n".join(bits))
        # Drafters 不能单独加载，禁用加载按钮
        is_drafter = row.get("category") == "Drafters"
        self.btn_load.state(["disabled"] if is_drafter else ["!disabled"])
        self.btn_unload.state(["!disabled"] if resident else ["disabled"])

    def selected_row(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        return self.app.lib_index.get(sel[0]) if sel else None

    # ------------------------------------------------------------ 动作
    def load_selected(self) -> None:
        """「加载」：把选中的模型装进路由。"""
        row = self.selected_row()
        if not row:
            messagebox.showinfo("先选一个模型", "请在表格里选中一行。",
                                parent=self.app.root)
            return
        # Drafters 不能单独加载，只能通过推测解码参数关联
        if row.get("category") == "Drafters":
            messagebox.showinfo(
                "不能单独加载草稿模型",
                "草稿模型（Drafter）不能单独加载到路由里。\n"
                "请在「推测解码」参数里配好草稿模型，它会跟随主 LLM 一起加载。",
                parent=self.app.root)
            return
        self.app.selected_model = row["path"]
        # 记一笔「最近使用」：重启后用它恢复上次的模型（原来靠「默认模型」）
        self.app.store.push_unique("recent_models", row["path"])
        self.app.store.save()
        self.app.router_load(row["path"])

    def edit_emb(self) -> None:
        """embedding 专属设置的弹窗入口。"""
        row = self.selected_row()
        if not row:
            messagebox.showinfo("先选一个模型", "请在表格里选中一行。",
                                parent=self.app.root)
            return
        self.app.show_emb_dialog(row["path"])

    def edit_load(self) -> None:
        row = self.selected_row()
        if not row:
            return
        self.app.selected_model = row["path"]
        if row.get("category") == "Embedding":
            self.app.show_emb_dialog(row["path"])
            return
        self.app.show("load")

    def edit_chat(self) -> None:
        row = self.selected_row()
        if not row:
            return
        self.app.selected_model = row["path"]
        if row.get("category") == "Embedding":
            self.app.show_emb_dialog(row["path"])
            return
        self.app.show("chat")

    def use_as_draft(self) -> None:
        row = self.selected_row()
        if row:
            self.app.set_draft_model(row["path"])

    def unload_selected(self) -> None:
        row = self.selected_row()
        if row:
            self.app.router_unload_path(row["path"])

    def open_dir(self) -> None:
        row = self.selected_row()
        if row:
            self.app.open_path(os.path.dirname(row["path"]))


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #

class App:
    def __init__(self, root: tk.Tk, base_dir: str,
                 start_minimized: bool = False) -> None:
        self.root = root
        self.base_dir = base_dir
        self._start_minimized = bool(start_minimized)
        self.store = Store(base_dir)
        # 让「每个模型一个配置文件」的文件名跟着模型库里的显示名走
        self.store.name_resolver = self._model_display_name
        self.manager = MG.Manager()
        self.rows: Dict[str, FieldRow] = {}
        self.pages: Dict[str, ttk.Frame] = {}
        self.lib_rows: List[Dict[str, Any]] = []
        self.lib_index: Dict[str, Dict[str, Any]] = {}
        self.selected_model = ""
        self.sort_col, self.sort_rev = self._load_sort()
        self.role = "llm"
        self.model_labels: List[ttk.Label] = []
        self.page = str(self.store.get("nav") or "lib")
        if self.page not in S.PAGE_ORDER:
            # 老配置里可能存着已经删掉的页面（imatrix / bench / quant），
            # 直接回退到模型库，否则下面取 PAGE_MODE 会 KeyError
            self.page = "lib"
        self.probe_by_mode: Dict[str, Dict[str, Any]] = {}
        # mode -> (exe 的 mtime:size, flag 集合)：启动前过滤不支持的参数用
        self.flags_cache: Dict[str, Tuple[str, set]] = {}
        self._flags_warned = False
        self.lora_panel: Optional[LoRAPanel] = None
        # (exe, mode) -> 已经提示过「这个 exe 没有这些参数」，避免预览每刷一次报一次
        self._skip_warned: set = set()
        self.scan_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.log_q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.lib: Optional[LibraryPage] = None
        # 运行时状态：轮询后端 /props + /slots，并接收网关转发时抓到的 timings
        self.runtime = RT.RuntimeStatus(log_fn=self.log_append_ts)
        self.rt_lbl: Optional[ttk.Label] = None
        self.rt_pct: Optional[ttk.Label] = None   # 已用上下文：纯文字
        # 多模型路由的后端：模型清单轮询 + 按类限流 + 空闲卸载。
        # ⚠️ 空闲卸载现在由它负责（LLM / Embedding 两套独立计时，见「驻留策略」），
        #    所以下面 UnifiedBackend 的 idle 那套关掉，免得两边抢着卸。
        self.router_policy = ROU.ResidencyPolicy()
        self.router = ROU.RouterMonitor(
            kind_of=self.router_kind_of,
            policy=self.router_policy,
            log=self.log_append_ts)
        self.runtime._router = self.router   # 让运行时能查子模型真实数据
        # 段名（= 客户端请求里写的 model 名）→ 模型路径 / 类别
        self.router_kind_map: Dict[str, str] = {}
        self.router_path_map: Dict[str, str] = {}
        # 本次会话加载过的模型（段名）——「重载」会把它们重新加载一遍
        self.router_history: List[str] = []
        self.unified = UnifiedBackend(
            self.manager, self.manager.bus,
            status_fn=lambda: self.manager.status().get("unified", {}),
            enabled_fn=lambda: True,
            log_fn=self.log_append_ts)
        self.api = ControlAPI(
            self.manager,
            token=str(self.store.control().get("token") or ""),
            models_provider=self.library_brief,
            upstream_key=self.upstream_api_key,
            unified=self.unified,
            default_model=self.default_model_for,
            row_resolver=self.row_for_path,
            runtime=self.runtime,
            # 路由监控：网关转发时把「这个模型刚被用过」记进去，空闲卸载靠它
            router=self.router)
        self._gateway_key = ("", 0)
        self._gw_after: Optional[str] = None
        self._gw_error = ""
        self.last_start_error = ""
        self.out_tail: Dict[str, List[str]] = {}   # 每个进程最近的输出，用于退出诊断
        self._vars: Dict[str, tk.Variable] = {}
        self._refresh_pending = False
        self._scan_thread: Optional[threading.Thread] = None
        self._pending_scan: Tuple[List[Dict[str, Any]], str] = ([], "")
        self._tick = 0
        self._closing = False
        self.tray = None                       # TrayIcon，建好后才用
        self._minimized = False               # 当前是否收在托盘

        self._build_window()
        self._build_topbar()
        self._build_body()
        self._build_bottom()
        self._build_footer()
        self._setup_tray()
        self._restore_session()
        self._bind_keys()
        self.root.after(120, self._pump)
        if self._start_minimized:
            self.minimize_to_tray()

    def _model_display_name(self, path: str) -> str:
        """配置文件名要用模型库里的显示名（好认）；拿不到就返回空，交给 store 兜底。"""
        if not path:
            return ""
        low = os.path.normcase(path)
        for r in (self.lib_rows or []):
            if os.path.normcase(str(r.get("path") or "")) == low:
                return str(r.get("name") or "")
        return ""

    def _load_sort(self) -> Tuple[str, bool]:
        raw = self.store.pref("table_sort", ["name", False])
        try:
            return str(raw[0]), bool(raw[1])
        except (IndexError, TypeError):
            return "name", False

    # ==================================================================== #
    # 窗口骨架
    # ==================================================================== #
    def _build_window(self) -> None:
        self.root.title(APP_TITLE)
        geo = str(self.store.pref("window", "") or "")
        if not geo:
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            w, h = min(1420, sw - 100), min(950, sh - 100)
            geo = "%dx%d+%d+%d" % (max(1120, w), max(720, h),
                                   max(0, (sw - w) // 2), max(0, (sh - h) // 3))
        try:
            self.root.geometry(geo)
        except tk.TclError:
            self.root.geometry("1400x940")
        self.root.minsize(1080, 700)
        self.root.configure(background=C["bg"])
        # 窗口图标（与托盘共用同一张 .ico）
        try:
            self.root.iconbitmap(TRAY._icon_path())
        except (tk.TclError, OSError):
            pass

        outer = ttk.Frame(self.root, style="TFrame")
        outer.pack(fill="both", expand=True)
        self.outer = outer
        self.topbar = ttk.Frame(outer, style="Bar.TFrame", padding=(14, 8))
        self.topbar.pack(fill="x")
        self.middle = ttk.Frame(outer, style="TFrame")
        self.middle.pack(fill="both", expand=True, padx=10, pady=(6, 2))
        self.middle.configure(height=430)
        self.middle.pack_propagate(False)
        self.bottom_host = ttk.Frame(outer, style="TFrame")
        self.bottom_host.pack(fill="both", expand=True)
        # 运行状态条：底部的命令行 / 日志上面一排。放在 bottom_host 内部，
        # 这样它和 _build_bottom 里的 PanedWindow 自动上下排布。
        self._build_runtime_bar(self.bottom_host)
        self.footer = ttk.Frame(outer, style="Bar.TFrame", padding=(14, 7))
        self.footer.pack(fill="x", side="bottom")

    # ==================================================================== #
    # 系统托盘 / 最小化到托盘
    # ==================================================================== #
    def _setup_tray(self) -> None:
        try:
            self.tray = TRAY.TrayIcon(
                self.root, APP_TITLE,
                on_restore=self.restore_from_tray,
                on_quit=self.real_quit)
        except Exception as exc:                # 托盘建不起来也不该拖垮整个界面
            self.tray = None
            self.log_append_ts("托盘初始化失败：%s" % exc, "warn")

    def minimize_to_tray(self) -> None:
        if self._minimized or self.tray is None:
            if self.tray is None and not self._minimized:
                # 连托盘都没有就退化为普通最小化
                self.root.iconify()
            return
        self._minimized = True
        self.root.withdraw()
        self.tray.show()
        self.tray.set_tooltip("%s（已收起到托盘）" % APP_TITLE)

    def restore_from_tray(self) -> None:
        if not self._minimized:
            # 本来就在前台（或只是任务栏最小化），提到最前即可
            try:
                self.root.deiconify()
            except tk.TclError:
                pass
            self.root.lift()
            self.root.focus_force()
            return
        self._minimized = False
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        if self.tray is not None:
            self.tray.hide()

    def real_quit(self) -> None:
        """托盘菜单里的「退出」：不管收没收起都真退出。"""
        self._minimized = False
        if self.tray is not None:
            self.tray.hide()
        # 复用 on_close 的保存与清理逻辑，但跳过「收起托盘」分支
        self._closing = True
        self.on_close(force_quit=True)

    def on_close(self, force_quit: bool = False) -> None:
        if self._minimized and not force_quit:
            # 按设计：X 永远=真退出，不会进托盘；这里只是兜底
            self.restore_from_tray()
            return
        if self.manager.any_running():
            if not messagebox.askyesno(
                    "确认退出", "还有推理进程在运行，退出会一并结束它们。"
                                "确定退出？", parent=self.root):
                return
        self.collect(self.page)
        self.store.set("engine_path", self.engine_var.get().strip())
        if self.lib:
            self.store.set("models_dir", self.lib.dir_var.get().strip())
        self.store.set_pref("autoscroll",
                            bool(self._vars["autoscroll"].get()))
        self.store.set_pref("cmd_multiline",
                            bool(self._vars["cmd_multiline"].get()))
        self.store.set_pref("table_sort", [self.sort_col, self.sort_rev])
        try:
            self.store.set_pref("window", self.root.winfo_geometry())
        except tk.TclError:
            pass
        self.store.save()
        try:
            self.api.stop()
        except Exception:
            pass
        try:
            self.manager.cleanup()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------ 设置弹窗
    def show_settings(self) -> None:
        if getattr(self, "_settings_win", None) and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        self._settings_win = win
        win.title("设置")
        win.transient(self.root)
        win.resizable(False, False)
        win.configure(background=C["bg"])
        f = ttk.Frame(win, padding=(20, 16))
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="设置", style="H2.TLabel",
                  background=C["bg"]).pack(anchor="w", pady=(0, 12))

        # 开机自动启动：实时写注册表（Run 键 + --minimized）
        auto_var = tk.BooleanVar(value=bool(TRAY.is_autostart()))
        row = ttk.Frame(f, style="TFrame")
        row.pack(fill="x", pady=2)
        cb = ttk.Checkbutton(
            row, text="开机自动启动（启动后最小化到系统托盘）",
            variable=auto_var, style="Card.TCheckbutton")
        cb.pack(side="left")

        def apply_autostart(*_a):
            ok = TRAY.set_autostart(bool(auto_var.get()))
            if ok:
                status.config(text="已保存" if auto_var.get() else "已关闭",
                              foreground=C["ok"])
            else:
                status.config(
                    text="写入注册表失败，请用管理员权限运行或检查杀软",
                    foreground=C["warn"])

        cb.configure(command=apply_autostart)
        status = ttk.Label(f, text="", style="Muted.TLabel")
        status.pack(anchor="w", pady=(2, 0))

        ttk.Button(f, text="关闭", style="TButton",
                   command=win.destroy).pack(side="right", pady=(14, 0))
        win.update_idletasks()
        win.geometry("+%d+%d" % (self.root.winfo_rootx() + 60,
                                self.root.winfo_rooty() + 60))

    # ------------------------------------------------------------ 顶栏
    def _build_topbar(self) -> None:
        bar = self.topbar
        bar.columnconfigure(1, weight=1)

        top = ttk.Frame(bar, style="Bar.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=APP_TITLE, style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(top, text="模型库 · 服务 · 对话，一套界面管到底", style="Muted.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0))
        right = ttk.Frame(top, style="Bar.TFrame")
        right.grid(row=0, column=2, sticky="e")
        self.dot = ttk.Label(right, text="\u25cf", style="Muted.TLabel",
                             font=("Segoe UI", 12))
        self.dot.pack(side="left")
        self.status_lbl = ttk.Label(right, text="未运行", style="Muted.TLabel")
        self.status_lbl.pack(side="left", padx=(4, 10))
        ttk.Button(right, text="控制 API", style="Mini.TButton",
                   command=self.show_api_dialog).pack(side="left")
        ttk.Button(right, text="参数对照表", style="Mini.TButton",
                   command=self.show_help).pack(side="left", padx=(6, 0))
        ttk.Button(right, text="最小化到托盘", style="Mini.TButton",
                   command=self.minimize_to_tray).pack(side="left", padx=(6, 0))
        ttk.Button(right, text="设置", style="Mini.TButton",
                   command=self.show_settings).pack(side="left", padx=(6, 0))

        ttk.Label(bar, text="引擎目录", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=3)
        eng = ttk.Frame(bar, style="Bar.TFrame")
        eng.grid(row=1, column=1, sticky="ew", pady=3)
        eng.columnconfigure(0, weight=1)
        self.engine_var = tk.StringVar(
            value=str(self.store.get("engine_path") or ""))
        e = ttk.Entry(eng, textvariable=self.engine_var)
        e.grid(row=0, column=0, sticky="ew")
        e.bind("<FocusOut>", lambda _e: self.commit_engine())
        e.bind("<Return>", lambda _e: self.commit_engine())
        ttk.Button(eng, text="浏览…", style="Mini.TButton",
                   command=self.pick_engine).grid(row=0, column=1, padx=(6, 0))
        mb = ttk.Menubutton(eng, text="最近", style="Mini.TMenubutton")
        menu = tk.Menu(mb, tearoff=0)
        mb.configure(menu=menu)
        mb.grid(row=0, column=2, padx=(4, 0))
        self.engine_menu = menu
        menu.configure(postcommand=self._fill_engine_menu)
        ttk.Button(eng, text="探测参数支持", style="Mini.TButton",
                   command=self.probe_now).grid(row=0, column=3, padx=(6, 0))
        self.engine_resolved = ttk.Label(eng, text="", style="MutedCard.TLabel")
        self.engine_resolved.grid(row=0, column=4, padx=(10, 0))
        Tooltip(e, "buun-llama-cpp 的编译输出目录（如 E:\\AID\\buun-llama-cpp 或 "
                   "…\\build\\bin\\Release），也可以直接选中某个 .exe。\n"
                   "「探测参数支持」会跑一次 --help，把当前 build 不认识的参数标出来。")

        # ---- 目标模型 + 已加载模型（多模型路由）
        #
        # 用户要求：删掉「选模型」「设为当前」，保留「卸载」，并且要能显示多个模型。
        # 所以这里不再是「一个角色一张卡」，而是：
        #   · 目标模型  —— 模型库里选中谁，它就是谁（不用再点按钮「设为当前」）
        #   · 已加载    —— 路由当前驻留的全部模型，每个都能单独卸载
        slots = ttk.Frame(bar, style="Bar.TFrame")
        slots.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        slots.columnconfigure(1, weight=1)

        card = ttk.Frame(slots, style="Alt.TFrame", padding=(10, 6))
        card.grid(row=0, column=0, sticky="w")
        ttk.Label(card, text="目标模型", style="Alt.TLabel",
                  font=T.FONT_UI_BOLD).grid(row=0, column=0, sticky="w")
        self.target_dot = ttk.Label(card, text="\u25cb", style="Alt.TLabel")
        self.target_dot.grid(row=0, column=1, sticky="w", padx=(8, 3))
        self.target_name = ttk.Label(card, text="未指定", style="Alt.TLabel")
        self.target_name.grid(row=0, column=2, sticky="w")
        self.target_state = ttk.Label(card, text="", style="MutedAlt.TLabel")
        self.target_state.grid(row=1, column=0, columnspan=3, sticky="w",
                               pady=(2, 0))
        Tooltip(card, "点「加载」装的就是这个模型。\n"
                      "在「模型库」里选中哪一行，目标就是哪个 —— "
                      "不用再点「设为当前」。\n"
                      "要同时装多个，就依次选中 → 点「加载」。")

        hold = ttk.Frame(slots, style="Alt.TFrame", padding=(10, 6))
        hold.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        hold.columnconfigure(0, weight=1)
        head = ttk.Frame(hold, style="Alt.TFrame")
        head.grid(row=0, column=0, sticky="w")
        ttk.Label(head, text="已加载模型", style="Alt.TLabel",
                  font=T.FONT_UI_BOLD).pack(side="left")
        self.loaded_count = ttk.Label(head, text="", style="MutedAlt.TLabel")
        self.loaded_count.pack(side="left", padx=(8, 0))
        Tooltip(head, "路由进程当前驻留的模型（每秒自动刷新）。\n"
                      "每个都能单独卸载 —— 卸载会把它的子进程整个收掉、"
                      "显存还回去，其他模型不受影响。\n"
                      "同时驻留几个、空闲多久自动卸，见「服务」页的「驻留策略」。")
        self.loaded_box = ttk.Frame(hold, style="Alt.TFrame")
        self.loaded_box.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.loaded_sig: Optional[Tuple] = None
        self.role_widgets: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------ 主体
    def _build_body(self) -> None:
        self.middle.columnconfigure(1, weight=1)
        self.middle.rowconfigure(0, weight=1)
        self.sidebar = Sidebar(self.middle, SIDEBAR_GROUPS,
                               on_select=self.show)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.content = ttk.Frame(self.middle, style="TFrame")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        self.lib = LibraryPage(self)
        self.pages["lib"] = self.lib
        for page in S.PAGES:
            if page[0] in ("lib", "lora"):
                continue
            self.pages[page[0]] = ParamPage(self, page[0])
        # LoRA 是独立页面（不是「加载参数」页里的小节了）
        self.lora_page = LoRAPage(self)
        self.lora_panel = self.lora_page.panel
        self.pages["lora"] = self.lora_page
        for frame in self.pages.values():
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_remove()
        self.sidebar.set_value(self.page)

    # ------------------------------------------------------------ 运行状态条
    _RT_HINT = ("数据全部来自引擎自己暴露的接口，不需要额外开关：\n"
                "· 后端 /props → 上下文长度、VBR 状态（阶梯/起始档/下限/预算）\n"
                "· 后端 /slots → 当前 KV 的 bits/value、当前上下文已用多少 token\n"
                "· 每次转发的响应体 timings → 提示处理与生成速度\n"
                "速度是「最近一次请求」的实测值；KV 档位由 kv_bpv 反查得出，\n"
                "没发生降级时按 --vbr-floor 的单价显示。")

    def _build_runtime_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent, style="Bar.TFrame", padding=(10, 5))
        bar.pack(fill="x", side="top")
        bar.columnconfigure(0, weight=1)
        self.rt_lbl = ttk.Label(bar, text="运行状态：未启动",
                                style="Muted.TLabel")
        self.rt_lbl.grid(row=0, column=0, sticky="w")
        right = ttk.Frame(bar, style="Bar.TFrame")
        right.grid(row=0, column=1, sticky="e")
        # 已用上下文长度：只用文字显示（不再用进度条）
        self.rt_pct = ttk.Label(right, text="", style="Muted.TLabel", width=30)
        self.rt_pct.pack(side="left", padx=(0, 6))
        Tooltip(self.rt_lbl, self._RT_HINT)
        Tooltip(self.rt_pct, "上下文占用：已用 token / 上下文长度（n_ctx）。")

    def _runtime_tick(self) -> None:
        if self.rt_lbl is None:
            return
        # 把「目标模型」的路由 id 喂给 runtime，让它去查子模型真实数据
        if self.router is not None and self.selected_model:
            self.runtime.set_target(self.router_name_for(self.selected_model))
        snap = self.runtime.snapshot()
        parts: List[str] = []
        if not snap.get("attached"):
            text = "运行状态：未启动"
        elif not snap.get("alive"):
            text = "运行状态：%s" % (snap.get("error") or "后端无响应")
        else:
            # 引擎 alive 就始终显示基础状态，不依赖是否有正在进行的请求
            if snap.get("gen_tps"):
                parts.append("生成 %.0f t/s" % snap["gen_tps"])
            if snap.get("prompt_tps"):
                parts.append("提示 %.0f t/s" % snap["prompt_tps"])
            if snap.get("kv_tier"):
                if snap.get("vbr_active"):
                    parts.append("KV vbr·当前 %s" % snap["kv_tier"])
                else:
                    parts.append("KV %s" % snap["kv_tier"])
            if snap.get("vbr_active"):
                if snap.get("realized_bpv"):
                    parts.append("已降级")
                elif snap.get("floor_bpv"):
                    parts.append("未降级（下限 %.3g bpv）" % snap["floor_bpv"])
            if snap.get("budget_bytes"):
                parts.append("KV 预算 %.2f GiB"
                             % (snap["budget_bytes"] / 1073741824.0))
            if snap.get("cache_n"):
                parts.append("缓存命中 %d tok" % snap["cache_n"])
            # 上下文长度（n_ctx）：引擎自动算出或手动设置的都在这——直接并进主状态，
            # 让「运行状态」里一定能看到（之前只放右侧 rt_pct，用户反馈没注意到）。
            if snap.get("n_ctx"):
                parts.append("上下文 %d" % int(snap["n_ctx"]))
            if not parts:
                # 引擎 alive 但还没采到任何数据（刚启动的前几秒 / 目标未加载）
                text = "运行状态：已就绪"
            else:
                text = "运行状态：" + " · ".join(parts)
        try:
            self.rt_lbl.configure(text=text)
        except tk.TclError:
            return
        # 已用上下文长度：文字显示（不再用进度条）。主标签已显示总上下文长度，
        # 这里只放「已用 token / 总长 · 百分比」，避免重复「上下文」字样。
        if self.rt_pct is not None:
            if snap.get("n_ctx"):
                np_ = int(snap.get("n_past") or 0)
                ratio = snap.get("used_ratio")
                t = "已用 %d/%d" % (np_, int(snap["n_ctx"]))
                if ratio is not None:
                    t += " · %.1f%%" % (ratio * 100)
                self.rt_pct.configure(text=t)
            else:
                self.rt_pct.configure(text="")

    # ------------------------------------------------------------ 底部
    def _build_bottom(self) -> None:
        self.panes = ttk.PanedWindow(self.bottom_host, orient="horizontal")
        self.panes.pack(fill="both", expand=True, padx=10, pady=(2, 2))

        left = ttk.Labelframe(self.panes, text=" 等价命令行 ",
                              style="Bg.TLabelframe", padding=(8, 4))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        lbar = ttk.Frame(left, style="Bg.TFrame")
        lbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(lbar, text="复制", style="Tiny.TButton",
                   command=self.copy_cmd).pack(side="left")
        ttk.Button(lbar, text="导出 .bat", style="Tiny.TButton",
                   command=self.export_bat).pack(side="left", padx=(4, 0))
        self.cmd_target = ttk.Label(lbar, text="", style="MutedBg.TLabel")
        self.cmd_target.pack(side="left", padx=(10, 0))
        ttk.Checkbutton(lbar, text="每行一个参数",
                        variable=self._autoscroll_var("cmd_multiline", False),
                        style="Bg.TCheckbutton",
                        command=self._refresh_cmd).pack(side="right")
        self.cmd_view = LogView(left, height=7)
        self.cmd_view.grid(row=1, column=0, sticky="nsew")
        self.cmd_view.text.configure(font=T.FONT_MONO_SMALL)

        right = ttk.Labelframe(self.panes, text=" 运行日志 ",
                               style="Bg.TLabelframe", padding=(8, 4))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        rbar = ttk.Frame(right, style="Bg.TFrame")
        rbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(rbar, text="清空", style="Tiny.TButton",
                   command=lambda: self.log_view.clear()).pack(side="left")
        ttk.Checkbutton(rbar, text="自动滚动",
                        variable=self._autoscroll_var("autoscroll", True),
                        style="Bg.TCheckbutton").pack(side="left",
                                                      padx=(8, 0))
        ttk.Button(rbar, text="打开 WebUI", style="Tiny.TButton",
                   command=lambda: self.open_webui(self.role)).pack(side="right")
        ttk.Button(rbar, text="复制 API 地址", style="Tiny.TButton",
                   command=lambda: self.copy_api(self.role)).pack(
            side="right", padx=(0, 4))
        self.log_view = LogView(right, height=7)
        self.log_view.grid(row=1, column=0, sticky="nsew")
        self.log_view.autoscroll = self._autoscroll_var("autoscroll", True)  # type: ignore[assignment]

        self.panes.add(left, weight=1)
        self.panes.add(right, weight=1)
        self._refresh_cmd()

    # ------------------------------------------------------------ 底栏
    def _build_footer(self) -> None:
        f = self.footer
        f.columnconfigure(1, weight=1)
        left = ttk.Frame(f, style="Bar.TFrame")
        left.grid(row=0, column=0, sticky="w")
        # 用户要求删掉底栏左下的「加载 LLM 默认模型」按钮（连带「默认模型」概念）。
        # 启动入口保留两个：模型库页的「加载」按钮，以及快捷键 F5 / Ctrl+Enter。
        self.btn_stop = ttk.Button(left, text="停止", style="Danger.TButton",
                                   command=self.stop_target)
        self.btn_stop.pack(side="left")
        self.btn_stop.state(["disabled"])
        Tooltip(self.btn_stop,
                "把所有已加载的模型都卸掉（显存还回去）。\n"
                "路由进程留着，随时能再「加载」——\n"
                "要单个卸载就用顶栏「已加载模型」里各自的按钮。")
        ttk.Button(left, text="重载", style="TButton",
                   command=self.restart_target).pack(side="left", padx=(8, 0))
        self.detached_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(left, text="新控制台窗口（可交互）",
                              variable=self.detached_var,
                              style="Card.TCheckbutton")
        chk.pack(side="left", padx=(14, 0))
        Tooltip(chk, "勾选后用独立控制台窗口启动，可以像平时那样在终端里交互"
                     "（llama-cli 的多轮对话需要这个）。\n"
                     "不勾选则在后台运行，日志显示在右下角「运行日志」里。")

        right = ttk.Frame(f, style="Bar.TFrame")
        right.grid(row=0, column=2, sticky="e")
        # 速度已统一在底部运行状态栏显示，不再单独放"最近速度"
        ttk.Button(right, text="切换模型（API）", style="Mini.TButton",
                   command=self.show_switch_dialog).pack(side="right",
                                                         padx=(0, 12))

    def _bind_keys(self) -> None:
        self.root.bind("<F5>", lambda _e: self.start_target())
        self.root.bind("<Shift-F5>", lambda _e: self.stop_target())
        self.root.bind("<Control-Return>", lambda _e: self.start_target())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ==================================================================== #
    # 小工具
    # ==================================================================== #
    def _autoscroll_var(self, key: str, default: bool) -> tk.Variable:
        if key not in self._vars:
            self._vars[key] = tk.BooleanVar(
                value=bool(self.store.pref(key, default)))
        return self._vars[key]

    # ==================================================================== #
    # 参数取值的上下文
    # ==================================================================== #
    def snapshot_for(self, page: str, role: str = "") -> Dict[str, Any]:
        if page == "server":
            return self.store.role_state(role or self.role)
        # 其余「运行模式」页面（对话 / 生成）各自一份运行参数
        if S.PAGE_KIND.get(page) == "run":
            return self.store.run_state(page)
        if page in ("load", "chat", "spec", "lora", "emb"):
            if not self.selected_model:
                return {}
            scope = "chat" if page == "chat" else (
                "emb" if page == "emb" else "load")
            return self.store.snapshot_ref(self.selected_model, scope)
        return {}

    def initial_state(self, f: S.F) -> Dict[str, Any]:
        st = self.snapshot_for(f.page).get(f.key)
        if isinstance(st, dict):
            return st
        default = f.default_value()
        if f.page == "server" and f.key == "port":
            # 两个角色默认端口不同，免得撞车
            default = S.ROLES[self.role]["default_port"]
        return {"on": bool(f.on), "value": default}

    def collect(self, page: str) -> None:
        snap = self.snapshot_for(page)
        for key, row in self.rows.items():
            if row.f.page == page:
                snap[key] = row.get()

    def effective_snapshot(self, mode: str, role: str = "llm",
                           model: str = "") -> Dict[str, Any]:
        snap = S.default_snapshot()
        srcs: List[Dict[str, Any]] = []
        if mode == "server":
            snap["port"] = {"on": True, "value": S.ROLES[role]["default_port"]}
            srcs.append(self.store.role_state(role))
        else:
            # 每个运行模式（对话 / 生成）各存一份运行参数
            srcs.append(self.store.run_state(mode))
        target = model or self.model_for(mode, role)
        if target:
            for scope in ("load", "chat", "emb"):
                got = self.store.state(target, scope)
                if got:
                    srcs.append(got)
        for src in srcs:
            for k, v in (src or {}).items():
                if k in snap and isinstance(v, dict):
                    snap[k] = {"on": bool(v.get("on")),
                               "value": v.get("value", "")}
        # 「驻留策略」是纯界面项，这里翻译成引擎真正要的 --models-max（兜底）
        snap = self._derive_residency(snap)
        # 挡位改名 / 类型变更后，老配置里的旧值要在这里就纠正过来，
        # 否则界面上会显示一个不存在的值、命令行里也会拼出引擎看不懂的东西
        for k, st in snap.items():
            f = S.FIELD_BY_KEY.get(k)
            if f is None:
                continue
            val = str(st.get("value") or "")
            fixed = val
            if f.kind in S.CHOICE_KINDS:
                fixed = B.valid_choice(f, val)
            elif f.kind in (S.K_INT, S.K_FLOAT) and val:
                # 原来是挡位、现在是数字框（如推理强度 → 思考 token 上限）
                if not B.numeric_ok(val, f.kind):
                    fixed = str(f.default_value() or "")
            if fixed != val:
                snap[k] = {"on": bool(st.get("on")), "value": fixed}
        return snap

    def model_for(self, mode: str, role: str = "llm") -> str:
        """当前「目标模型」。只有一套：模型库里选中的那行（没有「默认模型」）。"""
        return self.selected_model

    # ------------------------------------------------- 驻留策略 → --models-max
    def _residency_numbers(self) -> Tuple[int, int, bool]:
        """(LLM 上限, Embedding 上限, embedding 不计入)。"""
        def num(key: str, fallback: int) -> int:
            row = self.rows.get(key)
            raw = row.get().get("value") if row is not None else ""
            try:
                return max(0, int(str(raw).strip()))
            except (TypeError, ValueError):
                return fallback
        uncounted = False
        row = self.rows.get("res_emb_uncounted")
        if row is not None:
            uncounted = bool(row.get().get("on"))
        return num("res_llm_max", 1), num("res_emb_max", 2), uncounted

    def _derive_residency(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """把「驻留策略」那几个纯界面项翻译成引擎真正要的 ``--models-max``。

        引擎只有一个**全局**上限，而且它的 LRU 驱逐不看模型类别 ——
        传「LLM 上限」的话 embedding 一多就会被跨类踢掉。所以真正的按类限流
        （LLM / Embedding 分开记账、超了顶掉同类最久没用的、两类各自的空闲
        卸载计时）都在网关侧做；这个值只是**兜底**：

            LLM 上限 + Embedding 上限 + 2 的余量

        勾了「embedding 不计入驻留数量」就没有可算的上界了 → 传 0（不限制），
        显存完全靠网关那边保证。
        """
        def num(key: str, fallback: int) -> int:
            st = snap.get(key) or {}
            try:
                return max(0, int(str(st.get("value") or "").strip()))
            except (TypeError, ValueError):
                return fallback
        uncounted = bool((snap.get("res_emb_uncounted") or {}).get("on"))
        if uncounted:
            total = 0
        else:
            total = (num("res_llm_max", 1) + num("res_emb_max", 2) + 2)
        snap["models_max"] = {"on": True, "value": str(total)}
        return snap

    # ==================================================================== #
    # 导航
    # ==================================================================== #
    def show(self, page: str) -> None:
        if page not in self.pages:
            return
        if self.page in self.pages and self.page != page:
            self.collect(self.page)
        self.page = page
        self.store.set("nav", page)
        for pid, frame in self.pages.items():
            frame.grid() if pid == page else frame.grid_remove()
        self.sidebar.set_value(page)
        self._load_page(page)
        self._apply_probe(page)
        self.update_target_caption()
        self._refresh_cmd()

    def _load_page(self, page: str) -> None:
        if page == "lib":
            # 回到模型库时刷新一下：★（已单独调参）与扫描结果可能变了
            if self.lib is not None and self.lib_rows:
                self.lib.refresh_table()
            self.store.save()
            return
        if page in ("load", "chat", "spec", "lora") and not self.selected_model:
            self.log_append("还没选模型：请在「模型库」里选一行，"
                            "加载/对话/LoRA 参数都是按模型分别保存的。", "warn")
        for key, row in self.rows.items():
            if row.f.page == page:
                row.set(self.initial_state(row.f))
        if page == "server":
            self._sync_role_fields()
        if page == "spec":
            self._sync_spec_fields()
        if page == "lora" and self.lora_panel is not None:
            self.lora_panel.reload()
        if page == "load" and self.lora_panel is not None:
            self.lora_panel.reload()
        self.store.save()

    def _row_widgets(self, row: FieldRow) -> List[tk.Widget]:
        """一行涉及的所有 grid 子控件（隐藏整行时要一起收起来）。"""
        if row.f.kind == S.K_BOOL:
            return [row.chk, row.flag_lbl]
        out: List[tk.Widget] = [row.chk, row.lbl, row.flag_lbl, row.holder]
        return out

    def _sync_role_fields(self) -> None:
        for key, row in self.rows.items():
            if row.f.page != "server":
                continue
            visible = self.role in row.f.roles
            for w in self._row_widgets(row):
                try:
                    w.grid() if visible else w.grid_remove()
                except tk.TclError:
                    pass

    def _sync_spec_fields(self) -> None:
        master = self.rows.get("spec_enable")
        if master is None:
            return
        on = bool(master.on_var.get())
        for key, row in self.rows.items():
            if row.f.page != "spec" or key == "spec_enable":
                continue
            try:
                row.chk.configure(state="normal" if on else "disabled")
                if on:
                    row._apply_state()
                else:
                    for extra in (row.ctrl, row.browse):
                        if extra is not None:
                            extra.configure(state="disabled")
                    if row.seg is not None:
                        row.seg.set_enabled(False)
                    if row.text is not None:
                        row.text.configure(state="disabled")
            except tk.TclError:
                pass

    def update_target_caption(self) -> None:
        mode, role = self.target()
        model = self.model_for(mode, role)
        if mode == "server":
            host, port = self.gateway_endpoint()
            bits = ["统一端口 %s:%d" % (host, port)]
            u = self.unified
            bits.append("当前：%s" % (os.path.basename(u.model)
                                     if u.model else "未加载"))
            bits.append("目标：%s" % (os.path.basename(model) if model
                                     else "未指定"))
        else:
            bits = ["%s（%s）" % (S.MODES[mode]["zh"], S.MODES[mode]["exe"])]
            if mode == "server":
                bits.append("%s 角色" % S.ROLES[role]["zh"])
            bits.append(os.path.basename(model) if model else "未选择模型")
        try:
            self.cmd_target.configure(text=" · ".join(bits))
        except (AttributeError, tk.TclError):
            pass
        text = (os.path.basename(self.selected_model)
                if self.selected_model else "未选择（去「模型库」点一行）")
        for lbl in getattr(self, "model_labels", []):
            try:
                lbl.configure(text=text)
            except tk.TclError:
                pass

    def target(self) -> Tuple[str, str]:
        """当前导航位置对应的「要启动的目标」：运行模式 + 角色。"""
        if self.page in ("lib", "server"):
            return "server", self.role
        if self.page in ("load", "chat", "spec", "lora"):
            # 编辑类页面：这些页面的参数是「按模型」存的，拼的是服务命令行
            return "server", self.role
        return S.PAGE_MODE[self.page], ""

    # ==================================================================== #
    # 引擎与探测
    # ==================================================================== #
    def default_dir(self) -> str:
        if self.selected_model:
            return os.path.dirname(self.selected_model)
        d = str(self.store.get("models_dir") or "")
        if d and os.path.isdir(d):
            return d
        e = self.engine_var.get().strip()
        return e if e and os.path.isdir(e) else self.base_dir

    def pick_engine(self) -> None:
        cur = self.engine_var.get().strip()
        init = cur if cur and os.path.isdir(cur) else self.base_dir
        path = filedialog.askdirectory(parent=self.root, initialdir=init,
                                       title="选择 buun-llama-cpp 目录")
        if path:
            self.engine_var.set(os.path.normpath(path))
            self.commit_engine()

    def commit_engine(self) -> None:
        p = self.engine_var.get().strip()
        self.store.set("engine_path", p)
        if p:
            self.store.push_unique("engine_history", p)
        self.store.save()
        self._update_engine_label()
        self._refresh_cmd()

    def _fill_engine_menu(self) -> None:
        m = self.engine_menu
        m.delete(0, "end")
        hist = list(self.store.get("engine_history") or [])
        if not hist:
            m.add_command(label="（暂无记录）", state="disabled")
        for p in hist:
            m.add_command(label=p, command=lambda v=p: self._set_engine(v))

    def _set_engine(self, path: str) -> None:
        self.engine_var.set(path)
        self.commit_engine()

    def _update_engine_label(self) -> None:
        mode, _role = self.target()
        p = B.resolve_exe(self.engine_var.get().strip(), mode)
        if p:
            self.engine_resolved.configure(
                text="当前：" + os.path.basename(p), style="OkCard.TLabel")
        elif self.engine_var.get().strip():
            self.engine_resolved.configure(
                text="未找到 " + S.MODES[mode]["exe"], style="ErrCard.TLabel")
        else:
            self.engine_resolved.configure(text="")

    def probe_now(self) -> None:
        found = B.detect_exes(self.engine_var.get().strip())
        if not found:
            messagebox.showwarning(
                "没有找到可执行文件",
                "在「%s」及其 bin / build\\bin 子目录里都没找到 llama-*.exe。\n\n"
                "请确认是否已完成编译，或者直接把某个 .exe 拖进来。"
                % self.engine_var.get().strip(), parent=self.root)
            return
        self.log_append("正在探测 %d 个可执行文件的参数支持…" % len(found), "sys")

        def work() -> None:
            out = {}
            for mode, exe in found.items():
                out[mode] = PR.probe(exe)
            self.scan_q.put(("probe", out))

        threading.Thread(target=work, name="probe", daemon=True).start()

    def _probe_done(self, out: Dict[str, Dict[str, Any]],
                    quiet: bool = False) -> None:
        self.probe_by_mode = out
        for mode, info in out.items():
            exe = str(info.get("_exe") or "")
            if exe and info.get("flags"):
                self.flags_cache[mode] = (PR.stamp(exe), set(info["flags"]))
        if quiet:
            bad = sum(1 for m, i in out.items()
                      for f in S.FIELDS
                      if f.page and i.get("flags")
                      and f.all_flags()
                      and not any(c in i["flags"] for c in f.all_flags()))
            if bad:
                self.log_append(
                    "参数探测完成：当前 build 有 %d 个界面参数不支持，"
                    "已在参数页用 ⚠ 标出，启动时会自动跳过。" % bad, "warn")
            self._apply_probe(None)
            self._refresh_role_slots()
            return
        lines = []
        for mode in S.MODE_ORDER:
            info = out.get(mode)
            if info:
                lines.append("%s %s(%d)" % ("✓" if info["ok"] else "×",
                                            S.MODES[mode]["exe"],
                                            info["count"]))
        self.log_append("参数探测完成：" + ("；".join(lines) or "无结果"), "ok")
        ver = next((i.get("version") for i in out.values()
                    if i.get("version")), "")
        if ver:
            self.log_append("版本：%s" % ver, "muted")
        self._apply_probe(self.page)
        self._refresh_role_slots()
        self.store.save()

    def _apply_probe(self, page: Optional[str]) -> None:
        """``page`` 传 None 表示对所有已建行的页面生效（启动自动探测用）。"""
        bad = 0
        for key, row in self.rows.items():
            if page is not None and row.f.page != page:
                continue
            mode = S.PAGE_MODE.get(row.f.page, "server")
            flags = (self.probe_by_mode.get(mode) or {}).get("flags") or set()
            if not flags:
                continue
            if not row.f.all_flags():
                row.mark_unsupported(False)
                continue
            ok = PR.supports(flags, row.f)
            row.mark_unsupported(ok is False)
            if ok is False:
                bad += 1
        if bad:
            self.log_append("当前 build 不支持本页 %d 个参数，"
                            "已自动取消勾选并标红。" % bad, "warn")

    # ------------------------------------------------------- 参数兼容过滤
    def supported_flags(self, exe: str, mode: str = "server") -> set:
        """这台 exe 认识的 flag 集合，按「路径 + 修改时间 + 大小」缓存。

        **启动前必须拿到它。** 否则界面上任何一项「这份 exe 不认识的参数」
        都会让程序直接 `unknown argument` 退出 —— 实际踩过两次：
        `-kvu`（统一 KV 缓存）和 `-a`（模型别名）。

        注意：探测的是**这一模式对应的那个 exe**（llama-server / llama-cli），
        不同 exe 的参数集本来就不一样，不能混用。
        """
        if not exe or not os.path.isfile(exe):
            return set()
        stamp = PR.stamp(exe)
        hit = self.flags_cache.get(mode)
        if hit and hit[0] == stamp:
            return set(hit[1])
        info = PR.probe(exe)
        flags = set(info.get("flags") or set())
        self.flags_cache[mode] = (stamp, flags)
        self.probe_by_mode[mode] = info
        if not flags and not self._flags_warned:
            self._flags_warned = True
            self.log_append("没能读到 %s 的 --help，本次不做参数兼容性过滤；"
                            "如果启动报 unknown argument，手动点一次"
                            "「探测参数支持」。"
                            % (os.path.basename(exe) or "引擎"), "warn")
        return flags

    def dropped_flags(self, snap: Dict[str, Dict[str, Any]], mode: str,
                      role: str, flags: set) -> List[str]:
        """这份 exe 不认识、且用户勾上了的参数（中文名 + flag）。"""
        out: List[str] = []
        if not flags:
            return out
        for f in S.FIELDS:
            if mode not in f.modes:
                continue
            if f.page == "server" and role and role not in f.roles:
                continue
            if not f.all_flags():
                continue
            if not bool((snap.get(f.key) or {}).get("on")):
                continue
            if not any(c in flags for c in f.all_flags()):
                out.append("%s（%s）" % (f.zh, "/".join(f.all_flags())))
        return out

    def build_argv_checked(self, snap: Dict[str, Dict[str, Any]], mode: str,
                           model: str = "", role: str = "llm",
                           exe: str = "") -> List[str]:
        """生成命令行，并自动剔除这份 exe 不认识的参数。

        提示只报一次（按 exe 记），否则底部的命令行预览每刷新一次就刷一条日志；
        措辞也点明是**哪个程序**没有这个参数 —— 不是「build 不支持」，
        因为 llama-cli 的参数集本来就比 llama-server 少。
        """
        flags = self.supported_flags(exe, "server" if mode == "server"
                                     else mode)
        dropped = self.dropped_flags(snap, mode, role, flags)
        if dropped and exe:
            key = (exe, mode)
            if key not in self._skip_warned:
                self._skip_warned.add(key)
                self.log_append("%s 没有这些参数，已跳过：%s"
                                % (os.path.basename(exe), "、".join(dropped)),
                                "warn")
        return B.build_argv(snap, mode, model=model, role=role, flags=flags)

    def arch_of(self, path: str) -> str:
        """读 GGUF 头里的 general.architecture（模型库里有就用缓存）。"""
        if not path:
            return ""
        key = model_key(path)
        for r in self.lib_rows:
            if model_key(r.get("path", "")) == key:
                arch = str(r.get("arch") or "")
                if arch:
                    return arch
        try:
            info = GG.analyse(path)
        except Exception:  # noqa: BLE001
            return ""
        return str((info or {}).get("arch") or "")

    @staticmethod
    def _bad_arch_key(exe: str) -> str:
        # 用 exe 的 mtime:size 当键：更新过 buun-llama-cpp 就自动失效
        return PR.stamp(exe) or os.path.basename(exe)

    def known_bad_arch(self, exe: str, path: str) -> str:
        """这个 exe 之前加载失败过的架构名；没失败过返回空串。"""
        if not exe or not path:
            return ""
        arch = self.arch_of(path)
        if not arch:
            return ""
        learned = self.store.get("learned") or {}
        seq = ((learned.get("bad_arch") or {})
               .get(self._bad_arch_key(exe)) or [])
        return arch if arch in seq else ""

    def learn_bad_arch(self, exe: str, arch: str) -> None:
        """记住「这个 build 不认这个架构」，下次启动前先提醒。"""
        if not exe or not arch:
            return
        learned = dict(self.store.get("learned") or {})
        table = dict(learned.get("bad_arch") or {})
        key = self._bad_arch_key(exe)
        seq = list(table.get(key) or [])
        if arch not in seq:
            seq.append(arch)
        table[key] = seq
        learned["bad_arch"] = table
        self.store.set("learned", learned)
        self.store.save()

    def warn_if_bad_arch(self, exe: str, path: str) -> None:
        bad = self.known_bad_arch(exe, path)
        if bad:
            self.log_append(
                "提醒：上一次用这个 llama-server 加载「%s」架构就失败了"
                "（unknown model architecture）。如果之后没更新过 buun-llama-cpp，"
                "这次大概率还是起不来 —— 建议先更新引擎或换个模型。" % bad,
                "warn")

    def diagnose_exit(self, key: str, code: Any) -> None:
        """进程非 0 退出时，从它自己的输出里找出真正的原因。"""
        text = "\n".join((self.out_tail.get(key) or [])[-300:])
        for pat, title, advice in _EXIT_HINTS:
            m = pat.search(text)
            if not m:
                continue
            extra = ""
            if title == "模型架构不被支持":
                arch = m.group(1) if pat.groups else ""
                if arch:
                    extra = "：%s" % arch
                    self.learn_bad_arch(
                        B.resolve_exe(self.engine_var.get().strip(),
                                      "server"), arch)
            self.log_append("退出原因 = %s%s" % (title, extra), "err")
            self.log_append(advice, "warn")
            return
        self.log_append("排查提示：常见原因——显存不足、模型架构或参数不被本版本"
                        "支持、模型分片不完整。往上翻日志里的 error / failed 行"
                        "才是准的。", "warn")

    # ==================================================================== #
    # 模型库扫描
    # ==================================================================== #
    def scan_async(self, dirs: List[str], recursive: bool, depth: int,
                   on_done: Callable[[List[Dict[str, Any]], str], None]) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            self.log_append("上一次扫描还没结束，请稍候…", "warn")
            return

        def work() -> None:
            try:
                paths = SC.discover(dirs, recursive=recursive, depth=depth)
                mm = SC.index_mmproj(dirs, recursive=recursive, depth=depth)
                rows = SC.scan(paths, mm, self.store.configured_models(),
                               cache=self.store.cache(), roots=dirs,
                               progress=lambda i, n: self.scan_q.put(
                                   ("progress", (i, n))))
                self.store.prune_cache(paths)
                self.scan_q.put(("scan", (rows, "")))
            except Exception as exc:  # noqa: BLE001
                self.scan_q.put(("scan", ([], repr(exc))))
            self.scan_q.put(("scan_done", on_done))

        self._scan_thread = threading.Thread(target=work, name="scan",
                                             daemon=True)
        self._scan_thread.start()

    def library_brief(self) -> List[Dict[str, Any]]:
        return list(self.lib_rows)

    # ==================================================================== #
    # 角色
    # ==================================================================== #
    def autofill_mmproj(self, row: Dict[str, Any]) -> None:
        """视觉模型同目录有 mmproj 就自动填进 --mmproj，省得手找。"""
        mm = row.get("mmproj")
        if not mm:
            return
        snap = self.store.snapshot_ref(row["path"], "load")
        cur = snap.get("mmproj") or {}
        if str(cur.get("value") or "").strip() == mm:
            return
        snap["mmproj"] = {"on": True, "value": mm}
        self.store.save()
        self.log_append("已自动为 %s 配上多模态投影：%s"
                        % (row["name"], os.path.basename(mm)), "sys")
        if self.page == "load" and model_key(self.selected_model) == \
                model_key(row["path"]):
            rr = self.rows.get("mmproj")
            if rr is not None:
                rr.set(snap["mmproj"])

    def upstream_api_key(self, role: str = "llm") -> str:
        """网关转发时用的上游密钥：取该角色服务页上填的 --api-key。"""
        if role not in S.ROLES:
            role = "llm"
        snap = self.store.role_state(role)
        st = snap.get("api_key") or {}
        return str(st.get("value") or "").strip() if st.get("on") else ""

    # ----------------------------------------------------------- 运行方式
    def _role_setting(self, key: str, role: str = "llm") -> Tuple[bool, str]:
        """读服务页上属于「角色」的设置，没有就取 schema 默认值。"""
        st = self.store.role_state(role).get(key)
        if isinstance(st, dict):
            return bool(st.get("on")), str(st.get("value") or "")
        f = S.FIELD_BY_KEY.get(key)
        if f is None:
            return False, ""
        return bool(f.on), f.default_value()

    def gateway_endpoint(self) -> Tuple[str, int]:
        """统一模式的网关地址 = 服务页填的 host / port。"""
        _on, host = self._role_setting("host")
        _on2, port = self._role_setting("port")
        try:
            p = int(str(port).strip() or 8080)
        except (TypeError, ValueError):
            p = 8080
        return (host.strip() or "127.0.0.1"), p

    def default_model_for(self, role: str = "llm") -> str:
        """（历史名）网关给「没写 model 的请求」兜底的模型。

        不再有「默认模型」这一说 —— 直接用当前选中的目标模型。
        """
        return self.selected_model

    def role_for_path(self, path: str) -> str:
        """只有一个角色，服务参数也就只有一套。"""
        return "llm"

    @staticmethod
    def free_port(start: int = 18080, end: int = 18160) -> int:
        import socket
        for p in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start

    def row_for_path(self, path: str) -> Optional[Dict[str, Any]]:
        """给网关用：按路径找模型信息；模型库没扫过就现场解析 GGUF 头。

        这样即使一次都没扫描过模型库，`POST /switch {"model":"E:/.../x.gguf"}`
        也能直接把模型拉起来。
        """
        if not path:
            return None
        key = model_key(path)
        for r in self.lib_rows:
            if model_key(r.get("path", "")) == key:
                return r
        if not (os.path.isfile(path) and path.lower().endswith(".gguf")):
            return None
        warned = getattr(self, "_row_warned", None)
        if warned is None:
            warned = self._row_warned = set()
        if key not in warned:
            warned.add(key)
            self.log_append("模型库还没扫到 %s，按路径现场解析。"
                            % os.path.basename(path), "sys")
        info = G.analyse(path)
        if not info:
            return None
        try:
            return SC.build_row(path, info, "", False)
        except Exception:  # noqa: BLE001
            return None

    def short_name(self, path: str) -> str:
        """模型的短名字：模型库解析出来的名字优先，否则用文件名去后缀。"""
        if not path:
            return ""
        key = model_key(path)
        for r in self.lib_rows:
            if model_key(r.get("path", "")) == key:
                return str(r.get("name") or "")
        stem = os.path.splitext(os.path.basename(path))[0]
        return re.sub(r"-\d{5}-of-\d{5}$", "", stem)

    def set_target_model(self, path: str, start: bool = False) -> None:
        """把某个模型设为「目标模型」（= 模型库里选中的那个）。

        原来这里叫 ``assign_role``：会写一份持久化的「默认模型」。用户要求去掉
        「默认模型」这套东西，所以现在只做三件事 —— 设目标、记一笔「最近使用」
        （重启后用来恢复）、刷新界面。
        """
        if not path:
            return
        self.selected_model = path
        self.store.push_unique("recent_models", path)
        self.store.save()
        row = next((r for r in self.lib_rows
                    if model_key(r["path"]) == model_key(path)), None)
        if row:
            if row["category"] == "Embedding":
                self.log_append("提示：%s 是嵌入模型，本程序只跑 LLM，"
                                "它起不来（llama-server 会报架构不支持）。"
                                % row["name"], "warn")
            self.lib.refresh_table()
        self.log_append("目标模型已指定：%s" % os.path.basename(path), "sys")
        if self.page in ("load", "chat", "spec"):
            self._load_page(self.page)
        self._refresh_role_slots()
        self.update_target_caption()
        if start:
            self.start_role()

    def toggle_role(self, role: str) -> None:
        cur = self.unified.model
        want = self.selected_model
        if cur and want and model_key(cur) == model_key(want) \
                and self.manager.get("unified").running:
            self.stop_unified_backend()
        else:
            self.start_role(role)

    def start_role(self, role: str = "llm", model_override: str = "",
                   quiet: bool = False) -> bool:
        target = model_override or self.selected_model
        res = self.start_unified_backend(target)
        if res.get("ok"):
            self.last_start_error = ""
        else:
            self.last_start_error = str(res.get("error") or "启动失败")
        return bool(res.get("ok"))

    # ------------------------------------------------- 引擎环境与依赖体检
    def engine_env_for(self, exe: str) -> Dict[str, str]:
        """算出该给子进程注入的引擎环境，并把体检结果写进日志。

        buun 这套 exe 是 DLL 结构，而且 ggml-cuda.dll 直接引用 CUDA 运行时。
        缺 DLL 时 Windows 只会让进程以 0xC0000135 秒退、连日志都不留一行，
        所以启动前先静态解析一次导入表，把问题翻译成人话。
        """
        engine_path = self.engine_var.get().strip()
        cuda_dirs = E.cuda_bin_dirs()
        diag = E.diagnose_exe(exe, cuda_dirs)
        if not diag["ok"]:
            self.log_append("启动前体检：可执行文件缺运行库 —— 引擎很可能秒退。",
                            "err")
            for line in E.format_diagnosis(diag).splitlines():
                self.log_append(line, "err")
        else:
            note = E.runtime_report(engine_path, exe)
            if note:
                self.log_append(note, "sys")
        env = E.engine_env(engine_path, exe)
        cbs = [k for k in env if k.startswith("TURBO_TCQ")]
        if cbs:
            self.log_append("已注入 TCQ 码本环境变量：%s" % ", ".join(cbs), "sys")
        return env

    # ------------------------------------------------- 多模型路由（buun 独有）
    @staticmethod
    def _safe_ini_name(name: str) -> str:
        """INI 段名里不能出现 ] 和 :（: 会被当成量化后辍处理）。"""
        out = str(name or "").strip()
        for ch in ("]", "[", ":", "\n", "\r"):
            out = out.replace(ch, "-")
        return out.strip(" -") or "model"

    def router_model_name(self, path: str) -> str:
        """路由里这个模型叫什么（优先用模型库里的显示名）。"""
        low = os.path.normcase(path)
        for r in getattr(self, "lib_rows", []) or []:
            if os.path.normcase(str(r.get("path") or "")) == low:
                if r.get("name"):
                    return self._safe_ini_name(str(r["name"]))
        return self._safe_ini_name(
            os.path.splitext(os.path.basename(path))[0])

    def _classify_path(self, path: str) -> str:
        """这个模型算 LLM / embedding / drafter（按类限流 / 空闲卸载要用）。"""
        low = os.path.normcase(str(path))
        for r in getattr(self, "lib_rows", []) or []:
            if os.path.normcase(str(r.get("path") or "")) == low:
                cat = r.get("category")
                if cat == "Embedding":
                    return "emb"
                if cat == "Drafters":
                    return "drafter"
                return "llm"
        try:
            return "emb" if B.is_embedding_model(path) else "llm"
        except Exception:  # noqa: BLE001
            return "llm"

    def router_kind_of(self, name: str) -> str:
        """段名 → 类别。认不出来的按 LLM 算（只有 HF 缓存里冒出来的才会）。"""
        return str(getattr(self, "router_kind_map", {}).get(name) or "llm")

    def router_entries(self, current: str = "") -> List[Tuple[str, str, Dict[str, Any]]]:
        """预置文件里要声明的所有模型：**模型库里扫到的全部**（外加当前选中的）。

        为什么是「全部」而不是「只配过参数的」：路由**只能加载已经在预置文件里
        声明过的模型** —— `POST /models/load` 对不在清单里的名字直接返回
        not found。既然「加载」按钮要能加载任意一个模型，清单就得先铺全。
        每段至少写一行 `model = 路径`，成本极低（只有真正被 load 的才占显存）。

        顺便重建「段名 → 类别 / 路径」两张表，供按类限流与列表显示用。
        """
        paths: List[str] = []
        for r in getattr(self, "lib_rows", []) or []:
            p = str(r.get("path") or "")
            if p and p not in paths:
                paths.append(p)
        for p in ([current] if current else []) + \
                list(self.store.configured_models()):
            if p and p not in paths:
                paths.append(p)
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        used: Dict[str, str] = {}
        kinds: Dict[str, str] = {}
        pathmap: Dict[str, str] = {}
        for p in paths:
            if not os.path.isfile(p):
                continue
            name = self.router_model_name(p)
            if name in used:            # 同名就加后缀，免得预设互相覆盖
                i = 2
                while ("%s-%d" % (name, i)) in used:
                    i += 1
                name = "%s-%d" % (name, i)
            used[name] = p
            out.append((name, p, MG.merge_model_snapshot(self.store, p)))
            kinds[name] = self._classify_path(p)
            pathmap[name] = p
        self.router_kind_map = kinds
        self.router_path_map = pathmap
        return out

    def preset_ini_text(self, current: str = "") -> str:
        """生成路由预置 INI 的文本。

        ⚠️ 不往 ``[*]`` 里塞任何东西：它跟「路由器命令行」一样是**全局**的，
        引擎会把 ``[*]`` merge（覆盖）进每个模型段 —— 塞了某个模型的参数，
        所有模型就被锁成同一套了。模型参数只写各自的段。
        """
        return B.preset_ini(self.router_entries(current), None,
                            engine_hint=self.engine_var.get().strip())

    def export_router_preset(self) -> None:
        """「导出路由预置 INI」按钮：写到用户选的位置并把字段填上。"""
        cur = self.selected_model
        text = self.preset_ini_text(cur)
        init = self.store.config_dir
        path = filedialog.asksaveasfilename(
            parent=self.root, title="导出路由预置 INI", initialdir=init,
            initialfile="router-preset.ini",
            defaultextension=".ini",
            filetypes=[("INI 文件", "*.ini"), ("所有文件", "*.*")])
        if not path:
            return
        if not B.write_ini(path, text):
            messagebox.showerror("导出失败", "写不进这个位置：%s" % path,
                                 parent=self.root)
            return
        self.set_row_value("models_preset", path, on=True)
        self.log_append("已导出路由预置 INI：%s（%d 个模型）"
                        % (path, len(self.router_entries(cur))), "ok")

    def _prepare_router(self, snap: Dict[str, Any],
                        current: str) -> List[str]:
        """路由启动前：生成 preset INI 并把路径填进快照。

        每次都重新生成 —— 模型库扫到的东西和各自的参数都可能变了，
        反正只是写一个几 KB 的文本文件。
        """
        notes: List[str] = []
        entries = self.router_entries(current)
        text = B.preset_ini(entries, None,
                            engine_hint=self.engine_var.get().strip())
        path = os.path.join(self.store.config_dir, "router-preset.ini")
        if B.write_ini(path, text):
            snap["models_preset"] = {"on": True, "value": path}
            names = [n for n, _p, _s in entries]
            emb = sum(1 for n, p, _s in entries
                      if self._classify_path(p) == "emb")
            notes.append("已生成预置文件：%s（%d 个模型，其中 %d 个 embedding）"
                         % (path, len(entries), emb))
            if not names:
                notes.append("⚠ 模型库里一个模型都没有 —— 先到「模型库」页"
                             "选目录并扫描，否则路由起来也是空的。")
        else:
            notes.append("警告：生成预置文件失败（%s 不可写），"
                         "各模型会拿不到自己的参数。" % path)
        return notes

    def show_emb_dialog(self, path: str = "") -> None:
        """给 embedding / reranker 模型弹专属设置小窗口。"""
        target = path or self.selected_model
        if not target:
            messagebox.showinfo("先选模型",
                                "请在「模型库」里选中一个 embedding 模型。",
                                parent=self.root)
            return
        self.selected_model = target
        dlg = getattr(self, "_emb_dialog", None)
        if dlg is not None and dlg.winfo_exists():
            dlg.close()
        self._emb_dialog = EmbDialog(self, target)

    # ==================================================================== #
    # 多模型路由：加载 / 卸载 / 重载（需求 2 与 4）
    # ==================================================================== #
    def _row_int(self, key: str, fallback: int) -> int:
        row = self.rows.get(key)
        raw = row.get().get("value") if row is not None else ""
        try:
            return max(0, int(str(raw).strip()))
        except (TypeError, ValueError):
            return fallback

    def sync_residency_policy(self) -> None:
        """把「驻留策略」小节的当前取值推给 RouterMonitor。"""
        llm, emb, uncounted = self._residency_numbers()
        pol = self.router_policy
        pol.llm_max = llm
        pol.emb_max = emb
        pol.emb_uncounted = uncounted
        pol.llm_idle_min = self._row_int("res_llm_idle", 15)
        pol.emb_idle_min = self._row_int("res_emb_idle", 30)

    def rebuild_presets(self) -> bool:
        """重新生成预置 INI，并让引擎重读一遍（改完参数后必须做）。"""
        self.sync_residency_policy()
        path = os.path.join(self.store.config_dir, "router-preset.ini")
        entries = self.router_entries(self.selected_model)
        text = B.preset_ini(entries, None,
                            engine_hint=self.engine_var.get().strip())
        if not B.write_ini(path, text):
            self.log_append("写预置文件失败：%s" % path, "err")
            return False
        self.set_row_value("models_preset", path, on=True)
        if self.manager.get("unified").running:
            ok, msg = self.router.reload_presets()
            if not ok:
                self.log_append("引擎重读预置失败：%s" % msg, "warn")
        self.log_append("预置文件已重建（%d 个模型）" % len(entries), "sys")
        self.audit_preset_models(entries, path)
        return True

    # 体检时忽略这些 key：它们是**服务进程级**的（有没有预置文件、端口、
    # 引擎在不在），跟单个模型没关系 —— 拿它们当模型的问题会把日志刷满。
    _AUDIT_SKIP = frozenset(("models_preset", "port", "host", "__engine__",
                             "__model__"))

    def audit_preset_models(self, entries: List[Tuple[str, str, Dict[str, Any]]],
                            preset_path: str = "") -> None:
        """逐个模型体检，只记日志、不拦。

        路由模式下每个模型是**独立子进程**：某个模型的参数不成立时，只有它
        自己起不来，路由器本身没事 —— 所以这里不能拦，但必须说清楚。

        不这么做的话，用户在 API 那边只会看到引擎一句「failed to load /
        子进程退出码 1」，根本不知道是自己把推测方式设成了 DSpark 却没指定
        草稿模型。踩过：模型的存档里 spec-type=draft-dspark 但没有 model-draft。
        """
        exe = B.resolve_exe(self.engine_var.get().strip(), "server")
        if not exe:
            return
        warned = 0
        for name, model_path, snap in entries:
            # embedding 的「一次能吃多少」不看参数对不对，看模型自己的训练
            # 长度 —— 第三方向量软件传文档失败几乎都栽在这条：
            #   input (1008 tokens) is too large to process
            if B.is_embedding_model(model_path):
                _cap = B.model_ctx_train(model_path)
                if _cap and _cap < B.EMB_MIN_UBATCH:
                    self.log_append(
                        "「%s」是 embedding 模型，训练长度只有 %d tokens"
                        "（引擎会把上下文压到这个值）：第三方软件传文档时，"
                        "分块必须 ≤%d token，否则报 input too large；"
                        "要处理更长的文档请换窗口更大的 embedding 模型。"
                        % (name, _cap, _cap), "warn")
            audit = dict(snap or {})
            if preset_path:
                audit["models_preset"] = {"on": True, "value": preset_path}
            try:
                errs = [e for e in B.validate(audit, "server", exe,
                                              model=model_path, role="llm")
                        if e[0] not in self._AUDIT_SKIP]
            except Exception:      # noqa: BLE001
                continue
            if not errs:
                continue
            msg = str(errs[0][1]).splitlines()[0]
            hint = ""
            if ("草稿模型" in msg) and B._model_has_mtp(model_path):
                hint = ("（这个模型自带 MTP 预测头：[推测解码] 页把方式改成 "
                        "MTP 就不需要草稿模型了）")
            self.log_append("「%s」的参数有问题，它加载时会失败：%s%s"
                            % (name, msg, hint), "warn")
            warned += 1
            if warned >= 5:
                self.log_append("……还有更多模型参数有问题，先修上面这几个",
                                "warn")
                break

    def router_name_for(self, path: str) -> str:
        """模型路径 → 路由里的段名。"""
        want = os.path.normcase(str(path or ""))
        for name, p in (self.router_path_map or {}).items():
            if os.path.normcase(p) == want:
                return name
        # 表还没建（没扫描过 / 刚起来）→ 退化成按显示名算
        return self.router_model_name(path)

    def router_load(self, path: str = "") -> bool:
        """加载一个模型（默认「目标模型」）。后端没起来就顺手把整条路由拉起来。"""
        target = path or self.selected_model
        if not target or not os.path.isfile(target):
            messagebox.showinfo(
                "先选模型",
                "请在「模型库」里选中一行（或用「设为目标模型」指定），\n"
                "再点「加载」。", parent=self.root)
            return False
        # Drafters 不能单独加载，只能通过推测解码参数关联
        if self._classify_path(target) == "drafter":
            messagebox.showinfo(
                "不能单独加载草稿模型",
                "草稿模型（Drafter）不能单独加载到路由里。\n"
                "请在「推测解码」参数里配好草稿模型，它会跟随主 LLM 一起加载。",
                parent=self.root)
            return False
        base = os.path.basename(target)
        if not self.manager.get("unified").running:
            self.log_append("路由后端没在运行，先把它拉起来（顺便加载 %s）"
                            % base, "sys")
            self.selected_model = target
            return self.start_role("llm", model_override=target)
        if not self.rebuild_presets():
            return False
        name = self.router_name_for(target)
        ok, msg = self.router.load(name)
        self.log_append("加载「%s」：%s" % (name, msg), "ok" if ok else "err")
        if ok:
            self.mark_loaded(name)
        self.refresh_loaded_models()
        return ok

    def router_unload_name(self, name: str) -> None:
        ok, msg = self.router.unload(name)
        self.log_append("卸载「%s」：%s" % (name, msg), "warn" if ok else "err")
        self.refresh_loaded_models()

    def router_unload_path(self, path: str) -> None:
        """模型库页的「卸载」按钮：按路径找到段名再卸。"""
        name = self.router_name_for(path)
        if name not in (self.router.loaded_names() or []):
            messagebox.showinfo(
                "未加载",
                "「%s」当前没有驻留。\n（只有已加载的模型才能卸载）"
                % os.path.basename(path), parent=self.root)
            return
        self.router_unload_name(name)

    def router_unload_all(self) -> None:
        names = self.router.loaded_names()
        if not names:
            self.log_append("当前没有已加载的模型。", "sys")
            return
        n = self.router.unload_all()
        self.log_append("已请求卸载全部 %d 个模型（路由进程留着，随时可以再加载）"
                        % n, "warn")
        self.refresh_loaded_models()

    def mark_loaded(self, name: str) -> None:
        if name and name not in self.router_history:
            self.router_history.append(name)
        self.router.touch(name)

    def reload_models(self) -> None:
        """「重载」：重建预置 → 引擎重读 → 把加载过的模型重新加载一遍。

        目标 = 当前驻留的 ∪ 本次会话加载过的（含刚被策略顶掉 / 手动卸载的）。
        逐个重载 —— 会重新走一遍加载，所以新参数立刻生效，而且**不用重启
        路由进程**（其他模型全程不受影响）。
        超过「驻留策略」上限时，多出来的会被策略按 LRU 顶掉（日志里会写明）。
        """
        st = self.manager.get("unified")
        if not st.running:
            self.log_append("路由后端没在运行，重载改成「启动」：加载目标模型",
                            "sys")
            self.router_load()
            return
        targets: List[str] = []
        for name in (self.router.loaded_names() or []) + list(self.router_history):
            if name and name not in targets:
                targets.append(name)
        if not targets:
            self.log_append("还没有加载过任何模型，先点「加载」。", "warn")
            return
        if not self.rebuild_presets():
            return
        self.log_append("重载 %d 个模型（重新读 config\\models 里的参数）：%s"
                        % (len(targets), "、".join(targets)), "sys")
        for name in targets:
            ok, msg = self.router.reload_model(name)
            self.log_append("  重载「%s」：%s" % (name, msg),
                            "ok" if ok else "err")
            if ok:
                self.mark_loaded(name)
        self.sync_residency_policy()
        self.refresh_loaded_models()

    def refresh_loaded_models(self) -> None:
        """按 RouterMonitor 的最新清单重画「已加载模型」列表。"""
        box = getattr(self, "loaded_box", None)
        if box is None:
            return
        if getattr(self, "loaded_sig", None) == self._loaded_signature():
            return                       # 没变化就别重建控件（免得闪烁）
        self.loaded_sig = self._loaded_signature()
        for w in box.winfo_children():
            w.destroy()
        snap = self.router.snapshot()
        if not snap.get("active"):
            ttk.Label(box, text=("路由后端没在运行 —— 点「加载」会把它拉起来"),
                      style="Muted.TLabel").pack(side="left")
            return
        models = snap.get("models") or []
        resident = [m for m in models if m.get("status") in ROU.RESIDENT]
        # 过滤掉 Drafters：草稿模型跟随主 LLM 一起加载，不在顶栏单独显示
        resident = [m for m in resident
                    if self.router_kind_of(str(m.get("id"))) != "drafter"]
        if not resident:
            ttk.Label(box, text="当前没有已加载的模型（点「加载」把目标模型装进来）",
                      style="Muted.TLabel").pack(side="left")
            return
        pol = self.router_policy
        for m in resident:
            name = str(m.get("id"))
            kind = self.router_kind_of(name)
            st = str(m.get("status"))
            colour = C["warn"] if st in ("loading", "sleeping") else C["ok"]
            row = ttk.Frame(box, style="Alt.TFrame", padding=(8, 3))
            row.pack(side="left", padx=(0, 6))
            ttk.Label(row, text="\u25cf", style="Alt.TLabel",
                      foreground=colour).pack(side="left")
            ttk.Label(row, text=name[:34], style="Alt.TLabel",
                      font=T.FONT_UI_BOLD).pack(side="left", padx=(4, 6))
            limit = pol.limit_for(kind)
            tag = "%s · %s%s" % (
                "LLM" if kind == "llm" else "Embedding",
                ROU.STATUS_ZH.get(st, st),
                "" if limit is None else "（%s 上限 %d）"
                % ("LLM" if kind == "llm" else "Emb", limit))
            ttk.Label(row, text=tag, style="MutedAlt.TLabel").pack(side="left")
            ttk.Button(row, text="卸载", style="Tiny.TButton",
                       command=lambda n=name: self.router_unload_name(n)
                       ).pack(side="left", padx=(8, 0))

    def _loaded_signature(self) -> Tuple:
        snap = self.router.snapshot()
        return (bool(snap.get("active")),
                tuple((str(m.get("id")), str(m.get("status")))
                      for m in snap.get("models") or []))

    def _router_tick(self) -> None:
        """约每秒一次：同步驻留策略 → 重画目标/已加载/底栏。

        限流与空闲卸载不在这里做 —— 那在 RouterMonitor 的后台线程里
        （它才是唯一知道「每个模型最后一次被请求是什么时候」的地方）。
        """
        self.sync_residency_policy()
        self._refresh_role_slots()
        self._update_buttons()

    def set_row_value(self, key: str, value: str, on: bool = True) -> None:
        row = self.rows.get(key)
        if row is None:
            return
        if on:
            row.on_var.set(True)
        row.var.set(value)
        try:
            row._apply_state()
            row._update_flag_label()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------- 统一端口模式
    def start_unified_backend(self, path: str) -> Dict[str, Any]:
        """把统一后端切成 / 启动成 ``path`` 这个模型（主线程执行，只负责起进程）。"""
        if not path:
            return {"ok": False, "error": "没有指定模型"}
        if not os.path.isfile(path):
            return {"ok": False, "error": "模型文件不存在：%s" % path}
        role = "llm"
        exe = B.resolve_exe(self.engine_var.get().strip(), "server")
        if not exe:
            return {"ok": False,
                    "error": "没有找到 llama-server，请先设置引擎目录"}
        self.warn_if_bad_arch(exe, path)
        internal = self.free_port()
        snap = self.effective_snapshot("server", role, model=path)
        # 内部后端固定监听本机内部端口，对外只暴露网关那一个端口
        snap["host"] = {"on": True, "value": "127.0.0.1"}
        snap["port"] = {"on": True, "value": str(internal)}
        router = S.is_router(snap)
        if router:
            for note in self._prepare_router(snap, path):
                self.log_append("路由模式：" + note, "sys")
        # 路由模式下不做「针对这一个模型」的校验：路由器自己不带 -m，
        # 模型参数全在预置文件里，按目标模型卡住会让别的模型也起不来
        errs = B.validate(snap, "server", exe, model=path, role=role,
                          model_checks=not router)
        if errs:
            msg = "；".join(m for _k, m in errs[:3])
            self.log_append("启动失败：%s" % msg, "err")
            return {"ok": False, "error": msg}
        argv = self.build_argv_checked(snap, "server", model=path, role=role,
                                       exe=exe)
        st = self.manager.get("unified")
        if st.running:
            self.manager.stop("unified")
        cwd = os.path.dirname(exe)
        if router:
            self.log_append("运行方式：多模型路由（网关只做透传，"
                            "不参与换模型）", "sys")
        for note in B.kv_compat_notes(snap, path):
            self.log_append("提示：" + note, "warn")
        self.log_append("统一后端：加载 %s（%s 参数，内部端口 %d）"
                        % (os.path.basename(path), S.ROLES[role]["zh"],
                           internal), "sys")
        try:
            self.manager.start("unified", exe, argv, cwd, model=path,
                               port=internal, detached=False,
                               health_url="http://127.0.0.1:%d/health"
                                          % internal,
                               env_extra=self.engine_env_for(exe))
        except Exception as exc:  # noqa: BLE001
            self.log_append("启动失败：%r" % (exc,), "err")
            return {"ok": False, "error": repr(exc)}
        self._last_unified_argv = argv
        # 开始采集运行时状态（/props + /slots 都在这个内部端口上）
        self.runtime.attach(internal, os.path.basename(path))
        # 模型清单轮询 + 按类限流 + 空闲卸载都挂到这个内部端口上
        self.router.attach(internal)
        self._refresh_role_slots()
        self._update_buttons()
        return {"ok": True, "port": internal, "role": role, "model": path}

    def stop_unified_backend(self) -> None:
        st = self.manager.get("unified")
        if st.running:
            self.log_append("正在停止统一后端（PID %s）…" % st.runner.pid,
                            "warn")
            self.manager.stop("unified")
        self.unified.mark_stopped()
        self.runtime.detach()
        self.router.mark_stopped()
        self.log_append("路由已停止，所有模型子进程一并结束、显存已释放", "warn")
        self._refresh_role_slots()
        self._update_buttons()

    def stop_role(self, role: str) -> None:
        self.stop_unified_backend()

    def unload_role(self, role: str) -> None:
        self.stop_unified_backend()

    def unload_model(self, path: str) -> None:
        for role in S.ROLE_ORDER:
            st = self.manager.get(role)
            if st.running and st.model and model_key(st.model) == \
                    model_key(path):
                self.unload_role(role)
                return
        messagebox.showinfo("未加载", "这个模型当前没有被任何角色加载。",
                            parent=self.root)

    def switch_model(self, role: str, model: str,
                     source: str = "界面") -> Dict[str, Any]:
        if not model:
            return {"ok": False, "error": "缺少 model"}
        full = model
        if not os.path.isabs(model):
            hit = next((r for r in self.lib_rows
                        if os.path.basename(r["path"]).lower()
                        == model.lower()), None)
            if hit is None:
                hit = next((r for r in self.lib_rows
                            if model.lower() in r["name"].lower()), None)
            if hit is None:
                return {"ok": False,
                        "error": "模型库里找不到「%s」；请用完整路径，"
                                 "或先在模型库扫描目录" % model}
            full = hit["path"]
        if not os.path.isfile(full):
            return {"ok": False, "error": "文件不存在：%s" % full}
        st = self.manager.get(role)
        # 不再记「默认模型」：只把请求里的模型用起来
        self.selected_model = full
        self.store.push_unique("recent_models", full)
        self.store.save()
        self.log_append("[%s] 切换 %s 角色 → %s"
                        % (source, S.ROLES[role]["zh"],
                           os.path.basename(full)), "sys")
        if st.running:
            same = bool(st.model) and model_key(st.model) == model_key(full)
            self.stop_role(role)
            if not same:
                self.root.after(700, lambda: self.start_role(role))
                return {"ok": True, "accepted": True, "role": role,
                        "model": full, "mode": "restart",
                        "note": "已停止旧进程并用新模型启动。"}
            return {"ok": True, "role": role, "model": full,
                    "mode": "restart", "note": "已重启"}
        # 角色没在跑：直接把它拉起来（远程唤起）
        started = self.start_role(role, quiet=True)
        self._refresh_role_slots()
        if not started:
            return {"ok": False, "role": role, "model": full,
                    "error": self.last_start_error or "启动失败",
                    "hint": "详细过程在界面的「运行日志」里"}
        return {"ok": True, "accepted": True, "role": role, "model": full,
                "mode": "started",
                "note": "服务已启动、模型正在加载；用 GET /status 或 /health "
                        "查进度，就绪后即可用"}

    def set_draft_model(self, path: str) -> None:
        if not self.selected_model:
            messagebox.showinfo("先选模型",
                                "请先在模型库里选中「用哪个模型来推测」的目标模型。",
                                parent=self.root)
            return
        row = self.rows.get("model_draft")
        if row is None:
            return
        row.on_var.set(True)
        row.var.set(path)
        row._apply_state()
        row._update_flag_label()
        master = self.rows.get("spec_enable")
        if master is not None and not master.on_var.get():
            master.on_var.set(True)
        self.store.push_unique("recent_drafts", path)
        self.store.save()
        self.log_append("已把 %s 设为「%s」的推测解码草稿模型"
                        % (os.path.basename(path),
                           os.path.basename(self.selected_model)), "ok")
        self.show("spec")

    def _refresh_role_slots(self) -> None:
        self._refresh_unified_slots()

    def _refresh_unified_slots(self) -> None:
        """画「目标模型」卡片 + 「已加载模型」列表。

        ⚠️ 这里说的「目标模型」= 模型库里选中的那个（或上次作为角色模型启动的
        那个）。用户要求去掉「选模型 / 设为当前」两个按钮，所以目标完全跟着
        模型库的选中行走；要加载它只要点「加载」。
        """
        self.sync_residency_policy()
        host, port = self.gateway_endpoint()
        running = self.manager.get("unified").running
        model = self.selected_model
        if model:
            self.target_name.configure(text=os.path.basename(model)[:52])
        else:
            self.target_name.configure(text="未指定（去模型库选一行）")

        resident = self.router.loaded_names() if running else []
        if not running:
            state, colour = "路由未启动 · 点「加载」会把它拉起来", C["muted"]
        elif self.unified.loading:
            state, colour = "路由正在启动…", C["warn"]
        elif any(True for _ in resident):
            state, colour = ("已驻留 %d 个模型 · 统一端口 %s"
                             % (len(resident), port)), C["ok"]
        else:
            state, colour = "路由已就绪（还没有模型驻留）", C["ok"]
        self.target_dot.configure(text="\u25cf" if running else "\u25cb",
                                 foreground=colour)
        self.target_state.configure(text=state, foreground=colour)
        pol = self.router_policy
        llm_n = sum(1 for n in resident if self.router_kind_of(n) == "llm")
        emb_n = len(resident) - llm_n
        lim_llm = pol.limit_for("llm")
        lim_emb = pol.limit_for("emb")
        self.loaded_count.configure(
            text=("已驻留 LLM %d%s · Embedding %d%s"
                  % (llm_n, "" if lim_llm is None else "/%d" % lim_llm,
                     emb_n, "（不计入）" if lim_emb is None else "/%d" % lim_emb))
            if running else "")
        self.refresh_loaded_models()
        self._paint_global_status()

    def _paint_global_status(self) -> None:
        u = self.unified
        host, port = self.gateway_endpoint()
        if self.api.running:
            parts = ["统一端口 %s:%d" % (host, port), "网关在线"]
        else:
            parts = ["统一端口 %s:%d" % (host, port),
                     "网关未启动（外部 API 连不上）"]
        if u.model:
            state = "加载中" if u.loading else (
                "已就绪" if u.ready else "未就绪")
            parts.append("%s（%s）" % (os.path.basename(u.model), state))
        else:
            parts.append("空闲")
        if u.waiters:
            parts.append("排队 %d" % u.waiters)
        ready = bool(u.ready) or self.api.running
        self.dot.configure(foreground=C["ok"] if ready else C["err"])
        self.status_lbl.configure(
            text=" · ".join(parts),
            foreground=C["err"] if not self.api.running else (
                C["ok"] if u.ready else C["muted"]))

    # ==================================================================== #
    # 单次运行实例
    # ==================================================================== #
    def start_single(self, mode: str) -> bool:
        exe = B.resolve_exe(self.engine_var.get().strip(), mode)
        model = self.selected_model
        snap = self.effective_snapshot(mode, model=model)
        errs = B.validate(snap, mode, exe, model=model)
        if errs:
            self._show_errors(errs)
            return False
        argv = self.build_argv_checked(snap, mode, model=model, exe=exe)
        cwd = os.path.dirname(exe) if exe else None
        self.log_view.clear()
        self.log_append("=" * 70, "muted")
        self.log_append("启动：%s（%s）" % (S.MODES[mode]["zh"],
                                          os.path.basename(exe)), "sys")
        self.log_append(B.format_cmdline(exe, argv), "muted")
        self.log_append("=" * 70, "muted")
        try:
            self.manager.start("single", exe, argv, cwd, model=model, port=0,
                               detached=bool(self.detached_var.get()),
                               env_extra=self.engine_env_for(exe))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("启动失败", str(exc), parent=self.root)
            return False
        # 对话 / 生成 模式没有 HTTP 接口，运行状态条在这里没有可采的东西
        self.runtime.detach("「%s」模式没有 HTTP 状态接口" % S.MODES[mode]["zh"])
        self._paint_global_status()
        self._update_buttons()
        return True

    def stop_single(self) -> None:
        self.manager.stop("single")
        self._paint_global_status()
        self._update_buttons()

    # ==================================================================== #
    # 启动 / 停止 / 重启
    # ==================================================================== #
    def start_target(self) -> None:
        """底栏「加载」：服务模式装模型；对话/生成模式起一次性进程。"""
        self.collect(self.page)
        mode, role = self.target()
        if mode == "server":
            self.router_load()
        else:
            self.start_single(mode)

    def stop_target(self) -> None:
        """底栏「停止」：服务模式 = 卸载全部模型（路由进程留着）。"""
        mode, role = self.target()
        if mode == "server":
            self.router_unload_all()
        else:
            self.stop_single()

    def restart_target(self) -> None:
        """底栏「重载」：重新读 config\\models 里的参数并重装模型。

        服务模式 = 重建预置 → 引擎重读 → 把加载过的模型逐个重装（不重启进程）；
        对话 / 生成模式仍是一次性进程的「重启」。
        """
        mode, role = self.target()
        if mode == "server":
            self.reload_models()
        else:
            if self.manager.get("single").running:
                self.stop_single()
                self.root.after(900, lambda: self.start_single(mode))
            else:
                self.start_single(mode)

    def _show_errors(self, errs: List[Tuple[str, str]]) -> None:
        for row in self.rows.values():
            row.mark_error(False)
        for key, _msg in errs:
            row = self.rows.get(key)
            if row:
                row.mark_error(True)
        head = [m for _k, m in errs][:6]
        extra = "" if len(errs) <= 6 else "\n…还有 %d 条" % (len(errs) - 6)
        messagebox.showwarning("参数需要修正",
                               "\n".join("· " + m for m in head) + extra,
                               parent=self.root)

    def _update_buttons(self) -> None:
        """底栏三个按钮的可用性。

        ⚠️ 这里原来查的是 ``manager.get(role)``（role = "llm"），但后端进程实际
        注册在 ``"unified"`` 这个 key 上 —— 查不到就新建一个空的 RoleState，
        ``running`` 永远是 False，于是「停止」加载完模型也一直是灰的。
        这就是用户报的「停止按钮无作用」，修在取 key 这一步。
        """
        mode, _role = self.target()
        if mode == "server":
            busy = bool(self.router.loaded_names())
            self.btn_stop.state(["!disabled"] if busy else ["disabled"])
        else:
            running = self.manager.get("single").running
            self.btn_stop.state(["!disabled"] if running else ["disabled"])

    # ==================================================================== #
    # 命令行预览
    # ==================================================================== #
    def current_argv(self) -> Tuple[Optional[str], List[str]]:
        self.collect(self.page)
        mode, role = self.target()
        exe = B.resolve_exe(self.engine_var.get().strip(), mode)
        model = self.model_for(mode, role)
        snap = self.effective_snapshot(mode, role, model)
        return exe, self.build_argv_checked(snap, mode, model=model,
                                            role=role, exe=exe)

    def model_ini_preview(self) -> str:
        """当前选中模型在路由预置文件里的那一段（含状态提示）。"""
        path = self.selected_model
        self.collect(self.page)
        snap = MG.merge_model_snapshot(self.store, path)
        name = self.router_model_name(path)
        ini_path = os.path.join("config", "router-preset.ini")
        head = ("; ↓ 这一段会写进路由预置文件 %s\n"
                "; 服务只剩「多模型路由」一种跑法：模型参数**不进命令行**，\n"
                "; 因为路由器会把自身命令行 merge（覆盖）进所有模型。\n"
                "; 所以模型级参数只写在各自的段里。改完点底栏「重载」生效。\n\n"
                % ini_path)
        st = ""
        if self.manager.get("unified").running:
            cur = ""
            for m in (self.router.snapshot().get("models") or []):
                if m.get("id") == name:
                    cur = ROU.STATUS_ZH.get(str(m.get("status")), "")
                    break
            st = "\n\n; 该模型当前状态：%s" % (cur or "未加载")
        return head + B.preset_section(name, path, snap) + st

    def schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True
        self.root.after(180, self._do_refresh)

    def _do_refresh(self) -> None:
        self._refresh_pending = False
        for row in self.rows.values():
            row.mark_error(False)
        if self.page == "spec":
            self._sync_spec_fields()
        if self.page == "server":
            self._sync_role_fields()
        # 端口 / 运行方式改动后延迟重绑，避免边输边抢端口
        if self._gw_after is not None:
            try:
                self.root.after_cancel(self._gw_after)
            except tk.TclError:
                pass
        self._gw_after = self.root.after(1200, self._sync_gateway)
        self._refresh_cmd()

    def on_value_change(self) -> None:
        self.schedule_refresh()

    def _refresh_cmd(self) -> None:
        if not hasattr(self, "cmd_view"):
            return
        mode, _role = self.target()
        # 模型级页面（加载/对话/推测/LoRA/embedding）显示的是「这个模型会写进
        # 路由预置的那一段」—— 模型参数现在不进命令行了，看命令行没有意义。
        if self.page in ("load", "chat", "spec", "lora", "emb") \
                and self.selected_model:
            self.cmd_view.set_text(self.model_ini_preview())
            self._update_engine_label()
            self.update_target_caption()
            return
        exe, argv = self.current_argv()
        if exe:
            text = (B.format_multiline(os.path.basename(exe), argv)
                    if bool(self._vars["cmd_multiline"].get())
                    else B.format_cmdline(os.path.basename(exe), argv))
            env = self.engine_var.get().strip()
            if env and os.path.isdir(env):
                text = "cd /d \"%s\"\n\n%s" % (env, text)
            self.cmd_view.set_text(text)
        else:
            self.cmd_view.set_text(
                "（还没有定位到 %s）\n\n请在上方「引擎目录」里选择 "
                "buun-llama-cpp 的目录或可执行文件。"
                % (S.MODES[mode]["exe"] + B.EXE_SUFFIX))
        self._update_engine_label()
        self.update_target_caption()

    def copy_cmd(self) -> None:
        exe, argv = self.current_argv()
        if not exe:
            messagebox.showwarning("缺少可执行文件",
                                   "请先指定 buun-llama-cpp 目录。",
                                   parent=self.root)
            return
        self.clipboard(B.format_cmdline(exe, argv))
        self.log_append("命令行已复制到剪贴板", "ok")

    def export_bat(self) -> None:
        exe, argv = self.current_argv()
        if not exe:
            messagebox.showwarning("缺少可执行文件",
                                   "请先指定 buun-llama-cpp 目录。",
                                   parent=self.root)
            return
        mode, _role = self.target()
        init = str(self.store.pref("export_dir", "") or "") or self.default_dir()
        default = "%s_%s.bat" % (
            S.MODES[mode]["exe"],
            os.path.splitext(os.path.basename(self.selected_model)
                             or "run")[0])
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=init, initialfile=default,
            defaultextension=".bat",
            filetypes=[("批处理文件", "*.bat"), ("全部文件", "*.*")])
        if not path:
            return
        cwd = self.engine_var.get().strip()
        content = B.make_bat(exe, argv, cwd if os.path.isdir(cwd) else "")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.store.set_pref("export_dir", os.path.dirname(path))
        self.store.save()
        self.log_append("已导出启动脚本：%s" % path, "ok")

    def clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def copy_api(self, role: str = "llm") -> None:
        host, port = self.gateway_endpoint()
        url = "http://%s:%d/v1" % (host if host != "0.0.0.0"
                                   else "127.0.0.1", port)
        self.clipboard(url)
        self.log_append("OpenAI 兼容地址已复制：%s" % url, "ok")

    def open_webui(self, role: str = "llm") -> None:
        host, port = self.gateway_endpoint()
        url = "http://%s:%d/" % (host if host != "0.0.0.0" else "127.0.0.1",
                                 port)
        webbrowser.open(url)
        self.log_append("已在浏览器打开 %s" % url, "sys")

    def open_path(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ==================================================================== #
    # 事件泵
    # ==================================================================== #
    def log_append(self, line: str, tag: str = "") -> None:
        self.log_view.append(line, tag)

    def log_append_ts(self, line: str, tag: str = "") -> None:
        """线程安全的日志：网关/调度线程调用它，主线程 _pump 时再写控件。"""
        self.log_q.put((line, tag))

    def _pump(self) -> None:
        if self._closing:
            return
        self._tick += 1
        for key, st in self.manager.roles.items():
            for _ in range(300):
                try:
                    ev = st.runner.q.get_nowait()
                except queue.Empty:
                    break
                self._on_runner_event(key, ev)
        for _ in range(60):
            try:
                ev = self.manager.q.get_nowait()
            except queue.Empty:
                break
            self._on_manager_event(ev)
        for _ in range(20):
            try:
                kind, payload = self.scan_q.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                i, n = payload
                if self.lib:
                    self.lib.count_lbl.configure(
                        text="正在解析 %d / %d …" % (i, n))
            elif kind == "scan":
                self._pending_scan = payload
            elif kind == "scan_done":
                rows, err = self._pending_scan
                self.store.save()
                payload(rows, err)
            elif kind == "probe":
                self._probe_done(payload)
            elif kind == "probe-quiet":
                self._probe_done(payload, quiet=True)
        self.manager.bus.drain(self.handle_command)
        for _ in range(80):
            try:
                line, tag = self.log_q.get_nowait()
            except queue.Empty:
                break
            self.log_append(line, tag)
        if self._tick % 6 == 0:
            self._router_tick()
        if self._tick % 8 == 0:
            self._runtime_tick()
        self.root.after(150, self._pump)

    def _on_runner_event(self, key: str, ev: Any) -> None:
        who = MG.KEY_ZH.get(key) or S.ROLES.get(key, {}).get("zh") or key
        if isinstance(ev, P.LogLine):
            buf = self.out_tail.setdefault(key, [])
            buf.append(ev.text)
            if len(buf) > 400:
                del buf[:len(buf) - 400]
            self.log_view.append(ev.text, classify(ev.text))
            # 速度已统一在底部运行状态栏显示，不再从进程日志解析
        elif isinstance(ev, P.StartedEvent):
            self.out_tail[key] = []      # 新进程，旧的报错别串到这次的诊断里
            self.log_append("[%s] 进程已启动，PID %d" % (who, ev.pid), "ok")
        elif isinstance(ev, P.HealthEvent):
            st = self.manager.get(key)
            if ev.ok:
                self.log_append("[%s] 服务已就绪 → %s" % (who, st.base_url()),
                                "ok")
                if key in S.ROLES:
                    self.log_append("[%s] OpenAI 兼容地址：%s/v1"
                                    % (who, st.base_url()), "ok")
            else:
                self.log_append("[%s] 健康检查未通过：%s" % (who, ev.detail),
                                "warn")
        elif isinstance(ev, P.ExitEvent):
            self.log_append("=" * 70, "muted")
            self.log_append("[%s] 进程已退出，退出码 %s" % (who, ev.code),
                            "ok" if ev.code == 0 else "err")
            if ev.code not in (0, None):
                self.diagnose_exit(key, ev.code)
            self.out_tail.pop(key, None)
            self.manager.get(key).loaded_model = ""
            if key == "unified":
                self.runtime.detach("后端进程已退出")

    def _on_manager_event(self, ev: MG.ManagerEvent) -> None:
        who = MG.KEY_ZH.get(ev.role) or S.ROLES.get(ev.role, {}).get(
            "zh") or ev.role
        self.log_append(ev.detail, "ok" if ev.ok else "warn")
        self._refresh_role_slots()

    # ------------------------------------------------------ 控制 API 指令
    def handle_command(self, name: str,
                       payload: Dict[str, Any]) -> Dict[str, Any]:
        if name == "start_unified":
            res = self.start_unified_backend(str(payload.get("model") or ""))
            return res if res.get("ok") else {
                "ok": False, "error": res.get("error") or "启动失败"}
        if name == "stop_unified":
            self.stop_unified_backend()
            return {"ok": True}
        # 只有一个角色；老客户端传的 {"role":"llm"} 直接忽略掉就行
        str(payload.get("role") or "llm").lower()
        if name in ("switch", "load"):
            return self.switch_model("llm",
                                     str(payload.get("model") or "").strip(),
                                     source="API")
        if name == "unload":
            self.stop_unified_backend()
            return {"ok": True, "note": "已卸载当前模型，显存已释放"}
        if name == "stop":
            if str(payload.get("target") or "") == "single":
                self.stop_single()
                return {"ok": True, "note": "已停止单次运行实例"}
            self.stop_unified_backend()
            return {"ok": True, "note": "已停止 llama-server"}
        if name == "start":
            ok = self.start_role("llm", quiet=True)
            return {"ok": ok, "note": "已启动" if ok else "启动失败，看界面日志"}
        return {"ok": False, "error": "未知指令 %s" % name}

    # ==================================================================== #
    # 弹窗
    # ==================================================================== #
    def show_api_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("控制 API")
        win.geometry("760x560")
        win.configure(background=C["bg"])
        cfg = self.store.control()
        ttk.Label(win, text="控制 API · 用 HTTP 切换模型", style="H2.TLabel",
                  background=C["bg"]).pack(anchor="w", padx=16, pady=(14, 4))
        uh, up = self.gateway_endpoint()
        ttk.Label(
            win,
            text=("控制 API、模型列表、推理都在同一个端口上：http://%s:%d\n"
                  "端口在「服务」页的「统一端口」改；网关随程序启动，不用手动开。"
                  % (uh, up)),
            style="MutedBg.TLabel", justify="left").pack(anchor="w", padx=16,
                                                         pady=(0, 4))
        row = ttk.Frame(win, style="Bg.TFrame")
        row.pack(fill="x", padx=16, pady=(4, 6))
        ttk.Label(row, text="Token", style="MutedBg.TLabel").pack(
            side="left")
        token = tk.StringVar(value=str(cfg.get("token") or ""))
        ttk.Entry(row, textvariable=token, width=18).pack(side="left",
                                                          padx=(6, 0))
        state = ttk.Label(win, text=self.api_state_text(),
                          style="MutedBg.TLabel")
        state.pack(anchor="w", padx=16)
        body = LogView(win, height=18)
        body.pack(fill="both", expand=True, padx=16, pady=(6, 8))

        def refresh_text() -> None:
            t = token.get().strip()
            base = "http://%s:%d" % self.gateway_endpoint()
            head_note = ("# 统一端口：控制 API 与推理同一个口；"
                         "网关%s" % ("在线" if self.api.running
                                    else "未启动（外部连不上）"))
            lines = [
                head_note,
                "基地址：%s" % base,
                "鉴权：%s" % ("Authorization: Bearer %s" % t if t
                              else "未设置 Token，无需鉴权"),
                "",
                "# 网关是否活着（模型没加载也会返回 ok）",
                "curl %s/health" % base,
                "",
                "# 当前状态（谁在跑、跑的是哪个模型、空闲多久）",
                "curl %s/status" % base,
                "",
                "# 模型库（含分类 / 参数量 / 能力图标对应的布尔字段）",
                'curl "%s/models?category=LLMS"' % base,
                "",
                "# 唤起 / 切换模型 —— 会等到模型真的加载好才返回",
                "curl -X POST %s/switch \\" % base,
                '     -H "Content-Type: application/json" \\',
                '     -d "{\\"model\\":\\"Qwen3.8-27B-Uncensored-Heretic-v3\\"}"',
                "",
                "# 不想等就异步提交，再用 /status 或 /health 轮询",
                'curl -X POST "%s/switch?async=1" -d "{\\"model\\":\\"...\\"}"'
                % base,
                "",
                "# 卸载（释放显存）",
                "curl -X POST %s/unload -d \"{}\"" % base,
                "",
                "# 第三方软件：把 base_url 指到 %s/v1" % base,
                "curl %s/v1/models" % base,
                "",
                "model 可以写短模型名 / 文件名 / 完整路径；"
                "没扫过模型库也能直接按路径唤起。",
            ]
            body.set_text("\n".join(lines))

        def apply_cfg() -> None:
            self.store.control().update({"enabled": True,
                                         "token": token.get().strip()})
            self.store.save()
            self.apply_control_api()
            state.configure(text=self.api_state_text())
            refresh_text()

        btns = ttk.Frame(win, style="Bg.TFrame")
        btns.pack(fill="x", padx=16, pady=(0, 14))
        ttk.Button(btns, text="保存并应用", style="Accent.TButton",
                   command=apply_cfg).pack(side="left")
        ttk.Button(btns, text="复制示例", style="Mini.TButton",
                   command=lambda: self.clipboard(body.get_text())).pack(
            side="left", padx=(8, 0))
        ttk.Button(btns, text="关闭", style="Mini.TButton",
                   command=win.destroy).pack(side="right")
        refresh_text()

    def api_state_text(self) -> str:
        if self.api.running:
            mode = "统一端口（控制 + 推理同一个口）" if self._gateway_key \
                and self._gateway_key[0] == "unified" else "控制 API"
            return "运行中：http://%s:%d（%s）" % (self.api.host, self.api.port,
                                                 mode)
        return "未运行（点「保存并应用」启动）"

    def apply_control_api(self) -> None:
        """网关只绑统一端口那一个口，且要一直在线（否则外部没法唤起模型）。"""
        self._sync_gateway(force=True)

    def _sync_gateway(self, force: bool = False) -> None:
        cfg = self.store.control()
        self.api.token = str(cfg.get("token") or "")
        # 只有「统一端口」一种方式：网关必须一直在线，否则外部 API 连不上，
        # 也就没法远程唤起模型。所以这里不看「是否启用」，永远绑统一端口。
        host, port = self.gateway_endpoint()
        key = ("unified", host, port)
        if not force and self._gateway_key == key:
            return
        old = self._gateway_key
        self.api.stop()
        ok, err = self.api.start(host, port)
        self._gateway_key = key if ok else ("", 0)
        self._gw_error = "" if ok else err
        if ok:
            where = "统一端口" if key[0] == "unified" else "控制 API"
            self.log_append("%s已监听 http://%s:%d"
                            % (where, host, port), "ok")
            if key[0] == "unified":
                self.log_append("控制 / 模型列表 / 推理都在这个端口上；"
                                "llama-server 内部自己用别的端口。", "sys")
        else:
            self.log_append("监听 %s:%d 失败：%s" % (host, port, err), "err")
            if "10013" in str(err):
                self.log_append(
                    "WinError 10013：这个端口要么被别的程序占着，"
                    "要么落在 Windows 的保留端口区间里"
                    "（Hyper-V / WSL / Docker 经常会预留一段，8080 就在常见区间内）。",
                    "warn")
            if key[0] == "unified":
                # 统一端口是外部唯一的入口，起不来等于整个 API 断掉 —— 直接换一个能用的
                if self._autopick_port(host, port):
                    return
                self.log_append("试过 %s 都绑不上，网关没能起来。"
                                % "、".join(str(p) for p in
                                           self._fallback_ports(port)[:6]),
                                "err")
            self.log_append("处理办法：①先停掉占用端口的程序；"
                            "②或把「服务」页的「统一端口」改成别的空闲端口"
                            "（默认 1233，也可以试 1234 / 8000）。", "warn")
            self.log_append("网关没起来的话，外部 API 是连不上的，"
                            "也就没法远程唤起模型。", "warn")
        # 网关状态变了，底栏要立刻跟着变（否则会一直显示「网关未启动」）
        self._paint_global_status()
        try:
            self.api_lbl.configure(text=self.api_state_text())
        except (AttributeError, tk.TclError):
            pass

    def _fallback_ports(self, preferred: int) -> List[int]:
        """首选端口绑不上时依次试这些（去重、范围合法）。"""
        raw: List[int] = [1233, 1234, 8000, 8001, 9000, 1235, 1236]
        raw.extend(range(preferred + 1, preferred + 13))
        out: List[int] = []
        seen: set = set()
        for p in raw:
            if p == preferred:
                continue
            if 1024 <= p <= 65535 and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _autopick_port(self, host: str, preferred: int) -> bool:
        """统一端口绑不上时自动换一个能用的，并记回配置。

        返回 True 表示已经换好并起起来了。
        """
        for cand in self._fallback_ports(preferred):
            ok, _err = self.api.start(host, cand)
            if not ok:
                continue
            self.store.role_state("llm")["port"] = {"on": True,
                                                    "value": str(cand)}
            self.store.save()
            self._gateway_key = ("unified", host, cand)
            self._gw_error = ""
            self.log_append("%s:%d 起不来，已自动改用 %d" % (host, preferred,
                                                            cand), "warn")
            self.log_append("网关现在在 http://%s:%d —— 第三方软件的 "
                            "base_url 要填这个。"
                            % (host, cand), "ok")
            self.log_append("这个端口已经记住了；想换回来就去「服务」页改"
                            "「统一端口」。", "sys")
            row = self.rows.get("port")
            if row is not None:
                row.set(self.store.role_state("llm")["port"])
            self._refresh_role_slots()
            return True
        return False

    def show_switch_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("通过 API 切换模型")
        win.geometry("820x520")
        win.configure(background=C["bg"])
        ttk.Label(win, text="切换模型 · 同时只有一个模型驻留",
                  style="H2.TLabel", background=C["bg"]).pack(
            anchor="w", padx=16, pady=(14, 2))
        ttk.Label(win, text="切换 = 停掉当前 llama-server 再用新模型启动，"
                            "所以要等它重新加载完（模型越大越久）。\n"
                            "显存里同时只有一个模型；别的模型的请求会排队等切换完成。",
                  style="MutedBg.TLabel", justify="left").pack(
            anchor="w", padx=16, pady=(0, 8))

        names = [r["name"] for r in self.lib_rows] or ["（模型库为空）"]
        var = tk.StringVar(value=names[0])
        combo = ttk.Combobox(win, textvariable=var, values=names, width=64)
        combo.pack(fill="x", padx=16, pady=(10, 4))
        info = ttk.Label(win, text="", style="MutedBg.TLabel", justify="left")
        info.pack(anchor="w", padx=16)

        def paint(_e=None) -> None:
            row = next((r for r in self.lib_rows if r["name"] == var.get()),
                       None)
            if not row:
                info.configure(text="模型库为空，先到「模型库」页扫描一次。")
                return
            warn = ("这是嵌入模型，本程序只跑 LLM，起不来；"
                    if row["category"] == "Embedding" else "")
            info.configure(text="%s · %s · %s · %s\n%s能力：%s"
                                % (row["name"], row.get("arch"),
                                   row.get("quant"), row.get("size_text"),
                                   warn, row.get("pip", "")))
        combo.bind("<<ComboboxSelected>>", paint)
        paint()

        out = LogView(win, height=9)
        out.pack(fill="both", expand=True, padx=16, pady=(8, 6))

        def do_switch() -> None:
            row = next((r for r in self.lib_rows if r["name"] == var.get()),
                       None)
            path = row["path"] if row else var.get()
            res = self.switch_model("llm", path, source="界面")
            out.clear()
            out.append("结果：%s" % ("成功" if res.get("ok") else "失败"), 
                       "ok" if res.get("ok") else "err")
            for k in ("role", "mode", "model", "note", "error"):
                if res.get(k):
                    out.append("  %-6s %s" % (k, res[k]))
            out.append("")
            out.append("—— 等价 HTTP 请求 ——", "sys")
            if self.api.running:
                out.append('curl -X POST http://%s:%d/switch \\'
                           % (self.api.host, self.api.port))
                out.append('     -H "Content-Type: application/json" \\')
                out.append('     -d "{\\"model\\":\\"%s\\"}"'
                           % path.replace("\\", "/"))
            else:
                out.append("网关没起来；去「服务」页换个空闲的统一端口，"
                           "就能从外部用 HTTP 切换。", "warn")
            self._refresh_role_slots()
            if self.page in ("load", "chat", "spec"):
                self._load_page(self.page)

        btns = ttk.Frame(win, style="Bg.TFrame")
        btns.pack(fill="x", padx=16, pady=(0, 14))
        ttk.Button(btns, text="切换", style="Accent.TButton",
                   command=do_switch).pack(side="left")
        # 「设为当前模型并编辑参数」已删除（用户要求去掉「设置默认模型」入口）。
        ttk.Button(btns, text="关闭", style="Mini.TButton",
                   command=win.destroy).pack(side="right")

    def show_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("参数对照表")
        win.geometry("980x700")
        win.configure(background=C["bg"])
        ttk.Label(win, text="界面上每一项对应的命令行参数与作用域",
                  style="H2.TLabel").pack(anchor="w", padx=14, pady=(12, 4))
        ttk.Label(win,
                  text="勾选后才写进命令行；不勾选即使用引擎默认值。"
                       "带 ⚠ 的项表示当前 build 的 --help 里没有这个参数。\n"
                        "作用域：加载/对话参数跟着模型走（每个模型一份）；"
                        "服务页参数全局共一份。",
                  style="MutedBg.TLabel", justify="left",
                  wraplength=940).pack(anchor="w", padx=14)
        view = LogView(win, height=32)
        view.pack(fill="both", expand=True, padx=14, pady=12)
        lines: List[Tuple[str, str]] = []
        for page in S.pages_with_fields():
            fields = list(S.fields_for(page))
            if not fields:
                continue
            lines.append(("", ""))
            lines.append(("── %s ──" % S.PAGE_ZH[page], "sys"))
            sections = list(S.sections_for(page))
            if sections:
                for sid, sec_zh in sections:
                    group = [f for f in fields if f.section == sid]
                    if not group:
                        continue
                    lines.append(("   · %s" % sec_zh, "muted"))
                    for f in group:
                        lines.append(("     %-26s %-28s %s"
                                      % (f.zh, f.flag or "（位置参数）",
                                         f.en), ""))
            else:
                for f in fields:
                    lines.append(("   %-26s %-28s %s"
                                  % (f.zh, f.flag or "（位置参数）", f.en), ""))
        view.append_many(lines)
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=(0, 12))

    # ==================================================================== #
    # 会话恢复 / 退出
    # ==================================================================== #
    def _restore_session(self) -> None:
        self._update_engine_label()
        d = str(self.store.get("models_dir") or "")
        if self.lib and d:
            self.lib.dir_var.set(d)
        # 没有「默认模型」可读了 —— 恢复「上次用过的模型」就够了（最近使用列表）。
        recent = list(self.store.get("recent_models") or [])
        if not self.selected_model and recent and os.path.isfile(recent[0]):
            self.selected_model = recent[0]
        if self.selected_model:
            self.store.ensure_model(self.selected_model)
        self.detached_var.set(False)
        self.role = str(self.store.get("last_role") or "llm")
        if self.role not in S.ROLES:
            self.role = "llm"
        self.show(self.page if self.page in self.pages else "lib")
        self._refresh_role_slots()
        self._update_buttons()
        if self.store.pref("show_english"):
            for row in self.rows.values():
                row.set_label_lang(True)
            for frame in self.pages.values():
                if hasattr(frame, "set_section_lang"):
                    frame.set_section_lang(True)
        self.log_append("%s 已就绪。①模型库扫描目录 → ②选模型 → ③调参数 → "
                        "④启动。" % APP_TITLE, "sys")
        self.log_append("数据目录：%s（配置写在它下面的 config/ 里）"
                        % self.base_dir, "muted")
        self.log_append("F5 启动，Shift+F5 停止；底栏「切换模型（API）」"
                        "可以在运行中换模型。", "muted")
        self.apply_control_api()
        if self.store.get("models_dir"):
            self.lib.scan_now()
