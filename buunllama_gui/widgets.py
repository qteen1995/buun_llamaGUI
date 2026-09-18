# -*- coding: utf-8 -*-
"""可复用的小控件：气泡提示、可滚动容器、分段选择器、日志视图。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Sequence

from . import theme as T
from .theme import C, FONT_NAV, FONT_NAV_BOLD


# --------------------------------------------------------------------------- #
# 气泡提示
# --------------------------------------------------------------------------- #

class Tooltip:
    """鼠标悬停时显示「参数名 + 说明」。"""

    def __init__(self, widget: tk.Misc, text: str, delay: int = 420):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after: Optional[str] = None
        self._win: Optional[tk.Toplevel] = None
        if text:
            widget.bind("<Enter>", self._enter, add="+")
            widget.bind("<Leave>", self._leave, add="+")
            widget.bind("<ButtonPress>", self._leave, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _enter(self, _evt=None) -> None:
        self._cancel()
        if self.text:
            self._after = self.widget.after(self.delay, self._show)

    def _leave(self, _evt=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self) -> None:
        if self._win or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except tk.TclError:
            return
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        try:
            win.attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(win, background=C["border_strong"], bd=0)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=self.text, justify="left",
                 background="#fdfefe", foreground=C["text"],
                 font=T.FONT_UI_SMALL, padx=10, pady=7,
                 wraplength=460).pack(padx=1, pady=1)
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        if x + w > sw - 8:
            x = max(8, sw - w - 8)
        if y + h > sh - 8:
            y = max(8, self.widget.winfo_rooty() - h - 6)
        win.wm_geometry("+%d+%d" % (x, y))
        self._win = win

    def _hide(self) -> None:
        if self._win:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None


# --------------------------------------------------------------------------- #
# 可滚动容器
# --------------------------------------------------------------------------- #

class ScrollableFrame(ttk.Frame):
    """一个带竖向滚动条的容器，内容放在 .body 里。"""

    def __init__(self, parent, background: str = None, padding=(18, 12)):
        super().__init__(parent, style="Card.TFrame")
        bg = background or C["panel"]
        self.canvas = tk.Canvas(self, background=bg, highlightthickness=0,
                                borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas, style="Card.TFrame")
        self._win = self.canvas.create_window((0, 0), window=self.body,
                                              anchor="nw")
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.body.configure(padding=padding)

    def _on_scroll_set(self, first, last) -> None:
        # 内容不足一屏时隐藏滚动条，界面更干净
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.vbar.pack_forget()
        else:
            self.vbar.pack(side="right", fill="y")
        self.vbar.set(first, last)

    def _on_body_configure(self, _evt=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, evt) -> None:
        self.canvas.itemconfigure(self._win, width=evt.width)

    def _bind_wheel(self, _evt=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _evt=None) -> None:
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.canvas.unbind_all(seq)
            except Exception:
                pass

    def _on_wheel(self, evt) -> None:
        if not self.canvas.winfo_exists():
            return
        first, last = self.canvas.yview()
        if float(first) <= 0.0 and float(last) >= 1.0:
            return
        if getattr(evt, "num", None) == 4:
            delta = -1
        elif getattr(evt, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(evt.delta / 120) if evt.delta else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")


# --------------------------------------------------------------------------- #
# 分段选择器
# --------------------------------------------------------------------------- #

class Segmented(ttk.Frame):
    """一排互斥按钮，用于模式切换。"""

    def __init__(self, parent, items: Sequence[str],
                 on_change: Optional[Callable[[str], None]] = None,
                 style: str = "Bar.TFrame"):
        super().__init__(parent, style=style)
        self.items = list(items)
        self.buttons: Dict[str, ttk.Button] = {}
        self.value = self.items[0] if self.items else ""
        self._on_change = on_change
        for i, item in enumerate(self.items):
            btn = ttk.Button(self, text=item, style="TButton",
                             command=lambda it=item: self.select(it))
            btn.grid(row=0, column=i, sticky="ew",
                     padx=(0 if i == 0 else 1, 0))
            self.columnconfigure(i, weight=1)
            self.buttons[item] = btn
        self._refresh()

    def select(self, item: str, notify: bool = True) -> None:
        if item not in self.buttons or item == self.value:
            return
        self.value = item
        self._refresh()
        if notify and self._on_change:
            self._on_change(item)

    def set_value(self, item: str) -> None:
        """静默设置当前项（不触发回调）。"""
        if item in self.buttons:
            self.value = item
            self._refresh()

    def set_enabled(self, enabled: bool) -> None:
        for btn in self.buttons.values():
            try:
                btn.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _refresh(self) -> None:
        for name, btn in self.buttons.items():
            if name == self.value:
                btn.configure(style="Accent.TButton")
            else:
                btn.configure(style="TButton")


# --------------------------------------------------------------------------- #
# 左侧竖向导航
# --------------------------------------------------------------------------- #

class NavItem:
    def __init__(self, page: str, label: str, icon: str = "") -> None:
        self.page = page
        self.label = label
        self.icon = icon
        self.rows: List[tk.Widget] = []
        self.enabled = True
        self.badge = ""


class Sidebar(tk.Frame):
    """竖向导航：分组的页面列表，替代原来两排横向页签。"""

    def __init__(self, parent, groups, on_select, width: int = 194,
                 background: str = None) -> None:
        bg = background or C["panel_alt"]
        super().__init__(parent, background=bg, width=width)
        self.pack_propagate(False)
        self.bg = bg
        self._on_select = on_select
        self.items: Dict[str, NavItem] = {}
        self.order: List[str] = []
        self.value = ""

        for gi, (title, entries) in enumerate(groups):
            if gi:
                tk.Frame(self, background=bg, height=10).pack(fill="x")
            head = tk.Label(self, text=title, background=bg,
                            foreground=C["faint"], font=T.FONT_UI_SMALL,
                            anchor="w", padx=16)
            head.pack(fill="x", pady=(6, 2))
            for page, label, icon in entries:
                item = NavItem(page, label, icon)
                row = tk.Frame(self, background=bg, cursor="hand2")
                row.pack(fill="x")
                text = tk.Label(row, text=label, background=bg,
                                foreground=C["text"], font=FONT_NAV,
                                anchor="w", padx=14, pady=6)
                text.pack(side="left", fill="x", expand=True)
                if icon:
                    text.configure(text="%s  %s" % (icon, label))
                badge = tk.Label(row, text="", background=bg,
                                 foreground=C["ok"], font=T.FONT_UI_SMALL)
                badge.pack(side="right", padx=(0, 10))
                item.rows = [row, text, badge]
                self.items[page] = item
                self.order.append(page)
                for w in (row, text):
                    w.bind("<Button-1>", lambda _e, p=page: self.select(p))
                    w.bind("<Enter>", lambda _e, p=page: self._hover(p, True))
                    w.bind("<Leave>", lambda _e, p=page: self._hover(p, False))
        tk.Frame(self, background=bg).pack(fill="both", expand=True)

    # ------------------------------------------------------------- 交互
    def select(self, page: str, notify: bool = True) -> None:
        item = self.items.get(page)
        if not item or not item.enabled or page == self.value:
            return
        self.value = page
        self._paint()
        if notify:
            self._on_select(page)

    def set_value(self, page: str) -> None:
        if page in self.items:
            self.value = page
            self._paint()

    def set_enabled(self, page: str, enabled: bool) -> None:
        item = self.items.get(page)
        if item:
            item.enabled = enabled
            self._paint()

    def set_badge(self, page: str, text: str, colour: str = "") -> None:
        item = self.items.get(page)
        if not item:
            return
        item.badge = text
        badge = item.rows[2]
        badge.configure(text=text, foreground=colour or C["ok"])

    def _hover(self, page: str, on: bool) -> None:
        item = self.items.get(page)
        if not item or page == self.value or not item.enabled:
            return
        colour = C["accent_soft"] if on else self.bg
        for w in item.rows[:2]:
            w.configure(background=colour)

    def _paint(self) -> None:
        for page, item in self.items.items():
            selected = (page == self.value)
            if not item.enabled:
                row_bg, fg = self.bg, C["faint"]
            elif selected:
                row_bg, fg = C["accent"], "#ffffff"
            else:
                row_bg, fg = self.bg, C["text"]
            item.rows[0].configure(background=row_bg)
            item.rows[1].configure(background=row_bg, foreground=fg,
                                   font=FONT_NAV_BOLD if selected else FONT_NAV)
            item.rows[2].configure(background=row_bg)
            if not selected:
                item.rows[2].configure(foreground=C["ok"])


# --------------------------------------------------------------------------- #
# 日志 / 文本视图
# --------------------------------------------------------------------------- #

class LogView(ttk.Frame):
    MAX_LINES = 6000

    def __init__(self, parent, height: int = 10, wrap: str = "char"):
        super().__init__(parent, style="Card.TFrame")
        self.text = tk.Text(self, height=height, wrap=wrap, undo=False,
                            background=C["log_bg"], foreground=C["text"],
                            font=T.FONT_MONO, relief="flat", borderwidth=0,
                            insertbackground=C["text"], padx=8, pady=6,
                            selectbackground=C["accent_soft"],
                            selectforeground=C["text"])
        self.vbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self.text.yview)
        self.text.configure(yscrollcommand=self._on_scroll_set)
        self.vbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.autoscroll = tk.BooleanVar(value=True)
        self.text.tag_configure("err", foreground=C["err"])
        self.text.tag_configure("warn", foreground=C["warn"])
        self.text.tag_configure("sys", foreground=C["accent"])
        self.text.tag_configure("ok", foreground=C["ok"])
        self.text.tag_configure("muted", foreground=C["muted"])
        self.text.tag_configure("bold",
                                font=(T.FONT_MONO[0], T.FONT_MONO[1], "bold"))
        self.text.configure(state="disabled")

    def _on_scroll_set(self, first, last) -> None:
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.vbar.pack_forget()
        else:
            self.vbar.pack(side="right", fill="y")
        self.vbar.set(first, last)

    def append(self, line: str, tag: str = "") -> None:
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n", (tag,) if tag else ())
        self._trim()
        self.text.configure(state="disabled")
        if self.autoscroll.get():
            self.text.see("end")

    def append_many(self, lines: Sequence[str]) -> None:
        if not lines:
            return
        self.text.configure(state="normal")
        for line, tag in lines:
            self.text.insert("end", line + "\n", (tag,) if tag else ())
        self._trim()
        self.text.configure(state="disabled")
        if self.autoscroll.get():
            self.text.see("end")

    def _trim(self) -> None:
        try:
            n = int(self.text.index("end-1c").split(".")[0])
        except (ValueError, IndexError):
            return
        if n > self.MAX_LINES:
            self.text.delete("1.0", "%d.0" % (n - self.MAX_LINES + 1))

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def get_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def set_text(self, value: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        self.text.configure(state="disabled")


def classify(line: str) -> str:
    """粗略给日志行上色。"""
    low = line.lower()
    if any(k in low for k in ("error", "failed", "assert", "abort", "exception",
                              "cuda_error", "out of memory", "oom")):
        return "err"
    if "warn" in low:
        return "warn"
    if "tokens per second" in low or "eval time" in low:
        return "ok"
    if line.startswith(("load_tensors", "llama_model_loader", "print_info",
                        "main:", "llama_context", "llama_kv_cache")):
        return "muted"
    return ""
