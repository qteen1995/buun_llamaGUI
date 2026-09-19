# -*- coding: utf-8 -*-
"""配色与 ttk 样式（浅色主题，适合 Windows 桌面）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict

C: Dict[str, str] = {
    "bg": "#f2f4f7",
    "panel": "#ffffff",
    "panel_alt": "#f7f9fb",
    "border": "#d9dee5",
    "border_strong": "#c3ccd6",
    "text": "#1f2328",
    "muted": "#6b7280",
    "faint": "#9aa2ad",
    "accent": "#2f6feb",
    "accent_hover": "#1f5ad4",
    "accent_soft": "#e8f0fe",
    "ok": "#12813c",
    "warn": "#b45309",
    "err": "#c62828",
    "log_bg": "#ffffff",
    "log_bg_alt": "#fbfcfd",
}

FONT_UI = ("Microsoft YaHei UI", 10)
FONT_UI_SMALL = ("Microsoft YaHei UI", 9)
FONT_UI_BOLD = ("Microsoft YaHei UI", 10, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 12, "bold")
FONT_NAV = ("Microsoft YaHei UI", 10)
FONT_NAV_BOLD = ("Microsoft YaHei UI", 10, "bold")
FONT_MONO = ("Consolas", 9)
FONT_MONO_SMALL = ("Consolas", 8)


def apply(root: tk.Misc) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(background=C["bg"])

    # ------------------------------------------------------------- 容器
    style.configure(".", background=C["bg"], foreground=C["text"],
                    font=FONT_UI, borderwidth=0, focuscolor=C["accent_soft"])
    style.configure("TFrame", background=C["bg"])
    style.configure("Card.TFrame", background=C["panel"])
    style.configure("Alt.TFrame", background=C["panel_alt"])
    style.configure("Bar.TFrame", background=C["panel"])
    style.configure("Sep.TFrame", background=C["border"])

    # ------------------------------------------------------------- 文字
    style.configure("TLabel", background=C["bg"], foreground=C["text"])
    style.configure("Card.TLabel", background=C["panel"], foreground=C["text"])
    style.configure("Alt.TLabel", background=C["panel_alt"],
                    foreground=C["text"])
    style.configure("Title.TLabel", background=C["panel"], foreground=C["text"],
                    font=FONT_TITLE)
    style.configure("Muted.TLabel", background=C["panel"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    style.configure("MutedCard.TLabel", background=C["panel"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    style.configure("MutedBg.TLabel", background=C["bg"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    style.configure("ErrCard.TLabel", background=C["panel"],
                    foreground=C["err"], font=FONT_UI_SMALL)
    style.configure("ErrCard.TCheckbutton", background=C["panel"],
                    foreground=C["err"], font=FONT_UI_SMALL)
    style.map("ErrCard.TCheckbutton", background=[("active", C["panel"])])
    style.configure("Bg.TCheckbutton", background=C["bg"],
                    foreground=C["text"], font=FONT_UI_SMALL)
    style.map("Bg.TCheckbutton", background=[("active", C["bg"])])
    style.configure("H2.TLabel", background=C["bg"], foreground=C["text"],
                    font=FONT_TITLE)
    style.configure("OkCard.TLabel", background=C["panel"],
                    foreground=C["ok"], font=FONT_UI_SMALL)
    style.configure("Alt.TLabel", background=C["panel_alt"],
                    foreground=C["text"], font=FONT_UI_SMALL)
    style.configure("ErrAlt.TLabel", background=C["panel_alt"],
                    foreground=C["err"], font=FONT_UI_SMALL)
    # 「已加载模型」列表那几个小标签用的（panel_alt 底的次要文字 / 状态色）
    style.configure("MutedAlt.TLabel", background=C["panel_alt"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    style.configure("OkAlt.TLabel", background=C["panel_alt"],
                    foreground=C["ok"], font=FONT_UI_SMALL)
    style.configure("WarnAlt.TLabel", background=C["panel_alt"],
                    foreground=C["warn"], font=FONT_UI_SMALL)


    style.configure("Field.TLabel", background=C["bg"], foreground=C["text"])
    style.configure("WarnBg.TLabel", background=C["bg"], foreground=C["warn"],
                    font=FONT_UI_SMALL)

    # ------------------------------------------------------------ 按钮
    style.configure("TButton", background="#eef1f5", foreground=C["text"],
                    bordercolor=C["border_strong"], lightcolor="#eef1f5",
                    darkcolor="#eef1f5", relief="flat", padding=(12, 6),
                    font=FONT_UI_SMALL)
    style.map("TButton",
              background=[("pressed", "#dde3ea"), ("active", "#e3e8ee"),
                          ("disabled", "#f4f6f8")],
              foreground=[("disabled", C["faint"])],
              bordercolor=[("active", C["accent"])])

    style.configure("Accent.TButton", background=C["accent"],
                    foreground="#ffffff", bordercolor=C["accent"],
                    lightcolor=C["accent"], darkcolor=C["accent"],
                    padding=(16, 7), font=FONT_UI)
    style.map("Accent.TButton",
              background=[("pressed", C["accent_hover"]),
                          ("active", C["accent_hover"]),
                          ("disabled", "#a9c0f0")],
              foreground=[("disabled", "#f0f4fb")],
              bordercolor=[("disabled", "#a9c0f0")])

    style.configure("Danger.TButton", background="#fdecec",
                    foreground=C["err"], bordercolor="#f3c2c2",
                    lightcolor="#fdecec", darkcolor="#fdecec",
                    padding=(16, 7), font=FONT_UI)
    style.map("Danger.TButton",
              background=[("active", "#f9dcdc"), ("disabled", "#f6f7f9")],
              foreground=[("disabled", C["faint"])],
              bordercolor=[("disabled", C["border"])])

    style.configure("Mini.TButton", padding=(8, 3), font=FONT_UI_SMALL)
    style.configure("Tiny.TButton", padding=(6, 2), font=FONT_UI_SMALL)
    style.configure("Mini.TMenubutton", padding=(8, 3), font=FONT_UI_SMALL,
                    background="#eef1f5", foreground=C["text"],
                    bordercolor=C["border_strong"], lightcolor="#eef1f5",
                    darkcolor="#eef1f5", arrowcolor=C["muted"], relief="flat")
    style.map("Mini.TMenubutton",
              background=[("active", "#e3e8ee"), ("disabled", "#f4f6f8")],
              foreground=[("disabled", C["faint"])])


    # ---------------------------------------------------------- 选择框
    style.configure("TCheckbutton", background=C["bg"], foreground=C["text"],
                    focuscolor=C["accent_soft"], font=FONT_UI_SMALL)
    style.map("TCheckbutton",
              background=[("active", C["bg"])],
              foreground=[("disabled", C["faint"])])
    style.configure("Card.TCheckbutton", background=C["panel"],
                    foreground=C["text"], font=FONT_UI_SMALL)
    style.map("Card.TCheckbutton", background=[("active", C["panel"])])
    style.configure("Field.TCheckbutton", background=C["bg"],
                    foreground=C["text"], font=FONT_UI_SMALL)
    style.map("Field.TCheckbutton", background=[("active", C["bg"])])

    style.configure("TRadiobutton", background=C["bg"], foreground=C["text"])

    # ---------------------------------------------------------- 输入框
    style.configure("TEntry", fieldbackground=C["panel"],
                    background=C["panel"], bordercolor=C["border_strong"],
                    lightcolor=C["border"], darkcolor=C["border"],
                    insertcolor=C["text"], padding=4, relief="flat")
    style.map("TEntry",
              fieldbackground=[("disabled", "#f0f2f5"),
                               ("readonly", C["panel_alt"])],
              foreground=[("disabled", C["faint"])],
              bordercolor=[("focus", C["accent"])])

    style.configure("TCombobox", fieldbackground=C["panel"],
                    background=C["panel"], bordercolor=C["border_strong"],
                    lightcolor=C["border"], darkcolor=C["border"],
                    arrowcolor=C["muted"], padding=3, relief="flat")
    style.map("TCombobox",
              fieldbackground=[("disabled", "#f0f2f5"),
                               ("readonly", C["panel"])],
              background=[("active", C["panel"])],
              foreground=[("disabled", C["faint"])],
              arrowcolor=[("disabled", C["faint"])])

    # ---------------------------------------------------------- 标签页
    style.configure("TNotebook", background=C["bg"], bordercolor=C["border"],
                    tabmargins=(2, 4, 2, 0), borderwidth=0)
    style.configure("TNotebook.Tab", background="#e6eaef",
                    foreground=C["muted"], padding=(14, 6),
                    font=FONT_UI_SMALL, bordercolor=C["border"])
    style.map("TNotebook.Tab",
              background=[("selected", C["panel"]), ("active", "#eef2f7")],
              foreground=[("selected", C["accent"])],
              expand=[("selected", (0, 0, 0, 0))])

    # ------------------------------------------------------- LabelFrame
    style.configure("TLabelframe", background=C["panel"],
                    bordercolor=C["border"], borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=C["panel"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    style.configure("Bg.TLabelframe", background=C["bg"],
                    bordercolor=C["border"], borderwidth=1)
    style.configure("Bg.TLabelframe.Label", background=C["bg"],
                    foreground=C["muted"], font=FONT_UI_SMALL)

    # -------------------------------------------------------- PanedWindow
    style.configure("TPanedwindow", background=C["border"])
    style.configure("Sash", sashthickness=6, gripcount=0,
                    background=C["border"])

    # ---------------------------------------------------------- 滚动条
    style.configure("Vertical.TScrollbar", background="#dfe4ea",
                    troughcolor=C["panel_alt"], bordercolor=C["panel_alt"],
                    arrowcolor=C["muted"], relief="flat", width=11)
    style.map("Vertical.TScrollbar", background=[("active", "#cbd3dc")])
    style.configure("Horizontal.TScrollbar", background="#dfe4ea",
                    troughcolor=C["panel_alt"], bordercolor=C["panel_alt"],
                    arrowcolor=C["muted"], relief="flat")
    style.map("Horizontal.TScrollbar", background=[("active", "#cbd3dc")])

    style.configure("TSeparator", background=C["border"])

    # ---------------------------------------------------------- 表格
    style.configure("Treeview", background=C["panel"],
                    fieldbackground=C["panel"], foreground=C["text"],
                    rowheight=23, font=FONT_UI_SMALL, relief="flat",
                    bordercolor=C["border"], lightcolor=C["border"],
                    darkcolor=C["border"])
    style.map("Treeview",
              background=[("selected", C["accent_soft"])],
              foreground=[("selected", C["text"])])
    style.configure("Treeview.Heading", background="#e9edf2",
                    foreground=C["muted"], font=FONT_UI_SMALL,
                    relief="flat", padding=(6, 5), bordercolor=C["border"])
    style.map("Treeview.Heading",
              background=[("active", "#dbe3ec"), ("pressed", "#d0dae6")],
              foreground=[("active", C["accent"])])

    style.configure("Status.TLabel", background=C["bg"],
                    foreground=C["muted"], font=FONT_UI_SMALL)
    return style
