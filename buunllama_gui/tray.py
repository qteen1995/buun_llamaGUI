# -*- coding: utf-8 -*-
"""Windows 系统托盘 / 开机自启 / 单实例 —— 纯 ctypes，不引入任何第三方依赖。

守住「零依赖」约束（打包脚本里 PIL / pystray 都是明确排除的）：

- 托盘图标：Shell_NotifyIcon + 一个隐藏的 owner 窗口（tk 的 Toplevel），
  用 SetWindowLongPtr 接管它的 WndProc 来收托盘鼠标消息和「恢复」消息。
- 开机自启：写注册表 HKCU\\...\\Run（自启时附带 --minimized）。
- 单实例：CreateMutex；重复打开时给已运行实例发「恢复」消息并退出。
"""
from __future__ import annotations

import ctypes
import os
import sys

from ctypes import wintypes

# 个别 ctypes 版本里缺这个名字，补一个等价的（x64 上 LRESULT 是 64 位有符号）
if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = ctypes.c_longlong           # type: ignore[attr-defined]
LRESULT = wintypes.LRESULT

user32 = ctypes.windll.user32          # type: ignore[attr-defined]
kernel32 = ctypes.windll.kernel32      # type: ignore[attr-defined]
shell32 = ctypes.windll.shell32        # type: ignore[attr-defined]
advapi32 = ctypes.windll.advapi32      # type: ignore[attr-defined]

# ---- 常量 ---------------------------------------------------------------- #
WM_USER = 0x400
MYWM_TRAY = WM_USER + 1                # 托盘图标通知
MYWM_RESTORE = WM_USER + 2            # 第二个实例要求恢复窗口
WM_RBUTTONUP = 0x0205
WM_LBUTTONUP = 0x0202
WM_CONTEXTMENU = 0x007B
SW_RESTORE = 9
GWL_WNDPROC = -4
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x0010
LR_DEFAULTSIZE = 0x0040
NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIF_MESSAGE = 0x01
NIF_ICON = 0x02
NIF_TIP = 0x04

# 注册表
HKEY_CURRENT_USER = 0x80000001
KEY_READ = 0x20019
KEY_WRITE = 0x20006
REG_SZ = 1
ERROR_SUCCESS = 0

_WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT,
    wintypes.WPARAM, wintypes.LPARAM)


class _NOTIFYICONDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", wintypes.BYTE * 16),
        ("hBalloonIcon", wintypes.HICON),
    ]


# ---- ctypes 原型声明（不声明的话 HKEY/LPCWSTR 会被当成 int，注册表调用会错）#
def _protos():
    P = ctypes.POINTER
    advapi32.RegOpenKeyExW.argtypes = [
        wintypes.HKEY, wintypes.LPCWSTR, wintypes.DWORD,
        wintypes.DWORD, P(wintypes.HKEY)]
    advapi32.RegOpenKeyExW.restype = wintypes.LONG
    advapi32.RegQueryValueExW.argtypes = [
        wintypes.HKEY, wintypes.LPCWSTR, wintypes.LPDWORD,
        P(wintypes.DWORD), wintypes.LPWSTR, P(wintypes.DWORD)]
    advapi32.RegQueryValueExW.restype = wintypes.LONG
    advapi32.RegSetValueExW.argtypes = [
        wintypes.HKEY, wintypes.LPCWSTR, wintypes.DWORD,
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.DWORD]
    advapi32.RegSetValueExW.restype = wintypes.LONG
    advapi32.RegDeleteValueW.argtypes = [wintypes.HKEY, wintypes.LPCWSTR]
    advapi32.RegDeleteValueW.restype = wintypes.LONG
    advapi32.RegCloseKey.argtypes = [wintypes.HKEY]
    advapi32.RegCloseKey.restype = wintypes.LONG

    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.GetLastError.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterWindowMessageW.restype = wintypes.UINT
    user32.LoadImageW.argtypes = [
        wintypes.HANDLE, wintypes.LPCWSTR, wintypes.UINT,
        ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, _WNDPROC]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.CallWindowProcW.argtypes = [
        _WNDPROC, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = LRESULT
    user32.EnumWindows.argtypes = [_WNDPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.SendMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = wintypes.LRESULT
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL

    shell32.Shell_NotifyIconW.argtypes = [
        wintypes.DWORD, ctypes.POINTER(_NOTIFYICONDATA)]
    shell32.Shell_NotifyIconW.restype = wintypes.BOOL


_protos()


_REG_PATH = "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
_REG_VALUE = "buun_llama_gui"
_OWNER_TITLE = "buun_tray_owner"


def _icon_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", here)
        return os.path.join(base, "buunllama_gui", "app.ico")
    return os.path.join(here, "app.ico")


def load_hicon() -> wintypes.HICON:
    path = _icon_path()
    if not os.path.isfile(path):
        return wintypes.HICON(0)
    h = user32.LoadImageW(None, path, IMAGE_ICON, 0, 0,
                          LR_LOADFROMFILE | LR_DEFAULTSIZE)
    return wintypes.HICON(h) if h else wintypes.HICON(0)


# --------------------------------------------------------------------------- #
# 单实例
# --------------------------------------------------------------------------- #
def acquire_mutex(name: str):
    """拿到进程级互斥体。返回非空句柄=本进程是首个实例；返回 0=已有实例。"""
    h = kernel32.CreateMutexW(None, 1, name)
    if not h:
        return 0
    if kernel32.GetLastError() == 183:           # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(h)
        return 0
    return h


def find_window_by_title(title: str):
    result = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if buf.value == title:
            result.append(hwnd)
            return False
        return True

    user32.EnumWindows(cb, 0)
    return result[0] if result else None


def send_restore(hwnd) -> None:
    if hwnd:
        user32.SendMessageW(hwnd, MYWM_RESTORE, 0, 0)


class TrayIcon:
    """一行代码就能用的系统托盘。构造时建一个隐藏 owner 窗口并接管它的消息。"""

    def __init__(self, root, tooltip: str, on_restore, on_quit, on_click=None):
        self.root = root
        self.tooltip = (tooltip or "buun llama 启动器")[:127]
        self.on_restore = on_restore
        self.on_quit = on_quit
        self.on_click = on_click or on_restore
        self.hicon = load_hicon()
        self._added = False
        self._owner = None
        self._oldproc = None
        self._wndproc = None
        self._taskbar_msg = user32.RegisterWindowMessageW("TaskbarCreated")
        self._menu = None
        self._build_owner()

    # ---- owner 窗口（隐藏、无边框，专收消息）----------------------------- #
    def _build_owner(self):
        import tkinter as tk
        owner = tk.Toplevel(self.root)
        owner.title(_OWNER_TITLE)
        owner.overrideredirect(True)
        owner.geometry("1x1+4000+4000")
        owner.withdraw()                      # 不显示，但 hwnd 仍在、能收消息
        owner.update_idletasks()
        self._owner = owner
        hwnd = owner.winfo_id()
        self._wndproc = _WNDPROC(self._proc)
        self._oldproc = user32.SetWindowLongPtrW(
            hwnd, GWL_WNDPROC, self._wndproc)

    def _proc(self, hwnd, msg, wp, lp):
        if msg == MYWM_RESTORE:
            try:
                self.on_restore()
            except Exception:
                pass
            return 0
        if msg == self._taskbar_msg:          # 资源管理器重启 → 重新加图标
            if self._added:
                self._add()
            return 0
        if msg == MYWM_TRAY and wp == 1:      # uID
            evt = lp & 0xFFFF
            if evt in (WM_RBUTTONUP, WM_CONTEXTMENU):
                self._show_menu()
            elif evt == WM_LBUTTONUP:
                try:
                    self.on_click()
                except Exception:
                    pass
            return 0
        return user32.CallWindowProcW(
            ctypes.cast(self._oldproc, _WNDPROC), hwnd, msg, wp, lp)

    # ---- 托盘图标操作 ---------------------------------------------------- #
    def _data(self):
        d = _NOTIFYICONDATA()
        d.cbSize = ctypes.sizeof(_NOTIFYICONDATA)
        d.hWnd = self._owner.winfo_id()
        d.uID = 1
        d.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        d.uCallbackMessage = MYWM_TRAY
        d.hIcon = self.hicon
        d.szTip = self.tooltip
        return d

    def _add(self):
        d = self._data()
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(d))
        self._added = True

    def show(self):
        if self._added:
            d = self._data()
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(d))
        else:
            self._add()

    def hide(self):
        if not self._added:
            return
        d = self._data()
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(d))
        self._added = False

    def set_tooltip(self, tip: str):
        self.tooltip = (tip or self.tooltip)[:127]
        if self._added:
            self.show()

    # ---- 右键菜单 -------------------------------------------------------- #
    def _show_menu(self):
        import tkinter as tk
        if self._menu is None:
            self._menu = tk.Menu(self._owner, tearoff=0)
            self._menu.add_command(label="显示窗口", command=self.on_restore)
            self._menu.add_separator()
            self._menu.add_command(label="退出", command=self.on_quit)
        try:
            self._menu.tk_popup(self._cursor_x(), self._cursor_y())
        finally:
            try:
                self._menu.grab_release()
            except tk.TclError:
                pass

    @staticmethod
    def _cursor_x():
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x

    @staticmethod
    def _cursor_y():
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.y


# --------------------------------------------------------------------------- #
# 开机自启（注册表 Run；测试模式走文件，不碰注册表）
# --------------------------------------------------------------------------- #
def _test_store_path() -> str:
    import tempfile
    return os.path.join(tempfile.gettempdir(), "buun_autostart_test.json")


def _launch_command() -> str:
    """当前这份程序在自启时要运行的命令（带 --minimized）。"""
    if getattr(sys, "frozen", False):
        exe = os.path.abspath(sys.executable)
        return '"%s" --minimized' % exe
    launcher = os.path.abspath(sys.argv[0])
    return '"%s" --minimized' % launcher


def set_autostart(on: bool) -> bool:
    if os.environ.get("BUUN_SELFTEST"):
        p = _test_store_path()
        if on:
            import json
            json.dump({"cmd": _launch_command()}, open(p, "w", encoding="utf-8"))
        elif os.path.isfile(p):
            os.remove(p)
        return True
    return _reg_set(on)


def is_autostart() -> bool:
    if os.environ.get("BUUN_SELFTEST"):
        return os.path.isfile(_test_store_path())
    return _reg_get() is not None


def _reg_open(key, sub, access):
    hk = wintypes.HKEY()
    res = advapi32.RegOpenKeyExW(key, sub, 0, access, ctypes.byref(hk))
    if res != ERROR_SUCCESS:
        return None
    return hk


def _reg_get():
    hk = _reg_open(HKEY_CURRENT_USER, _REG_PATH, KEY_READ)
    if hk is None:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024 * 2)
        typ = wintypes.DWORD(0)
        res = advapi32.RegQueryValueExW(
            hk, _REG_VALUE, None, ctypes.byref(typ), buf, ctypes.byref(size))
        if res != ERROR_SUCCESS:
            return None
        return buf.value
    finally:
        advapi32.RegCloseKey(hk)


def _reg_set(on: bool) -> bool:
    hk = _reg_open(HKEY_CURRENT_USER, _REG_PATH, KEY_WRITE)
    if hk is None:
        return False
    try:
        if on:
            cmd = _launch_command()
            buf = ctypes.create_unicode_buffer(cmd)
            res = advapi32.RegSetValueExW(
                hk, _REG_VALUE, 0, REG_SZ, buf,
                (len(cmd) + 1) * ctypes.sizeof(wintypes.WCHAR))
        else:
            res = advapi32.RegDeleteValueW(hk, _REG_VALUE)
        return res == ERROR_SUCCESS
    finally:
        advapi32.RegCloseKey(hk)


# --------------------------------------------------------------------------- #
# 单实例：被调用方在 main() 里用
# --------------------------------------------------------------------------- #
def single_instance_guard(root_title: str) -> bool:
    """返回 True=本进程是首个实例可继续；False=已存在实例（已发恢复，调用方应退出）。"""
    h = acquire_mutex("buun_llama_gui_single_instance")
    if h:
        return True
    hwnd = find_window_by_title(_OWNER_TITLE)
    send_restore(hwnd)
    if hwnd is None:                       # 兜底：按主窗口标题找
        hwnd = find_window_by_title(root_title)
        if hwnd:
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
    return False
