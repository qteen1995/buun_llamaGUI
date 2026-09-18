# -*- coding: utf-8 -*-
"""把界面打包成**便携式 exe**（单文件，含 Python 与 tkinter，无需安装任何东西）。

用法（在项目根目录）：
    E:\\Python\\Python311\\python.exe build_portable.py

产物在 ``dist/``：

    dist/buun_llama_gui.exe          ← 双击即用（无控制台窗口）
    dist/buun_llama_gui_debug.exe    ← 带控制台，用来跑 --selftest / 看报错

**便携**：配置、日志都写在 exe **同级目录**的 ``config/`` 里，不碰注册表、
不写 %APPDATA%。把这两个 exe 拷到 U 盘或任何目录都能直接用。

打包工具（PyInstaller）装在独立环境里，不污染本项目的「零依赖」：
    E:\\Python\\Python311\\python.exe -m venv <某个目录>
    <venv>\\Scripts\\python.exe -m pip install pyinstaller
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(HERE, "buun_launcher.pyw")
DIST = os.path.join(HERE, "dist")
BUILD = os.path.join(HERE, "build")
SPECS = os.path.join(BUILD, "specs")

# 打包环境（PyInstaller 装在这里，不进项目）
VENV = r"C:\Users\27128\.workbuddy\binaries\python\envs\pyi311"
PYI_PY = os.path.join(VENV, "Scripts", "python.exe")


def _pyinstaller_python() -> str:
    if os.path.isfile(PYI_PY):
        return PYI_PY
    return sys.executable


def build(name: str, windowed: bool) -> int:
    args = [
        _pyinstaller_python(), "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile",
        "--name", name,
        "--paths", HERE,
        "--workpath", BUILD,
        "--specpath", SPECS,
        "--distpath", DIST,
        # 用不到的东西一律排除，包小一点、启动快一点
        "--exclude-module", "numpy", "--exclude-module", "PIL",
        "--exclude-module", "matplotlib", "--exclude-module", "unittest",
        "--exclude-module", "pydoc", "--exclude-module", "doctest",
        "--exclude-module", "test", "--exclude-module", "lib2to3",
        "--exclude-module", "distutils", "--exclude-module", "xmlrpc",
        "--exclude-module", "sqlite3", "--exclude-module", "curses",
        "--exclude-module", "multiprocessing",
        # 托盘/窗口图标：跟随 exe 一起打包，运行时从 sys._MEIPASS 取
        "--add-data",
        os.path.join(HERE, "buunllama_gui", "app.ico") + ";" +
        "buunllama_gui",
        "--console" if not windowed else "--noconsole",
        ENTRY,
    ]
    print("→ 打包 %s（%s）" % (name, "无控制台" if windowed else "带控制台"))
    return subprocess.call(args, cwd=HERE)


def main() -> int:
    if not os.path.isfile(ENTRY):
        print("找不到入口：%s" % ENTRY)
        return 1
    py = _pyinstaller_python()
    try:
        subprocess.check_output([py, "-c", "import PyInstaller"],
                                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("这个解释器里没有 PyInstaller：%s\n"
              "先装一下：\n  %s -m pip install pyinstaller" % (py, py))
        return 1

    os.makedirs(DIST, exist_ok=True)
    rc = build("buun_llama_gui", windowed=True)
    if rc == 0:
        rc = build("buun_llama_gui_debug", windowed=False)

    # 把说明和启动脚本一起放进 dist，拷走整个 dist 就能用
    for src, dst in (("便携版说明.txt", "便携版说明.txt"),):
        p = os.path.join(HERE, src)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(DIST, dst))

    print()
    if rc == 0:
        print("完成，产物在：%s" % DIST)
        for f in sorted(os.listdir(DIST)):
            full = os.path.join(DIST, f)
            if os.path.isfile(full):
                print("   %-28s %8.1f MB" % (f, os.path.getsize(full) / 1048576))
        print()
        print("配置会写在 exe 同级的 config\\：app.json（软件级）+ "
              "models\\<模型名>.json（每模型一份）+ cache\\scan.json。")
    else:
        print("打包失败，退出码 %s" % rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
