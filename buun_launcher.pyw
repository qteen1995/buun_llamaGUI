#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""buun-llama-cpp 图形化启动器 —— 双击运行入口。

如果 .pyw 已经关联到 pythonw.exe，直接双击本文件即可。
否则请运行同目录下的 start_gui.bat。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buunllama_gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
