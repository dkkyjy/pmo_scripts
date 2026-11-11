"""把 reconstruction_candidate/ 与仓库根目录放进 sys.path。

两者都需要：前者的模块用顶层名互相 import（`from vad2voc_fixed import vad2voc`），
后者提供 `logger_config`（驱动脚本 `run_parallel_reconstruction_fixed.py` 依赖它）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_MODULE_DIR)

for p in (_MODULE_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
