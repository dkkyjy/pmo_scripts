"""把 reconstruction_candidate/ 放进 sys.path。

该目录下的模块用顶层名互相 import（`from vad2voc_fixed import vad2voc`），
因此测试必须让这个目录本身可被 import，而不是把它当包导入。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.dirname(_HERE)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
