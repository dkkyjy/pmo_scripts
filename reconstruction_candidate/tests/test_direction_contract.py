"""方向重建链的契约测试。

运行：`pytest reconstruction_candidate/tests -q`

conftest.py 会把 `reconstruction_candidate/` 放进 sys.path，因为该目录下的模块
用顶层名互相 import（`from vad2voc_fixed import vad2voc`），不是真正的包。
"""

import pytest


def test_recons_direction_struct_exists():
    """U4 才引入的具名入口；U1 阶段此处应为红（import 失败）。"""
    from reconstruction_candidate_fluence_error_fixed import recons_direction_struct

    assert callable(recons_direction_struct)
