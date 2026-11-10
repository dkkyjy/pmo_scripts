"""U2/#12：把"20 元组下标只允许出现在 direction_result_from_tuple 里"变成会失败的断言。

局限（有意为之）：正则区分不了元组下标与普通 list/numpy 下标，所以只盯已知的
元组变量名（raw / _dir / r）。这是能守住回归的最小面；新的元组变量名出现时，
把它加进 _GUARDED 即可。
"""

import inspect
import re

import pytest

import reconstruction_candidate_fluence_error_fixed as m

_GUARDED = ("raw", "_dir", "_dir_result", "r", "out")
_ONLY_ALLOWED_IN = "direction_result_from_tuple"


def _strip_function(src, func_name):
    """把目标函数的整段源码切掉，返回其余部分（按 'def ' 缩进到下一个同级 def）。"""
    lines = src.splitlines()
    out, skipping, indent = [], False, 0
    for ln in lines:
        if re.match(rf"^\s*def {re.escape(func_name)}\b", ln):
            skipping, indent = True, len(ln) - len(ln.lstrip())
            continue
        if skipping:
            cur = len(ln) - len(ln.lstrip()) if ln.strip() else indent
            if ln.strip() and cur <= indent:
                skipping = False
        if not skipping:
            out.append(ln)
    return "\n".join(out)


def _index_hits(text):
    pat = re.compile(r"\b(" + "|".join(map(re.escape, _GUARDED)) + r")\[\s*\d+\s*\]")
    return [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines()) if pat.search(ln)]


def test_tuple_indexing_confined_to_one_function():
    rest = _strip_function(inspect.getsource(m), _ONLY_ALLOWED_IN)
    hits = _index_hits(rest)
    assert not hits, f"20 元组下标出现在 {_ONLY_ALLOWED_IN} 之外: {hits}"


def test_the_guard_actually_catches_a_violation():
    """反证：给守卫喂一段含 r[7] 的伪造源码，它必须命中（否则本测试恒真）。"""
    fake = "def f():\n    a = r[7]\n    b = raw[3]\n    c = out[9]\n"
    hits = _index_hits(fake)
    assert len(hits) == 3, hits
    assert _index_hits("def f():\n    return raw\n") == []


def test_strip_function_removes_only_the_target():
    """反证 _strip_function：给定函数被整段切掉，其余函数保留。"""
    src = "def a():\n    return raw[1]\n\ndef direction_result_from_tuple(t):\n    return t[0]\n\ndef b():\n    return 0\n"
    rest = _strip_function(src, "direction_result_from_tuple")
    assert "return t[0]" not in rest
    assert "return raw[1]" in rest and "def b():" in rest, rest
