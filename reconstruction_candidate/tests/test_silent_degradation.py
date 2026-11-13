"""U3：两处静默降级（`dw_err` 哨兵 / `du_nanoseconds` 置零）的契约测试。"""

import numpy as np
import pytest

from reconstruction_direction import (
    DirectionTuple,
    _adf_minuit,
    ADF_fit_analytic_fixdw,
)


def _synth(n=12, seed=7):
    rng = np.random.default_rng(seed)
    r = rng.uniform(50, 500, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x, y, z = r * np.cos(a), r * np.sin(a), np.zeros(n)
    dist = np.sqrt((0 - x) ** 2 + (0 - y) ** 2 + (7000 - z) ** 2)
    axis = np.array([np.sin(np.deg2rad(25)), 0.0, np.cos(np.deg2rad(25))])
    omega = np.arccos(np.clip(((-x) * axis[0] + (-y) * axis[1] + (7000 - z) * axis[2]) / dist, -1, 1))
    p = 1.0 / (1 + 4 * (((np.tan(omega) / np.tan(1.2)) ** 2 - 1) / 1.0) ** 2)
    return x, y, z, p


def test_dw_err_is_nan_when_dw_is_fixed():
    """固定 dw 时误差不可得 -> NaN（旧实现返回 0.0，会被读成"误差为零"）。"""
    x, y, z, p = _synth()
    r = _adf_minuit(x, y, z, p, 0.0, 0.0, 7000.0, 25.0, 0.0, 0.0,
                    dw0=1.2, ant_p_sigma=None, amp="profiled", fix_dw=True)
    assert np.isnan(r[7]), f"dw_err 应为 NaN，实际 {r[7]!r}"


def test_dw_err_is_nan_for_profiled_free_dw():
    """profiled + 自由 dw 时误差同样不可得（本来就不算误差）。"""
    x, y, z, p = _synth()
    r = _adf_minuit(x, y, z, p, 0.0, 0.0, 7000.0, 25.0, 0.0, 0.0,
                    dw0=1.2, ant_p_sigma=None, amp="profiled", fix_dw=False)
    assert np.isnan(r[7])


def test_adf_fit_analytic_fixdw_still_returns_five():
    """薄壳契约没变：仍返回 (theta, phi, dw, A, fval)。"""
    x, y, z, p = _synth()
    r = ADF_fit_analytic_fixdw(x, y, z, p, 0.0, 0.0, 7000.0, 25.0, 0.0, 1.2, 0.0)
    assert len(r) == 5
    assert all(np.isfinite(v) for v in r)


def test_direction_tuple_keeps_len_20_and_carries_dw_fixed():
    """DirectionTuple 仍按 20 元组工作，并额外带 dw_fixed。"""
    t = DirectionTuple(tuple(range(20)), dw_fixed=True)
    assert len(t) == 20
    assert t[17] == 17
    a, b = t[0], t[19]
    assert (a, b) == (0, 19)
    assert t.dw_fixed is True
    assert DirectionTuple(tuple(range(20))).dw_fixed is False


def test_coreas_truth_trees_use_coreas_coord_table(tmp_path, monkeypatch):
    """tefield/tvoltage 与 tadc 一样必须用 CoREAS 坐标表，不能落到生产表。

    R1 修好 tree 探测后这批文件才可达；若仍走 `else` 分支就会用 GP65 生产表，
    而 CoREAS 的 DU id 与生产表不重合 —— 位置静默错。
    """
    import reconstruction_candidate_fluence_error_fixed as m

    coords = tmp_path / "du_coords.txt"
    coords.write_text("1, 0, 0, 0\n")
    root = tmp_path / "efield_0-99_L0_0000.root"
    root.write_text("placeholder")

    class _FakeRoot:
        def keys(self):
            return ["tefield;7"]

        def close(self):
            pass

    monkeypatch.setattr(m.uproot, "open", lambda *a, **kw: _FakeRoot())
    monkeypatch.setattr(m, "find_event_tree_key", lambda keys, name: ("tefield;7", "tefield"))
    kind, path = m._load_du_position_for_input(str(root), None)
    assert kind == "tefield"
    assert path == str(coords), f"tefield 应使用 CoREAS 坐标表，实际 {path}"


def test_all_coreas_kinds_use_coreas_coord_table(tmp_path, monkeypatch):
    """三种 CoREAS kind 都要走 CoREAS 坐标表；trigger 仍走生产表（U1/#8）。"""
    import reconstruction_candidate_fluence_error_fixed as m

    coords = tmp_path / "du_coords.txt"
    coords.write_text("1, 0, 0, 0\n")
    root = tmp_path / "efield_0-99_L0_0000.root"
    root.write_text("placeholder")

    class _FakeRoot:
        def keys(self):
            return ["x;1"]

        def close(self):
            pass

    monkeypatch.setattr(m.uproot, "open", lambda *a, **kw: _FakeRoot())
    for kind in ("tadc", "tefield", "tvoltage"):
        monkeypatch.setattr(m, "find_event_tree_key", lambda keys, name, k=kind: ("x;1", k))
        got_kind, path = m._load_du_position_for_input(str(root), None)
        assert got_kind == kind
        assert path == str(coords), f"{kind} 应使用 CoREAS 坐标表，实际 {path}"

    monkeypatch.setattr(m, "find_event_tree_key", lambda keys, name: ("x;1", "teventadc"))
    _, path = m._load_du_position_for_input(str(root), None)
    assert path == m.DU_POS_TABLE, "trigger 仍应使用生产表"


def test_coreas_truth_tree_set_has_one_source():
    """"哪些 kind 是 CoREAS 真值树"只能有一处集合字面量（U1/#8）。"""
    import inspect
    import re
    import reconstruction_candidate_fluence_error_fixed as m

    src = inspect.getsource(m)
    assert m._COREAS_TRUTH_TREES == ('tefield', 'tvoltage')
    hits = [ln for ln in src.splitlines()
            if re.search(r"[\(\[]\s*['\"]tefield['\"]", ln) and 'tvoltage' in ln]
    assert len(hits) == 1, f"真值树集合字面量出现了 {len(hits)} 处: {hits}"
