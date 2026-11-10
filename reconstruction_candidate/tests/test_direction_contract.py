"""U4：具名结果（`DirectionResult`）与 20 元组位置契约的一致性。"""

import numpy as np
import pytest

from reconstruction_candidate_fluence_error_fixed import (
    DirectionResult,
    _TUPLE_FIELDS,
    direction_result_from_tuple,
    recons_direction_struct,
    recons_direction_swf_adf,
    reconstruct_direction_only,
)


def _synth_event(n=40, seed=11):
    """造一个够 SWF/ADF 跑起来的事例：球面波前 + 切伦科夫环幅度。"""
    rng = np.random.default_rng(seed)
    r = rng.uniform(50, 600, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x, y, z = r * np.cos(a), r * np.sin(a), np.zeros(n)
    xs, ys, zs = 120.0, -80.0, 6000.0
    dx, dy, dz = xs - x, ys - y, zs - z
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    th, ph = np.deg2rad(30.0), np.deg2rad(120.0)
    axis = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    omega = np.arccos(np.clip((dx * axis[0] + dy * axis[1] + dz * axis[2]) / dist, -1, 1))
    amp = 1.0 / (1 + 4 * (((np.tan(omega) / np.tan(1.2)) ** 2 - 1) / 1.2) ** 2)
    p = amp * (1.0 + rng.normal(0, 0.02, n))
    t = dist / 0.299792458 + rng.normal(0, 0.5, n)   # ns
    return x, y, z, t, p


def test_tuple_fields_length_is_20():
    assert len(_TUPLE_FIELDS) == 20


def test_field_map_matches_raw_indices():
    raw = tuple(float(i) for i in range(20))
    res = direction_result_from_tuple(raw)
    assert isinstance(res, DirectionResult)
    assert res.raw == raw
    by = dict(zip(_TUPLE_FIELDS, raw))
    assert res.zenith == by["zenith"] == raw[7]
    assert res.azimuth == raw[8]
    assert res.dw == raw[9]
    assert res.A == raw[10]
    assert res.wc == raw[12]
    assert res.x_max == raw[3] and res.y_max == raw[4] and res.z_max == raw[5]
    assert res.chi2_sph == raw[6] and res.fmin == raw[11]
    assert res.dw_err == raw[15] and res.A_err == raw[16]
    assert res.x_max_err == raw[17] and res.z_max_err == raw[19]
    assert res.dw_fixed is False


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        direction_result_from_tuple(tuple(range(19)))


def test_struct_matches_tuple_on_a_real_fit():
    x, y, z, t, p = _synth_event()
    init = (30.0, 120.0, 1.0)
    raw = recons_direction_swf_adf(x, y, z, t, p, init, 0.0)
    res = recons_direction_struct(x, y, z, t, p, init, 0.0)
    assert len(raw) == 20
    assert res.raw == tuple(raw)
    assert res.zenith == raw[7] and res.azimuth == raw[8]
    assert res.x_max == raw[3] and res.fmin == raw[11]
    assert res.dw_fixed == getattr(raw, "dw_fixed", False)


def test_direction_only_writes_nothing(tmp_path, monkeypatch):
    """U5：只算方向的入口不得建目录 / 写文件，且结果与方向核心一致。"""
    x, y, z, t, p = _synth_event()
    init = (30.0, 120.0, 1.0)

    monkeypatch.chdir(tmp_path)
    res = reconstruct_direction_only(x, y, z, t, p, init, 0.0)
    assert list(tmp_path.iterdir()) == [], f"方向入口写了文件: {list(tmp_path.iterdir())}"

    raw = recons_direction_swf_adf(x, y, z, t, p, init, 0.0)
    assert res.raw == tuple(raw)
    assert res.zenith_init == raw[0] and res.azimuth_init == raw[1]
    assert res.chi2_plane == raw[2]
