"""纯函数与文件解析层的单元测试（不依赖 ROOT 数据、不写盘）。"""

import os

import numpy as np
import pytest

import reconstruction_candidate_fluence_error_fixed as m


# --- 统计与时间 -------------------------------------------------------------

def test_sigma_is_zero_at_zero_dchi2_and_monotone():
    assert m.sigma(0.0, 1) == pytest.approx(0.0, abs=1e-9)
    s = [m.sigma(x, 1) for x in (1.0, 4.0, 9.0)]
    assert all(np.isfinite(s)) and s[0] < s[1] < s[2], s


def test_gps_to_lst_is_deterministic_and_in_range():
    gps = 1.5e9
    a = m.gps_to_lst(gps)
    b = m.gps_to_lst(gps)
    assert a == b
    lst = float(np.ravel(a)[0])
    assert 0.0 <= lst < 24.0, lst
    # 一整天后 LST 基本相同；恒星日 vs 太阳日差约 86400/365.25 s = 0.0657 h
    assert abs(float(np.ravel(m.gps_to_lst(gps + 86400.0))[0]) - lst) == pytest.approx(0.0657, abs=5e-3)


def test_synthesize_gps_times_is_monotone_and_sized(tmp_path):
    root = tmp_path / "sim_Dunhuang_20170331_220000_RUN1_CD_test" / "adc_0-9_L1_0000.root"
    root.parent.mkdir(parents=True)
    root.write_text("x")
    # 返回扁平三元组 (year, hhmmss, unix) 每个 DU 一组
    out = np.asarray(m._synthesize_gps_times(str(root), 3, 4))
    assert out.size == 3 * 4, out
    assert out[0] == 20170331 and out[1] == 220000
    assert np.all(out[2::3] == out[2])


# --- 几何 -------------------------------------------------------------------

def test_rotation_matrix_is_orthonormal():
    for theta, phi in ((0.0, 0.0), (25.0, 40.0), (80.0, 300.0)):
        r = np.asarray(m.rotation_m(np.radians(theta), np.radians(phi)))
        assert r.shape == (3, 3)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9), (theta, phi)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9), (theta, phi)


# --- 位置表解析 -------------------------------------------------------------

def test_read_du_position_table_comma_and_space(tmp_path):
    comma = tmp_path / "du_coords.txt"
    comma.write_text("0, 0.0, 0.0, 0.0\n1, -501.4150, 286.2280, 4.9852\n")
    a = np.asarray(m._read_du_position_table(str(comma)))
    assert a.shape == (2, 4)
    assert a[1, 1] == pytest.approx(-501.4150)

    space = tmp_path / "gp65.txt"
    space.write_text("0 0.0 0.0 0.0\n1 10.0 20.0 30.0\n")
    b = np.asarray(m._read_du_position_table(str(space)))
    assert b.shape == (2, 4)
    assert b[1, 3] == pytest.approx(30.0)


def test_read_xyz_coordinates_without_rotation(tmp_path, monkeypatch):
    table = tmp_path / "du_coords.txt"
    table.write_text("7, 11.0, 22.0, 33.0\n8, 44.0, 55.0, 66.0\n")
    x, y, z, rotated, all_rotated = m.read_xyz_coordinates(str(table), [7, 8], rotate=False)
    assert (list(x), list(y), list(z)) == ([11.0, 44.0], [22.0, 55.0], [33.0, 66.0])
    # rotate=False 时"旋转后"与原文一致
    assert np.allclose(np.asarray(rotated)[:, 0], x)
    assert np.allclose(np.asarray(all_rotated)[:, 0], x)


# --- 驱动脚本的纯工具 -------------------------------------------------------

def test_driver_safe_helpers():
    import run_parallel_reconstruction_fixed as d

    assert d._safe(np.float64(1.5)) == 1.5
    assert d._safe(np.nan) is None
    assert d._safe(None) is None
    assert d._safe_list(np.array([1.0, np.nan])) == [1.0, None]
    assert d._safe_int_list(np.array([3.0, 4.0])) == [3, 4]


def test_driver_extract_date_from_filename():
    import run_parallel_reconstruction_fixed as d

    assert d._extract_date_from_filename("Trigger_20250101_120000.root") == ("2025", "01", "01")
    with pytest.raises(ValueError):
        d._extract_date_from_filename("no_date.root")
