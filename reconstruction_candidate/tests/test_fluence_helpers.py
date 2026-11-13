"""`reconstruction_candidate_fluence_error_fixed.py` 里可独立验证的纯函数与轻量链路。

这些单元此前几乎全是 0 覆盖（模块整体 37%），但都不需要 `parameter_files/` 的 EFL 表：
频域↔时域换算、端口旋转与转角表、fluence/误差公式、Cherenkov 函数拟合、因果筛选、
CoREAS 坐标表解析。物理反演（`efield_recons_from_efield_PWF` / `energy_restruction`）
仍留在真实输入测试里，这里不碰。

不覆盖：`ADC_trace_to_Voc_time`（`vad2voc` 需要 `parameter_files/Z_ant_3.2m.csv`，仓库没有）。

断言取"公式层的不变量"，而不是把实现抄一遍：
rotation 保范数、Fluence = theta + phi 分量、零输入 → 零输出、拟合能回收到真值。
"""

import os

import numpy as np
import pytest

import reconstruction_candidate_fluence_error_fixed as m
import reconstruction_direction as rf

# ── 复数插值 ─────────────────────────────────────────────────────────────────


def test_interp_complex_1d_linear_is_exact_and_zero_outside():
    f_src = np.array([10.0, 20.0, 30.0, 40.0])
    y_src = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])

    # 落在源点上：线性插值应逐点还原
    same = m._interp_complex_1d(f_src, y_src, f_src, kind="linear")
    assert np.allclose(same, y_src)

    # 区间外的频点按契约置 0，而不是外插或抛错
    out = m._interp_complex_1d(f_src, y_src, np.array([5.0, 20.0, 100.0]), kind="linear")
    assert out[0] == 0 and out[2] == 0
    assert out[1] == pytest.approx(3 + 4j)


# ── 端口旋转 ─────────────────────────────────────────────────────────────────


def test_rotate_xy_ports_preserves_norm_and_leaves_z():
    rng = np.random.default_rng(3)
    voc = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))

    for angle in (-37.0, 0.0, 25.0, 90.0):
        rot = m.rotate_xy_ports(voc, angle)
        # Z 端口不动
        assert np.allclose(rot[:, 2], voc[:, 2])
        # 正交变换保范数（|Vx|²+|Vy|² 不变）
        assert np.allclose(np.abs(rot[:, 0]) ** 2 + np.abs(rot[:, 1]) ** 2,
                           np.abs(voc[:, 0]) ** 2 + np.abs(voc[:, 1]) ** 2)

    # 90 度：按文档公式 (V_X, V_Y) -> (V_Y, -V_X)
    rot90 = m.rotate_xy_ports(voc, 90.0)
    assert np.allclose(rot90[:, 0], voc[:, 1])
    assert np.allclose(rot90[:, 1], -voc[:, 0])


def test_rotate_xy_ports_handles_effective_length_axis():
    lec = np.zeros((5, 3, 3), dtype=complex)
    lec[:, 0, 0], lec[:, 0, 1] = 1.0, 0.0        # X 端口只沿 X
    rot = m.rotate_xy_ports(lec, 90.0, axis=1)
    # 端口轴旋 90 度后，X 端口的响应整体转到 Y 端口
    assert np.allclose(rot[:, 0, 0], 0.0) and np.allclose(rot[:, 1, 0], -1.0)


# ── 转角表 ───────────────────────────────────────────────────────────────────


def test_read_du_rotation_angles_converts_and_applies_overrides(tmp_path):
    table = tmp_path / "rotation.txt"
    np.savetxt(table, np.array([[1000.0, 0.0, 10.0, 0.0, 100.0],
                                [1001.0, 0.0, 40.0, 0.0, 40.0]]))
    angles = m.read_du_rotation_angles(str(table), usecols=(0, 2, 4))
    # phi = -0.5 * (ang_x + (ang_y - 90))
    assert angles[1000] == pytest.approx(-0.5 * (10.0 + 10.0))
    assert angles[1001] == pytest.approx(-0.5 * (40.0 - 50.0))

    adjusted = m.read_du_rotation_angles(str(table), usecols=(0, 2, 4),
                                         angle_overrides={1000: -2.0})
    assert adjusted[1000] == pytest.approx(angles[1000] - 2.0)
    assert adjusted[1001] == pytest.approx(angles[1001])


# ── CoREAS 坐标表解析 ────────────────────────────────────────────────────────


def test_resolve_coreas_det_pos_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("1000 0 0 0\n", encoding="utf-8")
    assert m._resolve_coreas_det_pos("/any/sim/adc_0.root", str(explicit)) == str(explicit)

    # 环境变量优先于同级文件
    env_path = tmp_path / "env.txt"
    env_path.write_text("1000 0 0 0\n", encoding="utf-8")
    monkeypatch.setenv("RECON_COREAS_DET_POS", str(env_path))
    root = tmp_path / "run" / "adc_0.root"
    root.parent.mkdir(parents=True)
    (root.parent / "du_coords.txt").write_text("1000 0 0 0\n", encoding="utf-8")
    assert m._resolve_coreas_det_pos(str(root)) == str(env_path)

    # 去掉环境变量 -> 同级 du_coords.txt
    monkeypatch.delenv("RECON_COREAS_DET_POS")
    assert m._resolve_coreas_det_pos(str(root)) == str(root.parent / "du_coords.txt")

    # 同级没有 -> 上一级；都没有 -> FileNotFoundError
    root2 = tmp_path / "run2" / "deep" / "adc_0.root"
    root2.parent.mkdir(parents=True)
    (tmp_path / "run2" / "du_coords.txt").write_text("1000 0 0 0\n", encoding="utf-8")
    assert m._resolve_coreas_det_pos(str(root2)) == str(tmp_path / "run2" / "du_coords.txt")

    missing = tmp_path / "empty" / "adc_0.root"
    missing.parent.mkdir()
    with pytest.raises(FileNotFoundError):
        m._resolve_coreas_det_pos(str(missing))


# ── 频域 Voc ─────────────────────────────────────────────────────────────────

def test_voc_fft_real_mhz_shape_and_band_only():
    n_time = 512
    rng = np.random.default_rng(11)
    voc = rng.normal(scale=1e-3, size=(n_time, 3))
    voc[40:60, :] *= 20                                    # 造一个峰，命中峰值判据
    freqs = np.fft.rfftfreq(n_time, d=2e-9) / 1e6
    adc = rng.normal(scale=1e-4, size=(n_time, 3))

    band = m.Voc_fft_real_MHz(voc, freqs, adc, m.time_wind)

    n_band = int(((freqs >= 30) & (freqs <= 250)).sum())
    assert band.shape == (n_band, 3)
    # 全零输入 -> 全零输出（窗口与变换都不产生虚假支撑）
    zeros = m.Voc_fft_real_MHz(np.zeros((n_time, 3)), freqs, np.zeros((n_time, 3)), m.time_wind)
    assert zeros.shape == (n_band, 3) and np.all(zeros == 0)


# ── Fluence 与误差公式 ───────────────────────────────────────────────────────


def test_calculate_energy_fluence_partition_and_zero_input():
    n_time = 256
    freqs = np.fft.rfftfreq(n_time, d=2e-9) / 1e6
    n_band = int(((freqs >= 30) & (freqs <= 250)).sum())
    rng = np.random.default_rng(7)
    et = rng.normal(size=n_band) * 1e-6
    ep = rng.normal(size=n_band) * 1e-6
    e_inv = np.zeros(3, dtype=complex)

    e_time, flu, f_th, f_phi, f_cross, f_sigma = m.calculate_energy_fluence_and_error(
        et, ep, e_inv, freqs, n_time)

    assert e_time.shape == (n_time, 2)
    # 两个分量正交相加：总 fluence = theta 分量 + phi 分量
    assert flu == pytest.approx(f_th + f_phi)
    # 交叉项是零延迟互相关：Cauchy-Schwarz 给出 |cross| <= sqrt(theta * phi)
    assert abs(f_cross) <= np.sqrt(f_th * f_phi) + 1e-30
    assert f_sigma >= 0

    zeros = m.calculate_energy_fluence_and_error(
        np.zeros(n_band), np.zeros(n_band), e_inv, freqs, n_time)
    _, flu0, f_th0, f_phi0, f_cross0, f_sigma0 = zeros
    assert flu0 == 0 and f_th0 == 0 and f_phi0 == 0 and f_cross0 == 0 and f_sigma0 == 0


# ── Cherenkov 函数拟合 ───────────────────────────────────────────────────────


def test_che_fit_chi2_recovers_synthetic_curve():
    omega = np.linspace(1e-4, 0.1, 80)
    amp_true, wc_true, dw_true = 10.0, 0.05, 0.5
    data = rf.f_Che([amp_true, wc_true, dw_true], omega)

    amp, wc, dw, chi2, amp_err, wc_err, dw_err = rf.che_fit_chi2(
        omega, data, amp_true, wc_true, dw_true, 0)

    assert amp == pytest.approx(amp_true, rel=0.05)
    assert wc == pytest.approx(wc_true, rel=0.05)
    assert dw == pytest.approx(dw_true, rel=0.10)
    assert chi2 < 1e-6
    assert amp_err > 0 and wc_err > 0 and dw_err >= 0


def test_che_fit_chi2_fix_dw_keeps_initial_width():
    omega = np.linspace(1e-4, 0.1, 60)
    data = rf.f_Che([10.0, 0.05, 0.5], omega)

    _, _, dw, chi2, _, _, _ = rf.che_fit_chi2(omega, data, 10.0, 0.04, 0.6, 1)

    assert dw == pytest.approx(0.6)          # 固定 dw：不随数据变化
    assert chi2 > 0                          # 宽度不对，残差不会归零


# ── 因果律筛选（图优化入口用替身）────────────────────────────────────────────


def _causality_inputs(rec_dir, n=3):
    du = np.array([1000, 1001, 1002])
    ns = np.array([-10, -20, 5])
    traces = [np.arange(n), np.arange(n) * 2, np.arange(n) * 3]
    pos = np.arange(3 * n, dtype=float).reshape(n, 3)
    return du, ns, traces[0], traces[1], traces[2], pos


def test_causality_filter_keeps_surviving_rows_and_cleans_temp_file(tmp_path, monkeypatch):
    rec_dir = tmp_path / "Reconstruction"
    rec_dir.mkdir()
    du, ns, t1, t2, t3, pos = _causality_inputs(rec_dir)

    seen = {}

    def fake_graph(file_path, detector_positions, min_detectors=5, **kw):
        seen["path"] = file_path
        seen["positions"] = detector_positions
        seen["min_detectors"] = min_detectors
        with open(file_path) as fh:
            seen["line"] = fh.read().strip()
        return [(1234, [("1000", -10), ("1002", 5)])]

    monkeypatch.setattr(m, "optimized_read_matching_times_graph", fake_graph)

    kept = m.causality_filter(str(rec_dir), 7, du, [1, 2, 1700000000],
                              ns, t1, t2, t3,
                              pos[:, 0], pos[:, 1], pos[:, 2], pos)

    du_ids, ns_out, a, b, c, pos_out, n_kept = kept
    assert n_kept == 2
    assert list(du_ids) == [1000, 1002]
    assert list(ns_out) == [-10, 5]
    assert a.tolist() == [0, 2] and b.tolist() == [0, 4] and c.tolist() == [0, 6]
    assert pos_out.tolist() == [pos[0].tolist(), pos[2].tolist()]
    # 传给图优化的是"未旋转的硬坐标"与文档规定的 4 个探测器下限
    assert seen["min_detectors"] == 4
    assert list(seen["positions"]) == ["1000", "1001", "1002"]
    assert seen["line"].startswith("1700000000: [")
    # 临时文件用完即删
    assert not os.path.exists(seen["path"])


def test_causality_filter_returns_none_when_graph_keeps_nothing(tmp_path, monkeypatch):
    du, ns, t1, t2, t3, pos = _causality_inputs(tmp_path)
    monkeypatch.setattr(m, "optimized_read_matching_times_graph",
                        lambda *a, **kw: [])

    result = m.causality_filter(str(tmp_path), 1, du, [1, 2, 3], ns, t1, t2, t3,
                                pos[:, 0], pos[:, 1], pos[:, 2], pos)
    assert result is None
    assert list(tmp_path.glob("temp_causality_event_*.txt")) == []
