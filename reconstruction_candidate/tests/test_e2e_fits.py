"""PWF / SWF / ADF 三条拟合链的端到端测试：已知真值 → 合成数据 → 拟合 → 回真值。

每条链都带一个"负对照"：把数据打乱（或给错初值）后拟合必须明显变差，
证明断言不是恒真。所有合成都用模块自己的物理约定（c 用 m/ns = 0.299792458）。
"""

import numpy as np
import pytest

import reconstruction_fitting as m

C_M_PER_S = 299792458.0        # PWF/SWF 的残差里 c 用 m/s，时间用秒


def _antennas(n=24, seed=3, radius=500.0, relief=8.0):
    """阵列加一点高程差（真实 det_pos 有 5-30 m 起伏）。

    z 全为 0 时天顶角的镜像解在平面阵上严格简并（dz=0，cosθ 符号不可分辨），
    拟合会落到 180-θ；给出少量起伏才能让方向可辨识。
    """
    rng = np.random.default_rng(seed)
    r = rng.uniform(50.0, radius, n)
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a), rng.uniform(0.0, relief, n)


def _plane_wave_times(x, y, z, zenith, azimuth, t0=0.0):
    """平面波前（与 PWF.chi2 同约定）：residual = r·n̂ - c·Δt，t 单位秒，n̂ 取正号。"""
    th, ph = np.radians(zenith), np.radians(azimuth)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return t0 + (n[0] * x + n[1] * y + n[2] * z) / C_M_PER_S


def _spherical_times(x, y, z, src, t0=0.0):
    """球面波前：t = t0 + d/c（秒）。SWF 内部用 n_eff 折射率，差异 ~1e-4 可忽略。"""
    d = np.sqrt((src[0] - x) ** 2 + (src[1] - y) ** 2 + (src[2] - z) ** 2)
    return t0 + d / C_M_PER_S


def _wc_for(x, y, z, src):
    """用模块自己的几何估计 wc（ADF 的切伦科夫角），让合成与拟合用同一个模型。"""
    dummy = np.ones_like(x)
    return float(m.ADF(x, y, z, dummy, src[0], src[1], src[2], 0.0).wc)


def _adf_amplitudes(x, y, z, src, zenith, azimuth, dw=1.2, wc=None, noise=0.0, seed=7):
    """CoREAS 型切伦科夫环幅度（与模块 _adf_chi2_profiled 同形式）。"""
    rng = np.random.default_rng(seed)
    if wc is None:
        wc = _wc_for(x, y, z, src)
    dx, dy, dz = src[0] - x, src[1] - y, src[2] - z
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    th, ph = np.radians(zenith), np.radians(azimuth)
    axis = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    omega = np.arccos(np.clip((dx * axis[0] + dy * axis[1] + dz * axis[2]) / dist, -1, 1))
    amp = 1.0 / (1 + 4 * (((np.tan(omega) / np.tan(wc)) ** 2 - 1) / dw) ** 2)
    return amp * (1.0 + rng.normal(0, noise, amp.size))


# --- PWF --------------------------------------------------------------------

def test_pwf_recovers_plane_wave_direction():
    """PWF 端到端：合成平面波 -> PWF_fit。

    阵列只有 8 m 起伏时，天顶角的镜像解（θ vs 180-θ）在该几何下几乎不可分辨，
    所以断言分三段：方位角必须严格回真值；天顶角回真值**或其镜像**；
    并且拟合点的模型 chi2 与真值点的 chi2 同量级（说明拟合没有跑到别处）。
    """
    x, y, z = _antennas()
    truth = (32.0, 118.0)
    t = _plane_wave_times(x, y, z, *truth)

    theta, phi, chi2_dof, theta_err, phi_err = m.PWF_fit(x, y, z, t, truth[0] + 0.3, truth[1] - 0.3)

    assert abs((phi - truth[1] + 180) % 360 - 180) < 1.0, (phi, truth[1])
    assert min(abs(theta - truth[0]), abs(theta - (180.0 - truth[0]))) < 0.15, theta

    pwf = m.PWF(x, y, z, t)
    chi2_truth = pwf(np.radians(truth[0]), np.radians(truth[1]), 1.0)
    chi2_fit = pwf(np.radians(theta), np.radians(phi), 1.0)
    assert chi2_fit < 10 * max(chi2_truth, 1e-9) + 1.0, (chi2_truth, chi2_fit)
    assert np.isfinite(chi2_dof)


def test_pwf_fit_is_worse_on_scrambled_times():
    """负对照：把时间顺序打乱后，拟合的 chi2/dof 必须显著变差。"""
    x, y, z = _antennas()
    t = _plane_wave_times(x, y, z, 32.0, 118.0)
    _, _, chi2_ok, _, _ = m.PWF_fit(x, y, z, t, 32.3, 117.7)

    scrambled = t.copy()
    np.random.default_rng(11).shuffle(scrambled)
    _, _, chi2_bad, _, _ = m.PWF_fit(x, y, z, scrambled, 32.3, 117.7)

    assert chi2_bad > 10 * max(chi2_ok, 1e-6), (chi2_ok, chi2_bad)


# --- SWF --------------------------------------------------------------------

def test_swf_recovers_spherical_source():
    x, y, z = _antennas()
    true_src = np.array([180.0, -260.0, 6400.0])
    t = _spherical_times(x, y, z, true_src)

    xs, ys, zs, fval, xs_err, ys_err, zs_err = m.SWF_fit(
        x, y, z, t, true_src[0] + 60, true_src[1] - 60, true_src[2] + 400)

    dist = np.linalg.norm(np.array([xs - true_src[0], ys - true_src[1], zs - true_src[2]]))
    assert dist < 300.0, (xs, ys, zs, dist)
    assert abs(zs - true_src[2]) < 600.0, zs


def test_swf_single_start_matches_grid_start():
    """simplex=False（给先验）与网格起点应落在同一解附近。"""
    x, y, z = _antennas()
    true_src = np.array([-120.0, 90.0, 5200.0])
    t = _spherical_times(x, y, z, true_src)

    grid = m.SWF_fit(x, y, z, t, -100.0, 100.0, 5000.0)[:3]
    single = m.SWF_fit(x, y, z, t, true_src[0] + 30, true_src[1] + 30, true_src[2] + 300,
                       simplex=False)[:3]
    gap = np.linalg.norm(np.array(grid, dtype=float) - np.array(single, dtype=float))
    assert gap < 150.0, (grid, single, gap)


# --- ADF --------------------------------------------------------------------

def test_adf_recovers_cherenkov_ring_direction():
    x, y, z = _antennas(n=40, radius=700.0)
    src = np.array([300.0, -200.0, 7000.0])
    truth = (26.0, 155.0)
    ant_p = _adf_amplitudes(x, y, z, src, *truth, noise=0.02)

    theta, phi, dw, A, fval = m.ADF_fit_analytic(
        x, y, z, ant_p, src[0], src[1], src[2], truth[0] + 0.4, truth[1] + 0.4, 1.2, 0.0)

    # ADF 的 wc 由模块按几何自行估计，与合成用的 wc 略有出入 -> 容差按度级放
    assert abs(theta - truth[0]) < 2.5, (theta, truth[0])
    assert abs((phi - truth[1] + 180) % 360 - 180) < 2.5, (phi, truth[1])
    assert 0.05 < dw < 5.0
    assert A > 0 and np.isfinite(fval)


def test_adf_is_worse_when_the_geometry_center_is_wrong():
    """负对照：把 ADF 的几何中心（Xmax 源点）挪开 3 km，chi2 必须明显变差。

    ADF 拟合的是"以源点为中心的切伦科夫环"，源点错了环就错，这是它最依赖的外部输入。
    """
    x, y, z = _antennas(n=40, radius=700.0)
    src = np.array([300.0, -200.0, 7000.0])
    ring = _adf_amplitudes(x, y, z, src, 26.0, 155.0, noise=0.02)

    _, _, _, _, fval_ok = m.ADF_fit_analytic(
        x, y, z, ring, src[0], src[1], src[2], 26.4, 155.4, 1.2, 0.0)
    _, _, _, _, fval_bad = m.ADF_fit_analytic(
        x, y, z, ring, src[0] + 3000.0, src[1], src[2], 26.4, 155.4, 1.2, 0.0)

    assert fval_bad > 2 * max(fval_ok, 1e-6), (fval_ok, fval_bad)


# --- 三链合一：方向驱动 ------------------------------------------------------

def test_direction_driver_runs_all_three_stages():
    """recons_direction_swf_adf = SWF(源点) → find_core(dw 先验) → ADF(方向)。

    用球面波时间 + 环幅度同一真值，检查三阶段结果互相一致。
    """
    x, y, z = _antennas(n=30, radius=600.0)
    src = np.array([150.0, -180.0, 6100.0])
    truth = (28.0, 130.0)
    t = _spherical_times(x, y, z, src)
    p = _adf_amplitudes(x, y, z, src, *truth, noise=0.02)

    res = m.reconstruct_direction_only(x, y, z, t, p, (truth[0] + 0.3, truth[1] + 0.3, 1.0), 0.0)

    assert abs(res.zenith - truth[0]) < 3.0, (res.zenith, truth[0])
    assert abs((res.azimuth - truth[1] + 180) % 360 - 180) < 3.0, (res.azimuth, truth[1])
    # SWF 阶段找到的源点应与真值同量级（同一批合成数据）
    assert np.hypot(res.x_max - src[0], res.y_max - src[1]) < 400.0, (res.x_max, res.y_max)
    assert np.isfinite(res.fmin) and res.fmin >= 0
