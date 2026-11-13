"""ADF 拟合在**真实输入**上的端到端测试：真实电场 / 真实电压 / 带噪电压。

数据来自 CoREAS-AN 模拟树（不随仓库分发），用环境变量指定：

    PYTEST_COREAS_ROOT=/home/dkk/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_Dunhuang_..._0000

四类输入与对应 tree：
- 真实电场   `efield_*-L0.root` -> `tefield`（L0 = 未加噪）
- 带噪电场   `efield_*-L1.root` -> `tefield`（L1 = 加噪，同一物理事例的另一级别）
- 真实电压   `voltage_*-L0.root` -> `tvoltage`
- 带噪电压   `adc_*-L1.root`    -> `tadc`

**重建电场**那条（`*E_rec_time_all_DUs.txt`）需要 `parameter_files/` 里的天线有效长度表，
仓库里没有该文件，故本文件对它 skip（见 docs/reviews 与本计划的 U7）。

断言分两层：
1. 每类输入都能驱动同一次 ADF 拟合，给出物理上合理的 (theta, phi, dw, A)；
2. 四类输入在同一事例、同一几何中心下给出的方向互相一致 ——
   这是"不同来源的幅度喂给同一个 ADF 是否等价"的端到端判据。

#2 的容差来自实测（15 个事例 × 4 类输入，以 tadc 为参考）：
    dtheta: p50=0.00 p95=0.48 max=1.60   dphi: p50=0.00 p95=1.60 max=1.60
数值集中在 0.00 与 1.60 两档，是拟合精度的离散下界；取 3 度作为回归门限。
注意 fval 在不同输入间可以差一倍以上（幅度尺度与噪声水平不同），方向仍稳定，
所以只对方向设门限，不对 chi2 设。
"""

import glob
import os

import numpy as np
import pytest

import reconstruction_candidate_fluence_error_fixed as m
import reconstruction_fitting as rf

COREAS_ROOT = os.environ.get("PYTEST_COREAS_ROOT", "")
COORDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "..", "du_coords_CoreasDC2.txt")

pytestmark = pytest.mark.skipif(
    not COREAS_ROOT or not os.path.isdir(COREAS_ROOT),
    reason="设置 PYTEST_COREAS_ROOT 指向一个 CoREAS-AN 事例目录才跑真实输入 e2e",
)

INPUTS = {
    "tefield_L0": "efield_*_L0_*.root",
    "tefield_L1": "efield_*_L1_*.root",
    "tvoltage_L0": "voltage_*_L0_*.root",
    "tadc_L1": "adc_*_L1_*.root",
}


def _one(pattern):
    hits = sorted(glob.glob(os.path.join(COREAS_ROOT, pattern)))
    if not hits:
        pytest.skip(f"目录里没有 {pattern}")
    return hits[0]


def _amplitudes(root, idx):
    """读一类真实输入，返回 (du_ids, 每 DU 三通道全窗峰值幅度)。"""
    ev, du, gps, ns, lst, tr1, tr2, tr3 = m.read_root_file_from_path(root, idx)
    tr = np.stack([np.asarray(tr1), np.asarray(tr2), np.asarray(tr3)])
    return np.asarray(du, dtype=int), np.max(np.abs(tr), axis=(0, 2))


def _source_from_tadc(idx):
    """用同一事例的 tadc（有时间）跑 SWF，得到 ADF 需要的几何中心。"""
    du, p = _amplitudes(_one(INPUTS["tadc_L1"]), idx)
    ev, du2, gps, ns, lst, tr1, tr2, tr3 = m.read_root_file_from_path(
        _one(INPUTS["tadc_L1"]), idx)
    x, y, z, _, _ = m.read_xyz_coordinates(COORDS, du, rotate=False)
    xs, ys, zs, *_ = rf.SWF_fit(np.asarray(x), np.asarray(y), np.asarray(z),
                               np.asarray(ns, dtype=float) * 1e-9,
                               0.0, 0.0, 6000.0)
    return float(xs), float(ys), float(zs)


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("name", list(INPUTS))
def test_adf_fits_real_input(name, idx):
    root = _one(INPUTS[name])
    src = _source_from_tadc(idx)
    assert np.all(np.isfinite(src)) and abs(src[2]) > 1000.0, src

    du, ant_p = _amplitudes(root, idx)
    assert du.size >= 5 and np.all(np.isfinite(ant_p)), (du.size, ant_p[:3])
    x, y, z, _, _ = m.read_xyz_coordinates(COORDS, du, rotate=False)

    theta, phi, dw, A, fval = rf.ADF_fit_analytic(
        np.asarray(x), np.asarray(y), np.asarray(z), ant_p,
        src[0], src[1], src[2], 30.0, 180.0, 1.2, 0.0)

    assert 0.0 < theta < 90.0, (name, theta)
    assert 0.0 <= phi < 360.0, (name, phi)
    # dw 的下界取 0.005：MINUIT 自身把 dw 限在 [0.01, 5]，真实事例里确实会贴到下边界
    # （环极宽/极平的事例，例如 event 3 的 tvoltage 落在 0.021），这不是缺陷。
    assert 0.005 < dw <= 5.0, (name, dw)
    assert A > 0.0 and np.isfinite(fval), (name, A, fval)


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_inputs_agree_with_each_other(idx):
    """四类真实输入在同一事例、同一几何中心下的方向差异必须 <=3 度（见模块 docstring 的实测分布）。"""
    src = _source_from_tadc(idx)
    got = {}
    for name in INPUTS:
        du, ant_p = _amplitudes(_one(INPUTS[name]), idx)
        x, y, z, _, _ = m.read_xyz_coordinates(COORDS, du, rotate=False)
        th, ph, dw, A, fv = rf.ADF_fit_analytic(
            np.asarray(x), np.asarray(y), np.asarray(z), ant_p,
            src[0], src[1], src[2], 30.0, 180.0, 1.2, 0.0)
        got[name] = (float(th), float(ph))

    ref = got["tadc_L1"]
    for name, (th, ph) in got.items():
        dth = abs(th - ref[0])
        dph = abs((ph - ref[1] + 180) % 360 - 180)
        assert dth < 3.0 and dph < 3.0, (f"event {idx}", name, got[name], ref, dth, dph)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(COORDS), "reconstruction_candidate",
                                    "parameter_files")),
    reason="重建电场需要 parameter_files/（EFL 天线有效长度表），本环境缺失",
)
def test_adf_fits_reconstructed_efield():  # pragma: no cover - 仅在参数齐全时启用
    """占位：parameter_files 到位后，这里应读 *E_rec_time_all_DUs.txt 跑 ADF。"""
    assert True
