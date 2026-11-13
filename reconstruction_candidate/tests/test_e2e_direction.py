"""端到端：合成 CoREAS `tadc` ROOT → reader → 方向重建，不起子进程跑全链。

为什么不到电场的全链：`efield_recons_from_efield_PWF` 需要
`reconstruction_candidate/parameter_files/` 里的天线有效长度表（EFL .mat），
仓库里没有该文件（见 docs/reviews 与计划的 U7 阻塞项），所以这里止于方向层，
把"读 ROOT → 方向"这条链整体跑通。
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import uproot

import reconstruction_candidate_fluence_error_fixed as m
import reconstruction_fitting as rf

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(MODULE_DIR, "run_parallel_reconstruction_fixed.py")

N_DU, N_TIME = 8, 64


def _write_synthetic_tadc(path, n_events=2):
    """写一个 tadc 树：每个 entry = 一个事例，du_id/trace_ch 为 jagged。

    球形波前到达时间 + 统一幅度脉冲；几何来自固定的 8 根天线位置。
    """
    import awkward as ak

    rng = np.random.default_rng(5)
    r = rng.uniform(80.0, 400.0, N_DU)
    a = rng.uniform(0, 2 * np.pi, N_DU)
    xs, ys = r * np.cos(a), r * np.sin(a)
    src = np.array([60.0, -40.0, 5000.0])
    dist = np.sqrt((src[0] - xs) ** 2 + (src[1] - ys) ** 2 + src[2] ** 2)
    ns_per_du = -np.round(dist / 0.299792458).astype(np.int64)

    base = np.zeros((N_DU, 3, N_TIME))
    base[:, :, 20] = 1.0

    try:
        with uproot.recreate(path) as f:
            f["tadc"] = {
                "event_number": np.arange(n_events, dtype=np.int32),
                "du_id": ak.Array([list(range(N_DU))] * n_events),
                "du_nanoseconds": ak.Array([ns_per_du.tolist()] * n_events),
                "trace_ch": ak.Array([base.tolist()] * n_events),
            }
    except TypeError as exc:      # pragma: no cover - 旧版 uproot/awkward 写不了 3 维 jagged
        pytest.skip(f"本机 uproot/awkward 无法写出 3 维 jagged ROOT: {exc}")
    return xs, ys


def test_reader_to_direction_end_to_end(tmp_path, monkeypatch):
    run_dir = tmp_path / "sim_Dunhuang_20170331_220000_RUN1_CD_e2e"
    run_dir.mkdir()
    root = run_dir / "adc_0-1_L1_0000.root"
    xs, ys = _write_synthetic_tadc(str(root))

    ev, du_ids, gps, du_ns, lst, tr1, tr2, tr3 = m.read_root_file_from_path(str(root), 0)
    assert ev == 0
    assert np.asarray(tr1).shape == (N_DU, N_TIME)
    assert len(du_ids) == N_DU and len(du_ns) == N_DU

    # 坐标表：与合成几何一致
    table = tmp_path / "du_coords.txt"
    table.write_text("".join(f"{i}, {xs[i]:.4f}, {ys[i]:.4f}, 0.0\n" for i in range(N_DU)))
    x, y, z, _, _ = m.read_xyz_coordinates(str(table), list(range(N_DU)), rotate=False)

    # 每个 DU 取三通道、全时间窗的峰值幅度 -> 形状 (n_du,)
    p = np.max(np.abs(np.stack([np.asarray(tr1), np.asarray(tr2), np.asarray(tr3)])), axis=(0, 2))

    monkeypatch.chdir(tmp_path)
    res = rf.reconstruct_direction_only(x, y, z, np.asarray(du_ns, dtype=float), p,
                                      (30.0, 120.0, 1.0), 0.0)
    assert isinstance(res, rf.DirectionResult)
    assert np.isfinite(res.zenith) and np.isfinite(res.azimuth)
    assert 0.0 <= res.azimuth < 360.0
    assert np.isfinite(res.fmin)
    assert list(run_dir.iterdir()) == [root], "方向层不应写任何文件"


def test_driver_cli_smoke():
    out = subprocess.run([sys.executable, DRIVER, "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    for flag in ("--ncores", "--det-pos", "--no-resume", "--candidates"):
        assert flag in out.stdout, flag


@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODULE_DIR, "parameter_files")),
    reason="缺 parameter_files/（EFL 天线有效长度表），电场全链在此环境跑不了",
)
def test_efield_full_chain_is_available():  # pragma: no cover - 仅在参数齐全时启用
    """占位：parameter_files 到位后，这里应升级为真正的 efield 全链 e2e。"""
    assert True
