"""驱动脚本 `run_parallel_reconstruction_fixed.py` 的单元与端到端测试。

单元层：路径与落盘文件的读取、断点续跑判据与字段映射、任务装载（txt / candidates YAML）、
汇总 YAML 的键与状态、`reconstruct_one` 的四种状态（0 成功 / 1 电场失败 / 2 能量失败 / 3 续跑）。

端到端两条：
- 进程内 `main()`：真实 argparse → 装载任务 → `Pool`（替换为同步执行）→ 汇总落盘；
- 子进程真实脚本：只走断点续跑路径（产物已落盘 → 不触发任何拟合）。这是本仓库
  唯一能整链跑通、且不需要 `parameter_files/` 的路径。

两处替身不可避免：`efield_recons_from_efield_PWF` 与 `energy_restruction` 需要
`reconstruction_candidate/parameter_files/` 里的 EFL 表（仓库缺失，见计划 U7 阻塞项）。
替身只按这两个函数的**落盘契约**写文件，被测的是驱动层如何读回、映射与汇总。
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

import run_parallel_reconstruction_fixed as d

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(MODULE_DIR)
DRIVER = os.path.join(MODULE_DIR, "run_parallel_reconstruction_fixed.py")

N_ANT = 3


# ── 落盘契约的夹具：与 fixed 版列顺序一致（18 列 fd / 35 列 so）────────────────


def _fd_rows(event_num=7):
    """`*reconstruction_results.txt`：18 列，每行一个天线。"""
    rows = []
    for i in range(N_ANT):
        row = np.zeros(18)
        row[0] = event_num
        row[1] = 0.0            # gps_time
        row[2] = 0.0            # LST
        row[3] = 1000 + i       # du_id
        row[4] = 30.0 + i       # theta_plane
        row[5] = 90.0 + i       # phi_plane
        row[6] = 1.5 + i        # chi2Plane/dof
        row[7] = -10 * i        # du_nanoseconds
        row[8], row[9], row[10] = 100.0 * i, 50.0 * i, 5.0
        row[11] = 10.0 + i      # Fluence
        row[16] = 0.11          # theta_err
        row[17] = 0.22          # phi_err
        rows.append(row)
    return np.array(rows)


def _so_row():
    """`shower_output.txt`：一维 35 列。"""
    so = np.zeros(35)
    so[0], so[1], so[2] = 12.0, -34.0, 8000.0     # SWF x/y/z_xmax
    so[3] = 2.0                                    # SWF chi2
    so[4], so[5] = 28.5, 0.3                       # ADF theta, err
    so[6], so[7] = 95.0, 0.4                       # ADF phi, err
    so[8], so[9] = 123.0, 4.5                      # ADF A, err
    so[10], so[12], so[13] = 900.0, 0.5, 0.02      # wc, dw, dw_err
    so[14] = 1.8                                   # ADF chi2
    so[15], so[16], so[17] = 1.0, 2.0, 3.0         # SWF x/y/z err
    return so


def _write_artifacts(rec_dir, event_num=7, include_arrays=True):
    """按主链路的命名写出一份完整产物。"""
    import glob  # noqa: F401  (保持与真实布局一致的 glob 命名)

    os.makedirs(rec_dir, exist_ok=True)
    np.savetxt(os.path.join(rec_dir, "x_reconstruction_results.txt"), _fd_rows(event_num))
    np.savetxt(os.path.join(rec_dir, "shower_output.txt"), _so_row())
    if include_arrays:
        np.savetxt(os.path.join(rec_dir, "y_E_rec_time_all_DUs.txt"), np.ones((N_ANT, 4)))
    return rec_dir


def _stub_physics(monkeypatch, calls, fail_field=False, fail_energy=False):
    """把两个重型入口替换为「按落盘契约写文件」的替身。"""

    def fake_efield(root_path, event_index, det_pos_path=None, valid_du_ids=None,
                    theta=None, phi=None, prior_times=None):
        calls.append(dict(fn="efield", det_pos_path=det_pos_path, valid_du_ids=valid_du_ids,
                          theta=theta, phi=phi, prior_times=prior_times))
        if fail_field:
            raise RuntimeError("field boom")
        _write_artifacts(d._event_rec_dir(root_path, event_index))

    def fake_energy(root_path, event_index, det_pos_path=None, x_xmax=None, y_xmax=None,
                    z_xmax=None, chi_square_external=None):
        calls.append(dict(fn="energy", x_xmax=x_xmax, y_xmax=y_xmax, z_xmax=z_xmax,
                          chi_square_external=chi_square_external))
        if fail_energy:
            raise RuntimeError("energy boom")
        return (np.array([1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0]),
                np.array([0.0, 1.0, 2.0]))

    monkeypatch.setattr(d, "efield_recons_from_efield_PWF", fake_efield)
    monkeypatch.setattr(d, "energy_restruction", fake_energy)


@pytest.fixture
def rec_root(tmp_path, monkeypatch):
    """把 `efield_rec_dir` 指向 tmp，避免写仓库里的 ./Reconstruction/。"""
    root = tmp_path / "Reconstruction"
    monkeypatch.setattr(d, "efield_rec_dir", str(root))
    return root


# ── 单元：路径与落盘文件读取 ──────────────────────────────────────────────────


def test_build_root_path_expands_date_and_passes_dirs():
    base = "/data/TD"
    assert d._build_root_path("/abs/x.root", base) == "/abs/x.root"
    assert d._build_root_path("sub/x.root", base) == "sub/x.root"
    assert d._build_root_path("Trigger_20250101_120000.root", base) == \
        "/data/TD/2025/01/01/Trigger_20250101_120000.root"
    # CoREAS shard 名不含 8 位日期 → 直接拼在 base_path 下
    assert d._build_root_path("adc_0-99_L1_0000.root", base) == "/data/TD/adc_0-99_L1_0000.root"


def test_readers_follow_fixed_column_layout(rec_root):
    _write_artifacts(str(rec_root / "sim_event_0"))

    fd = d._read_efield_txt("/any/sim.root", 0)
    assert fd.shape == (N_ANT, 18), fd.shape
    assert d._read_shower_output("/any/sim.root", 0).size == 35

    # 单天线：np.loadtxt 返回 1-D，读取侧应补成 (1, 18)
    single = rec_root / "single_event_0"
    single.mkdir(parents=True)
    np.savetxt(single / "single_reconstruction_results.txt", _fd_rows()[:1])
    assert d._read_efield_txt("/any/single.root", 0).shape == (1, 18)

    with pytest.raises(FileNotFoundError):
        d._read_efield_txt("/any/missing.root", 0)
    with pytest.raises(FileNotFoundError):
        d._read_shower_output("/any/missing.root", 0)


# ── 单元：断点续跑 ────────────────────────────────────────────────────────────


def test_load_completed_entry_requires_both_artifacts(rec_root):
    # 目录不存在
    assert d._load_completed_entry("/any/none.root", 0) is None

    # 只有 shower_output.txt
    half = rec_root / "half_event_0"
    half.mkdir(parents=True)
    np.savetxt(half / "shower_output.txt", _so_row())
    assert d._load_completed_entry("/any/half.root", 0) is None

    # 两个文件都在，但 shower_output 列数不足 (<15)
    short = rec_root / "short_event_0"
    short.mkdir(parents=True)
    np.savetxt(short / "x_reconstruction_results.txt", _fd_rows())
    np.savetxt(short / "shower_output.txt", np.zeros(10))
    assert d._load_completed_entry("/any/short.root", 0) is None


def test_load_completed_entry_rebuilds_summary_fields(rec_root):
    _write_artifacts(str(rec_root / "ok_event_2"), event_num=11)

    entry = d._load_completed_entry("/any/ok.root", 2)

    assert entry is not None
    assert entry["_status"] == d._STATUS_RESUMED == 3
    assert entry["event_num"] == 11
    assert entry["file_path"] == "/any/ok.root" and entry["event_index"] == 2
    assert entry["PWF"]["rec_theta_plane"] == pytest.approx(30.0)
    assert entry["PWF"]["rec_theta_plane_err"] == pytest.approx(0.11)
    assert entry["SWF"]["rec_x_xmax"] == pytest.approx(12.0)
    assert entry["SWF"]["rec_z_xmax_err"] == pytest.approx(3.0)
    assert entry["ADF"]["rec_theta_sph"] == pytest.approx(28.5)
    assert entry["ADF"]["rec_dw"] == pytest.approx(0.5)
    assert entry["antennas"]["du_ids"] == [1000, 1001, 1002]
    # 派生字段由 _recompute_antenna_fields 从落盘文件复现（能量/偏振都需要几何，这里只要求可复现）
    assert len(entry["antennas"]["omega_rad"]) == N_ANT
    assert len(entry["antennas"]["energy_flux"]) == N_ANT
    assert len(entry["antennas"]["polarization"]) == N_ANT

    # 缺少 *E_rec_time_all_DUs.txt 时复现失败 → 按契约回落为三个空列表，而不是丢掉整个 entry
    _write_artifacts(str(rec_root / "noarr_event_3"), event_num=12, include_arrays=False)
    fallback = d._load_completed_entry("/any/noarr.root", 3)
    assert fallback["_status"] == 3 and fallback["event_num"] == 12
    assert fallback["antennas"]["omega_rad"] == []
    assert fallback["antennas"]["energy_flux"] == []
    assert fallback["PWF"]["rec_theta_plane"] == pytest.approx(30.0)


# ── 单元：任务装载 ────────────────────────────────────────────────────────────


def test_load_tasks_from_txt_skips_malformed_lines(tmp_path):
    txt = tmp_path / "list.txt"
    txt.write_text(
        "# 注释行\n"
        "\n"
        "Trigger_20250101_120000.root 3\n"
        "只有一列\n"
        "adc_0-99_L1_0000.root 0\n",
        encoding="utf-8",
    )

    tasks = d.load_tasks_from_txt(str(txt), "/data/TD")

    assert [t.event_index for t in tasks] == [3, 0]
    assert tasks[0].root_path == "/data/TD/2025/01/01/Trigger_20250101_120000.root"
    assert tasks[1].root_path == "/data/TD/adc_0-99_L1_0000.root"
    assert all(t.valid_du_ids is None and t.theta is None and t.resume is True for t in tasks)


def test_load_tasks_from_candidates_yaml_maps_priors(tmp_path):
    cand = tmp_path / "candidates.yaml"
    yaml.safe_dump({
        "42": {"file": "Trigger_20250101_120000.root", "index": 1,
               "du_id": [1000, 1001], "time": {"1000": -10.0},
               "zenith": 31.5, "azimuth": 88.0,
               "x": 1.0, "y": 2.0, "z": 3.0, "chi_square": 4.5},
        "43": {"file": "x.root"},                      # 缺 index → 跳过
        "44": "not-a-mapping",                         # 非 mapping → 跳过
    }, open(cand, "w", encoding="utf-8"))

    tasks = d.load_tasks_from_candidates_yaml(str(cand), "/data/TD")

    assert len(tasks) == 1
    t = tasks[0]
    assert t.root_path == "/data/TD/2025/01/01/Trigger_20250101_120000.root"
    assert (t.event_index, t.valid_du_ids) == (1, [1000, 1001])
    assert t.prior_times == {"1000": -10.0}
    assert (t.theta, t.phi) == (31.5, 88.0)
    assert (t.x_xmax, t.y_xmax, t.z_xmax, t.chi_square_external) == (1.0, 2.0, 3.0, 4.5)


def test_candidates_yaml_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        d.load_tasks_from_candidates_yaml(str(bad), "/data/TD")


# ── 单元：reconstruct_one 的四种状态 ─────────────────────────────────────────


def test_reconstruct_one_success_maps_files(rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls)
    spec = d.TaskSpec(root_path="/any/e.root", event_index=0, det_pos_path="pos.txt",
                      valid_du_ids=[1000, 1001], theta=31.0, phi=88.0,
                      prior_times={"1000": -5.0}, x_xmax=1.0, y_xmax=2.0, z_xmax=3.0)

    entry = d.reconstruct_one(spec)

    assert entry["_status"] == 0
    assert entry["event_num"] == 7
    assert entry["PWF"]["rec_theta_plane"] == pytest.approx(30.0)
    assert entry["PWF"]["rec_phi_plane_err"] == pytest.approx(0.22)
    assert entry["SWF"]["rec_y_xmax"] == pytest.approx(-34.0)
    assert entry["ADF"]["rec_A"] == pytest.approx(123.0)
    assert entry["antennas"]["du_ids"] == [1000, 1001, 1002]
    assert entry["antennas"]["omega_rad"] == [1.0, 2.0, 3.0]
    assert entry["antennas"]["polarization"] == [0.0, 1.0, 2.0]
    # 先验透传给两个重型入口
    assert calls[0]["det_pos_path"] == "pos.txt"
    assert (calls[0]["theta"], calls[0]["phi"]) == (31.0, 88.0)
    assert calls[0]["prior_times"] == {"1000": -5.0}
    assert (calls[1]["x_xmax"], calls[1]["chi_square_external"]) == (1.0, None)


def test_reconstruct_one_field_failure_returns_status_1(rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls, fail_field=True)

    entry = d.reconstruct_one(d.TaskSpec(root_path="/any/f.root", event_index=4))

    assert entry["_status"] == 1
    assert entry["event_num"] is None
    assert entry["file_path"] == "/any/f.root" and entry["event_index"] == 4
    assert entry["PWF"]["rec_theta_plane"] is None
    assert entry["antennas"]["du_ids"] == []
    assert [c["fn"] for c in calls] == ["efield"]       # 电场失败后不再进入能量重建


def test_reconstruct_one_energy_failure_returns_status_2(rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls, fail_energy=True)

    entry = d.reconstruct_one(d.TaskSpec(root_path="/any/g.root", event_index=0))

    assert entry["_status"] == 2
    assert entry["PWF"]["rec_theta_plane"] == pytest.approx(30.0)   # 电场结果保留
    assert entry["SWF"]["rec_x_xmax"] is None
    assert entry["antennas"]["omega_rad"] == []


def test_reconstruct_one_resume_skips_physics(rec_root, monkeypatch):
    _write_artifacts(str(rec_root / "cached_event_0"), event_num=9)

    def boom(*a, **k):  # pragma: no cover - 一旦被调用即失败
        raise AssertionError("断点续跑不应触发重型重建")

    monkeypatch.setattr(d, "efield_recons_from_efield_PWF", boom)
    monkeypatch.setattr(d, "energy_restruction", boom)

    entry = d.reconstruct_one(d.TaskSpec(root_path="/any/cached.root", event_index=0))

    assert entry["_status"] == 3 and entry["event_num"] == 9


def test_reconstruct_one_resume_disabled_recomputes(rec_root, monkeypatch):
    _write_artifacts(str(rec_root / "cached_event_0"), event_num=9)
    calls = []
    _stub_physics(monkeypatch, calls)

    entry = d.reconstruct_one(d.TaskSpec(root_path="/any/cached.root", event_index=0, resume=False))

    assert entry["_status"] == 0 and entry["event_num"] == 7
    assert [c["fn"] for c in calls] == ["efield", "energy"]


def test_reconstruct_one_accepts_legacy_tuple(rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls)

    entry = d.reconstruct_one(("/any/h.root", "2", "pos.txt", False))

    assert entry["event_index"] == 2 and entry["_status"] == 0
    assert calls[0]["det_pos_path"] == "pos.txt"


# ── 单元：汇总 YAML ──────────────────────────────────────────────────────────


def test_assemble_summary_yaml_keys_statuses_and_duplicates(tmp_path):
    def entry(event_num, status, name):
        return dict(event_num=event_num, file_path=f"/any/{name}.root", event_index=0,
                    PWF={}, SWF={}, ADF={}, antennas={}, _status=status)

    out = tmp_path / "summary.yaml"
    summary, statuses = d.assemble_summary_yaml(
        [entry(7, 0, "a"), entry(7, 1, "b"), entry(None, 2, "c")], str(out))

    assert statuses == [0, 1, 2]
    # event_num 是 int，YAML 落盘后键仍是 int，重命名/兜底键是 str
    assert list(summary) == [7, "7_dup1", "unknown_c.root_0"]
    assert all("_status" not in v and "event_num" not in v for v in summary.values())
    on_disk = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert on_disk == summary


# ── 端到端 1：进程内 main()（argparse → 装载 → Pool → 汇总落盘）───────────────


def test_main_writes_summary_through_pool(tmp_path, rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls)

    txt = tmp_path / "list.txt"
    txt.write_text("Trigger_20250101_120000.root 0\nTrigger_20250101_120000.root 1\n",
                   encoding="utf-8")
    out = tmp_path / "summary.yaml"

    captured = {}

    class SyncPool:
        def __init__(self, processes=None):
            captured["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks):
            captured["tasks"] = list(tasks)
            return [fn(t) for t in tasks]

    monkeypatch.setattr(d, "Pool", SyncPool)
    monkeypatch.setattr(sys, "argv", [
        "driver", "--input", str(txt), "--base-path", str(tmp_path),
        "--output", str(out), "--ncores", "2", "--det-pos", "pos.txt",
    ])

    d.main()

    assert captured["processes"] == 2
    assert [t.event_index for t in captured["tasks"]] == [0, 1]
    assert all(t.resume is True and t.det_pos_path == "pos.txt" for t in captured["tasks"])
    summary = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(summary) == 2 and all("_status" not in v for v in summary.values())


def test_main_honours_no_resume_and_default_output(tmp_path, rec_root, monkeypatch):
    calls = []
    _stub_physics(monkeypatch, calls)
    # 两个事例的产物都已落盘：默认应全部命中续跑
    for idx in (0, 1):
        _write_artifacts(str(rec_root / f"Trigger_20250101_120000_event_{idx}"))

    txt = tmp_path / "Trigger_list.txt"
    txt.write_text("Trigger_20250101_120000.root 0\nTrigger_20250101_120000.root 1\n",
                   encoding="utf-8")

    captured = {}

    class SyncPool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks):
            captured["tasks"] = list(tasks)
            return [fn(t) for t in tasks]

    monkeypatch.setattr(d, "Pool", SyncPool)
    monkeypatch.setattr(sys, "argv", ["driver", "--input", str(txt),
                                      "--base-path", str(tmp_path), "--no-resume"])
    monkeypatch.chdir(tmp_path)

    d.main()

    assert all(t.resume is False for t in captured["tasks"])
    assert [c["fn"] for c in calls] == ["efield", "energy", "efield", "energy"]
    # 默认输出名 = {输入 stem}_reconstruction_summary.yaml，落在 CWD
    assert (tmp_path / "Trigger_list_reconstruction_summary.yaml").is_file()


# ── 门禁：0 个事例 ≠ 通过 ───────────────────────────────────────────────────


def test_adf_equivalence_gate_refuses_zero_cases():
    """等价性门禁必须区分"比较了 0 个"与"全部相等"。

    2026-11-13 的拆分 review 抓到过反例：TARGET 改指新模块后每个事例都
    AttributeError、被每事例 except 吞掉、比较 0 个却打印 PASS。这条测试把
    "说不出比了几个"钉成失败，读不到数据时必须是 NOT RUN 且退出非 0。
    """
    proc = subprocess.run(
        [sys.executable, os.path.join(MODULE_DIR, "tests", "run_adf_equiv.py"),
         "--root", "/dev/null", "--coords", "/dev/null", "--baseline", "/dev/null"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0, proc.stdout[-800:]
    assert "NOT RUN" in proc.stdout, proc.stdout[-800:]


# ── 端到端 2：真实子进程，只走断点续跑（无需 parameter_files/）───────────────


def test_driver_script_resume_end_to_end(tmp_path):
    workdir = tmp_path / "work"
    rec = _write_artifacts(str(workdir / "Reconstruction" / "sample_20250101_120000_event_0"),
                           event_num=7)
    assert os.path.isdir(rec)

    txt = workdir / "list.txt"
    txt.write_text("sample_20250101_120000.root 0\n", encoding="utf-8")
    out = workdir / "summary.yaml"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, MODULE_DIR] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    proc = subprocess.run(
        [sys.executable, DRIVER, "--input", str(txt), "--base-path", str(workdir),
         "--output", str(out), "--ncores", "1"],
        cwd=str(workdir), env=env, capture_output=True, text=True, timeout=300,
    )

    assert proc.returncode == 0, proc.stderr[-1500:]
    summary = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert list(summary) == [7]
    entry = summary[7]
    assert entry["PWF"]["rec_theta_plane"] == pytest.approx(30.0)
    assert entry["SWF"]["rec_z_xmax"] == pytest.approx(8000.0)
    assert entry["antennas"]["du_ids"] == [1000, 1001, 1002]
    assert "_status" not in entry
