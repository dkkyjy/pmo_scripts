"""U2 数值等价门禁：`5f0040f^` 的 ADF 变体 vs 当前 `_adf_minuit`（真实 CoREAS 事例）。

判据（KTD-7）：同一事例、同一初值下
    |Δtheta|,|Δphi| < 1e-6 deg, |Δdw|/dw < 1e-6, |Δfval|/fval < 1e-6, 以及 rec_x/y/z_xmax 一致。

用法：
    python reconstruction_candidate/tests/run_adf_equiv.py \
        --root /home/dkk/grand/DC2_Coreas/RFChain_v2/COREAS-AN/sim_..._0000/adc_0-99_L1_0000.root \
        --coords du_coords_CoreasDC2.txt --events 5 \
        --baseline reconstruction_candidate/tests/_baseline_5f0040f_parent.py

两次实现都吃**同一份**由当前 reader 读出的数组，所以差异只可能来自拟合本身。
基线模块在临时目录里被单独 import（兄弟模块用软链），不改动仓库。
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np

MOD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "reconstruction_direction.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_baseline(baseline_path):
    tmpdir = tempfile.mkdtemp(prefix="adf_equiv_")
    for entry in os.listdir(MOD_DIR):
        if entry in (TARGET, "__pycache__", "tests"):
            continue
        os.symlink(os.path.join(MOD_DIR, entry), os.path.join(tmpdir, entry))
    dst = os.path.join(tmpdir, TARGET)
    shutil.copyfile(baseline_path, dst)
    sys.path.insert(0, tmpdir)
    try:
        return load_module(dst, "rc_baseline"), tmpdir
    finally:
        sys.path.pop(0)


def read_coords(path):
    """du_id -> (x, y, z)"""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            if len(parts) < 4:
                continue
            try:
                out[int(float(parts[0]))] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return out


def event_arrays(cur, coords, root, index):
    """读一个事例，返回 (x, y, z, t_ns, p) 或 None（天线不足）。"""
    ev, du_ids, gps, du_ns, lst, tr1, tr2, tr3 = cur.read_root_file_from_path(root, index)
    ids, xs, ys, zs, ts, ps = [], [], [], [], [], []
    for k, du in enumerate(du_ids):
        du = int(du)
        if du not in coords:
            continue
        tr = np.vstack([np.atleast_2d(tr1[k]), np.atleast_2d(tr2[k]), np.atleast_2d(tr3[k])])
        amp = float(np.nanmax(np.abs(tr)))
        t = float(du_ns[k])
        if not np.isfinite(amp) or amp <= 0 or not np.isfinite(t):
            continue
        x, y, z = coords[du]
        ids.append(du); xs.append(x); ys.append(y); zs.append(z); ts.append(t); ps.append(amp)
    if len(ids) < 5:
        return None
    return (np.array(xs), np.array(ys), np.array(zs), np.array(ts), np.array(ps))


def compare(cur, base, arrays, label):
    x, y, z, t, p = arrays
    init = (30.0, 180.0, 1.0)          # 两边同一初值；物理值不重要，等价性才重要
    kw = dict(init_input=init, obs_altitude=0.0)
    r_cur = cur.recons_direction_swf_adf(x, y, z, t, p, **kw)
    r_base = base.recons_direction_swf_adf(x, y, z, t, p, **kw)

    d_theta = abs(r_cur[7] - r_base[7])
    d_phi = abs(r_cur[8] - r_base[8])
    d_dw = abs(r_cur[9] - r_base[9]) / max(abs(r_base[9]), 1e-12)
    d_f = abs(r_cur[11] - r_base[11]) / max(abs(r_base[11]), 1e-12)
    dx = max(abs(r_cur[3] - r_base[3]), abs(r_cur[4] - r_base[4]), abs(r_cur[5] - r_base[5]))
    ok = d_theta < 1e-6 and d_phi < 1e-6 and d_dw < 1e-6 and d_f < 1e-6 and dx < 1e-6
    print(f"{label} n_det={len(x):3d} "
          f"theta {r_base[7]:10.6f}->{r_cur[7]:10.6f} (d={d_theta:.2e}) "
          f"phi d={d_phi:.2e} dw d={d_dw:.2e} fval {r_base[11]:12.4f}->{r_cur[11]:12.4f} "
          f"(d={d_f:.2e}) xmax d={dx:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--events", type=int, default=5)
    ap.add_argument("--index-start", type=int, default=0)
    args = ap.parse_args()

    if MOD_DIR not in sys.path:
        sys.path.insert(0, MOD_DIR)
    # 新目标模块按扁平顶层名 import（logger_config 等），需要仓库根在 sys.path 上；
    # 这份路径由本工具自己负责，不写进被加载的模块。
    repo_root = os.path.dirname(MOD_DIR)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    cur = load_module(os.path.join(MOD_DIR, TARGET), "rc_current")
    # 读 ROOT 的能力留在旧模块（新模块刻意不做 I/O）：比较用 cur，读事例用 reader。
    reader = load_module(os.path.join(MOD_DIR, "reconstruction_io.py"), "rc_reader")
    base, tmpdir = load_baseline(args.baseline)
    coords = read_coords(args.coords)

    ok_all, done = True, 0
    try:
        for i in range(args.index_start, args.index_start + args.events):
            try:
                arrays = event_arrays(reader, coords, args.root, i)
            except (OSError, ValueError, IndexError, KeyError) as exc:   # 数据侧问题：跳过并计数
                print(f"event[{i}] read failed: {exc}")
                continue
            # 工具自身的 AttributeError/TypeError 不再被吞：那是 harness 坏了，不是数据坏了
            if arrays is None:
                print(f"event[{i}] skipped (n_det<5)")
                continue
            done += 1
            ok_all &= compare(cur, base, arrays, f"event[{i}]")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 0 个事例不是 PASS：这次 review 的 P1 正是"比较了 0 个却打印 PASS"。
    if done == 0:
        print(f"\nRESULT: NOT RUN (0 cases)  请检查 --root / --coords / --baseline")
        return 1
    if not ok_all:
        print(f"\nRESULT: MISMATCH  ({done}/{args.events} 事例参与对比)")
        return 1
    if done < args.events:
        print(f"\nRESULT: PARTIAL  ({done}/{args.events} 事例参与对比，部分事例未读出)")
        return 1
    print(f"\nRESULT: PASS  ({done}/{args.events} 事例参与对比)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
