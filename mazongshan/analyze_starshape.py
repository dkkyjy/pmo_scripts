#!/usr/bin/env python3
"""GP300 StarShape ROOT 文件探索性分析 (Mazongshan / Proton).

数据来源: GRAND GP300 StarShape ZHAireS 模拟, 每个 ROOT 文件含 3 棵树:
  - SimuCollection  / SimShowerBranch   : 原初/簇射真值 (能量, 天顶/方位角, Xmax, 磁场...)
  - EfieldCollection/ SimEfieldBranch    : 星形天线阵列的电场波形
  - SignalCollection/ SimSignalBranch    : 天线电压 p2p / t_0 及电压波形

参考同目录 READ.ME 与 _Readfile.cpp (Ma PengXiong, 2021)。

用法 (在 pmo001 上, py310 环境):
  /home/dkk/anaconda3/envs/py310/bin/python analyze_starshape.py \
      --file /home/mapx/mapx/GP300_StarShape_ROOTFiles/Proton/Proton_EeV_Mazongshan_stshp_2100Events.root \
      --out  /home/dkk/grand/mazongshan_test

为避免加载 12GB 波形数据, 本脚本仅读取标量分支 + p2p, 属于轻量探索。
"""
import argparse
import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FILE = (
    "/home/mapx/mapx/GP300_StarShape_ROOTFiles/Proton/"
    "Proton_EeV_Mazongshan_stshp_2100Events.root"
)


def summarize(f, outdir, filepath):
    lines = []

    def P(s):
        print(s)
        lines.append(s)

    P(f"=== {os.path.basename(filepath)} ===")

    # ---------- SimShower: 簇射真值 ----------
    sh = f["SimuCollection"]["SimShowerBranch"]
    d = sh.arrays(["evt_id", "prim_energy", "shower_zenith", "shower_azimuth",
                   "xmax_grams", "xmax_alt", "ground_alt", "magnetic_field",
                   "xmax_pos_shc"], library="np")
    E = np.asarray(d["prim_energy"], float)
    zen = np.asarray(d["shower_zenith"], float)
    azi = np.asarray(d["shower_azimuth"], float)
    xmg = np.asarray(d["xmax_grams"], float)
    xma = np.asarray(d["xmax_alt"], float)
    xmax_pos = np.array([np.asarray(v, float) for v in d["xmax_pos_shc"]])
    norm = np.linalg.norm(xmax_pos, axis=1)

    P(f"events (SimuCollection) = {len(E)}")
    P(f"prim_energy [EeV]  : min={E.min():.3f} med={np.median(E):.3f} max={E.max():.3f}")
    P(f"shower_zenith[deg]: min={zen.min():.2f} med={np.median(zen):.2f} max={zen.max():.2f}")
    P(f"shower_azimuth[deg]:min={azi.min():.2f} med={np.median(azi):.2f} max={azi.max():.2f}")
    P(f"xmax_grams [g/cm2]: min={xmg.min():.1f} med={np.median(xmg):.1f} max={xmg.max():.1f}")
    P(f"xmax_alt [m]       : min={xma.min():.0f} med={np.median(xma):.0f} max={xma.max():.0f}")
    P(f"|xmax_pos_shc| [m] : min={norm.min():.0f} med={np.median(norm):.0f} max={norm.max():.0f}")
    P(f"ground_alt[0]={d['ground_alt'][0]}  magnetic_field[0]={np.asarray(d['magnetic_field'][0])}")

    # xmax_alt 离群计数 (哨兵/异常值)
    n_bad_alt = int((xma > 1e5).sum())
    if n_bad_alt:
        P(f"[warn] xmax_alt > 1e5 m 的异常事件数 = {n_bad_alt} (疑似哨兵值)")

    # ---------- Efield: 天线布局 ----------
    ef = f["EfieldCollection"]["SimEfieldBranch"]
    ep = ef.arrays(["Detectors_det_pos_shc"], library="np")
    pos0 = np.array([np.asarray(p, float) for p in ep["Detectors_det_pos_shc"][0]])
    r0 = np.hypot(pos0[:, 0], pos0[:, 1])
    P(f"\nEfield ndet[0]={len(pos0)}  |pos_xy| min={r0.min():.1f} "
      f"med={np.median(r0):.1f} max={r0.max():.1f} m")

    # ---------- Signal: p2p ----------
    sg = f["SignalCollection"]["SimSignalBranch"]
    sd = sg.arrays(["Detectors_p2p"], library="np")
    p2p_list = [np.asarray(x, float) for x in sd["Detectors_p2p"]]
    p2p_max = np.array([x.max() if len(x) else np.nan for x in p2p_list])
    npass = np.array([int((x > 75).sum()) for x in p2p_list])   # _Readfile.cpp 阈值
    P(f"p2p: n_det/event min={min(len(x) for x in p2p_list)} max={max(len(x) for x in p2p_list)}")
    P(f"p2p_max [ADC] : med={np.nanmedian(p2p_max):.0f} max={np.nanmax(p2p_max):.0f}")
    P(f"n(p2p>75)/event: min={npass.min()} med={int(np.median(npass))} max={npass.max()}")

    # ---------- 图 ----------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(E, bins=50, color="C0"); ax[0, 0].set_xlabel("prim_energy [EeV]"); ax[0, 0].set_title("Energy")
    ax[0, 1].hist(zen, bins=50, color="C1"); ax[0, 1].set_xlabel("shower_zenith [deg]"); ax[0, 1].set_title("Zenith")
    ax[0, 2].hist(azi, bins=50, color="C2"); ax[0, 2].set_xlabel("shower_azimuth [deg]"); ax[0, 2].set_title("Azimuth")
    ax[1, 0].hist(xmg, bins=50, color="C3"); ax[1, 0].set_xlabel("Xmax [g/cm2]"); ax[1, 0].set_title("Xmax (slant depth)")
    ax[1, 1].scatter(pos0[:, 0], pos0[:, 1], s=8, c=r0, cmap="viridis"); ax[1, 1].set_aspect("equal")
    ax[1, 1].set_xlabel("W-E [m]"); ax[1, 1].set_ylabel("S-N [m]"); ax[1, 1].set_title("Antenna layout (star-shape, evt0)")
    ax[1, 2].hist(p2p_max, bins=50, color="C4"); ax[1, 2].set_xlabel("max p2p [ADC]"); ax[1, 2].set_title("Max p2p per event")
    plt.tight_layout()
    png = os.path.join(outdir, "proton_mazongshan_overview.png")
    plt.savefig(png, dpi=120)
    P(f"\nwrote {png}")

    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(outdir, "report.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--out", default="mazongshan_test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with uproot.open(args.file) as f:
        summarize(f, args.out, args.file)


if __name__ == "__main__":
    main()
