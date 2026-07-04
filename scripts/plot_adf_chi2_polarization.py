#!/usr/bin/env python3
"""
从 *_candidates_reconstruction_summary.yaml 中提取 ADF 的 chi2_per_dof（reduced chi2）
和 polarization（列表均值），画分布直方图和散点图。

用法:
    python scripts/plot_adf_chi2_polarization.py Trigger_20260604_RUN10380_candidates_reconstruction_summary.yaml
    python scripts/plot_adf_chi2_polarization.py Trigger_20260604_RUN10380_candidates_reconstruction_summary.yaml --chi2-cut 15 --pol-cut 40
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import scienceplots
plt.style.use(['science', 'grid', 'notebook'])


def load_and_extract(yaml_path: str) -> tuple[list[dict], list[float], list[float], list[float], list[float], list[float], list[int], list[dict]]:
    """读取 summary YAML，提取 reduced chi2、polarization 均值、所有单个值、omega_rad、每个事件的 DU 数。

    返回:
        records:       每个事件的信息列表
        reduced_chi2, pol_means, pol_all, omega_all, omega_pol_pairs_x, omega_pol_pairs_y, pol_lens: 数值数组
        adf_failed:    ADF 拟合失败（参数为 null）的事例列表
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    records: list[dict] = []
    adf_failed: list[dict] = []
    reduced_chi2: list[float] = []
    pol_means: list[float] = []
    pol_all: list[float] = []
    omega_all: list[float] = []
    omega_pol_x: list[float] = []   # omega_rad (for scatter vs polarization)
    omega_pol_y: list[float] = []   # polarization
    pol_lens: list[int] = []

    for evt_key, evt in data.items():
        # 提取 ADF 参数
        adf = evt.get("ADF", {})
        fmin = adf.get("chi2_per_dof")

        # 检测 ADF 拟合失败：关键参数为 null
        adf_params = [
            adf.get("rec_A"),
            adf.get("rec_wc"),
            adf.get("rec_dw"),
        ]
        if any(p is None for p in adf_params):
            adf_failed.append({
                'evt_key': evt_key,
                'file_path': evt.get('file_path', ''),
                'event_index': evt.get('event_index', 0),
            })
            continue

        if fmin is None:
            continue

        # 提取 antennas 中的 polarization / omega_rad
        ants = evt.get("antennas")
        pol_list = ants.get("polarization")
        omega_list = ants.get("omega_rad")
        if pol_list is None or len(pol_list) == 0:
            continue

        pol_mean = float(np.mean(pol_list))
        reduced_chi2.append(float(fmin))
        pol_means.append(pol_mean)
        pol_all.extend([float(v) for v in pol_list])
        pol_lens.append(len(pol_list))

        if omega_list:
            omega_all.extend([float(v) for v in omega_list])
            # 配对的 polarization-omega_rad 散点数据（按 DU 对齐）
            n = min(len(pol_list), len(omega_list))
            for i in range(n):
                omega_pol_x.append(float(omega_list[i]))
                omega_pol_y.append(float(pol_list[i]))

        # 判断该事件是否所有 DU 的 omega_rad 全在 rec_wc 同一侧
        rec_wc = float(adf.get("rec_wc"))
        omega_all_one_side = False
        if omega_list and len(omega_list) > 0:
            omega_vals = [float(v) for v in omega_list]
            omega_all_one_side = all(v > rec_wc for v in omega_vals) or all(v < rec_wc for v in omega_vals)

        records.append({
            'evt_key': evt_key,
            'chi2': float(fmin),
            'pol_mean': pol_mean,
            'file_path': evt.get('file_path', ''),
            'event_index': evt.get('event_index', 0),
            'omega_all_one_side': omega_all_one_side,
        })

    return records, reduced_chi2, pol_means, pol_all, omega_all, omega_pol_x, omega_pol_y, pol_lens, adf_failed


def parse_args():
    p = argparse.ArgumentParser(description="画 ADF reduced chi2 与 polarization 分布图")
    p.add_argument("yaml_file", help="*_candidates_reconstruction_summary.yaml 路径")
    p.add_argument("--chi2-max", type=float, default=100,
                   help="chi2 直方图的 x 轴上限（默认自动）")
    p.add_argument("--pol-max", type=float, default=90,
                   help="polarization 直方图的 x 轴上限（默认自动）")
    p.add_argument("--chi2-bins", type=int, default=60,
                   help="chi2 直方图 bins 数（默认 60）")
    p.add_argument("--pol-bins", type=int, default=60,
                   help="polarization 直方图 bins 数（默认 60）")
    p.add_argument("-o", "--output", default="",
                   help="输出图片基名（默认与输入 yaml 同名）")
    p.add_argument("--recon-dir", default="./Reconstruction",
                   help="Reconstruction 事例目录路径（默认 ./Reconstruction）")
    p.add_argument("--no-delete", action="store_true",
                   help="仅打印将要删除的目录，不实际删除")
    p.add_argument("--chi2-cut", type=float, default=10,
                   help="chi2 筛选阈值，超过则删除事例（默认 10）")
    p.add_argument("--pol-cut", type=float, default=30,
                   help="polarization mean 筛选阈值，超过则删除事例（默认 30）")
    return p.parse_args()


def main():
    args = parse_args()
    yaml_path = Path(args.yaml_file)

    records, reduced_chi2, pol_means, pol_all, omega_all, omega_pol_x, omega_pol_y, pol_lens, adf_failed = load_and_extract(str(yaml_path))
    print(f"读取事件数: {len(reduced_chi2)}")
    print(f"chi2:      min={np.min(reduced_chi2):.3f},  max={np.max(reduced_chi2):.3f},  "
          f"median={np.median(reduced_chi2):.3f},  mean={np.mean(reduced_chi2):.3f}")
    print(f"pol_mean:  min={np.min(pol_means):.3f},  max={np.max(pol_means):.3f},  "
          f"median={np.median(pol_means):.3f},  mean={np.mean(pol_means):.3f}")
    print(f"pol_all:   N={len(pol_all)},  min={np.min(pol_all):.3f},  max={np.max(pol_all):.3f},  "
          f"median={np.median(pol_all):.3f},  mean={np.mean(pol_all):.3f}")
    if omega_all:
        print(f"omega_all: N={len(omega_all)},  min={np.min(omega_all):.6f},  max={np.max(omega_all):.6f},  "
              f"median={np.median(omega_all):.6f},  mean={np.mean(omega_all):.6f}")

    # ── 画图前：删除 ADF 拟合失败的事例（参数为 null） ──
    if adf_failed:
        recon_dir = os.path.abspath(args.recon_dir)
        print(f"\nADF 拟合失败事例数: {len(adf_failed)}")
        if args.no_delete:
            print("--no-delete 模式，仅列出将要删除的目录：")
            for rec in adf_failed:
                basename = os.path.splitext(os.path.basename(rec['file_path']))[0]
                dir_name = f"{basename}_event_{int(rec['event_index'])}"
                dir_path = os.path.join(recon_dir, dir_name)
                print(f"  [DRY-RUN] {dir_path}")
        else:
            deleted = 0
            for rec in adf_failed:
                basename = os.path.splitext(os.path.basename(rec['file_path']))[0]
                dir_name = f"{basename}_event_{int(rec['event_index'])}"
                dir_path = os.path.join(recon_dir, dir_name)
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    deleted += 1
            print(f"  实际删除 {deleted} 个 ADF 失败事例目录")

    out_base = args.output if args.output else yaml_path.stem

    # ── 3×2 六子图布局 ──
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.28)

    # 图 1：reduced chi2 分布直方图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(np.clip(reduced_chi2, 0, args.chi2_max), bins=args.chi2_bins,
             color="steelblue", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("ADF reduced χ²")
    ax1.set_ylabel("Events")
    ax1.set_title(f"Reduced χ² (N={len(reduced_chi2)})")
    ax1.axvline(np.median(reduced_chi2), color="red", ls="--", lw=2,
                label=f"median={np.median(reduced_chi2):.1f}")
    ax1.legend()

    # 图 2：所有 polarization 值的整体分布
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(pol_all, bins=args.pol_bins,
             color="mediumseagreen", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Polarization Δφ (°)")
    ax2.set_ylabel("Counts")
    ax2.set_title(f"All polarization values (N={len(pol_all)})")
    ax2.axvline(np.median(pol_all), color="red", ls="--", lw=2,
                label=f"median={np.median(pol_all):.1f}°")
    ax2.legend()

    # 图 3：polarization 均值分布直方图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(np.clip(pol_means, 0, args.pol_max), bins=args.pol_bins,
             color="darkorange", edgecolor="white", alpha=0.85)
    ax3.set_xlabel("Polarization mean Δφ (°)")
    ax3.set_ylabel("Events")
    ax3.set_title(f"Polarization mean (N={len(pol_means)})")
    ax3.axvline(np.median(pol_means), color="red", ls="--", lw=2,
                label=f"median={np.median(pol_means):.1f}°")
    ax3.legend()

    # 图 4：omega_rad 分布直方图（所有 antennas）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(omega_all, bins=args.chi2_bins,
             color="mediumorchid", edgecolor="white", alpha=0.85)
    ax4.axvline(np.median(omega_all), color="red", ls="--", lw=2,
                label=f"median={np.median(omega_all):.4f} rad")
    ax4.legend()
    ax4.set_xlabel("ω (rad)")
    ax4.set_ylabel("Counts")
    ax4.set_title(f"ω_rad per antenna (N={len(omega_all)})")

    # 图 5：scatter: reduced chi2 vs polarization mean
    ax5 = fig.add_subplot(gs[2, 0])
    sc5 = ax5.scatter(pol_means, reduced_chi2, c=pol_lens, cmap="viridis",
                      alpha=0.6, s=12, edgecolors="none")
    ax5.set_xlabel("Polarization mean Δφ (°)")
    ax5.set_ylabel("ADF reduced χ²")
    ax5.set_title(f"χ² vs Polarization mean (N={len(reduced_chi2)})")
    cbar5 = fig.colorbar(sc5, ax=ax5)
    cbar5.set_label("Number of DUs")
    ax5.axvline(args.pol_cut, color="red", ls="--", lw=2,
                label=f"pol = {args.pol_cut:.1f}°")
    ax5.axhline(args.chi2_cut, color="blue", ls="--", lw=2,
                label=f"χ² = {args.chi2_cut:.1f}")
    ax5.set_ylim(0, args.chi2_max)
    ax5.set_xlim(0, args.pol_max)
    ax5.legend()

    # 图 6：scatter: polarization vs omega_rad（每根天线一个点）
    ax6 = fig.add_subplot(gs[2, 1])
    sc6 = ax6.scatter(omega_pol_x, omega_pol_y, alpha=0.3, s=8, edgecolors="none", color="teal")
    ax6.set_xlabel("ω (rad)")
    ax6.set_ylabel("Polarization Δφ (°)")
    ax6.set_title(f"Polarization vs ω (N={len(omega_pol_x)} antennas)")
    ax6.set_title("Polarization vs ω (no data)")
    ax6.axvline(np.median(omega_all), color="red", ls="--", lw=2,
                label=f"omega = {np.median(omega_all):.4f}")
    ax6.axhline(np.median(pol_all), color="blue", ls="--", lw=2,
                label=f"pol = {np.median(pol_all):.1f}°")
    ax6.legend()

    # fig.suptitle(yaml_path.stem, fontsize=13, y=1.01)
    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_base}.png")

    recon_dir = os.path.abspath(args.recon_dir)
    print(f"\n筛选条件: reduced χ² > {args.chi2_cut:.1f}  OR  polarization mean > {args.pol_cut:.1f}°")
    print(f"Reconstruction 目录: {recon_dir}")

    to_delete: list[str] = []
    kept: int = 0
    for rec in records:
        if rec['chi2'] > args.chi2_cut or rec['pol_mean'] > args.pol_cut:
            basename = os.path.splitext(os.path.basename(rec['file_path']))[0]
            dir_name = f"{basename}_event_{int(rec['event_index'])}"
            dir_path = os.path.join(recon_dir, dir_name)
            to_delete.append(dir_path)
        else:
            kept += 1

    # ── 额外删除：所有 DU 的 omega_rad 全在 rec_wc 同一侧的事例 ──
    omega_delete: list[str] = []
    for rec in records:
        if rec.get('omega_all_one_side'):
            basename = os.path.splitext(os.path.basename(rec['file_path']))[0]
            dir_name = f"{basename}_event_{int(rec['event_index'])}"
            dir_path = os.path.join(recon_dir, dir_name)
            if dir_path not in to_delete:
                omega_delete.append(dir_path)

    omega_one_side_count = sum(1 for rec in records if rec.get('omega_all_one_side'))
    print(f"ω 全在 rec_wc 同侧事例数: {omega_one_side_count},  额外删除: {len(omega_delete)}（已在前述删除列表的除外）")
    to_delete.extend(omega_delete)

    print(f"保留: {kept},  删除: {len(to_delete)}")

    if not to_delete:
        print("无需删除任何目录。")
        return

    if args.no_delete:
        print("\n--no-delete 模式，仅列出将要删除的目录：")
        for d in to_delete:
            print(f"  [DRY-RUN] {d}")
        return

    deleted = 0
    for dir_path in to_delete:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            deleted += 1
            # print(f"  已删除: {dir_path}")
    print(f"\n实际删除 {deleted} 个目录（{len(to_delete) - deleted} 个目录不存在）。")


if __name__ == "__main__":
    main()
