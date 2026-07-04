#!/usr/bin/env python3
"""单日数据处理完整流水线。

流程：
  1. 检查 TD/YYYY/MM/DD 目录是否存在及文件数
  2. stats_root_file_inventory.py — 文件数量和大小统计
  3. loop.py — 当天完整重建（并行）
  4. run_enrich_and_filter.py — 添加 signal 信息 + 挑选 slope>0 的事例（并行）
  5. run_merge_date.py — 合并数据（header + trace + results）

用法：
  python scripts/run_daily_pipeline.py 20260617
  python scripts/run_daily_pipeline.py 2026-06-17
  python scripts/run_daily_pipeline.py 2026/06/17
  python scripts/run_daily_pipeline.py 20260617 --jobs 80

环境变量（均可选）：
  BASE_PATH       — TD 根目录（默认 /mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD）
  RECO_DIR        — 重建输出目录（默认 ../Reco_Dir）
  JOBS            — 并行核数（默认 40，也可用 --jobs 参数指定）
  MIN_DETECTOR    — 匹配阶段最少探测器数（默认 6，也可用 --min-detector 参数指定）
  SKIP_INVENTORY  — 设为 1 跳过文件清点
  SKIP_LOOP       — 设为 1 跳过 loop.py
  SKIP_ENRICH     — 设为 1 跳过 enrich + slope filter
  SKIP_MERGE      — 设为 1 跳过 merge
  DRY_RUN         — 设为 1 只打印命令不执行
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_BASE_PATH = "/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD"


def parse_date(raw: str) -> str:
    """将 YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD 统一为 YYYYMMDD。"""
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", raw)
    if m:
        return m.group(1) + m.group(2) + m.group(3)
    if re.match(r"^\d{8}$", raw):
        return raw
    raise ValueError(
        f"Unsupported date format: {raw!r}. "
        f"Expected YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD."
    )


def human_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的大小字符串。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def resolve_repo_root() -> Path:
    """返回仓库根目录（scripts/ 的父目录）。"""
    return Path(__file__).resolve().parent.parent


def run_cmd(
    desc: str,
    cmd: List[str],
    *,
    dry_run: bool = False,
    date_label: str = "",
) -> int:
    """打印并执行命令。dry_run 时只打印不执行。"""
    from datetime import datetime

    ts = datetime.now().strftime("%H:%M:%S")
    print()
    print(f"[{ts}] >>> [{date_label}] {desc}")
    print(f"[{ts}] >>> CMD: {' '.join(cmd)}")
    if dry_run:
        print(f"[{ts}] >>> [DRY-RUN] skipped")
        return 0
    timeout = int(os.environ.get("CMD_TIMEOUT", "0")) or None
    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f">>> ERROR: command timed out after {timeout}s")
        return 124  # standard timeout exit code
    rc = result.returncode
    if rc != 0:
        print(f">>> ERROR: command failed with exit code {rc}")
    else:
        print(f">>> DONE: {desc}")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-day data processing pipeline"
    )
    parser.add_argument(
        "date",
        help="Date: YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD",
    )
    parser.add_argument(
        "--base-path",
        default=os.environ.get("BASE_PATH", DEFAULT_BASE_PATH),
        help="TD root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--reco-dir",
        default=os.environ.get("RECO_DIR", ""),
        help="Reconstruction output directory (default: ../Reco_Dir relative to repo)",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", "python"),
        help="Python interpreter (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("JOBS", "40")),
        help="Number of parallel workers for loop/enrich/slope steps (default: 40)",
    )
    parser.add_argument(
        "--min-detector",
        type=int,
        default=int(os.environ.get("MIN_DETECTOR", "6")),
        help="Minimum number of detectors for matching stage (default: 6)",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    # ── 日期解析 ────────────────────────────────────────────────────────
    try:
        yyyymmdd = parse_date(args.date)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    year = yyyymmdd[:4]
    month = yyyymmdd[4:6]
    day = yyyymmdd[6:8]
    date_slash = f"{year}/{month}/{day}"

    # ── 路径 ────────────────────────────────────────────────────────────
    repo_root = resolve_repo_root()
    base_path = args.base_path
    reco_dir = args.reco_dir or str(repo_root / ".." / "Reco_Dir")
    reco_dir = os.path.abspath(reco_dir)

    td_day_dir = os.path.join(base_path, date_slash)
    reco_day_dir = os.path.join(reco_dir, date_slash)
    inventory_output_dir = os.path.join(reco_dir, "root_file_inventory", date_slash)

    python_exe = args.python

    # 环境变量
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    skip_inventory = os.environ.get("SKIP_INVENTORY", "0") == "1"
    skip_loop = os.environ.get("SKIP_LOOP", "0") == "1"
    skip_enrich = os.environ.get("SKIP_ENRICH", "0") == "1"
    skip_merge = os.environ.get("SKIP_MERGE", "0") == "1"

    print("=" * 60)
    print(f"  Daily Pipeline: {yyyymmdd}")
    print("=" * 60)
    print(f"  TD dir:       {td_day_dir}")
    print(f"  Reco dir:     {reco_dir}")
    print(f"  Day subdir:   {reco_day_dir}")
    print(f"  Python:       {python_exe}")
    print(f"  Jobs:         {args.jobs}")
    print(f"  Min detector: {args.min_detector}")
    print(f"  Dry run:      {1 if dry_run else 0}")
    print("=" * 60)

    # ── Step 0: 检查 TD 目录 ────────────────────────────────────────────
    print()
    print("--- Step 0: Check TD directory ---")
    if not os.path.isdir(td_day_dir):
        print(f"ERROR: TD directory does not exist: {td_day_dir}")
        print("  (Data may not have arrived yet for this date)")
        sys.exit(1)

    trigger_count = len(list(Path(td_day_dir).glob("Trigger*.root")))
    calib_count = len(list(Path(td_day_dir).glob("Calibration*.root")))

    total_size = sum(
        entry.stat().st_size
        for entry in Path(td_day_dir).iterdir()
        if entry.is_file()
    )

    print(f"  Trigger files:     {trigger_count}")
    print(f"  Calibration files: {calib_count}")
    print(f"  Total size:        {human_size(total_size)}")

    if trigger_count == 0:
        print(f"WARNING: No Trigger files found in {td_day_dir}")
        print("  (Only Calibration or no data — pipeline cannot proceed with reconstruction)")
        sys.exit(0)

    # ── Step 1: 文件清点 ────────────────────────────────────────────────
    if not skip_inventory:
        rc = run_cmd(
            "Step 1: Root file inventory",
            [
                python_exe,
                str(repo_root / "scripts" / "stats_root_file_inventory.py"),
                "--base-path", os.path.dirname(base_path),
                "--date", f"{year}-{month}-{day}",
                "--output-dir", inventory_output_dir,
                "--daily-summary",
                "--hourly-summary",
            ],
            dry_run=dry_run,
            date_label=yyyymmdd,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print(">>> [SKIP] Step 1: Root file inventory")

    # ── Step 2: loop.py 当天完整重建（并行） ──────────────────────────────
    if not skip_loop:
        start_dt = f"{year}-{month}-{day}T00:00:00"
        end_dt = f"{year}-{month}-{day}T23:59:59"
        rc = run_cmd(
            "Step 2: Run loop.py (full day, parallel)",
            [
                python_exe,
                str(repo_root / "loop.py"),
                start_dt,
                end_dt,
                "--base-path", base_path,
                "--out-dir-base", reco_dir,
                "--min-detectors", str(args.min_detector),
                "--run",
                "--jobs", str(args.jobs),
            ],
            dry_run=dry_run,
            date_label=yyyymmdd,
        )
        rc = run_cmd(
            "Step 2: Run loop.py (full day, parallel)",
            [
                python_exe,
                str(repo_root / "loop.py"),
                start_dt,
                end_dt,
                "--base-path", base_path,
                "--out-dir-base", reco_dir,
                "--min-detectors", str(args.min_detector),
                "--with-signal",
                "--only-read",
                "--run",
                "--jobs", str(args.jobs),
            ],
            dry_run=dry_run,
            date_label=yyyymmdd,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print(">>> [SKIP] Step 2: loop.py")

    # ── Step 3: 添加 signal 信息 + slope 候选筛选（并行） ──────────────
    if not skip_enrich:
        rc = run_cmd(
            "Step 3: Enrich SWM with signal + slope filter (parallel)",
            [
                python_exe,
                str(repo_root / "scripts" / "run_enrich_and_filter.py"),
                "--reco-dir", reco_dir,
                "--date", yyyymmdd,
                "--overwrite",
                "--min-du-count", "6",
                "--jobs", str(args.jobs),
            ],
            dry_run=dry_run,
            date_label=yyyymmdd,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print(">>> [SKIP] Step 3: Enrich SWM with signal + slope filter")

    # ── Step 4: 合并数据 ────────────────────────────────────────────────
    if not skip_merge:
        rc = run_cmd(
            "Step 4: Merge data (header + trace + results)",
            [
                python_exe,
                str(repo_root / "scripts" / "run_merge_date.py"),
                "--target", "all",
                "--out-dir-base", reco_dir,
                "--run",
                yyyymmdd,
            ],
            dry_run=dry_run,
            date_label=yyyymmdd,
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print(">>> [SKIP] Step 4: Merge data")

    # ── 完成 ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Pipeline complete: {yyyymmdd}")
    print(f"  Output dir: {reco_dir}")
    print("  Merged files:")
    merged_files = list(Path(reco_dir).glob(
        f"Trigger_{yyyymmdd}_RUN*_merged*.yaml"
    ))
    if merged_files:
        for f in merged_files:
            sz = f.stat().st_size
            print(f"    {f.name}  ({human_size(sz)})")
    else:
        print("    (none)")
    # Check for candidate YAML files from enrich_and_filter
    cand_files = list(Path(reco_dir).glob(
        f"Trigger_{yyyymmdd}_RUN*_candidates.yaml"
    ))
    if cand_files:
        print(f"  Candidate YAML files: {len(cand_files)}")
    else:
        print("    (no candidate YAML files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
