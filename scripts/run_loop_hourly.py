#!/usr/bin/env python3
"""Run loop.py hour by hour for one day.

For each hour, the script first counts matching Trigger ROOT files, then
invokes loop.py with the requested time range and extra arguments.
"""

from __future__ import annotations

import sys
import subprocess
import argparse
import datetime as dt
from pathlib import Path

from logger_config import logger


DEFAULT_BASE_PATH = "/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run loop.py once per hour and count matching files first."
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Base directory containing yyyy/mm/dd ROOT files.",
    )
    parser.add_argument(
        "base_day",
        help="Target day in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format.",
    )
    args = parser.parse_args()

    # Normalize to YYYYMMDD
    raw = args.base_day.strip()
    # Try YYYY/MM/DD → YYYYMMDD
    if len(raw) == 10 and raw[4] == "/" and raw[7] == "/":
        args.base_day = raw.replace("/", "")
    # Try YYYY-MM-DD → YYYYMMDD
    elif len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        args.base_day = raw.replace("-", "")
    # Already YYYYMMDD
    elif len(raw) == 8 and raw.isdigit():
        args.base_day = raw
    else:
        parser.error(
            f"Expected YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD, got: {raw}"
        )
    # Validate the date is real (e.g. reject 20250229)
    try:
        dt.datetime.strptime(args.base_day, "%Y%m%d")
    except ValueError:
        parser.error(f"Invalid date: {raw}")
    return args


def count_matching_files(
    hour_dir: Path,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> int:
    """Count Trigger ROOT files whose embedded timestamp falls within the range."""
    if not hour_dir.is_dir():
        return 0

    count = 0
    for entry in hour_dir.iterdir():
        if not (entry.name.endswith(".root") and "Trigger" in entry.name):
            continue
        parts = entry.name.split("_")
        if len(parts) < 2:
            continue
        try:
            file_time = dt.datetime.strptime(parts[1][:14], "%Y%m%d%H%M%S")
        except ValueError:
            continue
        if start_dt <= file_time < end_dt:
            count += 1
    return count


def build_hour_bounds(base_day: str, hour: int) -> tuple[dt.datetime, dt.datetime]:
    """Return the start datetime and the next-hour boundary."""
    start_time = f"{base_day}{hour:02d}"
    start_dt = dt.datetime.strptime(start_time, "%Y%m%d%H")
    end_dt = start_dt + dt.timedelta(hours=1)
    return start_dt, end_dt


MAX_JOBS = 80


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_path)

    if not base_path.is_dir():
        logger.error("Base path does not exist: {}", base_path)
        return 1

    day_root = base_path / args.base_day[:4] / args.base_day[4:6] / args.base_day[6:8]

    for hour in range(24):
        start_dt, end_dt = build_hour_bounds(args.base_day, hour)
        start_time = start_dt.strftime("%Y%m%d%H")
        end_time = end_dt.strftime("%Y%m%d%H")
        raw_count = count_matching_files(day_root, start_dt, end_dt)
        file_count = min(raw_count, MAX_JOBS)

        logger.info(
            "[{}-{}] files={} jobs={}",
            start_time,
            end_time,
            raw_count,
            file_count,
        )

        if file_count == 0:
            continue

        # First pass: header-only (no signal)
        cmd = [
            sys.executable,
            "loop.py",
            start_time,
            end_time,
            "--base-path",
            str(base_path),
            "--jobs",
            str(file_count),
            "--run",
        ]
        subprocess.run(cmd, check=True)

        # Second pass: signal-amplitude (trace) mode
        cmd_signal = cmd + ["--only-read", "--with-signal"]
        subprocess.run(cmd_signal, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
