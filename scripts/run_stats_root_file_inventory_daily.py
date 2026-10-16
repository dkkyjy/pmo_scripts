#!/usr/bin/env python3
"""Run root-file-inventory script over an overall date range in daily chunks.

Range semantics:
- ``start_date`` is inclusive
- ``end_date`` is exclusive
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence


SCRIPT_REL_PATH = "scripts/stats_root_file_inventory.py"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split date range into daily chunks and run "
            "stats_root_file_inventory.py for each day."
        )
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date, e.g. 20260101 or 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive end date, e.g. 20270101 or 2027-01-01",
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="Base path passed to the inventory script",
    )
    parser.add_argument(
        "--output-base",
        default='../Reco_Dir/root_file_inventory',
        help="Base output directory; per-day output goes under YYYY/MM/DD subdirectories",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default="python",
        help="Python executable to run the downstream script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )
    return parser.parse_args(argv)


def normalize_date(raw: str) -> date:
    """Parse YYYYMMDD or YYYY-MM-DD into a date object."""
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {raw}")


def yyyymmdd(value: date) -> str:
    """Format date as YYYYMMDD for downstream scripts."""
    return value.strftime("%Y%m%d")


def ensure_script_exists(repo_dir: Path) -> None:
    """Validate the dependent script exists before processing."""
    script_path = repo_dir / SCRIPT_REL_PATH
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {SCRIPT_REL_PATH}")


def run_day(
    *,
    script_path: Path,
    python_bin: str,
    base_path: str,
    output_base: str,
    day: date,
    dry_run: bool,
) -> None:
    """Run the inventory script for a single day."""
    day_end = day + timedelta(days=1)
    print(f"Processing day: {yyyymmdd(day)}")

    day_output_dir = str(
        Path(output_base) / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
    )

    cmd = [
        python_bin,
        str(script_path),
        "--base-path", base_path,
        "--start-date", yyyymmdd(day),
        "--end-date", yyyymmdd(day_end),
        "--hourly-summary",
        "--output-dir", day_output_dir,
    ]
    cmd_text = shlex.join(cmd)

    if dry_run:
        print(f"DRY-RUN: {cmd_text}")
    else:
        print(f"RUN: {cmd_text}")
        subprocess.run(cmd, check=True)


def main(argv: Sequence[str]) -> int:
    """Program entrypoint."""
    args = parse_args(argv)

    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    if start_date >= end_date:
        raise ValueError("--start-date must be < --end-date (end-date is exclusive)")

    repo_dir = Path(__file__).resolve().parent.parent
    ensure_script_exists(repo_dir)
    script_path = repo_dir / SCRIPT_REL_PATH

    current = start_date
    failed_days: list[str] = []
    while current < end_date:
        try:
            run_day(
                script_path=script_path,
                python_bin=args.python_bin,
                base_path=args.base_path,
                output_base=args.output_base,
                day=current,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            day_text = yyyymmdd(current)
            print(f"Error processing day {day_text}: {exc}", file=sys.stderr)
            failed_days.append(day_text)
        current += timedelta(days=1)

    if failed_days:
        print(
            f"Done with {len(failed_days)} failed day(s): {', '.join(failed_days)}",
            file=sys.stderr,
        )
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)