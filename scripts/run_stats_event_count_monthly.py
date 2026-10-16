#!/usr/bin/env python3
"""Run monthly event-count scripts over an overall date range.

Range semantics follow the original shell script:
- ``start_date`` is inclusive
- ``end_date`` is exclusive
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Sequence


SCRIPT_REL_PATHS: tuple[str, ...] = (
    "scripts/stats_run_merged_event_count.py",
    "scripts/stats_run_matched_event_count.py",
    "scripts/stats_run_fingerprint_event_count.py",
    "scripts/stats_run_stage_event_count_compare.py",
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split date range into monthly chunks and run merged/matched/"
            "fingerprint/stage event-count scripts for each chunk."
        )
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date, e.g. 20250701 or 2025-07-01",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive end date, e.g. 20260401 or 2026-04-01",
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Reco directory passed to python scripts",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default="python",
        help="Python executable to run downstream scripts",
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


def next_month_start(value: date) -> date:
    """Return the first day of the month after ``value``."""
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def yyyymmdd(value: date) -> str:
    """Format date as YYYYMMDD for downstream scripts."""
    return value.strftime("%Y%m%d")


def ensure_scripts_exist(repo_dir: Path) -> None:
    """Validate all dependent scripts exist before processing."""
    for rel_path in SCRIPT_REL_PATHS:
        script_path = repo_dir / rel_path
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {rel_path}")


def run_chunk(
    *,
    repo_dir: Path,
    python_bin: str,
    reco_dir: str,
    chunk_start: date,
    chunk_end: date,
    dry_run: bool,
) -> None:
    """Run all target scripts for one monthly chunk."""
    print(f"Processing monthly chunk: [{yyyymmdd(chunk_start)}, "
          f"{yyyymmdd(chunk_end)})")

    for rel_path in SCRIPT_REL_PATHS:
        cmd = [
            python_bin,
            str(repo_dir / rel_path),
            "--reco-dir",
            reco_dir,
            "--start-date",
            yyyymmdd(chunk_start),
            "--end-date",
            yyyymmdd(chunk_end),
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
    ensure_scripts_exist(repo_dir)

    current = start_date
    while current < end_date:
        month_end = next_month_start(current)
        chunk_end = month_end if month_end <= end_date else end_date

        run_chunk(
            repo_dir=repo_dir,
            python_bin=args.python_bin,
            reco_dir=args.reco_dir,
            chunk_start=current,
            chunk_end=chunk_end,
            dry_run=args.dry_run,
        )
        current = chunk_end

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
