#!/usr/bin/env python3
"""Run root-file-inventory script over an overall date range in monthly chunks.

Range semantics:
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


SCRIPT_REL_PATH = "scripts/stats_root_file_inventory.py"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Split date range into monthly chunks and run "
            "stats_root_file_inventory.py for each chunk."
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
        default='root_file_inventory',
        help="Base output directory; per-month output goes under YYYY/MM subdirectories",
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


def next_month_start(value: date) -> date:
    """Return the first day of the month after ``value``."""
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def yyyymmdd(value: date) -> str:
    """Format date as YYYYMMDD for downstream scripts."""
    return value.strftime("%Y%m%d")


def ensure_script_exists(repo_dir: Path) -> None:
    """Validate the dependent script exists before processing."""
    script_path = repo_dir / SCRIPT_REL_PATH
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {SCRIPT_REL_PATH}")


def run_chunk(
    *,
    script_path: Path,
    python_bin: str,
    base_path: str,
    output_base: str,
    chunk_start: date,
    chunk_end: date,
    dry_run: bool,
) -> None:
    """Run the inventory script for one monthly chunk."""
    print(
        f"Processing monthly chunk: [{yyyymmdd(chunk_start)}, "
        f"{yyyymmdd(chunk_end)})"
    )

    month_output_dir = str(
        Path(output_base) / chunk_start.strftime("%Y") / chunk_start.strftime("%m")
    )

    cmd = [
        python_bin,
        str(script_path),
        "--base-path", base_path,
        "--start-date", yyyymmdd(chunk_start),
        "--end-date", yyyymmdd(chunk_end),
        "--daily-summary",
        "--output-dir", month_output_dir,
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
    failed_chunks: list[str] = []
    while current < end_date:
        month_end = next_month_start(current)
        chunk_end = month_end if month_end <= end_date else end_date

        try:
            run_chunk(
                script_path=script_path,
                python_bin=args.python_bin,
                base_path=args.base_path,
                output_base=args.output_base,
                chunk_start=current,
                chunk_end=chunk_end,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            chunk_label = f"[{yyyymmdd(current)}, {yyyymmdd(chunk_end)})"
            print(f"Error processing chunk {chunk_label}: {exc}", file=sys.stderr)
            failed_chunks.append(chunk_label)
        current = chunk_end

    if failed_chunks:
        print(
            f"Done with {len(failed_chunks)} failed chunk(s): {', '.join(failed_chunks)}",
            file=sys.stderr,
        )
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)