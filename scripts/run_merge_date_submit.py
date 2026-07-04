#!/usr/bin/env python3
"""Submit one Slurm job per day to run merge for each date.

Each job executes:
    scripts/run_merge_date.py <day> ...

Usage:
    python scripts/run_merge_date_submit.py -d YYYYMMDD [--end-date YYYYMMDD] [options]
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import List, Optional

from logger_config import logger


def parse_date(value: str) -> dt.date:
    """Parse YYYYMMDD date string."""
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}': must be YYYYMMDD"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Submit Slurm jobs for merge date processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --start-date 20260501
  %(prog)s --start-date 20260501 --end-date 20260503 --target trace --run
  %(prog)s --start-date 20260501 --target results --run
  %(prog)s --start-date 20260501 --end-date 20260503 --target all --run""",
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help="Single date (YYYYMMDD). Sets --start-date and --end-date automatically. "
             "Mutually exclusive with --start-date/--end-date.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="Start date (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date exclusive (default: start date + 1 day)",
    )
    parser.add_argument(
        "--target",
        choices=["header", "trace", "results", "both", "all"],
        default="both",
        help="Merge target mode (default: both; all = header + trace + results)",
    )
    parser.add_argument(
        "--channel",
        choices=["F", "X", "Y", "Z", "XY"],
        default="XY",
        help="Trace channel for merge_trace (default: XY)",
    )
    parser.add_argument(
        "--out-dir-base",
        default="../Reco_Dir",
        help="Passed to run_merge_date.py (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Passed to run_merge_date.py --python (default: python)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually submit sbatch jobs (default: dry-run, print only). "
             "When --run is given, --run is also passed to run_merge_date.py.",
    )
    parser.add_argument(
        "--partition",
        default="com",
        help="Slurm partition (default: com)",
    )
    parser.add_argument(
        "--time",
        default="12:00:00",
        help="Slurm time limit (default: 12:00:00)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Slurm cpus-per-task (default: 4)",
    )
    parser.add_argument(
        "--mem",
        default="64G",
        help="Slurm memory (default: 64G)",
    )
    return parser


def build_sbatch_command(
    repo_root: Path,
    log_dir: Path,
    cur_date: dt.date,
    target: str,
    out_dir_base: str,
    python_cmd: str,
    run: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    channel: str = "XY",
) -> List[str]:
    """Build sbatch command list for one date."""
    date_text = cur_date.strftime("%Y%m%d")
    job_name = f"merge_date_{target}_{date_text}"

    run_line = (
        f"cd {repo_root} || exit 1; "
        f"{python_cmd} scripts/run_merge_date.py {date_text} "
        f"--target {target} --out-dir-base {out_dir_base} --python {python_cmd} "
        f"--channel {channel}"
    )
    if run:
        run_line += " --run"

    return [
        "sbatch",
        "--job-name", job_name,
        "--partition", partition,
        "--time", time_limit,
        "--cpus-per-task", str(cpus),
        "--mem", mem,
        "--output", str(log_dir / f"{job_name}_%j.out"),
        "--error", str(log_dir / f"{job_name}_%j.err"),
        "--wrap", run_line,
    ]


def iter_dates(start_date: dt.date, end_date: dt.date):
    """Yield dates from start_date (inclusive) to end_date (exclusive)."""
    current = start_date
    while current < end_date:
        yield current
        current += dt.timedelta(days=1)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Resolve --date shorthand
    if args.date is not None:
        if args.start_date is not None or args.end_date is not None:
            logger.error(
                "Error: --date is mutually exclusive with --start-date/--end-date",
            )
            return 2
        args.start_date = args.date
        args.end_date = args.date + dt.timedelta(days=1)

    if args.start_date is None:
        logger.error("Error: either --date or --start-date is required")
        return 2

    start_date: dt.date = args.start_date
    end_date: dt.date = args.end_date or (start_date + dt.timedelta(days=1))

    if start_date >= end_date:
        logger.error(
            "Error: --start-date must be < --end-date (end-date is exclusive)",
        )
        return 2

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    log_dir = repo_root / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failed_dates: list[tuple[dt.date, int]] = []
    for cur_date in iter_dates(start_date, end_date):
        sbatch_cmd = build_sbatch_command(
            repo_root=repo_root,
            log_dir=log_dir,
            cur_date=cur_date,
            target=args.target,
            out_dir_base=args.out_dir_base,
            python_cmd=args.python,
            run=args.run,
            partition=args.partition,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            channel=args.channel,
        )

        logger.info(
            "Prepared command ({}): {}",
            cur_date.strftime('%Y%m%d'), " ".join(sbatch_cmd),
        )

        if not args.run:
            continue

        result = subprocess.run(sbatch_cmd, check=False)
        if result.returncode != 0:
            logger.error(
                "sbatch failed for {} with code {}",
                cur_date.strftime('%Y%m%d'), result.returncode,
            )
            failed_dates.append((cur_date, result.returncode))

    if failed_dates:
        for d, rc in failed_dates:
            logger.error(
                "FAILED: date={} returncode={}", d.strftime('%Y%m%d'), rc,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
