#!/usr/bin/env python3
"""Submit one Slurm job per day to run run_loop_hourly.py.

Each job executes:
    python scripts/run_loop_hourly.py --base-path <BASE_PATH> <YYYYMMDD>

Usage:
    python scripts/run_loop_hourly_submit.py -d YYYYMMDD [--end-date YYYYMMDD] [options]
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
        description="Submit Slurm jobs to run loop.py hourly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --start-date 20260501
  %(prog)s --start-date 20260501 --end-date 20260503 --run""",
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
        "--base-path",
        default="/share07/users/m/mapx/DunhuangData/ROOTFile/TD",
        help="Base path for ROOT data (default: /share07/.../TD)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually submit jobs with sbatch (dry-run by default)",
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
        default=20,
        help="Slurm cpus-per-task (default: 20)",
    )
    parser.add_argument(
        "--mem",
        default="64G",
        help="Slurm memory (default: 64G)",
    )
    parser.add_argument(
        "--python-cmd",
        default="python",
        help="Python command in Slurm --wrap (default: python)",
    )
    return parser


def build_sbatch_command(
    repo_root: Path,
    log_dir: Path,
    base_path: str,
    cur_date: dt.date,
    python_cmd: str,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
) -> List[str]:
    """Build sbatch command list for one date."""
    date_text = cur_date.strftime("%Y%m%d")
    job_name = f"loop_hourly_{date_text}"

    run_line = (
        f"cd {repo_root} || exit 1; "
        f"{python_cmd} {repo_root}/scripts/run_loop_hourly.py "
        f"--base-path {base_path} {date_text}"
    )

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
                "--date is mutually exclusive with --start-date/--end-date"
            )
            return 2
        args.start_date = args.date
        args.end_date = args.date + dt.timedelta(days=1)

    if args.start_date is None:
        logger.error("either --date or --start-date is required")
        return 2

    start_date: dt.date = args.start_date
    end_date: dt.date = args.end_date or (start_date + dt.timedelta(days=1))

    if start_date >= end_date:
        logger.error(
            "--start-date must be < --end-date (end-date is exclusive)"
        )
        return 2

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    log_dir = repo_root / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failed_dates: list[str] = []

    for cur_date in iter_dates(start_date, end_date):
        sbatch_cmd = build_sbatch_command(
            repo_root=repo_root,
            log_dir=log_dir,
            base_path=args.base_path,
            cur_date=cur_date,
            python_cmd=args.python_cmd,
            partition=args.partition,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
        )

        date_text = cur_date.strftime("%Y%m%d")
        logger.info("Prepared command ({}): {}", date_text, " ".join(sbatch_cmd))

        if args.run:
            result = subprocess.run(sbatch_cmd, check=False)
            if result.returncode != 0:
                logger.error(
                    "sbatch failed for {} with code {}",
                    date_text,
                    result.returncode,
                )
                failed_dates.append(date_text)

    if failed_dates:
        logger.error("Failed dates: {}", ", ".join(failed_dates))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
