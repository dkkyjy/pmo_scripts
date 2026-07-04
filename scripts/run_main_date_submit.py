#!/usr/bin/env python3
"""Submit one Slurm job per long-name config YAML file to run main.py.

Each job executes:
    main.py <yaml_file> --skip-fingerprint

Matches filenames with pattern:
    Trigger_YYYYMMDDhhmmss_RUN*_<config>.yaml

Usage:
    python scripts/run_main_date_submit.py -d YYYYMMDD [--end-date YYYYMMDD] [options]
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
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
        description="Submit Slurm jobs to run main.py on long-name config YAMLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --start-date 20260622
  %(prog)s --start-date 20260622 --end-date 20260624 --run""",
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
        "--out-dir-base",
        default="../Reco_Dir",
        help="Output directory containing merged YAMLs (default: ../Reco_Dir)",
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
        default=4,
        help="Slurm cpus-per-task (default: 4)",
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


def find_config_files(reco_dir: Path, cur_date: dt.date) -> List[Path]:
    """Find long-name config YAML files for a given date.

    Matches: Trigger_YYYYMMDDhhmmss_RUN*_<config>.yaml
    """
    date_text = cur_date.strftime("%Y%m%d")
    year, month, day = date_text[:4], date_text[4:6], date_text[6:8]
    date_dir = reco_dir / year / month / day
    if not date_dir.is_dir():
        return []

    # Regex: date prefix exactly `date_text` followed by 6 digits (hhmmss)
    date_prefix = re.escape(date_text)
    pat = re.compile(rf"^Trigger_{date_prefix}\d{{6}}_RUN\d+_.*\.yaml$")

    result: List[Path] = []
    for candidate in date_dir.glob(f"Trigger_{date_text}*_RUN*.yaml"):
        if pat.match(candidate.name):
            result.append(candidate)
    return sorted(result)


def build_sbatch_command(
    repo_root: Path,
    log_dir: Path,
    yaml_file: Path,
    cur_date: dt.date,
    run_number: str,
    python_cmd: str,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
) -> List[str]:
    """Build sbatch command list for one YAML file."""
    date_text = cur_date.strftime("%Y%m%d")
    job_name = f"main_{date_text}_run{run_number}"

    run_line = (
        f"cd {repo_root} || exit 1; "
        f"{python_cmd} main.py {yaml_file} --skip-fingerprint --det-pos _gp65_rtksort_2002_DU7.txt --force-recompute"
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


def extract_run_number(filename: str) -> str:
    """Extract RUN number from filename like Trigger_20260622012134_RUN10399_CD20dB-....yaml."""
    for part in filename.split("_"):
        if part.startswith("RUN"):
            return part[3:]  # strip "RUN" prefix
    return "unknown"


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
    reco_dir = (repo_root / args.out_dir_base).resolve()
    log_dir = repo_root / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not reco_dir.is_dir():
        logger.error("Reco directory not found: {}", reco_dir)
        return 2

    failed_submissions: list[tuple[str, str, int]] = []

    for cur_date in iter_dates(start_date, end_date):
        yaml_files = find_config_files(reco_dir, cur_date)
        date_text = cur_date.strftime("%Y%m%d")

        if not yaml_files:
            logger.warning("no config YAML found for {}", date_text)
            continue

        for yaml_file in yaml_files:
            run_number = extract_run_number(yaml_file.name)

            sbatch_cmd = build_sbatch_command(
                repo_root=repo_root,
                log_dir=log_dir,
                yaml_file=yaml_file,
                cur_date=cur_date,
                run_number=run_number,
                python_cmd=args.python_cmd,
                partition=args.partition,
                time_limit=args.time,
                cpus=args.cpus,
                mem=args.mem,
            )

            logger.info(
                "Prepared command ({}, RUN{}): {}",
                date_text,
                run_number,
                " ".join(sbatch_cmd),
            )

            if args.run:
                result = subprocess.run(sbatch_cmd, check=False)
                if result.returncode != 0:
                    logger.error(
                        "sbatch failed for {} RUN{} with code {}",
                        date_text,
                        run_number,
                        result.returncode,
                    )
                    failed_submissions.append(
                        (date_text, run_number, result.returncode)
                    )

    if failed_submissions:
        logger.error(
            "Failed submissions: {}",
            ", ".join(
                f"{d}/RUN{r} (rc={c})" for d, r, c in failed_submissions
            ),
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
