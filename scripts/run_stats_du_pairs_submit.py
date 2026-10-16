#!/usr/bin/env python3
"""Batch-submit stats_du_pairs.py by auto-discovering Trigger matched YAML.

This script scans Trigger matched YAML files and submits commands in the form:
    python stats/stats_du_pairs.py ../Reco_Dir/Trigger_YYYYMMDD_RUNxxx_matched.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

from logger_config import logger


DATE_RANGE_START = "20250701"
DATE_RANGE_END = "20260701"
TRIGGER_FILE_PATTERN = re.compile(
    r"^Trigger_(?P<date>\d{8})_RUN(?P<run>\d+)_matched\.yaml$"
)


def parse_date(value: str) -> dt.date:
    """Parse YYYYMMDD date string."""
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}': must be YYYYMMDD"
        )


def iter_dates(start_date: dt.date, end_date: dt.date):
    """Yield dates from start_date (inclusive) to end_date (exclusive)."""
    current = start_date
    while current < end_date:
        yield current
        current += dt.timedelta(days=1)


@dataclass(frozen=True)
class RunTarget:
    """One target identified by date and run number."""

    date_text: str
    run_number: int


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for batch stats_du_pairs submission."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch-submit stats/stats_du_pairs.py by discovering "
            "Trigger_YYYYMMDD_RUNxxx_matched.yaml."
        )
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help=(
            "Single date (YYYYMMDD). Submit jobs for this day only. "
            "Mutually exclusive with --start-date/--end-date."
        ),
    )
    parser.add_argument(
        "--run-number",
        default=None,
        type=int,
        help="Optional run number filter.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help=(
            "Start date YYYYMMDD (inclusive). "
            f"Default: {DATE_RANGE_START}. "
            "Mutually exclusive with --date."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help=(
            "End date YYYYMMDD (exclusive). "
            f"Default: {DATE_RANGE_END}. "
            "Mutually exclusive with --date."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of selected targets.",
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Directory containing Trigger_*_matched.yaml files.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run stats_du_pairs.py.",
    )
    parser.add_argument(
        "--script",
        default="stats/stats_du_pairs.py",
        help="Path to stats_du_pairs.py.",
    )
    parser.add_argument(
        "--det-pos",
        default=None,
        help="Forward --det-pos to child command.",
    )
    parser.add_argument(
        "--offset-file",
        default=None,
        help="Forward --offset-file to child command.",
    )
    parser.add_argument(
        "--start-datetime",
        default=None,
        help="Forward --start-datetime to child command.",
    )
    parser.add_argument(
        "--end-datetime",
        default=None,
        help="Forward --end-datetime to child command.",
    )
    parser.add_argument(
        "--jitter-delta-ns",
        type=float,
        default=None,
        help="Forward --jitter-delta-ns to child command.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Forward --no-plot to child command.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Forward --force-recompute to child command.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Submit with sbatch. Default is dry-run only.",
    )
    parser.add_argument(
        "--partition",
        default="com",
        help="Slurm partition (default: com).",
    )
    parser.add_argument(
        "--time",
        default="12:00:00",
        help="Slurm time limit HH:MM:SS (default: 12:00:00).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Slurm cpus-per-task (default: 1).",
    )
    parser.add_argument(
        "--mem",
        default="128G",
        help="Slurm memory (default: 128G).",
    )
    parser.add_argument(
        "--log-dir",
        default="slurm_logs",
        help="Directory for Slurm stdout/stderr logs.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="stats_du_pairs",
        help="Prefix for Slurm job name.",
    )
    return parser


def discover_targets(reco_dir: Path, date_text: Optional[str] = None) -> List[RunTarget]:
    """Discover all date/run targets from Trigger merged YAML filenames, optionally for a specific date."""
    unique_pairs = set()
    for file_path in sorted(reco_dir.glob("Trigger_*_matched.yaml")):
        matched = TRIGGER_FILE_PATTERN.match(file_path.name)
        if matched is None:
            continue
        matched_date = matched.group("date")
        if date_text is not None and matched_date != date_text:
            continue
        run_number = int(matched.group("run"))
        unique_pairs.add((matched_date, run_number))

    return [
        RunTarget(date_text=date_text, run_number=run_number)
        for date_text, run_number in sorted(unique_pairs)
    ]


def filter_targets(
    targets: Sequence[RunTarget],
    start_date: dt.date,
    end_date: dt.date,
    run_filter: Optional[int],
    limit: Optional[int],
) -> List[RunTarget]:
    """Apply date range, run filter, and limit to discovered targets."""
    filtered = [
        target
        for target in targets
        if start_date.strftime("%Y%m%d") <= target.date_text < end_date.strftime("%Y%m%d")
        and (run_filter is None or target.run_number == run_filter)
    ]

    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def build_yaml_path(reco_dir: Path, target: RunTarget) -> Path:
    """Build input YAML path from one target."""
    return reco_dir / (
        f"Trigger_{target.date_text}_RUN{target.run_number}_matched.yaml"
    )


def build_child_command(args: argparse.Namespace, yaml_path: Path) -> List[str]:
    """Build child command list for stats_du_pairs.py."""
    command = [
        args.python,
        args.script,
        str(yaml_path),
    ]

    if args.det_pos:
        command.extend(["--det-pos", args.det_pos])
    if args.offset_file:
        command.extend(["--offset-file", args.offset_file])
    if args.start_datetime:
        command.extend(["--start-datetime", args.start_datetime])
    if args.end_datetime:
        command.extend(["--end-datetime", args.end_datetime])
    if args.jitter_delta_ns is not None:
        command.extend(["--jitter-delta-ns", str(args.jitter_delta_ns)])
    if args.no_plot:
        command.append("--no-plot")
    if args.force_recompute:
        command.append("--force-recompute")

    return command


def build_sbatch_command(
    child_command: Sequence[str],
    target: RunTarget,
    args: argparse.Namespace,
    repo_root: Path,
    log_dir: Path,
) -> List[str]:
    """Build sbatch command wrapping one stats_du_pairs command."""
    job_name = f"{args.job_name_prefix}_{target.date_text}_run{target.run_number}"
    wrapped_cmd = (
        f"cd {shlex.quote(str(repo_root))} || exit 1; "
        + " ".join(shlex.quote(part) for part in child_command)
    )

    return [
        "sbatch",
        "--job-name",
        job_name,
        "--partition",
        args.partition,
        "--time",
        args.time,
        "--cpus-per-task",
        str(args.cpus),
        "--mem",
        args.mem,
        "--output",
        str(log_dir / f"{job_name}_%j.out"),
        "--error",
        str(log_dir / f"{job_name}_%j.err"),
        "--wrap",
        wrapped_cmd,
    ]


def submit_or_print(
    commands: Sequence[List[str]],
    do_run: bool,
) -> Tuple[int, int]:
    """Submit batch commands or print them in dry-run mode.

    Returns:
        (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0

    for index, command in enumerate(commands, start=1):
        command_text = " ".join(command)
        if not do_run:
            logger.info(
                "[dry-run][{}/{}] {}",
                index,
                len(commands),
                command_text,
            )
            success_count += 1
            continue

        logger.info("[run][{}/{}] {}", index, len(commands), command_text)
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            success_count += 1
        else:
            failure_count += 1
            logger.error(
                "Command failed with return code {}: {}",
                result.returncode,
                command_text,
            )

    return success_count, failure_count


def _process_day(
    args: argparse.Namespace,
    reco_dir: Path,
    repo_root: Path,
    log_dir: Path,
) -> Tuple[int, int]:
    """Discover and submit targets for a single date. Returns (success, failure)."""
    date_text = args.start_date.strftime("%Y%m%d")
    targets = discover_targets(reco_dir, date_text=date_text)
    selected_targets = filter_targets(
        targets,
        start_date=args.start_date,
        end_date=args.start_date + dt.timedelta(days=1),
        run_filter=args.run_number,
        limit=args.limit,
    )

    if not targets:
        logger.warning("No Trigger matched YAML files found for date {}", date_text)
        return 0, 0
    if not selected_targets:
        logger.info("No targets after filtering for date {}", date_text)
        return 0, 0

    logger.info(
        "Date={} discovered={} selected={}",
        date_text,
        len(targets),
        len(selected_targets),
    )

    sbatch_commands: List[List[str]] = []
    for target in selected_targets:
        yaml_path = build_yaml_path(reco_dir, target)
        if not yaml_path.exists():
            logger.warning("Skip missing YAML target: {}", yaml_path)
            continue
        child_command = build_child_command(args, yaml_path)
        sbatch_commands.append(
            build_sbatch_command(
                child_command,
                target,
                args,
                repo_root,
                log_dir,
            )
        )

    if not sbatch_commands:
        return 0, 0

    success_count, failure_count = submit_or_print(
        sbatch_commands,
        do_run=args.run,
    )
    return success_count, failure_count


def main() -> int:
    """Entry point for batch stats_du_pairs submit helper."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve date args: --date is mutually exclusive with --start-date/--end-date
    if args.date is not None:
        if args.start_date is not None or args.end_date is not None:
            parser.error("--date is mutually exclusive with --start-date/--end-date")
        args.start_date = args.date
        args.end_date = args.date + dt.timedelta(days=1)

    if args.start_date is None:
        args.start_date = dt.datetime.strptime(DATE_RANGE_START, "%Y%m%d").date()
    if args.end_date is None:
        args.end_date = dt.datetime.strptime(DATE_RANGE_END, "%Y%m%d").date()

    if args.start_date >= args.end_date:
        parser.error("--start-date must be < --end-date (end-date is exclusive)")
    if args.run_number is not None and args.run_number < 0:
        parser.error("--run-number must be >= 0")
    if args.cpus < 1:
        parser.error("--cpus must be >= 1")
    if not re.match(r"^(?:\d{1,2}-)?\d{2}:\d{2}:\d{2}$", args.time):
        parser.error(
            "--time must be in Slurm format: HH:MM:SS or D-HH:MM:SS"
        )

    reco_dir = Path(args.reco_dir).resolve()
    if not reco_dir.is_dir():
        parser.error(f"Reco directory does not exist: {reco_dir}")

    repo_root = Path(__file__).resolve().parents[1]

    # Validate --script path
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (repo_root / script_path).resolve()
    if not script_path.is_file():
        parser.error(f"Script not found: {script_path}")
    args.script = str(script_path)

    log_dir = (repo_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Slurm config partition={} time={} cpus={} mem={} log_dir={}",
        args.partition,
        args.time,
        args.cpus,
        args.mem,
        log_dir,
    )

    # Loop over dates: --date → one day; --start-date/--end-date → date range
    total_success = 0
    total_failure = 0
    for cur_date in iter_dates(args.start_date, args.end_date):
        day_args = argparse.Namespace(**vars(args))
        day_args.start_date = cur_date
        day_args.end_date = cur_date + dt.timedelta(days=1)
        success, failure = _process_day(day_args, reco_dir, repo_root, log_dir)
        total_success += success
        total_failure += failure

    logger.info(
        "Batch done. total_success={} total_failure={} mode={}",
        total_success,
        total_failure,
        "run" if args.run else "dry-run",
    )

    if total_failure > 0:
        return 2
    if total_success == 0:
        logger.warning("No targets submitted across all dates.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
