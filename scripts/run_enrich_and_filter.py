#!/usr/bin/env python3
"""Batch runner for enrich_and_filter_swm.py by date.

Scans ``--reco-dir/YYYY/MM/DD/`` directories for ``Trigger_*_SWM.yaml`` files
and calls ``enrich_and_filter_swm.py <path>`` for each file, which performs:

1. XY signal enrichment + 1/distance linear fit
2. Slope > 0 candidate filtering + candidate YAML + diagnostic plots

Also supports ``--input-pattern`` to process already-enriched files
(``*_SWM_with_signal.yaml``) for slope-filter-only mode.

Supports:
- ``--date YYYYMMDD`` — process a single day
- ``--start-date YYYYMMDD --end-date YYYYMMDD`` — process a date range
- ``--jobs N`` — parallel execution via multiprocessing.Pool
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

from logger_config import logger


def _parse_date(value: str) -> dt.date:
    """Parse date string in YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD format."""
    formats = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d")
    for fmt in formats:
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Unsupported date format: {value!r} "
        f"(expected YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD)"
    )


def _find_swm_files(date_dir: Path, glob_pattern: str) -> List[Path]:
    """Find SWM YAML files matching *glob_pattern* in a single date directory."""
    return sorted(date_dir.glob(glob_pattern))


def _iter_dates(start_date: dt.date, end_date: dt.date):
    """Yield dates from start_date (inclusive) to end_date (exclusive)."""
    current = start_date
    while current < end_date:
        yield current
        current += dt.timedelta(days=1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch enrichment and slope filtering by date."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run enrich_and_filter_swm.py for SWM files "
            "organized under YYYY/MM/DD directories."
        )
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Root directory containing date subdirectories (YYYY/MM/DD).",
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help=(
            "Single date (YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD). "
            "Mutually exclusive with --start-date/--end-date."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help=(
            "Start date (inclusive) for date-range mode. "
            "Must be used with --end-date. "
            "Mutually exclusive with --date."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help=(
            "End date (exclusive) for date-range mode. "
            "Must be used with --start-date. "
            "Mutually exclusive with --date."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run enrich_and_filter_swm.py.",
    )
    parser.add_argument(
        "--script",
        default="scripts/enrich_and_filter_swm.py",
        help="Path to enrich_and_filter_swm.py.",
    )
    parser.add_argument(
        "--input-pattern",
        default="Trigger_*_SWM.yaml",
        help=(
            "Glob pattern for input files under date directories. "
            "Default: 'Trigger_*_SWM.yaml'. "
            "Use 'Trigger_*_SWM_with_signal.yaml' for already-enriched files."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of files processed.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=20,
        help="Number of parallel workers (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Forward --output-dir to child script.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Forward --plot-dir to child script.",
    )
    parser.add_argument(
        "--det-pos-file",
        default=None,
        help="Forward --det-pos-file to child script.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Forward --no-plot to child script.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward --overwrite to child script.",
    )
    parser.add_argument(
        "--chi-square-threshold",
        type=float,
        default=None,
        help="Forward --chi-square-threshold to child script.",
    )
    parser.add_argument(
        "--swm-chi-square-max",
        type=float,
        default=None,
        help="Forward --swm-chi-square-max to child script.",
    )
    parser.add_argument(
        "--xmax-min-km",
        type=float,
        default=None,
        help="Forward --xmax-min-km to child script.",
    )
    parser.add_argument(
        "--xmax-max-km",
        type=float,
        default=None,
        help="Forward --xmax-max-km to child script.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Forward --suffix to child script.",
    )
    parser.add_argument(
        "--no-slope-filter",
        action="store_true",
        help="Forward --no-slope-filter to child script (skip slope candidate step).",
    )
    parser.add_argument(
        "--min-du-count",
        type=int,
        default=6,
        help="Forward --min-du-count to child script (default: 6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without executing.",
    )
    return parser


def _build_child_command(args: argparse.Namespace, swm_file: Path) -> List[str]:
    """Build command for one SWM file."""
    command = [
        args.python,
        args.script,
        str(swm_file),
    ]

    if args.overwrite:
        command.append("--overwrite")
    if args.no_plot:
        command.append("--no-plot")
    if args.no_slope_filter:
        command.append("--no-slope-filter")
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.plot_dir:
        command.extend(["--plot-dir", args.plot_dir])
    if args.det_pos_file:
        command.extend(["--det-pos-file", args.det_pos_file])
    if args.chi_square_threshold is not None:
        command.extend([
            "--chi-square-threshold",
            str(args.chi_square_threshold),
        ])
    if args.swm_chi_square_max is not None:
        command.extend([
            "--swm-chi-square-max",
            str(args.swm_chi_square_max),
        ])
    if args.xmax_min_km is not None:
        command.extend([
            "--xmax-min-km",
            str(args.xmax_min_km),
        ])
    if args.xmax_max_km is not None:
        command.extend([
            "--xmax-max-km",
            str(args.xmax_max_km),
        ])
    if args.suffix:
        command.extend(["--suffix", args.suffix])
    if args.min_du_count:
        command.extend(["--min-du-count", str(args.min_du_count)])

    return command


def _collect_swm_files(
    reco_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    glob_pattern: str,
    limit: Optional[int],
) -> List[Path]:
    """Collect SWM files across a date range, optionally limited."""
    all_files: List[Path] = []
    for cur_date in _iter_dates(start_date, end_date):
        date_dir = reco_dir / cur_date.strftime("%Y/%m/%d")
        if not date_dir.is_dir():
            logger.debug("Skipping non-existent date directory: {}", date_dir)
            continue
        files = _find_swm_files(date_dir, glob_pattern)
        all_files.extend(f.resolve() for f in files)

    total_before_limit = len(all_files)
    if limit is not None:
        all_files = all_files[:limit]
        if total_before_limit > limit:
            logger.info(
                "Limiting files from {} to {} (--limit={})",
                total_before_limit,
                len(all_files),
                limit,
            )
    return all_files


def _run_sequential(
    commands: List[List[str]],
    dry_run: bool,
) -> Tuple[int, int]:
    """Run commands one by one.

    Returns:
        (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0
    for idx, command in enumerate(commands, start=1):
        command_text = " ".join(command)
        if dry_run:
            logger.info(
                "[dry-run][{}/{}] {}",
                idx,
                len(commands),
                command_text,
            )
            success_count += 1
            continue

        logger.info("[run][{}/{}] {}", idx, len(commands), command_text)
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


def _run_one(command: Sequence[str]) -> int:
    """Run a single command via subprocess (pickleable for multiprocessing)."""
    result = subprocess.run(command, check=False)
    return result.returncode


def _run_parallel(
    commands: List[List[str]],
    jobs: int,
) -> Tuple[int, int]:
    """Run commands in parallel using multiprocessing.Pool.

    Returns:
        (success_count, failure_count)
    """
    from multiprocessing import Pool

    success_count = 0
    failure_count = 0
    with Pool(processes=jobs) as pool:
        results = pool.map(_run_one, commands)
        for idx, returncode in enumerate(results):
            if returncode == 0:
                success_count += 1
            else:
                failure_count += 1
                logger.error(
                    "Command failed with return code {}: {}",
                    returncode,
                    " ".join(commands[idx]),
                )
    return success_count, failure_count


def main() -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Validate mutually exclusive input modes ──
    date_mode = args.date is not None
    range_mode = args.start_date is not None or args.end_date is not None

    if date_mode and range_mode:
        parser.error(
            "--date and --start-date/--end-date are mutually exclusive"
        )
    if range_mode and (args.start_date is None or args.end_date is None):
        parser.error(
            "--start-date and --end-date must be used together"
        )

    # Resolve dates
    if date_mode:
        start_date = args.date
        end_date = args.date + dt.timedelta(days=1)
    elif range_mode:
        start_date = args.start_date
        end_date = args.end_date
    else:
        parser.error(
            "Either --date or --start-date/--end-date is required"
        )

    if start_date >= end_date:
        parser.error("start date must be < end date (end date is exclusive)")

    logger.info(
        "Batch mode reco_dir={} start={} end={} pattern={} jobs={} dry_run={}",
        args.reco_dir,
        start_date,
        end_date,
        args.input_pattern,
        args.jobs,
        args.dry_run,
    )

    reco_dir = Path(args.reco_dir).resolve()
    if not reco_dir.is_dir():
        parser.error(f"Reco directory does not exist: {reco_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = repo_root / script_path
    enrich_script = script_path.resolve()
    if not enrich_script.is_file():
        parser.error(f"Enrich script does not exist: {enrich_script}")
    # Use resolved absolute path for child commands
    args.script = str(enrich_script)

    # Collect SWM files
    swm_files = _collect_swm_files(
        reco_dir, start_date, end_date, args.input_pattern, args.limit,
    )
    if not swm_files:
        logger.warning(
            "No files matching '{}' found in reco_dir={} for date range [{}, {})",
            args.input_pattern,
            reco_dir,
            start_date,
            end_date,
        )
        return 1

    logger.info("Discovered SWM files count={}", len(swm_files))

    # Build commands
    commands = [_build_child_command(args, sf) for sf in swm_files]

    # Execute
    if args.jobs > 1 and not args.dry_run:
        success, failure = _run_parallel(commands, args.jobs)
    else:
        success, failure = _run_sequential(commands, dry_run=args.dry_run)

    logger.info(
        "Done. total={} success={} failure={} mode={}",
        len(commands),
        success,
        failure,
        "dry-run" if args.dry_run else "run",
    )

    if failure > 0:
        return 2
    if success == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
