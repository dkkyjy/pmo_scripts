#!/usr/bin/env python3
"""Run merge_header/merge_trace/merge_results for one date with one orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from logger_config import logger


TARGET_HEADER = "header"
TARGET_TRACE = "trace"
TARGET_RESULTS = "results"
TARGET_BOTH = "both"
TARGET_ALL = "all"

REQUIRED_TRACE_TYPES: Tuple[str, ...] = ("F", "X", "Y", "Z", "XY")
RESULTS_TYPES: Tuple[str, ...] = ("matched", "fingerprint", "PWM", "SWM")
RUN_PATTERN = re.compile(r"RUN(\d+)(?!\d)")
TRACE_SUFFIX_PATTERN = re.compile(r"_(F|X|Y|Z|XY)\.yaml$")
RESULTS_SUFFIX_PATTERN = re.compile(rf"_(?:{'|'.join(RESULTS_TYPES)})\.yaml$")


@dataclass
class DayPlan:
    """Planned commands and diagnostics for one date and one target."""

    code: int
    commands: List[List[str]]
    detected_runs: List[int]


def parse_day(value: str) -> dt.date:
    """Parse day string in supported formats."""
    formats = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d")
    for fmt in formats:
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported day format: {value}")


def _collect_trigger_files(date_dir: Path) -> List[Path]:
    """Collect trigger yaml files from one date directory."""
    return sorted(date_dir.glob("Trigger*.yaml"))


def _collect_runs_and_legacy(files: Sequence[Path]) -> Tuple[Set[int], List[Path]]:
    """Extract run numbers and legacy files without RUN segment."""
    runs: Set[int] = set()
    legacy_files: List[Path] = []
    for file_path in files:
        run_match = RUN_PATTERN.search(file_path.name)
        if run_match is None:
            legacy_files.append(file_path)
            continue
        runs.add(int(run_match.group(1)))
    return runs, legacy_files


def _collect_trace_run_types(files: Sequence[Path]) -> Dict[int, Set[str]]:
    """Collect available trace suffix types for each run number."""
    run_types: Dict[int, Set[str]] = {}
    for file_path in files:
        run_match = RUN_PATTERN.search(file_path.name)
        type_match = TRACE_SUFFIX_PATTERN.search(file_path.name)
        if run_match is None or type_match is None:
            continue
        run_number = int(run_match.group(1))
        run_types.setdefault(run_number, set()).add(type_match.group(1))
    return run_types


def _select_files_for_target(date_dir: Path, target: str) -> List[Path]:
    """Select input files for one target mode."""
    files = _collect_trigger_files(date_dir)
    if target == TARGET_TRACE:
        return files

    # Header / Results mode: strip trace channel files to avoid semantic pollution.
    return [
        file_path
        for file_path in files
        if TRACE_SUFFIX_PATTERN.search(file_path.name) is None
    ]


def _build_merge_command(
    python_exec: str,
    target: str,
    date_text: str,
    output_root: str,
    run_number: Optional[int],
    channel: str = "XY",
) -> List[str]:
    """Build merge command for one target/date/run tuple."""
    if target == TARGET_HEADER:
        script = "merge/merge_header.py"
    elif target == TARGET_TRACE:
        script = "merge/merge_trace.py"
    else:
        script = "merge/merge_results.py"
    command = [python_exec, script, date_text, "-o", output_root]
    if run_number is not None:
        command.extend(["--run-number", str(run_number)])
    if target == TARGET_TRACE:
        command.extend(["--channel", channel])
    return command


def _filter_runs_with_results_files(
    date_dir: Path, runs: List[int]
) -> List[int]:
    """Filter runs that have at least one results-type yaml in the date directory."""
    result: List[int] = []
    for run_number in runs:
        run_glob = f"Trigger*_RUN{run_number}_*"
        matching = list(date_dir.glob(run_glob))
        if any(RESULTS_SUFFIX_PATTERN.search(f.name) for f in matching):
            result.append(run_number)
        else:
            logger.info(
                "date={} target=results skip RUN{} (no results files)", date_dir.name, run_number,
            )
    return result


def plan_day_for_target(
    date_dir: Path,
    target: str,
    date_text: str,
    python_exec: str,
    output_root: str,
    channel: str = "XY",
) -> DayPlan:
    """Build target-specific command plan for one date."""
    if not date_dir.is_dir():
        return DayPlan(2, [], [])

    target_files = _select_files_for_target(date_dir, target)
    if not target_files:
        return DayPlan(1, [], [])

    detected_runs, legacy_files = _collect_runs_and_legacy(target_files)
    sorted_runs = sorted(detected_runs)

    if legacy_files and not detected_runs:
        logger.info(
            "date={} target={} legacy_files={} (no RUN files, merging all)",
            date_text, target, len(legacy_files),
        )

    if not sorted_runs:
        command = _build_merge_command(
            python_exec, target, date_text, output_root, run_number=None, channel=channel,
        )
        return DayPlan(0, [command], [])

    if target == TARGET_TRACE:
        commands = _build_trace_commands(
            python_exec, target, date_text, output_root, channel,
            sorted_runs, target_files,
        )
    elif target == TARGET_RESULTS:
        eligible_runs = _filter_runs_with_results_files(date_dir, sorted_runs)
        commands = [
            _build_merge_command(
                python_exec, target, date_text, output_root, run_number=rn, channel=channel,
            )
            for rn in eligible_runs
        ]
    elif target == TARGET_HEADER:
        commands = [
            _build_merge_command(
                python_exec, target, date_text, output_root, run_number=rn, channel=channel,
            )
            for rn in sorted_runs
        ]
    else:
        logger.error("date={} target={} unknown target", date_text, target)
        return DayPlan(2, [], sorted_runs)

    if legacy_files and detected_runs:
        logger.warning(
            "date={} target={} mixed RUN and legacy files; legacy files are ignored",
            date_text, target,
        )

    if not commands:
        return DayPlan(1, [], sorted_runs)
    return DayPlan(0, commands, sorted_runs)


def _build_trace_commands(
    python_exec: str,
    target: str,
    date_text: str,
    output_root: str,
    channel: str,
    sorted_runs: List[int],
    target_files: List[Path],
) -> List[List[str]]:
    """Build per-RUN trace commands, skipping RUNs missing required channel."""
    required_types = {channel}
    run_types = _collect_trace_run_types(target_files)
    commands: List[List[str]] = []
    for run_number in sorted_runs:
        available = run_types.get(run_number, set())
        if required_types.issubset(available):
            commands.append(
                _build_merge_command(
                    python_exec, target, date_text, output_root,
                    run_number, channel=channel,
                )
            )
        else:
            missing = sorted(required_types - available)
            logger.warning(
                "date={} target={} channel={} skip RUN{} due to missing types={}",
                date_text, target, channel, run_number, missing,
            )
    return commands


def build_arg_parser() -> argparse.ArgumentParser:
    """Build parser for one-date merge orchestration."""
    parser = argparse.ArgumentParser(
        description="Run merge_header/merge_trace/merge_results for one date"
    )
    parser.add_argument("date", help="Date: YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD")
    parser.add_argument(
        "--target",
        choices=[TARGET_HEADER, TARGET_TRACE, TARGET_RESULTS, TARGET_BOTH, TARGET_ALL],
        default=TARGET_BOTH,
        help="Target merge workflow (default: both; all = header + trace + results)",
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
        help="Output root used by merge tools -o (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run merge scripts",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Default is dry-run (print only).",
    )
    return parser


def _targets_in_order(target: str) -> List[str]:
    """Expand target into concrete execution order."""
    if target == TARGET_ALL:
        return [TARGET_HEADER, TARGET_TRACE, TARGET_RESULTS]
    if target == TARGET_BOTH:
        return [TARGET_HEADER, TARGET_TRACE]
    return [target]


def _run_merge_command(command: List[str]) -> Tuple[List[str], int, str]:
    """Execute one merge command, return (command, returncode, output)."""
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    parts: list[str] = []
    if result.stdout.strip():
        parts.append(result.stdout.strip())
    if result.stderr.strip():
        parts.append(f"[stderr] {result.stderr.strip()}")
    output = "\n".join(parts)
    return command, result.returncode, output


def _execute_commands(
    commands: Sequence[List[str]],
    timeout: Optional[float] = None,
) -> List[Tuple[List[str], int, str]]:
    """Execute command batch with parallel workers."""
    if not commands:
        return []
    max_workers = max(1, min(len(commands), os.cpu_count() or 1))
    with Pool(processes=max_workers) as pool:
        try:
            async_result = pool.map_async(_run_merge_command, commands)
            return async_result.get(timeout=timeout)
        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt, terminating pool...")
            pool.terminate()
            pool.join()
            raise


def _log_command_result(
    dry_run: bool,
    date_text: str,
    target: str,
    command: List[str],
    return_code: int = 0,
    output: str = "",
) -> None:
    """Log a single command result."""
    command_text = " ".join(command)
    prefix = "[DRY-RUN]" if dry_run else ""
    code_label = "NA" if dry_run else str(return_code)
    logger.info(
        "{}date={} target={} final_command={} returncode={}",
        f"{prefix} " if prefix else "",
        date_text, target, command_text, code_label,
    )
    if not dry_run and output:
        logger.info("  {}", output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    day = parse_day(args.date)
    output_root = Path(args.out_dir_base)
    date_text = day.strftime("%Y/%m/%d")
    date_dir = output_root / date_text

    for target in _targets_in_order(args.target):
        day_plan = plan_day_for_target(
            date_dir=date_dir,
            target=target,
            date_text=date_text,
            python_exec=args.python,
            output_root=str(output_root),
            channel=args.channel,
        )

        logger.info(
            "date={} target={} detected_runs={}",
            date_text, target, day_plan.detected_runs,
        )

        if day_plan.code != 0:
            logger.warning(
                "date={} target={} planning failed with code={} (dir={})",
                date_text, target, day_plan.code, date_dir,
            )
            return day_plan.code

        if not args.run:
            for command in day_plan.commands:
                _log_command_result(dry_run=True, date_text=date_text,
                                   target=target, command=command)
            continue

        results = _execute_commands(day_plan.commands)
        failed_commands: list[tuple[str, int]] = []
        for command, return_code, output in results:
            _log_command_result(dry_run=False, date_text=date_text,
                               target=target, command=command,
                               return_code=return_code, output=output)
            if return_code != 0:
                failed_commands.append((" ".join(command), return_code))
        if failed_commands:
            for cmd_text, rc in failed_commands:
                logger.error("date={} target={} FAILED: cmd={} returncode={}",
                             date_text, target, cmd_text, rc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
