#!/usr/bin/env python3
"""
loop.py

Generate and invoke:
    python read_header/read_header.py file_path base_path out_dir_base
    python main.py Reco_Dir/yyyy/mm/dd/Trigger_xxx.yaml
    
Usage examples:
    # Print commands only, do not execute
    ./loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59

    # Actually execute (be careful)
    ./loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 --run

Parameters:
    start Required positional datetime, format YYYY-MM-DDThh:mm:ss
    end Required positional datetime, format YYYY-MM-DDThh:mm:ss
    --out-dir-base Optional, passed to subcommands as --out-dir-base (default ../Reco_Dir)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from logger_config import logger

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass(frozen=True)
class TaskSpec:
    """Named task model for one ROOT file processing unit."""

    file_path: str
    date: str
    out_dir_base: str
    do_run: bool
    skip_read: bool
    only_read: bool
    with_signal: bool
    left: int
    right: int
    channel: str
    run_matching: bool
    run_pwm: bool
    run_swm: bool
    force_recompute: bool


TaskResult = Tuple[str, int]

def parse_datetime(s):
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # ISO
        "%Y%m%d%H%M%S",       # 20251127153000
        "%Y-%m-%d%H%M",       # 2025-11-271530
        "%Y/%m/%d%H%M",       # 2025/11/271530
    ]
    for fmt in formats:
        if len(s) < 14:
            s = f"{s:0<14}"
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime string: '{s}'")


def times_between(start: datetime.datetime, end: datetime.datetime) -> Iterable[datetime.datetime]:
    """Yield times from start to end inclusive, one per day at midnight."""
    cur = start.replace(hour=0, minute=0, second=0)
    one = datetime.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def get_file_list(file_start: int, file_end: int, date: str, base_path: str) -> list[str]:
    """
    Find .root Trigger files that match the criteria under base_path/date and filter after sorting by filename.
    Return full path list.
    """
    path = os.path.join(base_path, date)
    logger.info(f"Searching directory: {path}")
    logger.info(f"File from {file_start} to {file_end}")
    if not os.path.isdir(path):
        logger.warning(f"Directory does not exist: {path}")
        return []  # Return empty list instead of raising exception
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if (f.endswith(".root") and "Trigger" in f)
    ]
    files.sort()
    selected: list[str] = []
    seen_files = set()  # Prevent duplicate entries
    # logger.info(f"Found ROOT Trigger files: {len(files)}")
    for f in files:
        # It's safer to take the last part of the file path and split by '_'
        name = os.path.basename(f)
        parts = name.split("_")
        if len(parts) < 2:
            continue
        try:
            file_date = int(parts[1])
        except ValueError:
            continue
        if file_start <= file_date <= file_end and f not in seen_files:
            selected.append(f)
            seen_files.add(f)
    logger.info(f"Found {len(selected)} matching files in {path}")
    return selected


def get_file_bounds_for_date(
    start: datetime.datetime,
    end: datetime.datetime,
    current_date: datetime.datetime,
) -> tuple[int, int]:
    """Get file datetime bounds for one processing date."""
    file_start = int(start.strftime("%Y%m%d%H%M%S"))
    file_end = int(end.strftime("%Y%m%d%H%M%S"))

    tmp_start = file_start
    tmp_end = file_end
    if end - current_date > datetime.timedelta(days=1):
        tmp_end = int(current_date.strftime("%Y%m%d") + "240000")
    if current_date - start > datetime.timedelta(days=0):
        tmp_start = int(current_date.strftime("%Y%m%d") + "000000")
    return tmp_start, tmp_end


def make_tasks_for_file_paths(
    file_path_list: list[str],
    date: str,
    args: argparse.Namespace,
) -> list[TaskSpec]:
    """Build worker task tuples for one date and file list."""
    return [
        TaskSpec(
            file_path=file_path,
            date=date,
            out_dir_base=args.out_dir_base,
            do_run=args.run,
            skip_read=args.skip_read,
            only_read=args.only_read,
            with_signal=args.with_signal,
            left=args.left,
            right=args.right,
            channel=args.channel,
            run_matching=args.run_matching,
            run_pwm=args.run_pwm,
            run_swm=args.run_swm,
            force_recompute=args.force_recompute,
        )
        for file_path in file_path_list
    ]


def build_tasks(
    start: datetime.datetime,
    end: datetime.datetime,
    args: argparse.Namespace,
) -> list[TaskSpec]:
    """Build all worker tasks for the date range."""
    tasks: list[TaskSpec] = []
    for date in times_between(start, end):
        date_str = date.strftime("%Y/%m/%d")
        logger.info(f"Processing date: {date_str}")
        tmp_start, tmp_end = get_file_bounds_for_date(start, end, date)
        file_path_list = get_file_list(tmp_start, tmp_end, date_str, args.base_path)
        tasks.extend(make_tasks_for_file_paths(file_path_list, date_str, args))
    return tasks


def execute_tasks(tasks: list[TaskSpec], jobs: int) -> list[TaskResult]:
    """Execute tasks sequentially or in multiprocessing mode."""
    if jobs <= 1:
        return [process_date(task) for task in tasks]

    with multiprocessing.Pool(processes=jobs) as pool:
        return pool.map(process_date, tasks)


def make_readheader_command(file_path: str, date: str, out_dir_base: str) -> list[str]:
    """Construct the command for read_header with file_path, date, out_dir_base parameters."""
    cmd = [
        sys.executable,  # use current python interpreter
        "readroot/read_header.py",
        file_path,
        "--date",
        date,
        "--out_dir_base",
        out_dir_base,
    ]
    return cmd


def make_readtrace_command(file_path: str, date: str, out_dir_base: str, left: int = 0, right: int = 512) -> list[str]:
    """Construct the command for read_trace with file_path, left, right, date, out_dir_base parameters."""
    cmd = [
        sys.executable,  # use current python interpreter
        "readroot/read_trace.py",
        file_path,
        "--left",
        str(left),
        "--right",
        str(right),
        "--date",
        date,
        "--out_dir_base",
        out_dir_base,
    ]
    return cmd


def make_main_command(
    file_path: str,
    date: str,
    out_dir_base: str,
    with_signal: bool = False,
    channel: str = 'X',
    run_matching: bool = False,
    run_pwm: bool = False,
    run_swm: bool = False,
    force_recompute: bool = False,
) -> list[str]:
    """Construct the command for: python main.py Reco_Dir/yyyy/mm/dd/Trigger_xxx.yaml yyyy-mm-dd --out-dir-base"""

    out_dir = Path(out_dir_base) / date
    # print(out_dir)  # Removed debug print

    file_name = os.path.basename(file_path).replace('.root', '')

    if with_signal:
        matching_file = out_dir / (file_name + f'_{channel}.yaml')
    else:
        matching_file = out_dir / (file_name + '.yaml')
    # print(matching_file)  # Removed debug print

    cmd = [
        sys.executable,
        "main.py",
        str(matching_file),
        "--fig_name",
        str(file_name),
    ]
    if with_signal:
        cmd.append("--with-signal")
    if run_matching:
        cmd.append("--run-matching")
    if run_pwm:
        cmd.append("--run-pwm")
    if run_swm:
        cmd.append("--run-swm")
    if force_recompute:
        cmd.append("--force-recompute")
    return cmd


def merge_images_to_pdf(pdf_basename: str, with_signal: bool, search_dir: str | Path) -> bool:
    """Find images for the date (png) and merge into a single PDF at pdf_path.

    If `search_dir` is provided, look for PNGs under that directory (non-recursive) using
    patterns that include the date. If no files are found there, fall back to searching the
    current working directory as before.

    Returns True on success, False if nothing was done or PIL not available.
    """
    if not PIL_AVAILABLE:
        logger.warning(f"Pillow not available: skipping PDF creation. Install pillow to enable this feature.")
        return False

    pattern_suffix = "*_with_signal.png" if with_signal else "*.png"
    cwd_pattern = str(Path(f"{pdf_basename}{pattern_suffix}"))

    pdf_path = pdf_basename + ".pdf"
    png_files: list[str] = []
    if search_dir is not None:
        search_root = Path(search_dir)
        base_path = Path(pdf_basename)
        try:
            relative_base = base_path.relative_to(search_root)
        except ValueError:
            relative_base = base_path

        search_pattern = str(search_root / f"{relative_base}{pattern_suffix}")
        png_files.extend(glob.glob(search_pattern))
        if not png_files:
            logger.info(f"No PNG images found under search_dir={search_root}, falling back to cwd pattern")
            png_files.extend(glob.glob(cwd_pattern))
    else:
        png_files.extend(glob.glob(cwd_pattern))

    if not png_files:
        logger.warning(f"No PNG images found for {pdf_path}, skipping PDF creation")
        return False

    # remove duplicates and ensure sorted order
    png_files = sorted(set(png_files))
    logger.info(f"Found {len(png_files)} images to merge into PDF: {pdf_path}")

    images = []
    for f in png_files:
        try:
            if PIL_AVAILABLE:
                im = Image.open(f).convert("RGB")
                images.append(im)
        except Exception as e:
            logger.warning(f"Warning: failed to open image {f}: {e}")

    if not images:
        logger.info(f"No valid images to write for {pdf_path}")
        return False

    first, rest = images[0], images[1:]
    try:
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        if PIL_AVAILABLE:
            first.save(pdf_path, "PDF", save_all=True, append_images=rest)
        logger.info(f"Wrote PDF: {pdf_path} (from images: {len(images)})")
        return True
    except Exception as e:
        logger.error(f"Failed to write PDF {pdf_path}: {e}")
        return False


def process_date(task: TaskSpec) -> TaskResult:
    """Worker function for a single date.

    task: TaskSpec(file_path, date, out_dir_base, do_run, skip_read, only_read, with_signal, left, right, channel)
    Returns (date_iso_str, exit_code)
    """
    file_path = task.file_path
    date = task.date
    out_dir_base = task.out_dir_base
    do_run = task.do_run
    skip_read = task.skip_read
    only_read = task.only_read
    with_signal = task.with_signal
    left = task.left
    right = task.right
    channel = task.channel
    run_matching = task.run_matching
    run_pwm = task.run_pwm
    run_swm = task.run_swm
    force_recompute = task.force_recompute

    # Log/read command
    if skip_read:
        logger.info(f"Skipping read_header for {file_path}")
    else:
        if with_signal:
            readtrace_cmd = make_readtrace_command(file_path, date, out_dir_base, left, right)
            logger.info(f"{' '.join(readtrace_cmd)}")
            
            if do_run:
                try:
                    p = subprocess.run(readtrace_cmd, check=False)
                    if p.returncode != 0:
                        logger.error(f"Command failed for {file_path} with exit {p.returncode}")
                        logger.info(f"Aborting task before main.py due to read stage failure: {file_path}")
                        return (file_path, p.returncode)
                except FileNotFoundError:
                    logger.error(f"Executable not found when running: {readtrace_cmd[0]}")
                    logger.info(f"Aborting task before main.py due to read stage executable error: {file_path}")
                    return (file_path, 2)
        else:
            readheader_cmd = make_readheader_command(file_path, date, out_dir_base)
            logger.info(f"{' '.join(readheader_cmd)}")

            if do_run:
                try:
                    p = subprocess.run(readheader_cmd, check=False)
                    if p.returncode != 0:
                        logger.error(f"Command failed for {file_path} with exit {p.returncode}")
                        logger.info(f"Aborting task before main.py due to read stage failure: {file_path}")
                        return (file_path, p.returncode)
                except FileNotFoundError:
                    logger.error(f"Executable not found when running: {readheader_cmd[0]}")
                    logger.info(f"Aborting task before main.py due to read stage executable error: {file_path}")
                    return (file_path, 2)

    if only_read:
        logger.info(f"Skipping main.py for {file_path} (only-read)")
    else:
        main_cmd = make_main_command(
            file_path,
            date,
            out_dir_base,
            with_signal,
            channel,
            run_matching,
            run_pwm,
            run_swm,
            force_recompute,
        )
        logger.info(f"{' '.join(main_cmd)}")
        
        if do_run:
            try:
                p = subprocess.run(main_cmd, check=False)
                if p.returncode != 0:
                    logger.error(f"Command failed for {file_path} with exit {p.returncode}")
                    return (file_path, p.returncode)
            except FileNotFoundError:
                logger.error(f"Executable not found when running: {main_cmd[0]}")
                return (file_path, 2)
            try:
                basename = os.path.basename(file_path)
                pdf_basename = f"{out_dir_base}/{date}/{basename.replace('.root', '')}"
                logger.info("Attempting to merge images to PDF...")
                merge_images_to_pdf(pdf_basename, with_signal, out_dir_base)
            except Exception as e:
                logger.error(f"Error merging images to PDF for {basename}: {e}")
    return (file_path, 0)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for loop orchestration."""
    ap = argparse.ArgumentParser(description="Run read_header and find_events over dates")
    ap.add_argument("start", help="start time YYYY-MM-DDThh:mm:ss")
    ap.add_argument("end", help="end time YYYY-MM-DDThh:mm:ss")
    ap.add_argument("--base-path", default="/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD", help="value for --base-path passed to script")
    ap.add_argument("--out-dir-base", dest="out_dir_base", default="../Reco_Dir", help="value for --out-dir-base passed to script")
    ap.add_argument("--run", action="store_true", help="actually run the commands instead of printing them")
    ap.add_argument("--jobs", type=int, default=1, help="number of worker processes to use (1 = no parallelism)")
    ap.add_argument("--limit", type=int, default=0, help="limit number of files to show/run (0 = all)")
    ap.add_argument("--skip-read", action="store_true", help="skip the read_header stage")
    ap.add_argument("--only-read", action="store_true", help="Execute only the read_header stage (skip main.py and image merging)")
    ap.add_argument(
        "--with-signal",
        dest="with_signal",
        action="store_true",
        help="Whether to process signal amplitude related processes (such as signal amplitude fitting, signal related plotting, etc.). Adding this parameter will enable signal-related reading and analysis logic, disabled by default."
    )
    ap.add_argument("--left", type=int, default=0, help="left parameter for read_trace.py (default 0)")
    ap.add_argument("--right", type=int, default=512, help="right parameter for read_trace.py (default 512)")
    ap.add_argument("--channel", choices=['F', 'X', 'Y', 'Z', 'XY'], default='X', help="Channel suffix for matching files when --with-signal is set (default X)")
    ap.add_argument("--run-matching", action="store_true", help="pass --run-matching through to main.py")
    ap.add_argument("--run-pwm", action="store_true", help="pass --run-pwm through to main.py")
    ap.add_argument("--run-swm", action="store_true", help="pass --run-swm through to main.py")
    ap.add_argument("--force-recompute", action="store_true", help="pass --force-recompute through to main.py")
    return ap


def parse_cli_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.jobs < 1:
        ap.error("--jobs must be >= 1")

    return args


def apply_task_limit(tasks: list[TaskSpec], limit: int) -> list[TaskSpec]:
    """Apply optional task list cap from CLI."""
    if limit > 0:
        tasks = tasks[:limit]
        logger.info(f"Limited to {len(tasks)} tasks due to --limit flag")
    return tasks


def summarize_results(results: list[TaskResult], start: datetime.datetime, end: datetime.datetime) -> int:
    """Aggregate worker results and compute final process exit code."""
    count = len(results)
    exit_code = 0
    for ds, code in results:
        if code != 0:
            logger.error(f"Task for {ds} returned non-zero exit {code}")
            exit_code = 2

    logger.info(f"Processed {count} files from {start.strftime('%Y-%m-%dT%H:%M:%S')} to {end.strftime('%Y-%m-%dT%H:%M:%S')}")
    return exit_code


def main() -> int:
    """Run the loop CLI workflow and return a process exit code."""
    args = parse_cli_args()

    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    logger.info(f"Processing from {start.strftime('%Y-%m-%dT%H:%M:%S')} to {end.strftime('%Y-%m-%dT%H:%M:%S')}")
    if start > end:
        build_arg_parser().error("start must be <= end")

    file_start = int(start.strftime("%Y%m%d%H%M%S"))
    file_end = int(end.strftime("%Y%m%d%H%M%S"))
    logger.info(f"File range: {file_start} to {file_end}")
    
    # Validate that base path exists
    if not os.path.exists(args.base_path):
        logger.error(f"Base path does not exist: {args.base_path}")
        return 1
    tasks = build_tasks(start, end, args)
    tasks = apply_task_limit(tasks, args.limit)
    
    # When jobs==1 run sequentially; otherwise use multiprocessing.
    results = execute_tasks(tasks, args.jobs)

    return summarize_results(results, start, end)


if __name__ == "__main__":
    raise SystemExit(main())