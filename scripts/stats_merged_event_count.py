#!/usr/bin/env python3
"""Count events in RUN merged YAML files and plot daily trends.

The script scans files matching ``Trigger_YYYYMMDD_RUNxxx_merged.yaml``
under a Reco directory, counts top-level events in each file, writes a CSV,
and generates a time-series plot where each run number is a separate line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots
import yaml

from logger_config import logger


plt.style.use(["science", "grid", "notebook"])


FILE_PATTERN = re.compile(r"^Trigger_(\d{8})_RUN(\d+)_merged\.yaml$")
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@dataclass(frozen=True)
class EventCountRow:
    """One daily event-count data point for one run number."""

    date: dt.date
    run_number: int
    event_count: int
    file_name: str


def parse_cli_date(value: str) -> dt.date:
    """Parse date string in supported formats."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date format: {value!r}. Use YYYY-MM-DD, YYYY/MM/DD, or YYYYMMDD"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Count per-file events from Trigger_YYYYMMDD_RUNxxx_merged.yaml "
            "and plot per-run daily trends."
        )
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Directory containing merged YAML files (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--date",
        type=parse_cli_date,
        default=None,
        help="Single date to process (sets --start-date and --end-date automatically). "
             "e.g. 2026-01-01. Mutually exclusive with --start-date/--end-date.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_cli_date,
        default=None,
        help="Optional start date filter (inclusive)",
    )
    parser.add_argument(
        "--end-date",
        type=parse_cli_date,
        default=None,
        help="Optional end date filter (exclusive)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output CSV path. Default: <reco-dir>/"
            "run_merged_event_count_daily_<start-date>_<end-date>.csv"
        ),
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help=(
            "Output plot path. Default: <reco-dir>/"
            "run_merged_event_count_daily_<start-date>_<end-date>.png"
        ),
    )
    return parser


def extract_file_meta(file_path: Path) -> Optional[Dict[str, int | dt.date]]:
    """Extract date and run number from file name using strict pattern."""
    match = FILE_PATTERN.match(file_path.name)
    if match is None:
        return None

    date_text, run_text = match.groups()
    file_date = dt.datetime.strptime(date_text, "%Y%m%d").date()
    return {
        "date": file_date,
        "run_number": int(run_text),
    }


def in_date_range(
    file_date: dt.date,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> bool:
    """Check half-open date range match: start <= date < end."""
    if start_date is not None and file_date < start_date:
        return False
    if end_date is not None and file_date >= end_date:
        return False
    return True


def build_output_stem(
    prefix: str,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> str:
    """Build an output filename stem with an optional date suffix."""
    if start_date is None and end_date is None:
        return prefix

    start_text = start_date.strftime("%Y%m%d") if start_date is not None else "all"
    end_text = end_date.strftime("%Y%m%d") if end_date is not None else "all"
    return f"{prefix}_{start_text}_{end_text}"


def _count_events_line_scan(file_path: Path) -> int:
    """Count top-level YAML keys by scanning for unindented ``key:`` lines.

    For merged YAML files the top-level structure is a flat dict::

        '220921':
          datetime: ...
          du_id: ...
        '220922':
          ...

    We count lines that are non-empty, do not start with ``#`` (comment) or
    whitespace (indented = nested content), and end with ``:`` (key indicator).

    This avoids loading the entire YAML tree into memory and is ~100x faster
    than :func:`yaml.safe_load` for files > 100 MB.
    """
    count = 0
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            # Fast-path: check first char to skip most lines quickly
            if not line or line[0] in ("#", " ", "\n"):
                continue
            # Only lines like "'220921':" or "220921:" reach here
            if line.rstrip("\n").endswith(":"):
                count += 1
    return count


def count_events_in_yaml(file_path: Path) -> int:
    """Count top-level events in one merged YAML file.

    Uses a fast line-scan approach for large files and falls back to full
    YAML parsing for small / edge-case files.
    """
    # Use line-scan for files larger than 10 MB — full YAML parse is too slow.
    file_size = file_path.stat().st_size
    if file_size > 10 * 1024 * 1024:
        return _count_events_line_scan(file_path)

    # Small file: full parse for correctness (handles edge cases like
    # empty files, list payloads, etc.)
    with file_path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.load(file_obj, Loader=YAML_LOADER)

    if loaded is None:
        return 0
    if isinstance(loaded, dict):
        return len(loaded)
    if isinstance(loaded, list):
        return len(loaded)

    raise ValueError(
        f"Unsupported YAML payload type in {file_path}: {type(loaded).__name__}"
    )


def collect_event_rows(
    reco_dir: Path,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> List[EventCountRow]:
    """Collect event-count rows from all matching merged YAML files."""
    rows: List[EventCountRow] = []
    candidates = sorted(reco_dir.glob("Trigger_*_RUN*_merged.yaml"))
    total_candidates = len(candidates)

    logger.info(
        "Start scanning merged YAML files: reco_dir={}, candidates={}",
        reco_dir,
        total_candidates,
    )

    for index, file_path in enumerate(candidates, start=1):
        metadata = extract_file_meta(file_path)
        if metadata is None:
            continue

        file_date = metadata["date"]
        run_number = metadata["run_number"]
        if not isinstance(file_date, dt.date) or not isinstance(run_number, int):
            continue

        if not in_date_range(file_date, start_date, end_date):
            continue

        try:
            event_count = count_events_in_yaml(file_path)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Skip invalid YAML file {}: {}", file_path, exc)
            continue

        rows.append(
            EventCountRow(
                date=file_date,
                run_number=run_number,
                event_count=event_count,
                file_name=file_path.name,
            )
        )

        logger.info(
            "Counted file: [{}/{}] file={} date={} run={} events={}",
            index,
            total_candidates,
            file_path.name,
            file_date,
            run_number,
            event_count,
        )

        if index % 20 == 0 or index == total_candidates:
            logger.info(
                "Progress summary: processed={}/{} collected_rows={}",
                index,
                total_candidates,
                len(rows),
            )

    rows.sort(key=lambda row: (row.date, row.run_number, row.file_name))
    logger.info("Finished scanning files: collected_rows={}", len(rows))
    return rows


def write_rows_csv(rows: Iterable[EventCountRow], output_csv: Path) -> None:
    """Write aggregated rows to CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "date,run_number,event_count,file_name\n"

    with output_csv.open("w", encoding="utf-8") as file_obj:
        file_obj.write(header)
        for row in rows:
            file_obj.write(
                f"{row.date.strftime('%Y-%m-%d')},{row.run_number},"
                f"{row.event_count},{row.file_name}\n"
            )


def group_rows_by_run(rows: Sequence[EventCountRow]) -> DefaultDict[int, List[EventCountRow]]:
    """Group rows by run number."""
    grouped: DefaultDict[int, List[EventCountRow]] = defaultdict(list)
    for row in rows:
        grouped[row.run_number].append(row)

    for run_rows in grouped.values():
        run_rows.sort(key=lambda row: row.date)
    return grouped


def plot_rows(rows: Sequence[EventCountRow], output_png: Path) -> None:
    """Plot per-run daily event counts."""
    if not rows:
        raise ValueError("No rows available for plotting")

    grouped = group_rows_by_run(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    for run_number in sorted(grouped.keys()):
        run_rows = grouped[run_number]
        x_values = [dt.datetime.combine(row.date, dt.time()) for row in run_rows]
        y_values = [row.event_count for row in run_rows]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"RUN{run_number}",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Event Count")
    ax.set_title("Daily Event Count per RUN (merged YAML)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ── Resolve --date shorthand ──────────────────────────────────────────
    if args.date is not None:
        if args.start_date is not None or args.end_date is not None:
            parser.error("--date is mutually exclusive with --start-date/--end-date")
        args.start_date = args.date
        args.end_date = args.date + dt.timedelta(days=1)

    if args.start_date is not None and args.end_date is not None:
        if args.start_date >= args.end_date:
            parser.error("--start-date must be earlier than --end-date")

    reco_dir = Path(args.reco_dir)
    if not reco_dir.is_dir():
        parser.error(f"Reco directory does not exist: {reco_dir}")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else reco_dir
        / f"{build_output_stem('run_merged_event_count_daily', args.start_date, args.end_date)}.csv"
    )
    output_png = (
        Path(args.output_png)
        if args.output_png is not None
        else reco_dir
        / f"{build_output_stem('run_merged_event_count_daily', args.start_date, args.end_date)}.png"
    )

    logger.info(
        "Run config: reco_dir={} start_date={} end_date={} output_csv={} output_png={}",
        reco_dir,
        args.start_date,
        args.end_date,
        output_csv,
        output_png,
    )

    rows = collect_event_rows(reco_dir, args.start_date, args.end_date)
    if not rows:
        logger.warning("No matching files found under {}", reco_dir)
        return 1

    logger.info("Writing CSV output ...")
    write_rows_csv(rows, output_csv)
    logger.info("Rendering plot ...")
    plot_rows(rows, output_png)

    logger.info("Rows written: {}", len(rows))
    logger.info("CSV output: {}", output_csv)
    logger.info("Plot output: {}", output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
