#!/usr/bin/env python3
"""Plot monthly event-count distributions from run fingerprint CSV files.

The script reads monthly CSV files in a reco directory and renders a 3x3 grid
for months from 2025-07 to 2026-03 by default. In each subplot, x-axis is date,
y-axis is event count, and line colors represent run_number.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots

from logger_config import logger


plt.style.use(["science", "grid", "notebook"])


MONTH_FORMAT = "%Y-%m"
FILE_PATTERNS = "run_fingerprint_event_count_daily_{start}_{end}.csv"
MERGED_FILE_PATTERNS = "run_merged_event_count_daily_{start}_{end}.csv"

ROW_FIELDS = ["date", "run_number", "event_count", "file_name", "source_csv"]
EventRow = namedtuple("EventRow", ROW_FIELDS)


def parse_month(value: str) -> dt.date:
    """Parse month text to first day of month."""
    try:
        month_date = dt.datetime.strptime(value, MONTH_FORMAT).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid month format: {value!r}. Use YYYY-MM"
        ) from exc
    return month_date.replace(day=1)


def parse_csv_date(value: str) -> dt.date:
    """Parse CSV date field as YYYY-MM-DD."""
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def month_start_list(start_month: dt.date, end_month: dt.date) -> List[dt.date]:
    """Return month starts in an inclusive month range."""
    months: List[dt.date] = []
    cursor = start_month
    while cursor <= end_month:
        months.append(cursor)
        if cursor.month == 12:
            cursor = dt.date(cursor.year + 1, 1, 1)
        else:
            cursor = dt.date(cursor.year, cursor.month + 1, 1)
    return months


def next_month(month_start: dt.date) -> dt.date:
    """Return the first day of next month."""
    if month_start.month == 12:
        return dt.date(month_start.year + 1, 1, 1)
    return dt.date(month_start.year, month_start.month + 1, 1)


def resolve_month_csv(reco_dir: Path, month_start: dt.date) -> Optional[Path]:
    """Resolve the preferred monthly CSV path for one month."""
    start_text = month_start.strftime("%Y%m%d")
    end_text = next_month(month_start).strftime("%Y%m%d")

    candidate = reco_dir / FILE_PATTERNS.format(start=start_text, end=end_text)
    if candidate.is_file():
        return candidate

    return None


def resolve_merged_month_csv(reco_dir: Path, month_start: dt.date) -> Optional[Path]:
    """Resolve merged monthly CSV path for one month."""
    start_text = month_start.strftime("%Y%m%d")
    end_text = next_month(month_start).strftime("%Y%m%d")

    candidate = reco_dir / MERGED_FILE_PATTERNS.format(start=start_text, end=end_text)
    if candidate.is_file():
        return candidate

    return None


def read_month_rows(csv_path: Path, month_start: dt.date) -> List[EventRow]:
    """Read rows from one monthly CSV and keep rows within the month."""
    rows: List[EventRow] = []
    month_end = next_month(month_start)

    with csv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for raw in reader:
            try:
                row_date = parse_csv_date(raw["date"])
                run_number = int(raw["run_number"])
                event_count = int(raw["event_count"])
                file_name = raw.get("file_name", "")
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.warning("Skip invalid CSV row in {}: {}", csv_path, exc)
                continue

            if row_date < month_start or row_date >= month_end:
                continue

            rows.append(
                EventRow(
                    date=row_date,
                    run_number=run_number,
                    event_count=event_count,
                    file_name=file_name,
                    source_csv=csv_path.name,
                )
            )

    return rows


def read_month_keys(csv_path: Path, month_start: dt.date) -> set[tuple[dt.date, int]]:
    """Read (date, run_number) keys from one monthly CSV within month range."""
    keys: set[tuple[dt.date, int]] = set()
    month_end = next_month(month_start)

    with csv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for raw in reader:
            try:
                row_date = parse_csv_date(raw["date"])
                run_number = int(raw["run_number"])
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.warning("Skip invalid CSV row in {}: {}", csv_path, exc)
                continue

            if row_date < month_start or row_date >= month_end:
                continue

            keys.add((row_date, run_number))

    return keys


def fill_missing_rows_from_merged(
    month_rows: List[EventRow],
    merged_keys: set[tuple[dt.date, int]],
    source_csv_name: str,
) -> List[EventRow]:
    """Fill missing fingerprint rows with zero count using merged keys as reference."""
    fingerprint_keys = {(row.date, row.run_number) for row in month_rows}
    missing_keys = sorted(merged_keys - fingerprint_keys)

    if not missing_keys:
        return month_rows

    filled_rows = list(month_rows)
    for row_date, run_number in missing_keys:
        filled_rows.append(
            EventRow(
                date=row_date,
                run_number=run_number,
                event_count=0,
                file_name="",
                source_csv=source_csv_name,
            )
        )

    return filled_rows


def collect_rows_by_month(
    reco_dir: Path,
    months: Sequence[dt.date],
) -> DefaultDict[dt.date, List[EventRow]]:
    """Collect data rows grouped by month start date."""
    grouped: DefaultDict[dt.date, List[EventRow]] = defaultdict(list)

    for month_start in months:
        csv_path = resolve_month_csv(reco_dir, month_start)
        if csv_path is None:
            logger.warning(
                "Monthly CSV not found for {} (expected *_{}_{}.csv)",
                month_start.strftime(MONTH_FORMAT),
                month_start.strftime("%Y%m%d"),
                next_month(month_start).strftime("%Y%m%d"),
            )
            continue

        logger.info("Reading monthly CSV: {}", csv_path)
        rows = read_month_rows(csv_path, month_start)

        merged_csv_path = resolve_merged_month_csv(reco_dir, month_start)
        if merged_csv_path is not None:
            logger.info("Reading merged reference CSV: {}", merged_csv_path)
            merged_keys = read_month_keys(merged_csv_path, month_start)
            rows = fill_missing_rows_from_merged(rows, merged_keys, csv_path.name)
            logger.info(
                "Aligned rows with merged keys for {}: {}",
                month_start.strftime(MONTH_FORMAT),
                len(rows),
            )
        else:
            logger.warning(
                "Merged reference CSV not found for {} (skip zero-fill)",
                month_start.strftime(MONTH_FORMAT),
            )

        rows.sort(key=lambda row: (row.date, row.run_number, row.file_name))
        grouped[month_start].extend(rows)

        logger.info(
            "Loaded rows for {}: {}",
            month_start.strftime(MONTH_FORMAT),
            len(rows),
        )

    return grouped


def plot_monthly_grid(
    grouped: DefaultDict[dt.date, List[EventRow]],
    months: Sequence[dt.date],
    output_png: Path,
) -> None:
    """Render a 3x3 monthly grid plot."""
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 12),
        squeeze=False,
        sharey="row",
    )

    row_y_limits: Dict[int, tuple[float, float]] = {}
    for row_idx in range(3):
        row_months = months[row_idx * 3 : (row_idx + 1) * 3]
        row_event_counts = [
            row.event_count
            for month_start in row_months
            for row in grouped.get(month_start, [])
        ]

        if row_event_counts:
            y_min = min(row_event_counts)
            y_max = max(row_event_counts)
            if y_min == y_max:
                y_margin = max(1.0, y_max * 0.05)
            else:
                y_margin = (y_max - y_min) * 0.05
            row_y_limits[row_idx] = (y_min - y_margin, y_max + y_margin)
        else:
            row_y_limits[row_idx] = (0.0, 1.0)

    for idx, month_start in enumerate(months):
        ax = axes[idx // 3][idx % 3]
        month_rows = grouped.get(month_start, [])
        panel_number = idx + 1

        ax.set_title(month_start.strftime("%Y-%m"))
        if panel_number in (7, 8, 9):
            ax.set_xlabel("Date")
        else:
            ax.set_xlabel("")

        if panel_number in (1, 4, 7):
            ax.set_ylabel("Event Count")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(row_y_limits[idx // 3])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelrotation=30)

        if not month_rows:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.8,
            )
            continue

        by_run: DefaultDict[int, List[EventRow]] = defaultdict(list)
        for row in month_rows:
            by_run[row.run_number].append(row)

        for run_number in sorted(by_run.keys()):
            run_rows = sorted(by_run[run_number], key=lambda row: row.date)
            x_values = [dt.datetime.combine(row.date, dt.time()) for row in run_rows]
            y_values = [row.event_count for row in run_rows]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=f"RUN{run_number}",
            )

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Monthly Event Count Distribution by RUN\n"
        "(2025-07 to 2026-03)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Read monthly run_fingerprint_event_count CSV files and plot a 3x3 monthly "
            "distribution grid with run-number colors."
        )
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Directory containing run_fingerprint_event_count CSV files",
    )
    parser.add_argument(
        "--start-month",
        type=parse_month,
        default=dt.date(2025, 7, 1),
        help="Inclusive start month in YYYY-MM (default: 2025-07)",
    )
    parser.add_argument(
        "--end-month",
        type=parse_month,
        default=dt.date(2026, 3, 1),
        help="Inclusive end month in YYYY-MM (default: 2026-03)",
    )
    parser.add_argument(
        "--output-png",
        default="../Reco_Dir/run_fingerprint_event_count_monthly_202507_202603_grid.png",
        help="Output PNG path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.start_month > args.end_month:
        parser.error("--start-month must be earlier than or equal to --end-month")

    reco_dir = Path(args.reco_dir)
    if not reco_dir.is_dir():
        parser.error(f"Reco directory does not exist: {reco_dir}")

    months = month_start_list(args.start_month, args.end_month)
    if len(months) != 9:
        parser.error(
            "This script renders a fixed 3x3 grid, month count must be 9 "
            "(for example 2025-07 to 2026-03)."
        )

    logger.info(
        "Run config: reco_dir={} start_month={} end_month={} output_png={}",
        reco_dir,
        args.start_month.strftime(MONTH_FORMAT),
        args.end_month.strftime(MONTH_FORMAT),
        args.output_png,
    )

    grouped = collect_rows_by_month(reco_dir, months)
    output_png = Path(args.output_png)
    plot_monthly_grid(grouped, months, output_png)

    total_rows = sum(len(rows) for rows in grouped.values())
    logger.info("Total loaded rows: {}", total_rows)
    logger.info("Plot output: {}", output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
