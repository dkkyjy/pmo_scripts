#!/usr/bin/env python3
"""Compare event counts across merged/matched/fingerprint CSV files.

The script reads stage daily CSV files under a Reco directory and compares
event counts by the same ``(date, run_number)`` key:
- run_merged_event_count_daily_*.csv
- run_matched_event_count_daily_*.csv
- run_fingerprint_event_count_daily_*.csv

It writes a compare CSV and generates per-run trend plots.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots

from logger_config import logger


plt.style.use(["science", "grid", "notebook"])


STAGE_NAMES: Tuple[str, str, str, str] = (
    "merged", "matched", "fingerprint", "swm",
)
STAGE_CSV_PATTERNS: Dict[str, str] = {
    "merged": "run_merged_event_count_daily_{start}_{end}.csv",
    "matched": "run_matched_event_count_daily_{start}_{end}.csv",
    "fingerprint": "run_fingerprint_event_count_daily_{start}_{end}.csv",
    "swm": "run_swm_event_count_daily_{start}_{end}.csv",
}

OUTPUT_FILE_STEM = "run_stage_event_count_compare"

STAGE_RECORD_FIELDS = ["date", "run_number", "stage", "event_count", "source_csv"]
COMPARE_ROW_FIELDS = [
    "date",
    "run_number",
    "merged_count",
    "matched_count",
    "fingerprint_count",
    "swm_count",
]

StageRecord = namedtuple("StageRecord", STAGE_RECORD_FIELDS)
CompareRow = namedtuple("CompareRow", COMPARE_ROW_FIELDS)

CSV_DATE_FORMAT = "%Y-%m-%d"
CSV_DATE_FORMAT_COMPACT = "%Y%m%d"
CSV_DATE_FORMAT_SLASH = "%Y/%m/%d"

CSV_OUTPUT_HEADERS = (
    "date",
    "run_number",
    "merged_count",
    "matched_count",
    "fingerprint_count",
    "swm_count",
)


def parse_cli_date(value: str) -> dt.date:
    """Parse date string in supported formats."""
    for fmt in (CSV_DATE_FORMAT, CSV_DATE_FORMAT_SLASH, CSV_DATE_FORMAT_COMPACT):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date format: {value!r}. Use YYYY-MM-DD, YYYY/MM/DD, or YYYYMMDD"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare stage event counts from run_*_event_count_daily*.csv files "
            "and plot per-run trends over date."
        )
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Directory containing run_*_event_count_daily*.csv files (default: ../Reco_Dir)",
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
            "Output CSV path. "
            "Default: <reco-dir>/run_stage_event_count_compare_<start-date>_<end-date>.csv"
        ),
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help=(
            "Output plot path. "
            "Default: <reco-dir>/run_stage_event_count_compare_<start-date>_<end-date>.png"
        ),
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Filter to a specific RUN number for plotting (e.g. 10386). "
             "If not given, all RUNs are plotted.",
    )
    return parser


def in_date_range(
    file_date: dt.date,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> bool:
    """Check half-open date range: start <= date < end."""
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


def resolve_stage_csv(
    reco_dir: Path,
    stage: str,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> Optional[Path]:
    """Resolve the best-matching stage CSV path for current date range."""
    pattern = STAGE_CSV_PATTERNS[stage]

    if start_date is not None and end_date is not None:
        start_text = start_date.strftime(CSV_DATE_FORMAT_COMPACT)
        end_text = end_date.strftime(CSV_DATE_FORMAT_COMPACT)
        candidate = reco_dir / pattern.format(start=start_text, end=end_text)
        if candidate.is_file():
            return candidate

    # Fallback: glob-match any CSV for this stage (ignoring date range suffix).
    for csv_path in sorted(reco_dir.glob(f"run_{stage}_event_count*.csv")):
        if csv_path.is_file():
            return csv_path

    return None


def parse_csv_row_date(value: str) -> dt.date:
    """Parse date field from CSV row."""
    for fmt in (CSV_DATE_FORMAT, CSV_DATE_FORMAT_SLASH, CSV_DATE_FORMAT_COMPACT):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid CSV date value: {value!r}")


def read_stage_records(
    csv_path: Path,
    stage: str,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> List[StageRecord]:
    """Read and filter stage records from one CSV file."""
    records: List[StageRecord] = []

    with csv_path.open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for raw in reader:
            try:
                row_date = parse_csv_row_date(raw["date"])
                run_number = int(raw["run_number"])
                event_count = int(raw["event_count"])
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.warning("Skip invalid CSV row in {}: {}", csv_path, exc)
                continue

            if not in_date_range(row_date, start_date, end_date):
                continue

            records.append(
                StageRecord(
                    date=row_date,
                    run_number=run_number,
                    stage=stage,
                    event_count=event_count,
                    source_csv=csv_path.name,
                )
            )

    return records


def collect_stage_records(
    reco_dir: Path,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> List[StageRecord]:
    """Collect event counts from merged/matched/fingerprint CSV files."""
    records: List[StageRecord] = []

    logger.info(
        "Start scanning stage CSV files: reco_dir={} start_date={} end_date={}",
        reco_dir,
        start_date,
        end_date,
    )

    for stage in STAGE_NAMES:
        csv_path = resolve_stage_csv(reco_dir, stage, start_date, end_date)
        if csv_path is None:
            logger.warning("Stage CSV not found for stage={}", stage)
            continue

        logger.info("Reading stage CSV: stage={} file={}", stage, csv_path)
        stage_records = read_stage_records(csv_path, stage, start_date, end_date)
        logger.info(
            "Loaded stage records: stage={} rows={} source={}",
            stage,
            len(stage_records),
            csv_path,
        )
        records.extend(stage_records)

    records.sort(key=lambda r: (r.date, r.run_number, r.stage))
    logger.info("Finished scanning stage CSV files: accepted_records={}", len(records))
    return records


def build_compare_rows(records: Sequence[StageRecord]) -> List[CompareRow]:
    """Build compare rows using merged keys as the canonical key set."""
    merged_map: Dict[Tuple[dt.date, int], int] = {}
    matched_map: Dict[Tuple[dt.date, int], int] = {}
    fingerprint_map: Dict[Tuple[dt.date, int], int] = {}
    swm_map: Dict[Tuple[dt.date, int], int] = {}

    for record in records:
        key = (record.date, record.run_number)
        if record.stage == "merged":
            merged_map[key] = record.event_count
        elif record.stage == "matched":
            matched_map[key] = record.event_count
        elif record.stage == "fingerprint":
            fingerprint_map[key] = record.event_count
        elif record.stage == "swm":
            swm_map[key] = record.event_count

    all_keys = sorted(
        merged_map.keys()
        | matched_map.keys()
        | fingerprint_map.keys()
        | swm_map.keys()
    )
    rows: List[CompareRow] = []
    for date_value, run_number in all_keys:
        key = (date_value, run_number)
        rows.append(
            CompareRow(
                date=date_value,
                run_number=run_number,
                merged_count=merged_map.get(key, 0),
                matched_count=matched_map.get(key, 0),
                fingerprint_count=fingerprint_map.get(key, 0),
                swm_count=swm_map.get(key, 0),
            )
        )

    return rows


def write_compare_csv(rows: Iterable[CompareRow], output_csv: Path) -> None:
    """Write compare rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(CSV_OUTPUT_HEADERS)
        for row in rows:
            writer.writerow([
                row.date.strftime(CSV_DATE_FORMAT),
                row.run_number,
                row.merged_count,
                row.matched_count,
                row.fingerprint_count,
                row.swm_count,
            ])


def _to_plot_y(value: int) -> float:
    """Convert count to plottable float."""
    return float(value)


def plot_compare_rows(
    rows: Sequence[CompareRow],
    output_png: Path,
    run_number: Optional[int] = None,
) -> None:
    """Plot stage-count comparison grouped by run number.

    Parameters
    ----------
    run_number : int, optional
        If given, only the specified RUN is plotted (single-panel).
    """
    if not rows:
        raise ValueError("No compare rows available for plotting")

    by_run: Dict[int, List[CompareRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_number, []).append(row)

    if run_number is not None:
        if run_number not in by_run:
            raise ValueError(
                f"RUN{run_number} not found in data (available: {sorted(by_run.keys())})"
            )
        by_run = {run_number: by_run[run_number]}

    run_numbers = sorted(by_run.keys())
    total_runs = len(run_numbers)
    ncols = 2 if total_runs > 1 else 1
    nrows = math.ceil(total_runs / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.8 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    stage_colors = {
        "merged": "tab:blue",
        "matched": "tab:orange",
        "fingerprint": "tab:green",
        "swm": "tab:red",
    }

    for idx, run_number in enumerate(run_numbers):
        ax = axes[idx // ncols][idx % ncols]
        run_rows = sorted(by_run[run_number], key=lambda row: row.date)
        x_values = [dt.datetime.combine(row.date, dt.time()) for row in run_rows]

        merged_values = [_to_plot_y(row.merged_count) for row in run_rows]
        matched_values = [_to_plot_y(row.matched_count) for row in run_rows]
        fingerprint_values = [_to_plot_y(row.fingerprint_count) for row in run_rows]
        swm_values = [_to_plot_y(row.swm_count) for row in run_rows]

        ax.plot(
            x_values,
            merged_values,
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            color=stage_colors["merged"],
            label="merged",
        )
        ax.plot(
            x_values,
            matched_values,
            marker="s",
            linewidth=1.6,
            markersize=3.5,
            color=stage_colors["matched"],
            label="matched",
        )
        ax.plot(
            x_values,
            fingerprint_values,
            marker="^",
            linewidth=1.6,
            markersize=3.5,
            color=stage_colors["fingerprint"],
            label="fingerprint",
        )
        ax.plot(
            x_values,
            swm_values,
            marker="d",
            linewidth=1.6,
            markersize=3.5,
            color=stage_colors["swm"],
            label="swm",
        )

        ax.set_title(f"RUN{run_number}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Event Count")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(loc="best", fontsize=8)

    total_axes = nrows * ncols
    for idx in range(total_runs, total_axes):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")

    if run_number is not None:
        suptitle = (
            f"Event Count Comparison — RUN{run_number} "
            "(merged vs matched vs fingerprint vs swm)"
        )
    else:
        suptitle = (
            "Event Count Comparison by RUN "
            "(merged vs matched vs fingerprint vs swm)"
        )
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=200, bbox_inches="tight")
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

    output_stem = build_output_stem(OUTPUT_FILE_STEM, args.start_date, args.end_date)
    if args.run_number is not None:
        output_stem += f"_RUN{args.run_number}"

    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else reco_dir / f"{output_stem}.csv"
    )
    output_png = (
        Path(args.output_png)
        if args.output_png is not None
        else reco_dir / f"{output_stem}.png"
    )

    logger.info(
        "Run config: reco_dir={} start_date={} end_date={} output_csv={} output_png={}",
        reco_dir,
        args.start_date,
        args.end_date,
        output_csv,
        output_png,
    )

    records = collect_stage_records(reco_dir, args.start_date, args.end_date)
    if not records:
        logger.warning("No matching stage files found under {}", reco_dir)
        return 1

    rows = build_compare_rows(records)
    logger.info("Built compare rows: {}", len(rows))

    logger.info("Writing CSV output ...")
    write_compare_csv(rows, output_csv)

    logger.info("Rendering plot ...")
    plot_compare_rows(rows, output_png, run_number=args.run_number)

    logger.info("CSV output: {}", output_csv)
    logger.info("Plot output: {}", output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
