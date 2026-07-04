#!/usr/bin/env python3
"""Scan TD/yyyy/mm/dd directories and inventory ROOT files by category.

Classifies ``.root`` files into ``Trigger``, ``Calibration``, or ``other``
based on filename prefix.  For each category + date + run-number combination
the script writes one CSV row with file count and total size, then generates
per-run time-series plots (count vs time, size vs time).

Filename conventions
--------------------
- Trigger:  ``Trigger_<YYYYMMDDHHMMSS>_<RUN>_...``
- Calibration: ``Calibration_..._<YYYYMMDDHHMMSS>_..._<RUN>_...``

Trigger timestamp is extracted from field index 2 (1-based), RUN from field 3.
Calibration timestamp is extracted from field index 6 (1-based), RUN from field 7.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots

from logger_config import logger


plt.style.use(["science", "grid", "notebook"])

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"

# Category definitions: (prefix, timestamp_field_index_1based, run_field_index_1based)
CATEGORY_DEFS: Dict[str, Tuple[int, int]] = {
    "trigger": (2, 3),
    "calibration": (6, 7),
}


@dataclass(frozen=True)
class InventoryRow:
    """One aggregated inventory data point for one category + date + run."""

    timestamp: str
    date: str
    category: str
    run_number: int
    file_count: int
    total_size_bytes: int
    total_size_human: str


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
            "Scan TD/yyyy/mm/dd directories and inventory ROOT files "
            "by category (Trigger / Calibration / other)."
        )
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="Base path containing TD/ directory (default: .)",
    )
    parser.add_argument(
        "--date",
        type=parse_cli_date,
        default=None,
        help="Single date to inventory (sets --start-date and --end-date automatically). "
             "e.g. 2026-01-01. Mutually exclusive with --start-date/--end-date.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_cli_date,
        default=None,
        help="Start date (inclusive), e.g. 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=parse_cli_date,
        default=None,
        help="End date (exclusive), e.g. 2026-02-01",
    )
    parser.add_argument(
        "--output-dir",
        default='../Reco_Dir/root_file_inventory',
        help="Output directory for CSV and PNG files (default: root_file_inventory)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation, only produce CSV",
    )
    parser.add_argument(
        "--daily-summary",
        action="store_true",
        help="Read existing inventory CSV and produce daily aggregation",
    )
    parser.add_argument(
        "--hourly-summary",
        action="store_true",
        help="Read existing inventory CSV and produce hourly aggregation",
    )
    return parser


def classify_file(name: str) -> Optional[str]:
    """Classify a .root file by its filename prefix.

    Returns ``"trigger"``, ``"calibration"``, ``"other"``, or ``None``
    if the file is not a ``.root`` file.
    """
    if not name.endswith(".root"):
        return None
    if name.startswith("Trigger"):
        return "trigger"
    if name.startswith("Calibration"):
        return "calibration"
    return "other"


def parse_timestamp_from_filename(name: str, category: str) -> Optional[str]:
    """Extract timestamp from filename based on category.

    Trigger: field index 2 (1-based) after splitting by ``_``.
    Calibration: field index 6 (1-based) after splitting by ``_``.
    """
    field_index = CATEGORY_DEFS.get(category)
    if field_index is None:
        return None

    idx = field_index[0] - 1  # convert to 0-based
    parts = name.split("_")
    if idx >= len(parts):
        return None

    raw = parts[idx]
    # Validate the raw value looks like a timestamp
    if len(raw) == 14 and raw.isdigit():
        return raw
    return None


def parse_run_from_filename(name: str, category: str) -> Optional[int]:
    """Extract RUN number from filename based on category.

    Trigger: field index 3 (1-based) after splitting by ``_``.
    Calibration: field index 7 (1-based) after splitting by ``_``.
    """
    field_def = CATEGORY_DEFS.get(category)
    if field_def is None:
        return None

    idx = field_def[1] - 1  # convert to 0-based
    parts = name.split("_")
    if idx >= len(parts):
        return None

    raw = parts[idx]
    # Strip .root suffix and non-digit prefix (e.g. "RUN001" → "001" → 1)
    raw = raw.replace(".root", "")
    # Find the first digit sequence
    digits = ""
    for ch in raw:
        if ch.isdigit():
            digits += ch
    if not digits:
        return None
    return int(digits)


def to_human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string (MB or GB)."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def scan_day_directory(
    day_path: Path,
) -> List[Tuple[str, str, int, int]]:
    """Scan a single day directory and return file info tuples.

    Returns list of ``(category, timestamp, run_number, file_size)``.
    """
    results: List[Tuple[str, str, int, int]] = []

    if not day_path.is_dir():
        return results

    for entry in sorted(day_path.iterdir()):
        if not entry.is_file():
            continue

        category = classify_file(entry.name)
        if category is None:
            continue  # not a .root file

        if category == "other":
            # For "other" files, use file mtime as timestamp, no run number
            stat = entry.stat()
            mtime = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
            ts = mtime.strftime(TIMESTAMP_FORMAT)
            results.append((category, ts, 0, stat.st_size))
            logger.info("Other file: {} (size={})", entry.name, stat.st_size)
            continue

        ts = parse_timestamp_from_filename(entry.name, category)
        if ts is None:
            logger.warning(
                "Cannot parse timestamp from {} (category={}), skipping",
                entry.name,
                category,
            )
            continue

        run_number = parse_run_from_filename(entry.name, category)
        if run_number is None:
            logger.warning(
                "Cannot parse RUN from {} (category={}), skipping",
                entry.name,
                category,
            )
            continue

        results.append((category, ts, run_number, entry.stat().st_size))

    return results


def aggregate_rows(
    file_infos: List[Tuple[str, str, int, int]],
) -> List[InventoryRow]:
    """Aggregate file infos by (timestamp, category, run_number).

    Returns sorted list of InventoryRow.
    """
    # Group by (timestamp, category, run_number)
    groups: DefaultDict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for category, ts, run_number, size in file_infos:
        groups[(ts, category, run_number)].append(size)

    rows: List[InventoryRow] = []
    for (ts, category, run_number), sizes in groups.items():
        total_bytes = sum(sizes)
        # Parse timestamp to date
        try:
            ts_dt = dt.datetime.strptime(ts, TIMESTAMP_FORMAT)
            date_str = ts_dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = ts[:8]  # fallback to YYYYMMDD

        rows.append(
            InventoryRow(
                timestamp=ts,
                date=date_str,
                category=category,
                run_number=run_number,
                file_count=len(sizes),
                total_size_bytes=total_bytes,
                total_size_human=to_human_size(total_bytes),
            )
        )

    rows.sort(key=lambda r: (r.timestamp, r.category, r.run_number))
    return rows


_INVENTORY_CSV_FIELDS = (
    "timestamp", "date", "category", "run_number",
    "file_count", "total_size_bytes", "total_size_human",
)


def write_csv(rows: Iterable[InventoryRow], output_path: Path) -> None:
    """Write inventory rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_INVENTORY_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp": row.timestamp,
                "date": row.date,
                "category": row.category,
                "run_number": row.run_number,
                "file_count": row.file_count,
                "total_size_bytes": row.total_size_bytes,
                "total_size_human": row.total_size_human,
            })


def read_csv_rows(csv_path: Path) -> List[InventoryRow]:
    """Read inventory CSV back into InventoryRow list."""
    rows: List[InventoryRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            rows.append(
                InventoryRow(
                    timestamp=line["timestamp"],
                    date=line["date"],
                    category=line["category"],
                    run_number=int(line["run_number"]),
                    file_count=int(line["file_count"]),
                    total_size_bytes=int(line["total_size_bytes"]),
                    total_size_human=line["total_size_human"],
                )
            )
    return rows


def aggregate_daily_rows(rows: Sequence[InventoryRow]) -> List[InventoryRow]:
    """Aggregate inventory rows by (date, category, run_number)."""
    groups: DefaultDict[Tuple[str, str, int], Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in rows:
        key = (r.date, r.category, r.run_number)
        total_count, total_bytes = groups[key]
        groups[key] = (total_count + r.file_count, total_bytes + r.total_size_bytes)

    daily: List[InventoryRow] = []
    for (date_str, category, run_number), (total_count, total_bytes) in groups.items():
        daily.append(
            InventoryRow(
                timestamp=date_str,
                date=date_str,
                category=category,
                run_number=run_number,
                file_count=total_count,
                total_size_bytes=total_bytes,
                total_size_human=to_human_size(total_bytes),
            )
        )

    daily.sort(key=lambda r: (r.date, r.category, r.run_number))
    return daily


def aggregate_hourly_rows(rows: Sequence[InventoryRow]) -> List[InventoryRow]:
    """Aggregate inventory rows by (date, hour, category, run_number).

    Hour is extracted from the timestamp field (HH portion of YYYYMMDDHHMMSS).
    The hour is stored in the ``timestamp`` field as ``YYYYMMDDHH`` for grouping.
    """
    groups: DefaultDict[Tuple[str, str, int], Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in rows:
        hour_key = r.timestamp[:10]  # YYYYMMDDHH
        key = (hour_key, r.category, r.run_number)
        total_count, total_bytes = groups[key]
        groups[key] = (total_count + r.file_count, total_bytes + r.total_size_bytes)

    hourly: List[InventoryRow] = []
    for (hour_key, category, run_number), (total_count, total_bytes) in groups.items():
        date_str = f"{hour_key[:4]}-{hour_key[4:6]}-{hour_key[6:8]}"
        hourly.append(
            InventoryRow(
                timestamp=hour_key,
                date=date_str,
                category=category,
                run_number=run_number,
                file_count=total_count,
                total_size_bytes=total_bytes,
                total_size_human=to_human_size(total_bytes),
            )
        )

    hourly.sort(key=lambda r: (r.timestamp, r.category, r.run_number))
    return hourly


_DAILY_CSV_FIELDS = (
    "date", "category", "run_number",
    "file_count", "total_size_bytes", "total_size_human",
)


def write_daily_csv(rows: Sequence[InventoryRow], output_path: Path) -> None:
    """Write daily-aggregated rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DAILY_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "date": r.date,
                "category": r.category,
                "run_number": r.run_number,
                "file_count": r.file_count,
                "total_size_bytes": r.total_size_bytes,
                "total_size_human": r.total_size_human,
            })


_HOURLY_CSV_FIELDS = (
    "hour_key", "date", "category", "run_number",
    "file_count", "total_size_bytes", "total_size_human",
)


def write_hourly_csv(rows: Sequence[InventoryRow], output_path: Path) -> None:
    """Write hourly-aggregated rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HOURLY_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "hour_key": r.timestamp,
                "date": r.date,
                "category": r.category,
                "run_number": r.run_number,
                "file_count": r.file_count,
                "total_size_bytes": r.total_size_bytes,
                "total_size_human": r.total_size_human,
            })


def group_rows_for_plot(
    rows: Sequence[InventoryRow],
) -> DefaultDict[str, DefaultDict[int, List[InventoryRow]]]:
    """Group rows by (category, run_number) for per-run plotting."""
    grouped: DefaultDict[str, DefaultDict[int, List[InventoryRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row.category][row.run_number].append(row)

    for cat in grouped:
        for run in grouped[cat]:
            grouped[cat][run].sort(key=lambda r: r.timestamp)
    return grouped


# ── Time-series plot configuration presets ──────────────────────────────────

_PlotConfig = Dict[str, object]


def _make_count_config() -> _PlotConfig:
    """Return plot config for file-count time series."""
    return {
        "marker": "o",
        "ylabel": "File Count",
    }


def _make_size_config() -> _PlotConfig:
    """Return plot config for total-size time series."""
    return {
        "marker": "s",
        "ylabel": "Total Size (MB)",
    }


def _make_time_config() -> _PlotConfig:
    """Return plot config for raw-timestamp (``TIMESTAMP_FORMAT``) x-axis."""
    return {
        "xlabel": "Time",
        "x_parser": lambda r: dt.datetime.strptime(r.timestamp, TIMESTAMP_FORMAT),
        "date_fmt": "%m-%d %H:%M",
        "figsize_scale": 12,
    }


def _make_day_config() -> _PlotConfig:
    """Return plot config for daily-aggregated x-axis (``date`` field)."""
    return {
        "xlabel": "Date",
        "x_parser": lambda r: dt.datetime.strptime(r.date, "%Y-%m-%d"),
        "date_fmt": "%m-%d",
        "figsize_scale": 14,
    }


def _make_hour_config() -> _PlotConfig:
    """Return plot config for hourly-aggregated x-axis (``timestamp`` as YYYYMMDDHH)."""
    return {
        "xlabel": "Hour",
        "x_parser": lambda r: dt.datetime.strptime(r.timestamp, "%Y%m%d%H"),
        "date_fmt": "%m-%d %H:00",
        "figsize_scale": 16,
    }


def _plot_time_series(
    rows: Sequence[InventoryRow],
    output_png: Path,
    count_or_size: str,
    time_granularity: str,
) -> None:
    """Shared plot engine for all time-series plots.

    Parameters
    ----------
    rows:
        Inventory rows to plot.
    output_png:
        Destination path for the PNG file.
    count_or_size:
        ``"count"`` or ``"size"`` — selects y-axis data and marker style.
    time_granularity:
        ``"time"``, ``"day"``, or ``"hour"`` — selects x-axis parser and date format.
    """
    if not rows:
        raise ValueError(f"No rows available for {time_granularity} {count_or_size} plot")

    # Resolve config presets
    y_cfg = _make_count_config() if count_or_size == "count" else _make_size_config()
    x_cfg = {
        "time": _make_time_config,
        "day": _make_day_config,
        "hour": _make_hour_config,
    }[time_granularity]()

    grouped = group_rows_for_plot(rows)
    n_categories = len(grouped)
    figsize_w: int = x_cfg["figsize_scale"]  # type: ignore[assignment]
    fig, axes = plt.subplots(
        nrows=n_categories, ncols=1,
        figsize=(figsize_w, 5 * n_categories),
        squeeze=False,
    )

    x_parser = x_cfg["x_parser"]
    date_fmt: str = x_cfg["date_fmt"]  # type: ignore[assignment]
    xlabel: str = x_cfg["xlabel"]  # type: ignore[assignment]
    ylabel: str = y_cfg["ylabel"]  # type: ignore[assignment]
    marker: str = y_cfg["marker"]  # type: ignore[assignment]

    for idx, category in enumerate(sorted(grouped.keys())):
        ax = axes[idx][0]
        run_groups = grouped[category]

        for run_number in sorted(run_groups.keys()):
            run_rows = run_groups[run_number]
            x_values = [x_parser(r) for r in run_rows]
            if count_or_size == "count":
                y_values = [r.file_count for r in run_rows]
            else:
                y_values = [r.total_size_bytes / (1024 ** 2) for r in run_rows]
            ax.plot(
                x_values, y_values,
                marker=marker, linewidth=1.8, markersize=4,
                label=f"RUN{run_number}" if run_number > 0 else "other",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{category.capitalize()} — {ylabel} Over Time")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=200)
    plt.close(fig)


def plot_count_vs_time(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Plot per-category, per-run file count over time."""
    _plot_time_series(rows, output_png, "count", "time")


def plot_size_vs_time(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Plot per-category, per-run total size over time."""
    _plot_time_series(rows, output_png, "size", "time")


def plot_count_vs_day(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Line chart: daily file count per category, per run_number."""
    _plot_time_series(rows, output_png, "count", "day")


def plot_size_vs_day(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Line chart: daily total size per category, per run_number."""
    _plot_time_series(rows, output_png, "size", "day")


def plot_count_vs_hour(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Line chart: hourly file count per category, per run_number."""
    _plot_time_series(rows, output_png, "count", "hour")


def plot_size_vs_hour(rows: Sequence[InventoryRow], output_png: Path) -> None:
    """Line chart: hourly total size per category, per run_number."""
    _plot_time_series(rows, output_png, "size", "hour")


# ── Unified per-category plot (count + size subplots) ────────────────────────


def _plot_category_time_series(
    rows: Sequence[InventoryRow],
    category: str,
    output_png: Path,
    time_granularity: str,
) -> None:
    """Plot count and size as subplots for a single category.

    Creates a figure with two vertically-stacked subplots:
    - Top: file count vs time
    - Bottom: total size vs time

    Parameters
    ----------
    rows:
        All inventory rows (will be filtered to *category*).
    category:
        Category to plot (``"trigger"``, ``"calibration"``, etc.).
    output_png:
        Destination PNG path.
    time_granularity:
        ``"time"``, ``"day"``, or ``"hour"``.
    """
    cat_rows = [r for r in rows if r.category == category]
    if not cat_rows:
        logger.warning("No rows for category={}, skipping plot", category)
        return

    x_cfg = {
        "time": _make_time_config,
        "day": _make_day_config,
        "hour": _make_hour_config,
    }[time_granularity]()

    x_parser = x_cfg["x_parser"]
    date_fmt: str = x_cfg["date_fmt"]  # type: ignore[assignment]
    xlabel: str = x_cfg["xlabel"]  # type: ignore[assignment]
    figsize_w: int = x_cfg["figsize_scale"]  # type: ignore[assignment]

    # Group by run_number
    run_groups: DefaultDict[int, List[InventoryRow]] = defaultdict(list)
    for r in cat_rows:
        run_groups[r.run_number].append(r)
    for run in run_groups:
        run_groups[run].sort(key=lambda r: r.timestamp)

    fig, (ax_count, ax_size) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(figsize_w, 8),
        sharex=True,
    )

    for run_number in sorted(run_groups.keys()):
        run_rows = run_groups[run_number]
        x_values = [x_parser(r) for r in run_rows]
        label = f"RUN{run_number}" if run_number > 0 else "other"

        # Count subplot
        y_count = [r.file_count for r in run_rows]
        ax_count.plot(
            x_values, y_count,
            marker="o", linewidth=1.8, markersize=4,
            label=label,
        )

        # Size subplot
        y_size = [r.total_size_bytes / (1024 ** 2) for r in run_rows]
        ax_size.plot(
            x_values, y_size,
            marker="s", linewidth=1.8, markersize=4,
            label=label,
        )

    ax_count.set_ylabel("File Count")
    ax_count.set_title(f"{category.capitalize()} — File Count vs {xlabel}")
    ax_count.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2, fontsize=8)
    ax_count.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    ax_size.set_xlabel(xlabel)
    ax_size.set_ylabel("Total Size (MB)")
    ax_size.set_title(f"{category.capitalize()} — Total Size vs {xlabel}")
    ax_size.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2, fontsize=8)
    ax_size.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=200)
    plt.close(fig)


def plot_category_daily(rows: Sequence[InventoryRow], category: str, output_png: Path) -> None:
    """Daily file count + size subplots for one category."""
    _plot_category_time_series(rows, category, output_png, "day")


def plot_category_hourly(rows: Sequence[InventoryRow], category: str, output_png: Path) -> None:
    """Hourly file count + size subplots for one category."""
    _plot_category_time_series(rows, category, output_png, "hour")


def build_output_stem(
    prefix: str,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    """Build an output filename stem with date suffix."""
    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")
    return f"{prefix}_{start_text}_{end_text}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ── Resolve --date shorthand ──────────────────────────────────────────────
    if args.date is not None:
        if args.start_date is not None or args.end_date is not None:
            parser.error("--date is mutually exclusive with --start-date/--end-date")
        args.start_date = args.date
        args.end_date = args.date + dt.timedelta(days=1)
    elif args.start_date is None or args.end_date is None:
        parser.error("Either --date or both --start-date/--end-date are required")

    if args.start_date >= args.end_date:
        parser.error("--start-date must be earlier than --end-date")

    output_dir = Path(args.output_dir)

    # ── Summary modes (read existing CSV, produce aggregated output) ──────────
    if args.daily_summary or args.hourly_summary:
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = build_output_stem("root_file_inventory", args.start_date, args.end_date)
        input_csv = output_dir / f"{stem}.csv"

        if not input_csv.is_file():
            logger.info(
                "Inventory CSV not found ({}), running inventory scan first ...",
                input_csv,
            )
            base_path = Path(args.base_path)
            td_path = base_path / "TD"
            if not td_path.is_dir():
                parser.error(f"TD directory does not exist: {td_path}")

            all_file_infos: List[Tuple[str, str, int, int]] = []
            current = args.start_date
            while current < args.end_date:
                day_path = (
                    td_path
                    / current.strftime("%Y")
                    / current.strftime("%m")
                    / current.strftime("%d")
                )
                if day_path.is_dir():
                    logger.info("Scanning: {}", day_path)
                    file_infos = scan_day_directory(day_path)
                    all_file_infos.extend(file_infos)
                else:
                    logger.warning("Directory not found, skipping: {}", day_path)
                current += dt.timedelta(days=1)

            if not all_file_infos:
                logger.warning(
                    "No .root files found in date range [{}, {})",
                    args.start_date,
                    args.end_date,
                )
                return 1

            rows = aggregate_rows(all_file_infos)
            logger.info("Writing inventory CSV: {} rows", len(rows))
            write_csv(rows, input_csv)

        logger.info("Reading inventory CSV: {}", input_csv)
        raw_rows = read_csv_rows(input_csv)

        if args.daily_summary:
            logger.info("Aggregating daily summary ...")
            daily_rows = aggregate_daily_rows(raw_rows)
            daily_stem = build_output_stem("root_file_inventory_daily", args.start_date, args.end_date)
            daily_csv = output_dir / f"{daily_stem}.csv"
            write_daily_csv(daily_rows, daily_csv)
            logger.info("Daily CSV: {} ({} rows)", daily_csv, len(daily_rows))

            if not args.no_plot:
                for cat in ("trigger", "calibration"):
                    cat_png = output_dir / f"{daily_stem}_{cat}.png"
                    logger.info("Rendering daily {} count+size chart ...", cat)
                    plot_category_daily(daily_rows, cat, cat_png)
                    logger.info("Daily {} plot: {}", cat, cat_png)

        if args.hourly_summary:
            logger.info("Aggregating hourly summary ...")
            hourly_rows = aggregate_hourly_rows(raw_rows)
            hourly_stem = build_output_stem("root_file_inventory_hourly", args.start_date, args.end_date)
            hourly_csv = output_dir / f"{hourly_stem}.csv"
            write_hourly_csv(hourly_rows, hourly_csv)
            logger.info("Hourly CSV: {} ({} rows)", hourly_csv, len(hourly_rows))

            if not args.no_plot:
                for cat in ("trigger", "calibration"):
                    cat_png = output_dir / f"{hourly_stem}_{cat}.png"
                    logger.info("Rendering hourly {} count+size chart ...", cat)
                    plot_category_hourly(hourly_rows, cat, cat_png)
                    logger.info("Hourly {} plot: {}", cat, cat_png)

        logger.info("Done.")
        return 0

    # ── Normal inventory mode ─────────────────────────────────────────────────
    base_path = Path(args.base_path)
    td_path = base_path / "TD"

    if not td_path.is_dir():
        parser.error(f"TD directory does not exist: {td_path}")

    stem = build_output_stem("root_file_inventory", args.start_date, args.end_date)
    output_csv = output_dir / f"{stem}.csv"

    logger.info(
        "Run config: td_path={} start_date={} end_date={} output_csv={}",
        td_path,
        args.start_date,
        args.end_date,
        output_csv,
    )

    # Scan day directories
    all_file_infos: List[Tuple[str, str, int, int]] = []
    current = args.start_date
    while current < args.end_date:
        day_path = td_path / current.strftime("%Y") / current.strftime("%m") / current.strftime("%d")
        if not day_path.is_dir():
            logger.warning("Directory not found, skipping: {}", day_path)
            current += dt.timedelta(days=1)
            continue

        logger.info("Scanning: {}", day_path)
        file_infos = scan_day_directory(day_path)
        all_file_infos.extend(file_infos)
        current += dt.timedelta(days=1)

    if not all_file_infos:
        logger.warning("No .root files found in date range [{}, {})", args.start_date, args.end_date)
        return 1

    # Aggregate and write CSV
    rows = aggregate_rows(all_file_infos)
    logger.info("Writing CSV output: {} rows", len(rows))
    write_csv(rows, output_csv)

    logger.info("CSV output: {}", output_csv)
    logger.info("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())