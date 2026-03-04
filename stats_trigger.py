#!/usr/bin/env python3
"""Compute per-event DU counts and per-second event rates from Trigger YAML.

Required payload fields:
1) ``gps_time`` for event second
2) ``du_ns`` for DU nanosecond map
3) optional ``date``/``time`` for event clock labels
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from logger_config import logger


def parse_event_second_from_payload(
    event_key: str,
    payload: Dict[str, Any],
) -> int:
    """Parse event second strictly from ``payload['gps_time']``.

    Raises:
        ValueError: If ``gps_time`` is missing or invalid.
    """
    if "gps_time" not in payload:
        raise ValueError(f"Missing gps_time for event {event_key}")

    gps_time = payload.get("gps_time")
    try:
        return int(gps_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid gps_time={gps_time} for event {event_key}") from exc


def parse_event_date_time(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Parse event date/time strings from payload.

    ``date`` is returned as-is stringified.
    ``time`` is stringified and zero-padded to 6 digits when numeric.
    """
    date_value = payload.get("date", "")
    time_value: Optional[Any] = payload.get("time", "")
    if isinstance(time_value, dict):
        time_value = ""

    date_str = str(date_value) if date_value is not None else ""
    time_str = str(time_value) if time_value is not None else ""
    if time_str.isdigit() and len(time_str) < 6:
        time_str = time_str.zfill(6)

    return date_str, time_str


def get_du_ns_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get DU nanosecond map from ``payload['du_ns']``."""
    du_ns = payload.get("du_ns")
    if isinstance(du_ns, dict):
        return du_ns

    return {}


def parse_event_records(
    data: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, int, int, int, int, str, str]]:
    """Parse records for each event.

    Returns:
        [
            (
                event_key,
                event_second,
                du_count,
                event_number,
                index,
                event_date,
                event_time,
            ),
            ...,
        ]
    """
    records: List[Tuple[str, int, int, int, int, str, str]] = []

    for event_key, payload in data.items():
        event_second = parse_event_second_from_payload(event_key, payload)

        du_ids = payload.get("du_id", [])
        if isinstance(du_ids, list):
            du_count = len(du_ids)
        else:
            du_count = len(get_du_ns_map(payload))

        event_number = int(payload.get("event_number", -1))
        index = int(payload.get("index", -1))
        event_date, event_time = parse_event_date_time(payload)

        records.append(
            (
                event_key,
                event_second,
                du_count,
                event_number,
                index,
                event_date,
                event_time,
            )
        )

    records.sort(key=lambda item: (item[1], item[4], item[3]))
    return records


def build_rate_per_second(
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> List[Tuple[int, int, float]]:
    """Compute event rates per second (Hz)."""
    second_counter = Counter(record[1] for record in records)
    return [(sec, count, float(count)) for sec, count in sorted(second_counter.items())]


def build_du_trigger_rate(
    data: Dict[str, Dict[str, Any]],
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> List[Tuple[str, int, float]]:
    """Compute trigger count and trigger rate (Hz) for each DU.

    Trigger rate is defined as:
        trigger_count / observation_seconds
    where observation_seconds = max_second - min_second + 1.
    """
    if not records:
        return []

    min_second = min(record[1] for record in records)
    max_second = max(record[1] for record in records)
    observation_seconds = max_second - min_second + 1
    if observation_seconds <= 0:
        observation_seconds = 1

    du_counter: Counter[str] = Counter()
    for payload in data.values():
        du_ids = payload.get("du_id", [])
        if isinstance(du_ids, list):
            for du_id in du_ids:
                du_counter[str(du_id)] += 1
        else:
            du_ns_map = get_du_ns_map(payload)
            for du_id in du_ns_map.keys():
                du_counter[str(du_id)] += 1

    rows: List[Tuple[str, int, float]] = []
    for du_id, count in du_counter.items():
        rows.append((du_id, count, count / float(observation_seconds)))

    rows.sort(key=lambda row: int(row[0]) if row[0].isdigit() else row[0])
    return rows


def build_du_counts_per_second(
    data: Dict[str, Dict[str, Any]],
) -> Dict[int, Counter[str]]:
    """Count DU triggers per second."""
    per_second_du: Dict[int, Counter[str]] = {}
    for event_key, payload in data.items():
        second = parse_event_second_from_payload(event_key, payload)
        if second not in per_second_du:
            per_second_du[second] = Counter()

        du_ids = payload.get("du_id", [])
        if isinstance(du_ids, list):
            for du_id in du_ids:
                per_second_du[second][str(du_id)] += 1
        else:
            du_ns_map = get_du_ns_map(payload)
            for du_id in du_ns_map.keys():
                per_second_du[second][str(du_id)] += 1

    return per_second_du


def build_du_rate_per_second_rows(
    per_second_du: Dict[int, Counter[str]],
) -> List[Tuple[int, str, int, float]]:
    """Flatten per-second DU trigger-rate records (Hz)."""
    rows: List[Tuple[int, str, int, float]] = []
    for second in sorted(per_second_du.keys()):
        du_counter = per_second_du[second]
        sorted_du_ids = sorted(
            du_counter.keys(), key=lambda item: int(item) if item.isdigit() else item
        )
        for du_id in sorted_du_ids:
            count = du_counter[du_id]
            rows.append((second, du_id, count, float(count)))
    return rows


def build_window_averages(
    records: List[Tuple[str, int, int, int, int, str, str]],
    per_second_du: Dict[int, Counter[str]],
    window_seconds: int,
) -> Tuple[List[Tuple[int, int, int, float]], List[Tuple[int, int, str, int, float]]]:
    """Compute average event/DU trigger rates over fixed windows.

    Windows are represented as half-open intervals:
        [window_start_second, window_end_second)
    including the start and excluding the end.
    """
    if not records:
        return [], []

    start_second = min(record[1] for record in records)
    end_second_inclusive = max(record[1] for record in records)
    stop_second_exclusive = end_second_inclusive + 1

    event_counter = Counter(record[1] for record in records)
    all_du_ids = sorted(
        {
            du_id
            for second_map in per_second_du.values()
            for du_id in second_map.keys()
        },
        key=lambda item: int(item) if item.isdigit() else item,
    )

    event_rows: List[Tuple[int, int, int, float]] = []
    du_rows: List[Tuple[int, int, str, int, float]] = []

    window_start = start_second
    while window_start < stop_second_exclusive:
        window_end_exclusive = min(window_start + window_seconds, stop_second_exclusive)
        duration = window_end_exclusive - window_start

        event_count = 0
        for second in range(window_start, window_end_exclusive):
            event_count += event_counter.get(second, 0)
        event_rows.append(
            (window_start, window_end_exclusive, event_count, event_count / duration)
        )

        for du_id in all_du_ids:
            trigger_count = 0
            for second in range(window_start, window_end_exclusive):
                trigger_count += per_second_du.get(second, Counter()).get(du_id, 0)
            du_rows.append(
                (
                    window_start,
                    window_end_exclusive,
                    du_id,
                    trigger_count,
                    trigger_count / duration,
                )
            )

        window_start += window_seconds

    return event_rows, du_rows


def write_event_csv(
    path: Path,
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> None:
    """Write per-event DU-count CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "event_key",
                "event_second",
                "event_date",
                "event_time",
                "du_count",
                "event_number",
                "index",
            ]
        )
        rows = [
            (record[0], record[1], record[5], record[6], record[2], record[3], record[4])
            for record in records
        ]
        writer.writerows(rows)


def write_rate_csv(path: Path, rates: List[Tuple[int, int, float]]) -> None:
    """Write per-second event-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["event_second", "event_count", "event_rate_hz"])
        writer.writerows(rates)


def write_du_rate_csv(path: Path, du_rates: List[Tuple[str, int, float]]) -> None:
    """Write per-DU trigger count/rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["du_id", "trigger_count", "trigger_rate_hz"])
        writer.writerows(du_rates)


def write_du_per_second_csv(path: Path, rows: List[Tuple[int, str, int, float]]) -> None:
    """Write per-second per-DU trigger-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["event_second", "du_id", "trigger_count", "trigger_rate_hz"])
        writer.writerows(rows)


def write_avg_event_csv(path: Path, rows: List[Tuple[int, int, int, float]]) -> None:
    """Write window-averaged event-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "window_start_second",
                "window_end_second_exclusive",
                "event_count",
                "avg_event_rate_hz",
            ]
        )
        writer.writerows(rows)


def write_avg_du_csv(path: Path, rows: List[Tuple[int, int, str, int, float]]) -> None:
    """Write window-averaged per-DU trigger-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "window_start_second",
                "window_end_second_exclusive",
                "du_id",
                "trigger_count",
                "avg_trigger_rate_hz",
            ]
        )
        writer.writerows(rows)


def plot_du_count_histogram(
    path: Path,
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> None:
    """Plot histogram of DU counts per event."""
    du_counts = [record[2] for record in records]
    if not du_counts:
        logger.warning("No event data; skip DU-count histogram")
        return

    max_du = max(du_counts)
    bins = [value - 0.5 for value in range(1, max_du + 2)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(du_counts, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_xlabel("DU count per event")
    ax.set_ylabel("Event count")
    ax.set_title("Histogram of Trigger DU Count")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_xticks(range(1, max_du + 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rate_line(path: Path, rates: List[Tuple[int, int, float]]) -> None:
    """Plot line chart of event rate per second."""
    if not rates:
        logger.warning("No event-rate data; skip per-second event-rate plot")
        return

    seconds = [row[0] for row in rates]
    event_rates = [row[2] for row in rates]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seconds, event_rates, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("Event second (GPS)")
    ax.set_ylabel("Event rate (Hz)")
    ax.set_title("Event Rate Per Second")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_du_trigger_rate(path: Path, du_rates: List[Tuple[str, int, float]]) -> None:
    """Plot bar chart of trigger rate for each DU."""
    if not du_rates:
        logger.warning("No DU trigger-rate data; skip DU trigger-rate plot")
        return

    du_ids = [row[0] for row in du_rates]
    trigger_rates = [row[2] for row in du_rates]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(du_ids, trigger_rates, color="tab:blue", alpha=0.85)
    ax.set_xlabel("DU ID")
    ax.set_ylabel("Trigger rate (Hz)")
    ax.set_title("Trigger Rate per DU")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_du_rate_per_second_heatmap(
    path: Path,
    per_second_du: Dict[int, Counter[str]],
) -> None:
    """Plot heatmap of per-second trigger rate for each DU."""
    if not per_second_du:
        logger.warning("No per-second DU data; skip DU time-series heatmap")
        return

    seconds = sorted(per_second_du.keys())
    du_ids = sorted(
        {
            du_id
            for second_map in per_second_du.values()
            for du_id in second_map.keys()
        },
        key=lambda item: int(item) if item.isdigit() else item,
    )

    matrix = np.zeros((len(du_ids), len(seconds)), dtype=float)
    for sec_idx, second in enumerate(seconds):
        for du_idx, du_id in enumerate(du_ids):
            matrix[du_idx, sec_idx] = float(per_second_du[second].get(du_id, 0))

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Trigger rate (Hz)")

    ax.set_xlabel("Event second index")
    ax.set_ylabel("DU ID")
    ax.set_title("Per-second Trigger Rate per DU")

    if len(du_ids) <= 40:
        ax.set_yticks(range(len(du_ids)))
        ax.set_yticklabels(du_ids)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_event_rate(path: Path, rows: List[Tuple[int, int, int, float]], window_seconds: int) -> None:
    """Plot line chart of window-averaged event rate."""
    if not rows:
        logger.warning("No window-averaged event-rate data; skip plotting")
        return

    x_values = [row[0] for row in rows]
    y_values = [row[3] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_values, y_values, marker="o", linewidth=1)
    ax.set_xlabel("Window start second (GPS)")
    ax.set_ylabel("Average event rate (Hz)")
    ax.set_title(f"Average Event Rate (window={window_seconds}s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_du_rate_heatmap(
    path: Path,
    rows: List[Tuple[int, int, str, int, float]],
    window_seconds: int,
) -> None:
    """Plot heatmap of window-averaged trigger rate for each DU."""
    if not rows:
        logger.warning("No window-averaged DU trigger-rate data; skip plotting")
        return

    windows = sorted({(row[0], row[1]) for row in rows})
    du_ids = sorted(
        {row[2] for row in rows},
        key=lambda item: int(item) if item.isdigit() else item,
    )

    window_index = {window: idx for idx, window in enumerate(windows)}
    du_index = {du_id: idx for idx, du_id in enumerate(du_ids)}

    matrix = np.zeros((len(du_ids), len(windows)), dtype=float)
    for row in rows:
        win_key = (row[0], row[1])
        du_id = row[2]
        matrix[du_index[du_id], window_index[win_key]] = row[4]

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average trigger rate (Hz)")

    ax.set_xlabel(f"Window index ({window_seconds}s)")
    ax.set_ylabel("DU ID")
    ax.set_title(f"Average Trigger Rate per DU (window={window_seconds}s)")

    if len(du_ids) <= 40:
        ax.set_yticks(range(len(du_ids)))
        ax.set_yticklabels(du_ids)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_du_rate_topn_lines(
    path: Path,
    rows: List[Tuple[int, int, str, int, float]],
    window_seconds: int,
    top_n: int,
) -> None:
    """Plot window-averaged trigger-rate lines for Top-N active DUs."""
    if not rows:
        logger.warning("No window-averaged DU trigger-rate data; skip Top-N line plot")
        return
    if top_n <= 0:
        logger.warning("Top-N <= 0; skip Top-N line plot")
        return

    du_total_counts: Counter[str] = Counter()
    for row in rows:
        du_total_counts[row[2]] += row[3]

    top_du_ids = [du_id for du_id, _ in du_total_counts.most_common(top_n)]
    if not top_du_ids:
        logger.warning("No Top-N DU selected; skip line plot")
        return

    window_starts = sorted({row[0] for row in rows})
    rate_map: Dict[str, Dict[int, float]] = {du_id: {} for du_id in top_du_ids}
    for row in rows:
        window_start, _, du_id, _, avg_rate = row
        if du_id in rate_map:
            rate_map[du_id][window_start] = avg_rate

    fig, ax = plt.subplots(figsize=(13, 6))
    for du_id in top_du_ids:
        y_values = [rate_map[du_id].get(window_start, 0.0) for window_start in window_starts]
        ax.plot(window_starts, y_values, marker="o", linewidth=1, label=f"DU {du_id}")

    ax.set_xlabel("Window start second (GPS)")
    ax.set_ylabel("Average trigger rate (Hz)")
    ax.set_title(f"Top-{len(top_du_ids)} DU Average Trigger Rate (window={window_seconds}s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute DU counts and per-second event rates from Trigger YAML"
    )
    parser.add_argument("yaml_file", help="Input YAML file path")
    parser.add_argument(
        "--event-out",
        help="Per-event stats CSV output path (default: <input_stem>_event_du_count.csv)",
    )
    parser.add_argument(
        "--rate-out",
        help="Per-second event-rate CSV output path (default: <input_stem>_rate_per_second.csv)",
    )
    parser.add_argument(
        "--du-rate-out",
        help="Per-DU trigger-rate CSV output path (default: <input_stem>_du_trigger_rate.csv)",
    )
    parser.add_argument(
        "--du-second-out",
        help="Per-second per-DU trigger-rate CSV output path (default: <input_stem>_du_rate_per_second.csv)",
    )
    parser.add_argument(
        "--avg-event-out",
        help="Window-averaged event-rate CSV output path (default: <input_stem>_avg_event_rate.csv)",
    )
    parser.add_argument(
        "--avg-du-out",
        help="Window-averaged per-DU trigger-rate CSV output path (default: <input_stem>_avg_du_trigger_rate.csv)",
    )
    parser.add_argument(
        "--du-hist-out",
        help="DU-count histogram PNG output path (default: <input_stem>_du_count_hist.png)",
    )
    parser.add_argument(
        "--rate-plot-out",
        help="Per-second event-rate line plot PNG output path (default: <input_stem>_rate_per_second.png)",
    )
    parser.add_argument(
        "--du-rate-plot-out",
        help="Per-DU trigger-rate bar plot PNG output path (default: <input_stem>_du_trigger_rate.png)",
    )
    parser.add_argument(
        "--du-second-plot-out",
        help="Per-second per-DU trigger-rate heatmap PNG output path (default: <input_stem>_du_rate_per_second_heatmap.png)",
    )
    parser.add_argument(
        "--avg-event-plot-out",
        help="Window-averaged event-rate line plot PNG output path (default: <input_stem>_avg_event_rate.png)",
    )
    parser.add_argument(
        "--avg-window",
        type=int,
        default=0,
        help="Averaging window length in seconds; output window averages when >0 (e.g., 60 means 60-second average)",
    )
    parser.add_argument(
        "--avg-du-window",
        type=int,
        default=0,
        help="Averaging window in seconds for per-DU trigger rates; output per-DU averages when >0 (default follows --avg-window)",
    )
    parser.add_argument(
        "--avg-du-plot-out",
        help="Window-averaged per-DU trigger-rate heatmap PNG output path (default: <input_stem>_avg_du_trigger_rate_heatmap.png)",
    )
    parser.add_argument(
        "--avg-du-topn",
        type=int,
        default=10,
        help="Number of most active DUs shown in window-averaged per-DU line plot (default: 10, set to 0 to disable)",
    )
    parser.add_argument(
        "--avg-du-topn-plot-out",
        help="Top-N window-averaged per-DU trigger-rate line plot PNG output path (default: <input_stem>_avg_du_trigger_rate_topn.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Export CSV only, do not generate plots",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Preview first N rows in terminal (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    """Main workflow."""
    args = parse_args()
    yaml_path = Path(args.yaml_file)

    if not yaml_path.exists():
        logger.error(f"Input file does not exist: {yaml_path}")
        return 2

    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if not isinstance(data, dict):
        logger.error("YAML top-level structure is not a dict; cannot parse by event key")
        return 2

    records = parse_event_records(data)
    rates = build_rate_per_second(records)
    du_rates = build_du_trigger_rate(data, records)
    per_second_du = build_du_counts_per_second(data)
    du_second_rows = build_du_rate_per_second_rows(per_second_du)

    avg_event_rows: List[Tuple[int, int, int, float]] = []
    avg_du_rows: List[Tuple[int, int, str, int, float]] = []

    effective_avg_du_window = args.avg_du_window if args.avg_du_window > 0 else args.avg_window

    if args.avg_window > 0:
        avg_event_rows, _ = build_window_averages(records, per_second_du, args.avg_window)

    if effective_avg_du_window > 0:
        _, avg_du_rows = build_window_averages(records, per_second_du, effective_avg_du_window)

    event_out = Path(args.event_out) if args.event_out else yaml_path.with_name(
        f"{yaml_path.stem}_event_du_count.csv"
    )
    rate_out = Path(args.rate_out) if args.rate_out else yaml_path.with_name(
        f"{yaml_path.stem}_rate_per_second.csv"
    )
    du_rate_out = Path(args.du_rate_out) if args.du_rate_out else yaml_path.with_name(
        f"{yaml_path.stem}_du_trigger_rate.csv"
    )
    du_second_out = Path(args.du_second_out) if args.du_second_out else yaml_path.with_name(
        f"{yaml_path.stem}_du_rate_per_second.csv"
    )
    avg_event_out = Path(args.avg_event_out) if args.avg_event_out else yaml_path.with_name(
        f"{yaml_path.stem}_avg_event_rate.csv"
    )
    avg_du_out = Path(args.avg_du_out) if args.avg_du_out else yaml_path.with_name(
        f"{yaml_path.stem}_avg_du_trigger_rate.csv"
    )
    du_hist_out = Path(args.du_hist_out) if args.du_hist_out else yaml_path.with_name(
        f"{yaml_path.stem}_du_count_hist.png"
    )
    rate_plot_out = Path(args.rate_plot_out) if args.rate_plot_out else yaml_path.with_name(
        f"{yaml_path.stem}_rate_per_second.png"
    )
    du_rate_plot_out = (
        Path(args.du_rate_plot_out)
        if args.du_rate_plot_out
        else yaml_path.with_name(f"{yaml_path.stem}_du_trigger_rate.png")
    )
    du_second_plot_out = (
        Path(args.du_second_plot_out)
        if args.du_second_plot_out
        else yaml_path.with_name(f"{yaml_path.stem}_du_rate_per_second_heatmap.png")
    )
    avg_event_plot_out = (
        Path(args.avg_event_plot_out)
        if args.avg_event_plot_out
        else yaml_path.with_name(f"{yaml_path.stem}_avg_event_rate.png")
    )
    avg_du_plot_out = (
        Path(args.avg_du_plot_out)
        if args.avg_du_plot_out
        else yaml_path.with_name(f"{yaml_path.stem}_avg_du_trigger_rate_heatmap.png")
    )
    avg_du_topn_plot_out = (
        Path(args.avg_du_topn_plot_out)
        if args.avg_du_topn_plot_out
        else yaml_path.with_name(f"{yaml_path.stem}_avg_du_trigger_rate_topn.png")
    )

    write_event_csv(event_out, records)
    write_rate_csv(rate_out, rates)
    write_du_rate_csv(du_rate_out, du_rates)
    write_du_per_second_csv(du_second_out, du_second_rows)
    if args.avg_window > 0:
        write_avg_event_csv(avg_event_out, avg_event_rows)
    if effective_avg_du_window > 0:
        write_avg_du_csv(avg_du_out, avg_du_rows)

    if not args.no_plot:
        plot_du_count_histogram(du_hist_out, records)
        plot_rate_line(rate_plot_out, rates)
        plot_du_trigger_rate(du_rate_plot_out, du_rates)
        plot_du_rate_per_second_heatmap(du_second_plot_out, per_second_du)
        if args.avg_window > 0:
            plot_avg_event_rate(avg_event_plot_out, avg_event_rows, args.avg_window)
        if effective_avg_du_window > 0:
            plot_avg_du_rate_heatmap(avg_du_plot_out, avg_du_rows, effective_avg_du_window)
            if args.avg_du_topn > 0:
                plot_avg_du_rate_topn_lines(
                    avg_du_topn_plot_out,
                    avg_du_rows,
                    effective_avg_du_window,
                    args.avg_du_topn,
                )

    logger.info(f"Total events: {len(records)}")
    logger.info(f"Per-event DU-count stats written to: {event_out}")
    logger.info(f"Per-second event-rate stats written to: {rate_out}")
    logger.info(f"Per-DU trigger-rate stats written to: {du_rate_out}")
    logger.info(f"Per-second per-DU trigger-rate stats written to: {du_second_out}")
    if args.avg_window > 0:
        logger.info(f"Window-averaged event-rate stats written to: {avg_event_out}")
    if effective_avg_du_window > 0:
        logger.info(f"Window-averaged per-DU trigger-rate stats written to: {avg_du_out}")
    if not args.no_plot:
        logger.info(f"DU-count histogram written to: {du_hist_out}")
        logger.info(f"Per-second event-rate line plot written to: {rate_plot_out}")
        logger.info(f"Per-DU trigger-rate bar plot written to: {du_rate_plot_out}")
        logger.info(f"Per-second per-DU trigger-rate heatmap written to: {du_second_plot_out}")
        if args.avg_window > 0:
            logger.info(f"Window-averaged event-rate line plot written to: {avg_event_plot_out}")
        if effective_avg_du_window > 0:
            logger.info(f"Window-averaged per-DU trigger-rate heatmap written to: {avg_du_plot_out}")
            if args.avg_du_topn > 0:
                logger.info(
                    "Top-N window-averaged per-DU trigger-rate line plot "
                    f"written to: {avg_du_topn_plot_out}"
                )

    preview_n = max(args.head, 0)
    if preview_n > 0:
        print("\n[Per-event DU count preview]")
        for row in records[:preview_n]:
            print(row)

        print("\n[Per-second event-rate preview]")
        for row in rates[:preview_n]:
            print(row)

        print("\n[Per-DU trigger-rate preview]")
        for row in du_rates[:preview_n]:
            print(row)

        print("\n[Per-second per-DU trigger-rate preview]")
        for row in du_second_rows[:preview_n]:
            print(row)

        if args.avg_window > 0:
            print(f"\n[Window-averaged event-rate preview] (window={args.avg_window}s)")
            for row in avg_event_rows[:preview_n]:
                print(row)

        if effective_avg_du_window > 0:
            print(
                f"\n[Window-averaged per-DU trigger-rate preview] "
                f"(window={effective_avg_du_window}s)"
            )
            for row in avg_du_rows[:preview_n]:
                print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
