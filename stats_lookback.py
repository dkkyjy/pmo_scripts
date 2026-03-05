#!/usr/bin/env python3
"""Statistics for previous-N-event common DU-pair delta differences.

For each event in time order, this script compares it with up to ``N``
previous events and only keeps DU pairs that are present in both events.
For every shared DU pair ``(du_a, du_b)``, it computes:

``delta_diff = (du_b_ns - du_a_ns)_curr - (du_b_ns - du_a_ns)_prev``

It exports per-sample rows to CSV and optionally plots a histogram.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import scienceplots
import yaml

from logger_config import logger

plt.style.use(["science", "grid", "notebook"])

DELTA_PLOT_MIN_NS = -2e7
DELTA_PLOT_MAX_NS = 2e7

EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]
EventRecord = Tuple[str, int, int, int, Dict[str, float], str]
PairKey = Tuple[str, str]
PairDeltaMap = Dict[PairKey, float]
OffsetMap = Dict[str, float]
AdjacentPairRow = Tuple[
    int,
    int,
    int,
    int,
    str,
    str,
    float,
    float,
    float,
    float,
    str,
]


def sort_du_id_key(du_id: str) -> Tuple[int, str]:
    """Sort DU IDs numerically when possible, else lexicographically."""
    return (0, f"{int(du_id):012d}") if du_id.isdigit() else (1, du_id)


def read_yaml_events(yaml_path: Path) -> YamlData:
    """Read Trigger YAML and validate top-level structure."""
    with yaml_path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj)

    if loaded is None:
        logger.warning("Input YAML is empty: {}", yaml_path)
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Input YAML top-level must be a dict")
    return loaded


def parse_du_ns_map(payload: EventPayload) -> Dict[str, float]:
    """Parse numeric ``du_ns`` map from one event payload."""
    du_ns = payload.get("du_ns")
    if not isinstance(du_ns, dict):
        return {}

    parsed: Dict[str, float] = {}
    for key, value in du_ns.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            logger.debug("Skip non-numeric du_ns value for DU {}: {}", key, value)
    return parsed


def parse_event_datetime(payload: EventPayload) -> str:
    """Parse and format event datetime from payload date/time fields."""
    date_value = payload.get("date", "")
    time_value = payload.get("time", "")

    date_str = str(date_value) if date_value is not None else ""
    time_str = str(time_value) if time_value is not None else ""
    if time_str.isdigit() and len(time_str) < 6:
        time_str = time_str.zfill(6)

    if not date_str or not time_str:
        return ""

    try:
        dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return f"{date_str} {time_str}".strip()


def load_du_time_offsets(offset_file: Path) -> OffsetMap:
    """Load DU time offsets from file, format: du_id, offset, sigma."""
    offsets: OffsetMap = {}
    if not offset_file.exists():
        logger.warning("Offset file not found, skip correction: {}", offset_file)
        return offsets

    with offset_file.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                logger.warning("Skip invalid offset line: {}", line)
                continue

            du_id = str(parts[0])
            try:
                offset_ns = float(parts[1])
            except (TypeError, ValueError):
                logger.warning("Skip invalid offset value for DU {}: {}", du_id, parts[1])
                continue

            offsets[du_id] = offset_ns

    logger.info("Loaded DU offsets: {} from {}", len(offsets), offset_file)
    return offsets


def build_event_records(data: YamlData) -> List[EventRecord]:
    """Build event records sorted by ``gps_time``, ``index``, ``event_number``."""
    records: List[EventRecord] = []

    for event_key, payload in data.items():
        try:
            event_second = int(payload.get("gps_time", -1))
        except (TypeError, ValueError):
            logger.debug("Skip invalid gps_time for event {}", event_key)
            continue

        if event_second < 0:
            continue

        try:
            event_number = int(payload.get("event_number", -1))
        except (TypeError, ValueError):
            event_number = -1

        try:
            index = int(payload.get("index", -1))
        except (TypeError, ValueError):
            index = -1

        du_ns_map = parse_du_ns_map(payload)
        event_datetime = parse_event_datetime(payload)
        records.append(
            (event_key, event_second, index, event_number, du_ns_map, event_datetime)
        )

    records.sort(key=lambda row: (row[1], row[2], row[3], row[0]))
    logger.info("Built event records: {}", len(records))
    return records


def build_pair_delta_map(
    du_ns_map: Dict[str, float],
    offset_map: OffsetMap | None = None,
) -> PairDeltaMap:
    """Build DU-pair in-event delta map: ``du_b_ns - du_a_ns``."""
    pair_map: PairDeltaMap = {}
    du_ids = sorted(du_ns_map.keys(), key=sort_du_id_key)
    if offset_map is None:
        offset_map = {}

    for du_a, du_b in itertools.combinations(du_ids, 2):
        du_a_corrected = du_ns_map[du_a] - offset_map.get(du_a, 0.0)
        du_b_corrected = du_ns_map[du_b] - offset_map.get(du_b, 0.0)
        pair_map[(du_a, du_b)] = du_b_corrected - du_a_corrected

    return pair_map


def build_adjacent_common_pair_rows(
    records: Sequence[EventRecord],
    lookback: int = 10,
    offset_map: OffsetMap | None = None,
) -> List[AdjacentPairRow]:
    """Build rows for shared DU pairs between current and previous events."""
    if len(records) < 2:
        return []

    lookback = max(1, int(lookback))
    if offset_map is None:
        offset_map = {}

    rows: List[AdjacentPairRow] = []
    for row_index in range(1, len(records)):
        (
            _,
            curr_second,
            curr_index,
            curr_event_number,
            curr_du_ns,
            curr_event_datetime,
        ) = records[row_index]
        curr_pair_map = build_pair_delta_map(curr_du_ns, offset_map)

        start_index = max(0, row_index - lookback)
        for prev_row_index in range(start_index, row_index):
            _, prev_second, prev_index, prev_event_number, prev_du_ns, _ = records[
                prev_row_index
            ]
            prev_pair_map = build_pair_delta_map(prev_du_ns, offset_map)
            common_pairs = sorted(set(prev_pair_map) & set(curr_pair_map))

            for du_a, du_b in common_pairs:
                prev_delta = prev_pair_map[(du_a, du_b)]
                curr_delta = curr_pair_map[(du_a, du_b)]
                adjacent_delta = curr_delta - prev_delta
                rows.append(
                    (
                        prev_event_number,
                        curr_event_number,
                        prev_second,
                        curr_second,
                        du_a,
                        du_b,
                        prev_delta,
                        curr_delta,
                        adjacent_delta,
                        abs(adjacent_delta),
                        curr_event_datetime,
                    )
                )

            logger.debug(
                "Compared {}->{}: common DU pairs={} (index {}->{})",
                prev_event_number,
                curr_event_number,
                len(common_pairs),
                prev_index,
                curr_index,
            )

    logger.info(
        "Built previous-N common DU-pair rows: {} (lookback={})",
        len(rows),
        lookback,
    )
    return rows


def write_adjacent_pair_csv(path: Path, rows: Sequence[AdjacentPairRow]) -> None:
    """Write adjacent common DU-pair rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "prev_event_number",
                "curr_event_number",
                "prev_gps_time",
                "curr_gps_time",
                "du_a",
                "du_b",
                "prev_pair_delta_ns",
                "curr_pair_delta_ns",
                "adjacent_pair_delta_ns",
                "abs_adjacent_pair_delta_ns",
                "curr_event_datetime",
            ]
        )
        writer.writerows(rows)


def read_adjacent_pair_csv(path: Path) -> List[AdjacentPairRow]:
    """Read adjacent/common DU-pair rows from CSV cache."""
    rows: List[AdjacentPairRow] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                (
                    int(row["prev_event_number"]),
                    int(row["curr_event_number"]),
                    int(row["prev_gps_time"]),
                    int(row["curr_gps_time"]),
                    str(row["du_a"]),
                    str(row["du_b"]),
                    float(row["prev_pair_delta_ns"]),
                    float(row["curr_pair_delta_ns"]),
                    float(row["adjacent_pair_delta_ns"]),
                    float(row["abs_adjacent_pair_delta_ns"]),
                    str(row.get("curr_event_datetime", "")),
                )
            )
    return rows


def apply_time_ticks(
    ax: plt.Axes,
    gps_times: Sequence[int],
    datetime_labels: Sequence[str],
    max_ticks: int = 8,
) -> None:
    """Apply compact x-axis ticks as gps_time + datetime label."""
    if not gps_times:
        return

    label_map: Dict[int, str] = {}
    for gps_time, datetime_label in zip(gps_times, datetime_labels):
        if gps_time not in label_map and datetime_label:
            label_map[gps_time] = datetime_label

    unique_gps_times = sorted(set(gps_times))
    tick_count = min(max_ticks, len(unique_gps_times))
    if tick_count == 0:
        return

    if tick_count == 1:
        tick_positions = [unique_gps_times[0]]
    else:
        index_step = (len(unique_gps_times) - 1) / float(tick_count - 1)
        indices = [int(round(i * index_step)) for i in range(tick_count)]
        tick_positions = [unique_gps_times[index] for index in indices]

    tick_labels: List[str] = []
    for gps_time in tick_positions:
        datetime_label = label_map.get(gps_time, "")
        if datetime_label:
            tick_labels.append(f"{gps_time}\n{datetime_label}")
        else:
            tick_labels.append(str(gps_time))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")


def plot_adjacent_pair_delta_vs_time(
    path: Path,
    rows: Sequence[AdjacentPairRow],
) -> None:
    """Plot adjacent DU-pair delta scatter versus curr_event time."""
    if not rows:
        logger.warning("No adjacent common DU-pair rows; skip time scatter plotting")
        return

    gps_times = [row[3] for row in rows]
    delta_values = [row[8] for row in rows]
    datetime_labels = [row[10] for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(gps_times, delta_values, s=8, alpha=0.6)
    ax.set_xlabel("Curr event time (gps_time)")
    ax.set_ylabel("Adjacent DU-pair delta difference (ns)")
    ax.set_title("Adjacent DU-pair delta difference vs curr event time")
    ax.set_ylim(DELTA_PLOT_MIN_NS, DELTA_PLOT_MAX_NS)
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_time_ticks(ax, gps_times, datetime_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_adjacent_pair_delta_histogram(
    path: Path,
    rows: Sequence[AdjacentPairRow],
    bins: int,
) -> None:
    """Plot histogram for adjacent DU-pair delta differences."""
    if not rows:
        logger.warning("No adjacent common DU-pair rows; skip plotting")
        return

    deltas = [row[8] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        deltas,
        bins=max(1, bins),
        edgecolor="black",
        range=(DELTA_PLOT_MIN_NS, DELTA_PLOT_MAX_NS),
    )
    ax.set_xlabel("Adjacent DU-pair delta difference (ns)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of adjacent-event common DU-pair delta differences")
    ax.set_xlim(DELTA_PLOT_MIN_NS, DELTA_PLOT_MAX_NS)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute distribution of DU-pair delta differences between adjacent "
            "events, using only DU pairs shared by adjacent events"
        )
    )
    parser.add_argument("yaml_file", help="Input Trigger YAML file path")
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Histogram bins (default: 120)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Export CSV only, do not generate plot",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Compare each event with up to N previous events (default: 10)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing CSV cache and recompute from YAML",
    )
    parser.add_argument(
        "--offset-file",
        default="2025-10-28_beacon_25Hz_offset.txt",
        help="DU time offset file path. Format: du_id, offset, sigma",
    )
    return parser.parse_args()


def main() -> int:
    """Main workflow."""
    args = parse_args()
    yaml_path = Path(args.yaml_file)
    normalized_lookback = max(1, args.lookback)

    csv_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_distribution.csv"
    )
    plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_hist.png"
    )
    time_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_vs_time.png"
    )

    if csv_out.exists() and not args.force_recompute:
        logger.info("Found CSV cache: {}", csv_out)
        rows = read_adjacent_pair_csv(csv_out)
        logger.info("Loaded rows from CSV cache: {}", len(rows))

        if not args.no_plot:
            plot_adjacent_pair_delta_histogram(plot_out, rows, args.bins)
            plot_adjacent_pair_delta_vs_time(time_plot_out, rows)
            if rows:
                logger.info("Plot written: {}", plot_out)
                logger.info("Plot written: {}", time_plot_out)
        else:
            logger.info("Plotting is disabled by --no-plot")

        return 0

    if csv_out.exists() and args.force_recompute:
        logger.info("Force recompute enabled, ignore existing CSV cache: {}", csv_out)

    if not yaml_path.exists():
        logger.error("Input file does not exist: {}", yaml_path)
        return 2

    try:
        data = read_yaml_events(yaml_path)
    except Exception as exc:
        logger.error("Failed to read YAML: {}", exc)
        return 2

    offset_map = load_du_time_offsets(Path(args.offset_file))
    records = build_event_records(data)
    rows = build_adjacent_common_pair_rows(
        records,
        lookback=args.lookback,
        offset_map=offset_map,
    )

    write_adjacent_pair_csv(csv_out, rows)
    logger.info("CSV written: {}", csv_out)

    if not args.no_plot:
        plot_adjacent_pair_delta_histogram(plot_out, rows, args.bins)
        plot_adjacent_pair_delta_vs_time(time_plot_out, rows)
        if rows:
            logger.info("Plot written: {}", plot_out)
            logger.info("Plot written: {}", time_plot_out)
    else:
        logger.info("Plotting is disabled by --no-plot")

    logger.info(
        "Summary: events={}, compared_event_pairs<=N={}, samples={}, lookback={}",
        len(records),
        sum(min(i, normalized_lookback) for i in range(len(records))),
        len(rows),
        normalized_lookback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())