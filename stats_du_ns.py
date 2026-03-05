#!/usr/bin/env python3
"""Statistics for DU-pair observed/theoretical time-delta distributions.

This script reads Trigger YAML (event payloads containing ``du_ns``), then:
1) Builds observed DU-pair time deltas for each event.
2) Builds (or reuses) a shared theoretical DU-pair delta cache from detector
    positions, where theoretical delta = distance / c.
3) Exports a single distribution CSV containing both observed and theoretical
    time deltas for each observed DU pair sample.
4) Optionally plots observed vs theoretical distributions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import yaml

from find_event.io import load_data_from_file
from logger_config import logger

plt.style.use(["science", "grid", "notebook"])

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
NS_PER_SECOND = 1e9

EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]
OffsetMap = Dict[str, float]
ObservedPairDelta = Tuple[int, int, int, str, str, float, float]
ExpectedPairDelta = Tuple[str, str, float, float]
PairDistributionRow = Tuple[int, int, int, str, str, float, float, float, float, str]

EventTimeLabelMap = Dict[int, str]


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
    """Parse ``du_ns`` map from one event payload."""
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


def load_du_time_offsets(offset_file: Path) -> OffsetMap:
    """Load DU time offsets from a file.

    Expected line format (comma-separated):
        du_id, offset, sigma
    where only ``du_id`` and ``offset`` are used; ``sigma`` is ignored.
    """
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


def parse_event_date_time(payload: EventPayload) -> Tuple[str, str]:
    """Parse event date/time strings from payload."""
    date_value = payload.get("date", "")
    time_value = payload.get("time", "")
    if isinstance(time_value, dict):
        time_value = ""

    date_str = str(date_value) if date_value is not None else ""
    time_str = str(time_value) if time_value is not None else ""
    if time_str.isdigit() and len(time_str) < 6:
        time_str = time_str.zfill(6)
    return date_str, time_str


def format_event_datetime(date_str: str, time_str: str) -> str:
    """Format event date/time into human-readable datetime text."""
    if not date_str or not time_str:
        return ""

    try:
        dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return f"{date_str} {time_str}".strip()


def build_event_time_label_map(data: YamlData) -> EventTimeLabelMap:
    """Build mapping: gps_time -> formatted date/time string."""
    label_map: EventTimeLabelMap = {}
    for payload in data.values():
        try:
            gps_time = int(payload.get("gps_time", -1))
        except (TypeError, ValueError):
            continue
        if gps_time < 0 or gps_time in label_map:
            continue

        date_str, time_str = parse_event_date_time(payload)
        label_map[gps_time] = format_event_datetime(date_str, time_str)

    return label_map


def build_event_time_label_map_from_rows(
    rows: Sequence[PairDistributionRow],
) -> EventTimeLabelMap:
    """Build mapping: gps_time -> datetime label from cached rows."""
    label_map: EventTimeLabelMap = {}
    for row in rows:
        gps_time = int(row[2])
        if gps_time < 0 or gps_time in label_map:
            continue
        if len(row) > 9:
            label_map[gps_time] = str(row[9])
    return label_map


def apply_time_ticks(
    ax: plt.Axes,
    gps_times: Sequence[int],
    event_time_label_map: Optional[EventTimeLabelMap],
    max_ticks: int = 8,
) -> None:
    """Apply compact x-axis ticks showing gps_time and date/time label."""
    gps_time_array = np.asarray(gps_times).ravel()
    if gps_time_array.size == 0:
        return

    unique_gps_times = sorted(set(gps_time_array.tolist()))
    tick_count = min(max_ticks, len(unique_gps_times))
    if tick_count == 0:
        return

    if tick_count == 1:
        tick_positions = [unique_gps_times[0]]
    else:
        indices = np.linspace(0, len(unique_gps_times) - 1, tick_count, dtype=int)
        tick_positions = [unique_gps_times[index] for index in indices]

    tick_labels: List[str] = []
    for gps_time in tick_positions:
        datetime_label = ""
        if event_time_label_map is not None:
            datetime_label = event_time_label_map.get(gps_time, "")

        if datetime_label:
            tick_labels.append(f"{gps_time}\n{datetime_label}")
        else:
            tick_labels.append(str(gps_time))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")


def non_zero_floor_magnitude(values: Sequence[float], fallback: float = 1e-12) -> float:
    """Return ``10^k`` where ``k = floor(log10(min(values > 0)))``.

    This is used as a stable lower bound before log-scaling when data may
    include zeros or negative values.
    """
    array_values = np.asarray(values, dtype=float).ravel()
    positive_values = array_values[array_values > 0]
    if positive_values.size == 0:
        return fallback

    min_positive = float(np.min(positive_values))
    magnitude_power = int(np.floor(np.log10(min_positive)))
    return float(10 ** magnitude_power)


def build_observed_pair_deltas(
    data: YamlData,
    offsets: Optional[OffsetMap] = None,
) -> List[ObservedPairDelta]:
    """Build observed DU-pair time-delta rows from each event.

    Delta is computed as: ``du_ns(du_b) - du_ns(du_a)`` with ordered pair
    ``du_a < du_b`` under ``sort_du_id_key``.
    """
    rows: List[ObservedPairDelta] = []
    if offsets is None:
        offsets = {}

    for _, payload in data.items():
        event_number = int(payload.get("event_number", -1))
        index = int(payload.get("index", -1))
        event_time = int(payload.get("gps_time", -1))
        du_ns_map = parse_du_ns_map(payload)

        du_ids = sorted(du_ns_map.keys(), key=sort_du_id_key)
        for du_a, du_b in itertools.combinations(du_ids, 2):
            du_a_offset = offsets.get(du_a, 0.0)
            du_b_offset = offsets.get(du_b, 0.0)
            du_a_corrected = du_ns_map[du_a] - du_a_offset
            du_b_corrected = du_ns_map[du_b] - du_b_offset
            delta_ns = du_b_corrected - du_a_corrected
            rows.append(
                (
                    event_number,
                    index,
                    event_time,
                    du_a,
                    du_b,
                    delta_ns,
                    abs(delta_ns),
                )
            )

    rows.sort(
        key=lambda row: (row[2], row[1], sort_du_id_key(row[3]), sort_du_id_key(row[4]))
    )
    return rows


def build_expected_pair_deltas(
    detector_positions: Dict[str, np.ndarray],
    du_ids_in_data: Optional[Iterable[str]] = None,
) -> List[ExpectedPairDelta]:
    """Build expected DU-pair deltas from geometry as ``distance / c``."""
    if du_ids_in_data is None:
        usable_ids = sorted(detector_positions.keys(), key=sort_du_id_key)
    else:
        usable_ids = sorted(
            [du_id for du_id in set(du_ids_in_data) if du_id in detector_positions],
            key=sort_du_id_key,
        )

    rows: List[ExpectedPairDelta] = []
    for du_a, du_b in itertools.combinations(usable_ids, 2):
        pos_a = detector_positions[du_a]
        pos_b = detector_positions[du_b]
        distance_m = float(np.linalg.norm(pos_b - pos_a))
        expected_delta_ns = distance_m / SPEED_OF_LIGHT_M_PER_S * NS_PER_SECOND
        rows.append((du_a, du_b, distance_m, expected_delta_ns))

    return rows


def theoretical_cache_path(det_pos_path: Path) -> Path:
    """Return shared cache CSV path for theoretical DU-pair deltas."""
    return det_pos_path.with_name(f"{det_pos_path.stem}_du_pair_theoretical.csv")


def write_theoretical_cache(path: Path, rows: Sequence[ExpectedPairDelta]) -> None:
    """Write shared theoretical DU-pair cache CSV."""
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["du_a", "du_b", "distance_m", "theoretical_delta_ns"])
        writer.writerows(rows)


def read_theoretical_cache(path: Path) -> List[ExpectedPairDelta]:
    """Read shared theoretical DU-pair cache CSV."""
    rows: List[ExpectedPairDelta] = []
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            du_a = str(row["du_a"])
            du_b = str(row["du_b"])
            distance_m = float(row["distance_m"])
            theoretical_delta_ns = float(row["theoretical_delta_ns"])
            rows.append((du_a, du_b, distance_m, theoretical_delta_ns))
    return rows


def load_or_build_theoretical_rows(
    det_pos_path: Path,
    detector_positions: Dict[str, np.ndarray],
) -> List[ExpectedPairDelta]:
    """Load shared theoretical cache if exists; otherwise build and persist."""
    cache_path = theoretical_cache_path(det_pos_path)
    if cache_path.exists():
        logger.info("Using theoretical DU-pair cache: {}", cache_path)
        return read_theoretical_cache(cache_path)

    rows = build_expected_pair_deltas(detector_positions)
    write_theoretical_cache(cache_path, rows)
    logger.info("Wrote theoretical DU-pair cache: {}", cache_path)
    return rows


def build_pair_distribution_rows(
    observed_rows: Sequence[ObservedPairDelta],
    expected_rows: Sequence[ExpectedPairDelta],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> List[PairDistributionRow]:
    """Join observed event-level pair deltas with theoretical pair deltas."""
    expected_map: Dict[Tuple[str, str], Tuple[float, float]] = {
        (du_a, du_b): (distance_m, expected_ns)
        for du_a, du_b, distance_m, expected_ns in expected_rows
    }

    rows: List[PairDistributionRow] = []
    for event_number, index, event_time, du_a, du_b, delta_ns, abs_delta_ns in observed_rows:
        expected = expected_map.get((du_a, du_b))
        if expected is None:
            continue
        distance_m, theoretical_delta_ns = expected
        event_datetime = ""
        if event_time_label_map is not None:
            event_datetime = event_time_label_map.get(event_time, "")
        rows.append(
            (
                event_number,
                index,
                event_time,
                du_a,
                du_b,
                delta_ns,
                abs_delta_ns,
                theoretical_delta_ns,
                distance_m,
                event_datetime,
            )
        )

    rows.sort(
        key=lambda row: (row[2], row[1], sort_du_id_key(row[3]), sort_du_id_key(row[4]))
    )
    return rows


def write_pair_distribution_csv(path: Path, rows: Sequence[PairDistributionRow]) -> None:
    """Write DU-pair distribution CSV with observed and theoretical deltas."""
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "event_number",
                "index",
                "event_time",
                "du_a",
                "du_b",
                "observed_delta_ns",
                "observed_abs_delta_ns",
                "theoretical_delta_ns",
                "distance_m",
                "event_datetime",
            ]
        )
        writer.writerows(rows)


def read_pair_distribution_csv(path: Path) -> List[PairDistributionRow]:
    """Read DU-pair distribution CSV with observed and theoretical deltas."""
    rows: List[PairDistributionRow] = []
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                (
                    int(row["event_number"]),
                    int(row["index"]),
                    int(row["event_time"]),
                    str(row["du_a"]),
                    str(row["du_b"]),
                    float(row["observed_delta_ns"]),
                    float(row["observed_abs_delta_ns"]),
                    float(row["theoretical_delta_ns"]),
                    float(row["distance_m"]),
                    str(row.get("event_datetime", "")),
                )
            )
    return rows


def plot_pair_delta_distribution(
    path: Path,
    observed_abs_delta: Sequence[float],
) -> None:
    """Plot observed DU-pair time-delta distribution only."""
    observed_values = np.asarray(observed_abs_delta, dtype=float)
    if observed_values.size == 0:
        logger.warning("No pair delta values available; skip pair delta plot")
        return

    floor_value = non_zero_floor_magnitude(observed_values)
    observed_values[observed_values <= 0] = floor_value
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        np.log10(observed_values),
        bins=80,
        alpha=0.8,
        label="Observed |Δt| (ns)",
        edgecolor="black",
    )

    ax.set_xlabel("log(Time delta (ns))")
    ax.set_ylabel("Count")
    ax.set_title("Observed DU-pair time delta distribution")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_observed_expected_ratio(path: Path, rows: Sequence[PairDistributionRow]) -> None:
    """Plot distribution of observed/theoretical ratio for DU-pair deltas."""
    if not rows:
        logger.warning("No pair delta values available; skip ratio plot")
        return

    ratios = np.array([
        row[6] / row[7]
        for row in rows
        if row[7] > 0
    ])
    if len(ratios) == 0:
        logger.warning("No positive theoretical deltas available; skip ratio plot")
        return

    floor_value = non_zero_floor_magnitude(ratios)
    ratios[ratios <= 0] = floor_value
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(np.log10(ratios), bins=80, edgecolor="black", alpha=0.8)
    ax.set_xlabel("log(Observed |Δt| / Theoretical Δt)")
    ax.set_ylabel("Count")
    ax.set_title("Observed/Theoretical DU-pair delta ratio")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    # ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_delta_vs_event_time(
    path: Path,
    rows: Sequence[PairDistributionRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> None:
    """Plot observed absolute delta scatter against event time."""
    samples = [(row[2], row[6]) for row in rows if row[2] >= 0]
    if not samples:
        logger.warning("No event-time delta samples available; skip delta-vs-time plot")
        return

    x_values = np.array([sample[0] for sample in samples])
    y_values = np.array([sample[1] for sample in samples])

    floor_value = non_zero_floor_magnitude(y_values)
    y_values[y_values <= 0] = floor_value
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_values, y_values, s=8, alpha=0.6)
    ax.set_xlabel("Event time (gps_time)")
    ax.set_ylabel("Observed |Δt| (ns)")
    ax.set_title("Observed DU-pair delta vs event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, x_values, event_time_label_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ratio_vs_event_time(
    path: Path,
    rows: Sequence[PairDistributionRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> None:
    """Plot observed/theoretical ratio scatter against event time."""
    samples = [
        (row[2], row[6] / row[7])
        for row in rows
        if row[2] >= 0 and row[7] > 0
    ]
    if not samples:
        logger.warning("No event-time ratio samples available; skip ratio-vs-time plot")
        return

    x_values = np.array([sample[0] for sample in samples])
    y_values = np.array([sample[1] for sample in samples])

    floor_value = non_zero_floor_magnitude(y_values)
    y_values[y_values <= 0] = floor_value
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_values, y_values, s=8, alpha=0.6)
    ax.set_xlabel("Event time (gps_time)")
    ax.set_ylabel("Observed |Δt| / Theoretical Δt")
    ax.set_title("Observed/Theoretical DU-pair ratio vs event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, x_values, event_time_label_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute DU du_ns distributions and DU-pair observed/expected "
            "time-delta distributions from Trigger YAML"
        )
    )
    parser.add_argument("yaml_file", help="Input Trigger YAML path")
    parser.add_argument(
        "--det-pos",
        default="_gp65_rtksort.txt",
        help="Detector position file path (default: _gp65_rtksort.txt)",
    )
    parser.add_argument(
        "--offset-file",
        default="2025-10-28_beacon_25Hz_offset.txt",
        help=(
            "DU time offset file path. Format: du_id, offset, sigma "
            "(sigma column ignored)."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Export CSV only, do not generate plots",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Force recomputation from YAML and overwrite existing "
            "du_pair_delta_distribution.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    yaml_path = Path(args.yaml_file)
    stem = yaml_path.stem
    root = yaml_path.parent
    distribution_csv = root / f"{stem}_du_pair_delta_distribution.csv"

    if distribution_csv.exists() and not args.force_recompute:
        logger.info("Found distribution CSV cache: {}", distribution_csv)
        distribution_rows = read_pair_distribution_csv(distribution_csv)
        logger.info("Loaded distribution rows from cache: {}", len(distribution_rows))
        event_time_label_map = build_event_time_label_map_from_rows(distribution_rows)

        if not args.no_plot:
            pair_delta_plot = root / f"{stem}_du_pair_delta_distribution.png"
            ratio_plot = root / f"{stem}_observed_expected_ratio.png"
            delta_time_plot = root / f"{stem}_du_pair_delta_vs_time.png"
            ratio_time_plot = root / f"{stem}_observed_expected_ratio_vs_time.png"
            plot_pair_delta_distribution(
                pair_delta_plot,
                np.array([row[6] for row in distribution_rows]),
            )
            plot_observed_expected_ratio(ratio_plot, distribution_rows)
            plot_delta_vs_event_time(
                delta_time_plot,
                distribution_rows,
                event_time_label_map,
            )
            plot_ratio_vs_event_time(
                ratio_time_plot,
                distribution_rows,
                event_time_label_map,
            )
            logger.info("Plot written: {}", pair_delta_plot)
            logger.info("Plot written: {}", ratio_plot)
            logger.info("Plot written: {}", delta_time_plot)
            logger.info("Plot written: {}", ratio_time_plot)
        else:
            logger.info("Plotting disabled by --no-plot")

        return 0

    if distribution_csv.exists() and args.force_recompute:
        logger.info(
            "Force recompute enabled, ignore existing distribution CSV: {}",
            distribution_csv,
        )

    if not yaml_path.exists():
        logger.error("Input YAML file not found: {}", yaml_path)
        return 2

    try:
        data = read_yaml_events(yaml_path)
    except ValueError as exc:
        logger.error("Invalid YAML format: {}", exc)
        return 2

    event_time_label_map = build_event_time_label_map(data)

    det_pos_path = Path(args.det_pos)
    detector_positions = load_data_from_file(args.det_pos)

    offset_map = load_du_time_offsets(Path(args.offset_file))
    observed_rows = build_observed_pair_deltas(data, offset_map)

    du_ids_in_data = [du_id for _, _, _, du_a, du_b, _, _ in observed_rows for du_id in (du_a, du_b)]
    missing_position_ids = sorted(
        set(du_ids_in_data) - set(detector_positions.keys()),
        key=sort_du_id_key,
    )
    if missing_position_ids:
        logger.warning(
            "{} DU IDs from YAML not found in detector position file, examples: {}",
            len(missing_position_ids),
            ",".join(missing_position_ids[:10]),
        )

    expected_rows = load_or_build_theoretical_rows(det_pos_path, detector_positions)
    distribution_rows = build_pair_distribution_rows(
        observed_rows,
        expected_rows,
        event_time_label_map,
    )

    write_pair_distribution_csv(distribution_csv, distribution_rows)

    logger.info("Observed pair rows: {}", len(observed_rows))
    logger.info("Theoretical pair rows: {}", len(expected_rows))
    logger.info("Distribution rows: {} -> {}", len(distribution_rows), distribution_csv)

    if not args.no_plot:
        pair_delta_plot = root / f"{stem}_du_pair_delta_distribution.png"
        ratio_plot = root / f"{stem}_observed_expected_ratio.png"
        delta_time_plot = root / f"{stem}_du_pair_delta_vs_time.png"
        ratio_time_plot = root / f"{stem}_observed_expected_ratio_vs_time.png"
        plot_pair_delta_distribution(
            pair_delta_plot,
            np.array([row[6] for row in distribution_rows]),
        )
        plot_observed_expected_ratio(ratio_plot, distribution_rows)
        plot_delta_vs_event_time(delta_time_plot, distribution_rows, event_time_label_map)
        plot_ratio_vs_event_time(ratio_time_plot, distribution_rows, event_time_label_map)
        logger.info("Plot written: {}", pair_delta_plot)
        logger.info("Plot written: {}", ratio_plot)
        logger.info("Plot written: {}", delta_time_plot)
        logger.info("Plot written: {}", ratio_time_plot)
    else:
        logger.info("Plotting disabled by --no-plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
