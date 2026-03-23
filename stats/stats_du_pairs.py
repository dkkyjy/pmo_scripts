#!/usr/bin/env python3
"""Statistics for DU-pair observed/theoretical time-delta distributions.

This script reads Trigger YAML (event payloads containing ``time``), then:
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
import json
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import yaml

from find_event.io import load_data_from_file
from logger_config import logger
from du_pair_theoretical import load_or_build_theoretical_rows
from stats import common as scommon

plt.style.use(["science", "grid", "notebook"])

def make_named_row(fields: Iterable[str], **values: Any):
    """Create one named row based on a predefined field schema using namedtuple."""
    tuple_type = {
        OBSERVED_PAIR_DELTA_FIELDS: ObservedPairDelta,
        EXPECTED_PAIR_DELTA_FIELDS: ExpectedPairDelta,
        PAIR_DISTRIBUTION_FIELDS: PairDistributionRow,
        EVENT_PAIR_COUNT_FIELDS: EventPairCountRow,
        EVENT_MAX_RATIO_FIELDS: EventMaxRatioRow,
    }.get(fields)
    if tuple_type is None:
        raise ValueError(f"Unknown row fields: {fields}")
    return tuple_type(**values)


OBSERVED_PAIR_DELTA_FIELDS = (
    "event_number",
    "event_time",
    "du_a",
    "du_b",
    "observed_delta_ns",
    "observed_abs_delta_ns",
)
EXPECTED_PAIR_DELTA_FIELDS = (
    "du_a",
    "du_b",
    "distance_m",
    "theoretical_delta_ns",
)
PAIR_DISTRIBUTION_FIELDS = (
    "event_number",
    "event_time",
    "du_a",
    "du_b",
    "observed_delta_ns",
    "observed_abs_delta_ns",
    "theoretical_delta_ns",
    "distance_m",
    "event_datetime",
)
EVENT_PAIR_COUNT_FIELDS = (
    "event_number",
    "event_time",
    "pair_count",
    "event_datetime",
)
EVENT_MAX_RATIO_FIELDS = (
    "event_number",
    "event_time",
    "max_observed_theoretical_ratio",
)

# Define namedtuple row models
ObservedPairDelta = namedtuple("ObservedPairDelta", OBSERVED_PAIR_DELTA_FIELDS)
ExpectedPairDelta = namedtuple("ExpectedPairDelta", EXPECTED_PAIR_DELTA_FIELDS)
PairDistributionRow = namedtuple("PairDistributionRow", PAIR_DISTRIBUTION_FIELDS)
EventPairCountRow = namedtuple("EventPairCountRow", EVENT_PAIR_COUNT_FIELDS)
EventMaxRatioRow = namedtuple("EventMaxRatioRow", EVENT_MAX_RATIO_FIELDS)

DATETIME_FORMAT = scommon.DATETIME_FORMAT
DATETIME_HELP_FORMAT = scommon.DATETIME_HELP_FORMAT


def normalize_cli_datetime_text(value: Optional[datetime]) -> str:
    """Normalize optional CLI datetime to deterministic text for cache meta."""
    return scommon.normalize_cli_datetime_text(value)


def cache_file_state(path: Path) -> Dict[str, Any]:
    """Build lightweight file signature for cache validation."""
    return scommon.cache_file_state(path)


def distribution_meta_path(distribution_csv: Path) -> Path:
    """Return meta file path for one distribution cache CSV."""
    return distribution_csv.with_name(f"{distribution_csv.stem}.meta.json")


def build_distribution_cache_meta(
    yaml_path: Path,
    det_pos_path: Path,
    offset_path: Path,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[str, Any]:
    """Build cache metadata signature from inputs and filter parameters."""
    return {
        "schema_version": 1,
        "yaml": cache_file_state(yaml_path),
        "det_pos": cache_file_state(det_pos_path),
        "offset": cache_file_state(offset_path),
        "start_datetime": normalize_cli_datetime_text(start_dt),
        "end_datetime": normalize_cli_datetime_text(end_dt),
    }


def write_distribution_cache_meta(path: Path, meta: Dict[str, Any]) -> None:
    """Write distribution cache metadata file."""
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(meta, file_obj, ensure_ascii=True, sort_keys=True)


def read_distribution_cache_meta(path: Path) -> Dict[str, Any]:
    """Read distribution cache metadata file."""
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError("Distribution cache meta must be a JSON object")
    return loaded


def distribution_cache_meta_is_valid(
    cached_meta: Dict[str, Any],
    expected_meta: Dict[str, Any],
) -> tuple[bool, str]:
    """Check whether cached meta matches expected runtime signature."""
    if int(cached_meta.get("schema_version", -1)) != int(
        expected_meta.get("schema_version", -2)
    ):
        return False, "schema_version mismatch"

    for key in ("yaml", "det_pos", "offset"):
        cached_file_state = cached_meta.get(key)
        expected_file_state = expected_meta.get(key)
        if cached_file_state != expected_file_state:
            return False, f"{key} signature mismatch"

    for key in ("start_datetime", "end_datetime"):
        if str(cached_meta.get(key, "")) != str(expected_meta.get(key, "")):
            return False, f"{key} mismatch"

    return True, ""


def sort_du_id_key(du_id: str) -> tuple[int, str]:
    """Sort DU IDs numerically when possible, else lexicographically."""
    return scommon.sort_du_id_key(du_id)


def read_yaml_events(yaml_path: Path) -> YamlData:
    """Read Trigger YAML and validate top-level structure."""
    return scommon.read_yaml_events(yaml_path, logger)


def parse_du_ns_map(payload: EventPayload) -> Dict[str, float]:
    """Parse ``time`` map from one event payload.

    Supports both scalar values and list values; for list values, the last
    numeric sample is used to keep backward-compatible per-event scalar math.
    """
    return scommon.parse_du_ns_map(payload, logger)


def load_du_time_offsets(offset_file: Path) -> OffsetMap:
    """Load DU time offsets from a file.

    Expected line format (comma-separated):
        du_id, offset, sigma
    where only ``du_id`` and ``offset`` are used; ``sigma`` is ignored.
    """
    return scommon.load_du_time_offsets(offset_file, logger)


def parse_event_date_time(payload: EventPayload) -> tuple[str, str]:
    """Parse event date/time strings from payload ``datetime``."""
    datetime_value = payload.get("datetime", "")
    datetime_text = str(datetime_value) if datetime_value is not None else ""
    if not datetime_text:
        return "", ""

    try:
        dt_obj = datetime.strptime(datetime_text, DATETIME_FORMAT)
        return dt_obj.strftime("%Y%m%d"), dt_obj.strftime("%H%M%S")
    except ValueError:
        return "", ""


def format_event_datetime(date_str: str, time_str: str) -> str:
    """Format event date/time into human-readable datetime text."""
    if not date_str or not time_str:
        return ""

    normalized_datetime = normalize_event_datetime_text(date_str, time_str)
    try:
        dt_obj = datetime.strptime(normalized_datetime, DATETIME_FORMAT)
        return dt_obj.strftime(DATETIME_FORMAT)
    except ValueError:
        return f"{date_str} {time_str}".strip()


def normalize_event_datetime_text(date_str: str, time_str: str) -> str:
    """Normalize compact date/time text into ``DATETIME_FORMAT``."""
    if date_str.isdigit() and len(date_str) == 8:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    if time_str.isdigit() and len(time_str) == 6:
        time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"

    return f"{date_str}T{time_str}"


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime argument as ``YYYY-MM-DDTHH:MM:SS``."""
    return scommon.parse_cli_datetime(value)


def parse_event_datetime(payload: EventPayload) -> Optional[datetime]:
    """Parse one event payload datetime from ``datetime`` field."""
    return scommon.parse_payload_datetime(payload)


def parse_row_datetime(event_datetime: str) -> Optional[datetime]:
    """Parse datetime text from distribution row cache field."""
    if not event_datetime:
        return None

    try:
        return datetime.strptime(event_datetime, DATETIME_FORMAT)
    except ValueError:
        return None


def in_datetime_range(
    event_dt: datetime,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> bool:
    """Check whether event datetime is inside an inclusive range."""
    return scommon.in_datetime_range(event_dt, start_dt, end_dt)


def filter_events_by_datetime_range(
    data: YamlData,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> YamlData:
    """Filter YAML events by datetime range from payload ``datetime``."""
    return scommon.filter_data_by_datetime_range(
        data,
        start_dt,
        end_dt,
        logger=logger,
        parse_payload_datetime_fn=parse_event_datetime,
    )


def filter_distribution_rows_by_datetime_range(
    rows: Sequence[PairDistributionRow],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> List[PairDistributionRow]:
    """Filter distribution rows by datetime range from ``event_datetime``."""
    if start_dt is None and end_dt is None:
        return list(rows)

    filtered_rows: List[PairDistributionRow] = []
    invalid_datetime_count = 0

    for row in rows:
        event_dt = parse_row_datetime(str(row.event_datetime))
        if event_dt is None:
            invalid_datetime_count += 1
            continue
        if in_datetime_range(event_dt, start_dt, end_dt):
            filtered_rows.append(row)

    logger.info(
        "Datetime range filter on cached rows: {} -> {}",
        len(rows),
        len(filtered_rows),
    )
    if invalid_datetime_count > 0:
        logger.warning(
            "Skipped {} cached rows with invalid/missing event_datetime "
            "while filtering",
            invalid_datetime_count,
        )

    return filtered_rows


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
        gps_time = int(row.event_time)
        if gps_time < 0 or gps_time in label_map:
            continue
        event_datetime = str(row.event_datetime)
        if event_datetime:
            label_map[gps_time] = event_datetime
    return label_map


def apply_time_ticks(
    ax: plt.Axes,
    gps_times: Sequence[int],
    event_time_label_map: Optional[EventTimeLabelMap],
    max_ticks: int = 8,
) -> None:
    """Apply compact x-axis ticks showing gps_time and date/time label."""
    scommon.apply_gps_datetime_ticks(
        ax,
        list(np.asarray(gps_times).ravel().tolist()),
        event_time_label_map,
        max_ticks=max_ticks,
    )


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

    Delta is computed as: ``time(du_b) - time(du_a)`` with ordered pair
    ``du_a < du_b`` under ``sort_du_id_key``.
    """
    rows: List[ObservedPairDelta] = []
    if offsets is None:
        offsets = {}

    for _, payload in data.items():
        event_number = payload.get("event_number", -1)
        event_time = payload.get("gps_time", -1)
        du_ns_map = parse_du_ns_map(payload)

        du_ids = sorted(du_ns_map.keys(), key=sort_du_id_key)
        for du_a, du_b in itertools.combinations(du_ids, 2):
            du_a_offset = offsets.get(du_a, 0.0)
            du_b_offset = offsets.get(du_b, 0.0)
            du_a_corrected = du_ns_map[du_a] - du_a_offset
            du_b_corrected = du_ns_map[du_b] - du_b_offset
            delta_ns = du_b_corrected - du_a_corrected
            rows.append(
                make_named_row(
                    OBSERVED_PAIR_DELTA_FIELDS,
                    event_number=event_number,
                    event_time=event_time,
                    du_a=du_a,
                    du_b=du_b,
                    observed_delta_ns=delta_ns,
                    observed_abs_delta_ns=abs(delta_ns),
                )
            )

    rows.sort(
        key=lambda row: (
            int(row.event_time),
            int(row.event_number),
            sort_du_id_key(str(row.du_a)),
            sort_du_id_key(str(row.du_b)),
        )
    )
    return rows



def build_pair_distribution_rows(
    observed_rows: Sequence[ObservedPairDelta],
    expected_rows: Sequence[ExpectedPairDelta],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> List[PairDistributionRow]:
    """Join observed event-level pair deltas with theoretical pair deltas."""
    expected_map: Dict[tuple[str, str], tuple[float, float]] = {
        (str(row.du_a), str(row.du_b)): (
            float(row.distance_m),
            float(row.theoretical_delta_ns),
        )
        for row in expected_rows
    }

    rows: List[PairDistributionRow] = []
    for observed_row in observed_rows:
        event_number = int(observed_row.event_number)
        event_time = int(observed_row.event_time)
        du_a = str(observed_row.du_a)
        du_b = str(observed_row.du_b)
        delta_ns = float(observed_row.observed_delta_ns)
        abs_delta_ns = float(observed_row.observed_abs_delta_ns)
        expected = expected_map.get((du_a, du_b))
        if expected is None:
            continue
        distance_m, theoretical_delta_ns = expected
        event_datetime = ""
        if event_time_label_map is not None:
            event_datetime = event_time_label_map.get(event_time, "")
        rows.append(
            make_named_row(
                PAIR_DISTRIBUTION_FIELDS,
                event_number=event_number,
                event_time=event_time,
                du_a=du_a,
                du_b=du_b,
                observed_delta_ns=delta_ns,
                observed_abs_delta_ns=abs_delta_ns,
                theoretical_delta_ns=theoretical_delta_ns,
                distance_m=distance_m,
                event_datetime=event_datetime,
            )
        )

    rows.sort(
        key=lambda row: (
            int(row.event_time),
            int(row.event_number),
            sort_du_id_key(str(row.du_a)),
            sort_du_id_key(str(row.du_b)),
        )
    )
    return rows


def write_pair_distribution_csv(path: Path, rows: Sequence[PairDistributionRow]) -> None:
    """Write DU-pair distribution CSV with observed and theoretical deltas."""
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "event_number",
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
        writer.writerows(tuple(row) for row in rows)


def read_pair_distribution_csv(path: Path) -> List[PairDistributionRow]:
    """Read DU-pair distribution CSV with observed and theoretical deltas."""
    rows: List[PairDistributionRow] = []
    invalid_row_count = 0
    required_fields = {
        "event_number",
        "event_time",
        "du_a",
        "du_b",
        "observed_delta_ns",
        "observed_abs_delta_ns",
        "theoretical_delta_ns",
        "distance_m",
        "event_datetime",
    }

    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        header_fields = set(reader.fieldnames or [])
        if not required_fields.issubset(header_fields):
            missing_fields = sorted(required_fields - header_fields)
            raise ValueError(
                "Distribution cache missing required columns: "
                f"{','.join(missing_fields)}"
            )

        for row_index, row in enumerate(reader, start=2):
            try:
                rows.append(
                    make_named_row(
                        PAIR_DISTRIBUTION_FIELDS,
                        event_number=int(row["event_number"]),
                        event_time=int(row["event_time"]),
                        du_a=str(row["du_a"]),
                        du_b=str(row["du_b"]),
                        observed_delta_ns=float(row["observed_delta_ns"]),
                        observed_abs_delta_ns=float(row["observed_abs_delta_ns"]),
                        theoretical_delta_ns=float(row["theoretical_delta_ns"]),
                        distance_m=float(row["distance_m"]),
                        event_datetime=str(row.get("event_datetime", "")),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                invalid_row_count += 1
                logger.warning(
                    "Skip invalid distribution cache row at line {}: {}",
                    row_index,
                    exc,
                )
                continue

    if invalid_row_count > 0:
        logger.warning(
            "Skipped {} invalid rows while reading distribution cache: {}",
            invalid_row_count,
            path,
        )

    return rows


def plot_pair_delta_distribution(
    path: Path,
    rows: Sequence[PairDistributionRow],
) -> None:
    """Plot observed DU-pair time-delta distribution only."""
    observed_values =  np.array([float(row.observed_abs_delta_ns) for row in rows])
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
        float(row.observed_abs_delta_ns) / float(row.theoretical_delta_ns)
        for row in rows
        if float(row.theoretical_delta_ns) > 0
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
    samples = [
        (int(row.event_time), float(row.observed_abs_delta_ns))
        for row in rows
        if int(row.event_time) >= 0
    ]
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
        (
            int(row.event_time),
            float(row.observed_abs_delta_ns) / float(row.theoretical_delta_ns),
        )
        for row in rows
        if int(row.event_time) >= 0 and float(row.theoretical_delta_ns) > 0
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


def plot_theoretical_delta_vs_event_time(
    path: Path,
    rows: Sequence[PairDistributionRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> None:
    """Plot theoretical delta scatter against event time."""
    samples = [
        (int(row.event_time), float(row.theoretical_delta_ns))
        for row in rows
        if int(row.event_time) >= 0 and float(row.theoretical_delta_ns) > 0
    ]
    if not samples:
        logger.warning(
            "No event-time theoretical-delta samples available; "
            "skip theoretical-delta-vs-time plot"
        )
        return

    x_values = np.array([sample[0] for sample in samples])
    y_values = np.array([sample[1] for sample in samples], dtype=float)

    floor_value = non_zero_floor_magnitude(y_values)
    y_values[y_values <= 0] = floor_value
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_values, y_values, s=8, alpha=0.6)
    ax.set_xlabel("Event time (gps_time)")
    ax.set_ylabel("Theoretical Δt (ns)")
    ax.set_title("Theoretical DU-pair delta vs event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, x_values, event_time_label_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_theoretical_delta_histogram(
    path: Path,
    rows: Sequence[PairDistributionRow],
) -> None:
    """Plot histogram of theoretical delta values."""
    theoretical_values = np.array(
        [
            float(row.theoretical_delta_ns)
            for row in rows
            if float(row.theoretical_delta_ns) > 0
        ],
        dtype=float,
    )
    if theoretical_values.size == 0:
        logger.warning(
            "No positive theoretical deltas available; "
            "skip theoretical-delta histogram"
        )
        return

    floor_value = non_zero_floor_magnitude(theoretical_values)
    theoretical_values[theoretical_values <= 0] = floor_value

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(np.log10(theoretical_values), bins=80, edgecolor="black", alpha=0.8)
    ax.set_xlabel("log(Theoretical Δt (ns))")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of theoretical DU-pair delta")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_event_max_ratio_rows(
    rows: Sequence[PairDistributionRow],
) -> List[EventMaxRatioRow]:
    """Build per-event maximum observed/theoretical ratio rows."""
    max_ratio_map: Dict[tuple[int, int], float] = {}

    for row in rows:
        event_number = int(row.event_number)
        event_time = int(row.event_time)
        observed_abs_delta_ns = float(row.observed_abs_delta_ns)
        theoretical_delta_ns = float(row.theoretical_delta_ns)
        if event_time < 0 or theoretical_delta_ns <= 0:
            continue

        ratio = observed_abs_delta_ns / theoretical_delta_ns
        key = (event_number, event_time)
        previous_max = max_ratio_map.get(key)
        if previous_max is None or ratio > previous_max:
            max_ratio_map[key] = ratio

    result_rows: List[EventMaxRatioRow] = [
        make_named_row(
            EVENT_MAX_RATIO_FIELDS,
            event_number=event_number,
            event_time=event_time,
            max_observed_theoretical_ratio=max_ratio,
        )
        for (event_number, event_time), max_ratio in max_ratio_map.items()
    ]
    result_rows.sort(key=lambda item: (item.event_time, item.event_number))
    return result_rows


def build_event_pair_count_rows(
    rows: Sequence[PairDistributionRow],
) -> List[EventPairCountRow]:
    """Build per-event DU-pair count rows from distribution rows."""
    grouped_counts: Dict[tuple[int, int], int] = defaultdict(int)
    event_datetime_map: Dict[tuple[int, int], str] = {}

    for row in rows:
        event_number = int(row.event_number)
        event_time = int(row.event_time)
        event_datetime = str(row.event_datetime)
        key = (event_number, event_time)
        grouped_counts[key] += 1
        if key not in event_datetime_map and event_datetime:
            event_datetime_map[key] = event_datetime

    result_rows: List[EventPairCountRow] = []
    for event_number, event_time in sorted(grouped_counts.keys(), key=lambda item: (item[1], item[0])):
        result_rows.append(
            make_named_row(
                EVENT_PAIR_COUNT_FIELDS,
                event_number=event_number,
                event_time=event_time,
                pair_count=grouped_counts[(event_number, event_time)],
                event_datetime=event_datetime_map.get((event_number, event_time), ""),
            )
        )

    return result_rows


def plot_event_max_ratio_vs_time(
    path: Path,
    rows: Sequence[EventMaxRatioRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> None:
    """Plot per-event maximum ratio against event time."""
    if not rows:
        logger.warning("No per-event max-ratio samples available; skip time plot")
        return

    x_values = np.array([int(row.event_time) for row in rows])
    y_values = np.array(
        [float(row.max_observed_theoretical_ratio) for row in rows],
        dtype=float,
    )

    floor_value = non_zero_floor_magnitude(y_values)
    y_values[y_values <= 0] = floor_value

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_values, y_values, s=10, alpha=0.7)
    ax.set_xlabel("Event time (gps_time)")
    ax.set_ylabel("Per-event max(Observed |Δt| / Theoretical Δt)")
    ax.set_title("Per-event maximum ratio vs event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, x_values, event_time_label_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_event_max_ratio_histogram(
    path: Path,
    rows: Sequence[EventMaxRatioRow],
) -> None:
    """Plot histogram of per-event maximum ratio values."""
    if not rows:
        logger.warning("No per-event max-ratio samples available; skip histogram")
        return

    ratio_values = np.array(
        [float(row.max_observed_theoretical_ratio) for row in rows],
        dtype=float,
    )
    floor_value = non_zero_floor_magnitude(ratio_values)
    ratio_values[ratio_values <= 0] = floor_value

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(np.log10(ratio_values), bins=80, edgecolor="black", alpha=0.8)
    ax.set_xlabel("log(Per-event max(Observed |Δt| / Theoretical Δt))")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-event maximum ratio")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_event_pair_count_vs_time(
    path: Path,
    rows: Sequence[EventPairCountRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> None:
    """Plot per-event DU-pair count against event time."""
    samples = [
        (int(row.event_time), float(row.pair_count))
        for row in rows
        if int(row.event_time) >= 0
    ]
    if not samples:
        logger.warning("No per-event pair-count samples available; skip time plot")
        return

    x_values = np.array([sample[0] for sample in samples])
    y_values = np.array([sample[1] for sample in samples], dtype=float)

    floor_value = non_zero_floor_magnitude(y_values)
    y_values[y_values <= 0] = floor_value

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_values, y_values, s=10, alpha=0.7)
    ax.set_xlabel("Event time (gps_time)")
    ax.set_ylabel("DU-pair count per event")
    ax.set_title("DU-pair count per event vs event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, x_values, event_time_label_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_event_pair_count_histogram(
    path: Path,
    rows: Sequence[EventPairCountRow],
) -> None:
    """Plot histogram of per-event DU-pair counts."""
    if not rows:
        logger.warning("No per-event pair-count samples available; skip histogram")
        return

    count_values = np.array([float(row.pair_count) for row in rows], dtype=float)
    floor_value = non_zero_floor_magnitude(count_values)
    count_values[count_values <= 0] = floor_value

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(np.log10(count_values), bins=80, edgecolor="black", alpha=0.8)
    ax.set_xlabel("log(DU-pair count per event)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of DU-pair count per event")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
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
    parser.add_argument(
        "--start-datetime",
        type=parse_cli_datetime,
        help=(
            "Start datetime (inclusive), format: "
            f"{DATETIME_HELP_FORMAT}"
        ),
    )
    parser.add_argument(
        "--end-datetime",
        type=parse_cli_datetime,
        help=(
            "End datetime (inclusive), format: "
            f"{DATETIME_HELP_FORMAT}"
        ),
    )
    return parser.parse_args()


def render_plots(
    root: Path,
    stem: str,
    distribution_rows: Sequence[PairDistributionRow],
    event_time_label_map: Optional[EventTimeLabelMap] = None,
) -> List[Path]:
    """Render all DU-pair plots and return generated plot paths."""
    pair_delta_plot = root / f"{stem}_du_pair_delta_distribution.png"
    ratio_plot = root / f"{stem}_observed_expected_ratio.png"
    delta_time_plot = root / f"{stem}_du_pair_delta_vs_time.png"
    ratio_time_plot = root / f"{stem}_observed_expected_ratio_vs_time.png"
    theoretical_time_plot = root / f"{stem}_theoretical_delta_vs_time.png"
    theoretical_hist_plot = root / f"{stem}_theoretical_delta_hist.png"
    event_pair_count_time_plot = root / f"{stem}_event_du_pair_count_vs_time.png"
    event_pair_count_hist_plot = root / f"{stem}_event_du_pair_count_hist.png"
    event_max_ratio_time_plot = root / f"{stem}_event_max_ratio_vs_time.png"
    event_max_ratio_hist_plot = root / f"{stem}_event_max_ratio_hist.png"

    event_pair_count_rows = build_event_pair_count_rows(distribution_rows)
    event_max_ratio_rows = build_event_max_ratio_rows(distribution_rows)

    plot_pair_delta_distribution(
        pair_delta_plot,
        distribution_rows)
    plot_observed_expected_ratio(ratio_plot, distribution_rows)
    plot_delta_vs_event_time(delta_time_plot, distribution_rows, event_time_label_map)
    plot_ratio_vs_event_time(ratio_time_plot, distribution_rows, event_time_label_map)
    plot_theoretical_delta_vs_event_time(
        theoretical_time_plot,
        distribution_rows,
        event_time_label_map,
    )
    plot_theoretical_delta_histogram(theoretical_hist_plot, distribution_rows)
    plot_event_pair_count_vs_time(
        event_pair_count_time_plot,
        event_pair_count_rows,
        event_time_label_map,
    )
    plot_event_pair_count_histogram(
        event_pair_count_hist_plot,
        event_pair_count_rows,
    )
    plot_event_max_ratio_vs_time(
        event_max_ratio_time_plot,
        event_max_ratio_rows,
        event_time_label_map,
    )
    plot_event_max_ratio_histogram(
        event_max_ratio_hist_plot,
        event_max_ratio_rows,
    )

    return [
        pair_delta_plot,
        ratio_plot,
        delta_time_plot,
        ratio_time_plot,
        theoretical_time_plot,
        theoretical_hist_plot,
        event_pair_count_time_plot,
        event_pair_count_hist_plot,
        event_max_ratio_time_plot,
        event_max_ratio_hist_plot,
    ]


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    start_datetime = getattr(args, "start_datetime", None)
    end_datetime = getattr(args, "end_datetime", None)

    if (
        start_datetime is not None
        and end_datetime is not None
        and start_datetime > end_datetime
    ):
        logger.error("start-datetime must be earlier than or equal to end-datetime")
        return 2

    yaml_path = Path(args.yaml_file)
    stem = yaml_path.stem
    root = yaml_path.parent
    distribution_csv = root / f"{stem}_du_pair_delta_distribution.csv"
    det_pos_path = Path(args.det_pos)
    offset_path = Path(args.offset_file)
    distribution_meta = distribution_meta_path(distribution_csv)

    if distribution_csv.exists() and not args.force_recompute:
        logger.info("Found distribution CSV cache: {}", distribution_csv)
        expected_meta = build_distribution_cache_meta(
            yaml_path,
            det_pos_path,
            offset_path,
            start_datetime,
            end_datetime,
        )
        cached_rows = None

        if not distribution_meta.exists():
            logger.info(
                "Cache meta missing for {}, fallback to recompute",
                distribution_csv,
            )
        else:
            try:
                cached_meta = read_distribution_cache_meta(distribution_meta)
                is_valid, reason = distribution_cache_meta_is_valid(
                    cached_meta,
                    expected_meta,
                )
                if not is_valid:
                    logger.info(
                        "Cache meta mismatch for {}: {}, fallback to recompute",
                        distribution_csv,
                        reason,
                    )
                else:
                    cached_rows = read_pair_distribution_csv(distribution_csv)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Invalid distribution cache or meta: {} / {} ({}). "
                    "Falling back to recompute.",
                    distribution_csv,
                    distribution_meta,
                    exc,
                )
                cached_rows = None

        if cached_rows is None:
            logger.info("Skip invalid cache and continue recompute path")
        else:
            distribution_rows = filter_distribution_rows_by_datetime_range(
                cached_rows,
                start_datetime,
                end_datetime,
            )
            logger.info("Loaded distribution rows from cache: {}", len(distribution_rows))
            event_time_label_map = build_event_time_label_map_from_rows(distribution_rows)

            if not args.no_plot:
                plot_paths = render_plots(
                    root,
                    stem,
                    distribution_rows,
                    event_time_label_map,
                )
                for plot_path in plot_paths:
                    logger.info("Plot written: {}", plot_path)
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

    data = filter_events_by_datetime_range(data, start_datetime, end_datetime)

    event_time_label_map = build_event_time_label_map(data)

    try:
        detector_positions = load_data_from_file(args.det_pos)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(
            "Failed to load detector position file: {} ({})",
            det_pos_path,
            exc,
        )
        return 2

    offset_map = load_du_time_offsets(Path(args.offset_file))
    observed_rows = build_observed_pair_deltas(data, offset_map)

    du_ids_in_data = [
        str(du_id)
        for row in observed_rows
        for du_id in (row.du_a, row.du_b)
    ]
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
    expected_meta = build_distribution_cache_meta(
        yaml_path,
        det_pos_path,
        offset_path,
        start_datetime,
        end_datetime,
    )
    write_distribution_cache_meta(distribution_meta, expected_meta)

    logger.info("Observed pair rows: {}", len(observed_rows))
    logger.info("Theoretical pair rows: {}", len(expected_rows))
    logger.info("Distribution rows: {} -> {}", len(distribution_rows), distribution_csv)

    if not args.no_plot:
        plot_paths = render_plots(
            root,
            stem,
            distribution_rows,
            event_time_label_map,
        )
        for plot_path in plot_paths:
            logger.info("Plot written: {}", plot_path)
    else:
        logger.info("Plotting disabled by --no-plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
