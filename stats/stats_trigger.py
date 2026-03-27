#!/usr/bin/env python3
"""Compute per-event DU counts and per-second event rates from Trigger YAML.

Required payload fields:
1) ``gps_time`` for event second
2) ``time`` for DU nanosecond map
3) optional ``datetime`` for event clock labels
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple, NamedTuple

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import yaml

from logger_config import logger
from stats import common as scommon

plt.style.use(["science", "grid", "notebook"])


SecondDateTimeMap = Dict[int, str]
DATETIME_FORMAT = scommon.DATETIME_FORMAT
DATETIME_HELP_FORMAT = scommon.DATETIME_HELP_FORMAT

EVENT_RECORD_FIELDS = (
    "event_key",
    "event_second",
    "du_count",
    "event_number",
    "event_datetime",
)
RATE_ROW_FIELDS = ("event_second", "event_count", "event_rate_hz")
DELTA_DIST_FIELDS = ("delta_second", "event_pair_count")
DU_RATE_FIELDS = ("du_id", "trigger_count", "trigger_rate_hz")
DU_RATE_PER_SECOND_FIELDS = (
    "event_second",
    "du_id",
    "trigger_count",
    "trigger_rate_hz",
)
TOTAL_DU_PER_SECOND_FIELDS = (
    "event_second",
    "total_du_trigger_count",
    "total_du_trigger_rate_hz",
)
AVG_TOTAL_DU_FIELDS = (
    "window_start_second",
    "window_end_second_exclusive",
    "total_du_trigger_count",
    "avg_total_du_trigger_rate_hz",
)
AVG_EVENT_FIELDS = (
    "window_start_second",
    "window_end_second_exclusive",
    "event_count",
    "avg_event_rate_hz",
)
AVG_DU_FIELDS = (
    "window_start_second",
    "window_end_second_exclusive",
    "du_id",
    "trigger_count",
    "avg_trigger_rate_hz",
)


# Define namedtuple row models
EventRecordRow = namedtuple("EventRecordRow", EVENT_RECORD_FIELDS)
RateRow = namedtuple("RateRow", RATE_ROW_FIELDS)
DeltaDistRow = namedtuple("DeltaDistRow", DELTA_DIST_FIELDS)
DuRateRow = namedtuple("DuRateRow", DU_RATE_FIELDS)
DuRatePerSecondRow = namedtuple("DuRatePerSecondRow", DU_RATE_PER_SECOND_FIELDS)
TotalDuPerSecondRow = namedtuple("TotalDuPerSecondRow", TOTAL_DU_PER_SECOND_FIELDS)
AvgTotalDuRow = namedtuple("AvgTotalDuRow", AVG_TOTAL_DU_FIELDS)
AvgEventRow = namedtuple("AvgEventRow", AVG_EVENT_FIELDS)
AvgDuRow = namedtuple("AvgDuRow", AVG_DU_FIELDS)

def make_named_row(fields: Iterable[str], **values: Any):
    """Create one named row based on a predefined field schema using namedtuple."""
    tuple_type = {
        EVENT_RECORD_FIELDS: EventRecordRow,
        RATE_ROW_FIELDS: RateRow,
        DELTA_DIST_FIELDS: DeltaDistRow,
        DU_RATE_FIELDS: DuRateRow,
        DU_RATE_PER_SECOND_FIELDS: DuRatePerSecondRow,
        TOTAL_DU_PER_SECOND_FIELDS: TotalDuPerSecondRow,
        AVG_TOTAL_DU_FIELDS: AvgTotalDuRow,
        AVG_EVENT_FIELDS: AvgEventRow,
        AVG_DU_FIELDS: AvgDuRow,
    }.get(fields)
    if tuple_type is None:
        raise ValueError(f"Unknown row fields: {fields}")
    return tuple_type(**values)


def build_second_datetime_map(
    records: List[Any],
) -> SecondDateTimeMap:
    """Build mapping: gps second -> datetime string."""
    second_map: SecondDateTimeMap = {}
    for record in records:
        second = int(record.event_second)
        if second in second_map:
            continue
        datetime_str = str(record.event_datetime)
        second_map[second] = datetime_str
    logger.debug(f"Built second-datetime map with {len(second_map)} unique seconds")
    return second_map


def get_second_datetime(
    second: int,
    second_datetime_map: Optional[SecondDateTimeMap],
) -> str:
    """Get datetime string for a gps second from mapping."""
    if second_datetime_map is None:
        return ""
    return second_datetime_map.get(second, "")


def apply_gps_datetime_ticks(
    ax: plt.Axes,
    seconds: List[int],
    second_datetime_map: Optional[SecondDateTimeMap],
    max_ticks: int = 8,
) -> None:
    """Apply compact x ticks showing both GPS second and date/time."""
    scommon.apply_gps_datetime_ticks(
        ax,
        seconds,
        second_datetime_map,
        max_ticks=max_ticks,
    )


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


def parse_event_datetime(payload: Dict[str, Any]) -> str:
    """Parse event datetime string from payload ``datetime``."""
    datetime_value = payload.get("datetime", "")
    datetime_text = str(datetime_value) if datetime_value is not None else ""
    if not datetime_text:
        return ""  # pragma: no cover

    try:
        dt_obj = datetime.strptime(datetime_text, DATETIME_FORMAT)
        return dt_obj.strftime(DATETIME_FORMAT)
    except ValueError:
        return ""  # pragma: no cover


def parse_payload_datetime(payload: Dict[str, Any]) -> Optional[datetime]:
    """Parse payload datetime as ``datetime`` object for range filtering."""
    return scommon.parse_payload_datetime(payload)


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime argument as ``YYYY-MM-DDTHH:MM:SS``."""
    return scommon.parse_cli_datetime(value)


def normalize_cli_datetime_text(value: Optional[datetime]) -> str:
    """Normalize optional CLI datetime to deterministic text."""
    return scommon.normalize_cli_datetime_text(value)


def cache_file_state(path: Path) -> Dict[str, Any]:
    """Build lightweight file signature for cache validation."""
    return scommon.cache_file_state(path)


def event_cache_meta_path(event_csv_path: Path) -> Path:
    """Return sidecar meta path for one event CSV cache."""
    return event_csv_path.with_name(f"{event_csv_path.stem}.meta.json")


def build_event_cache_meta(
    yaml_path: Path,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[str, Any]:
    """Build cache metadata signature from runtime inputs."""
    return {
        "schema_version": 1,
        "yaml": cache_file_state(yaml_path),
        "start_datetime": normalize_cli_datetime_text(start_dt),
        "end_datetime": normalize_cli_datetime_text(end_dt),
    }


def write_event_cache_meta(path: Path, meta: Dict[str, Any]) -> None:
    """Write event CSV cache metadata file."""
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(meta, file_obj, ensure_ascii=True, sort_keys=True)


def read_event_cache_meta(path: Path) -> Dict[str, Any]:
    """Read event CSV cache metadata file."""
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError("Event CSV cache meta must be a JSON object")
    return loaded


def event_cache_meta_is_valid(
    cached_meta: Dict[str, Any],
    expected_meta: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check whether cached meta matches expected runtime signature."""
    if int(cached_meta.get("schema_version", -1)) != int(
        expected_meta.get("schema_version", -2)
    ):
        return False, "schema_version mismatch"

    if cached_meta.get("yaml") != expected_meta.get("yaml"):
        return False, "yaml signature mismatch"  # pragma: no cover

    for key in ("start_datetime", "end_datetime"):
        if str(cached_meta.get(key, "")) != str(expected_meta.get(key, "")):
            return False, f"{key} mismatch"

    return True, ""


def in_datetime_range(
    event_dt: datetime,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> bool:
    """Check whether event datetime is inside an inclusive range."""
    return scommon.in_datetime_range(event_dt, start_dt, end_dt)


def filter_data_by_datetime_range(
    data: Dict[str, Dict[str, Any]],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[str, Dict[str, Any]]:
    """Filter YAML event payloads by datetime range from ``datetime``."""
    return scommon.filter_data_by_datetime_range(
        data,
        start_dt,
        end_dt,
        logger=logger,
        parse_payload_datetime_fn=parse_payload_datetime,
    )


def get_du_ns_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get DU nanosecond map from ``payload['time']``."""
    time_map = payload.get("time")
    if isinstance(time_map, dict):
        return time_map

    return {}


def get_event_du_ids(payload: Dict[str, Any]) -> List[str]:
    """Get event DU IDs, preferring ``du_id`` and falling back to ``time``."""
    du_ids = payload.get("du_id", [])
    if isinstance(du_ids, list):
        return [str(du_id) for du_id in du_ids]

    return [str(du_id) for du_id in get_du_ns_map(payload).keys()]


def parse_event_records(
    data: Dict[str, Dict[str, Any]],
) -> List[Any]:
    """Parse records for each event as namedtuple rows."""
    records: List[Any] = []
    for event_key, payload in data.items():
        event_second = parse_event_second_from_payload(event_key, payload)
        du_count = len(get_event_du_ids(payload))
        event_number = payload.get("event_number", -1)
        event_datetime = parse_event_datetime(payload)
        records.append(
            make_named_row(
                EVENT_RECORD_FIELDS,
                event_key=event_key,
                event_second=event_second,
                du_count=du_count,
                event_number=event_number,
                event_datetime=event_datetime,
            )
        )
    records.sort(
        key=lambda item: (
            int(item.event_second),
            int(item.event_number),
        )
    )
    if records:
        logger.info(
            "Parsed event records: total={}, second_range=[{}, {}]",
            len(records),
            records[0].event_second,
            records[-1].event_second,
        )
    else:
        logger.info("Parsed event records: total=0")
    return records


def build_rate_per_second(
    records: List[Any],
) -> List[Any]:
    """Compute event rates per second (Hz) as namedtuple rows."""
    second_counter = Counter(int(record.event_second) for record in records)
    rates = [
        make_named_row(
            RATE_ROW_FIELDS,
            event_second=sec,
            event_count=count,
            event_rate_hz=float(count),
        )
        for sec, count in sorted(second_counter.items())
    ]
    logger.debug(f"Built per-second event rates for {len(rates)} seconds")
    return rates


def build_adjacent_time_deltas(
    records: List[Any],
) -> List[int]:
    """Compute second-level deltas between adjacent events."""
    if len(records) < 2:
        return []
    deltas: List[int] = []
    for index in range(1, len(records)):
        delta_second = (
            int(records[index].event_second)
            - int(records[index - 1].event_second)
        )
        deltas.append(delta_second)
    logger.debug(f"Built adjacent event time deltas: {len(deltas)}")
    return deltas


def build_adjacent_time_delta_distribution(
    deltas: List[int],
) -> List[Any]:
    """Build distribution rows as namedtuple."""
    counter = Counter(deltas)
    rows = [
        make_named_row(
            DELTA_DIST_FIELDS,
            delta_second=delta_second,
            event_pair_count=count,
        )
        for delta_second, count in sorted(counter.items())
    ]
    logger.debug(f"Built adjacent time-delta distribution rows: {len(rows)}")
    return rows


def build_du_trigger_rate(
    data: Dict[str, Dict[str, Any]],
    records: List[Any],
) -> List[Any]:
    """Compute trigger count and trigger rate (Hz) for each DU as namedtuple rows."""
    if not records:
        return []
    min_second = min(int(record.event_second) for record in records)
    max_second = max(int(record.event_second) for record in records)
    observation_seconds = max_second - min_second + 1
    if observation_seconds <= 0:
        observation_seconds = 1
    du_counter: Counter[str] = Counter()
    for payload in data.values():
        for du_id in get_event_du_ids(payload):
            du_counter[du_id] += 1
    rows: List[Any] = []
    for du_id, count in du_counter.items():
        rows.append(
            make_named_row(
                DU_RATE_FIELDS,
                du_id=du_id,
                trigger_count=count,
                trigger_rate_hz=count / float(observation_seconds),
            )
        )
    rows.sort(
        key=lambda row: (
            int(str(row.du_id)) if str(row.du_id).isdigit() else str(row.du_id)
        )
    )
    logger.debug(
        "Built per-DU trigger rates: du_count={}, observation_seconds={}",
        len(rows),
        observation_seconds,
    )
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

        for du_id in get_event_du_ids(payload):
            per_second_du[second][du_id] += 1

    logger.debug(f"Built per-second DU counters for {len(per_second_du)} seconds")
    return per_second_du


def build_du_rate_per_second_rows(
    per_second_du: Dict[int, Counter[str]],
) -> List[NamedTuple]:
    """Flatten per-second DU trigger-rate records (Hz)."""
    rows: List[NamedTuple] = []
    for second in sorted(per_second_du.keys()):
        du_counter = per_second_du[second]
        sorted_du_ids = sorted(
            du_counter.keys(), key=lambda item: int(item) if item.isdigit() else item
        )
        for du_id in sorted_du_ids:
            count = du_counter[du_id]
            rows.append(
                make_named_row(
                    DU_RATE_PER_SECOND_FIELDS,
                    event_second=second,
                    du_id=du_id,
                    trigger_count=count,
                    trigger_rate_hz=float(count),
                )
            )
    logger.debug(f"Expanded per-second DU rate rows: {len(rows)}")
    return rows


def build_total_du_trigger_per_second_rows(
    per_second_du: Dict[int, Counter[str]],
) -> List[NamedTuple]:
    """Build per-second total DU trigger count rows."""
    rows: List[NamedTuple] = []
    for second in sorted(per_second_du.keys()):
        total_count = sum(int(count) for count in per_second_du[second].values())
        rows.append(
            make_named_row(
                TOTAL_DU_PER_SECOND_FIELDS,
                event_second=second,
                total_du_trigger_count=total_count,
                total_du_trigger_rate_hz=float(total_count),
            )
        )
    logger.debug(f"Built per-second total DU trigger rows: {len(rows)}")
    return rows


def build_avg_total_du_trigger_rows(
    total_du_rows: List[NamedTuple],
    window_seconds: int,
) -> List[NamedTuple]:
    """Build window-averaged total DU trigger count/rate rows."""
    if not total_du_rows:
        return []

    second_to_total_count: Dict[int, int] = {
        int(row.event_second): int(row.total_du_trigger_count)
        for row in total_du_rows
    }
    start_second = min(second_to_total_count.keys())
    end_second_inclusive = max(second_to_total_count.keys())
    stop_second_exclusive = end_second_inclusive + 1

    rows: List[NamedTuple] = []
    window_start = start_second
    while window_start < stop_second_exclusive:
        window_end_exclusive = min(window_start + window_seconds, stop_second_exclusive)
        duration = window_end_exclusive - window_start

        total_count = 0
        for second in range(window_start, window_end_exclusive):
            total_count += second_to_total_count.get(second, 0)

        rows.append(
            make_named_row(
                AVG_TOTAL_DU_FIELDS,
                window_start_second=window_start,
                window_end_second_exclusive=window_end_exclusive,
                total_du_trigger_count=total_count,
                avg_total_du_trigger_rate_hz=total_count / duration,
            )
        )
        window_start += window_seconds

    logger.debug(
        "Built window-averaged total DU rows: rows={}, window_seconds={}",
        len(rows),
        window_seconds,
    )
    return rows


def build_window_averages(
    records: List[NamedTuple],
    per_second_du: Dict[int, Counter[str]],
    window_seconds: int,
) -> Tuple[List[NamedTuple], List[NamedTuple]]:
    """Compute average event/DU trigger rates over fixed windows.

    Windows are represented as half-open intervals:
        [window_start_second, window_end_second)
    including the start and excluding the end.
    """
    if not records:
        return [], []

    start_second = min(int(record.event_second) for record in records)
    end_second_inclusive = max(int(record.event_second) for record in records)
    stop_second_exclusive = end_second_inclusive + 1

    event_counter = Counter(int(record.event_second) for record in records)
    all_du_ids = sorted(
        {
            du_id
            for second_map in per_second_du.values()
            for du_id in second_map.keys()
        },
        key=lambda item: int(item) if item.isdigit() else item,
    )

    event_rows: List[NamedTuple] = []
    du_rows: List[NamedTuple] = []

    window_start = start_second
    while window_start < stop_second_exclusive:
        window_end_exclusive = min(window_start + window_seconds, stop_second_exclusive)
        duration = window_end_exclusive - window_start

        event_count = 0
        for second in range(window_start, window_end_exclusive):
            event_count += event_counter.get(second, 0)
        event_rows.append(
            make_named_row(
                AVG_EVENT_FIELDS,
                window_start_second=window_start,
                window_end_second_exclusive=window_end_exclusive,
                event_count=event_count,
                avg_event_rate_hz=event_count / duration,
            )
        )

        for du_id in all_du_ids:
            trigger_count = 0
            for second in range(window_start, window_end_exclusive):
                trigger_count += per_second_du.get(second, Counter()).get(du_id, 0)
            du_rows.append(
                make_named_row(
                    AVG_DU_FIELDS,
                    window_start_second=window_start,
                    window_end_second_exclusive=window_end_exclusive,
                    du_id=du_id,
                    trigger_count=trigger_count,
                    avg_trigger_rate_hz=trigger_count / duration,
                )
            )

        window_start += window_seconds

    logger.debug(
        "Built window averages: event_rows={}, du_rows={}, window_seconds={}",
        len(event_rows),
        len(du_rows),
        window_seconds,
    )
    return event_rows, du_rows


def serialize_du_ids(du_ids: List[str]) -> str:
    """Serialize DU IDs to a compact CSV-safe string field."""
    return "|".join(du_ids)


def deserialize_du_ids(value: str) -> List[str]:
    """Deserialize DU IDs from compact CSV field."""
    if not value:
        return []
    return [item for item in value.split("|") if item]


def build_event_du_ids_map(
    data: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Build mapping from event_number string to DU ID list."""
    event_du_ids_map: Dict[str, List[str]] = {}
    for event_key, payload in data.items():
        event_number = payload.get("event_number", event_key)
        event_du_ids_map[str(event_number)] = get_event_du_ids(payload)
    return event_du_ids_map


def build_data_from_cached_events(
    records: List[NamedTuple],
    event_du_ids_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Rebuild minimal payloads from cache for DU-based aggregations."""
    cached_data: Dict[str, Dict[str, Any]] = {}
    for record in records:
        event_number_key = str(record.event_number)
        cached_data[event_number_key] = {
            "gps_time": int(record.event_second),
            "du_id": event_du_ids_map.get(event_number_key, []),
        }
    return cached_data


def build_output_paths(yaml_path: Path) -> Dict[str, Path]:
    """Build all output paths derived from one input YAML path."""
    stem = yaml_path.stem
    return {
        "event_csv": yaml_path.with_name(f"{stem}_event_du_count.csv"),
        "du_hist_png": yaml_path.with_name(f"{stem}_du_count_hist.png"),
        "event_du_time_png": yaml_path.with_name(
            f"{stem}_event_du_count_over_time.png"
        ),
        "rate_png": yaml_path.with_name(f"{stem}_rate_per_second.png"),
        "du_rate_png": yaml_path.with_name(f"{stem}_du_trigger_rate.png"),
        "total_du_rate_png": yaml_path.with_name(
            f"{stem}_total_du_trigger_per_second.png"
        ),
        "delta_hist_png": yaml_path.with_name(
            f"{stem}_adjacent_time_delta_hist.png"
        ),
        "du_second_heatmap_png": yaml_path.with_name(
            f"{stem}_du_rate_per_second_heatmap.png"
        ),
        "avg_event_png": yaml_path.with_name(f"{stem}_avg_event_rate.png"),
        "avg_total_du_png": yaml_path.with_name(
            f"{stem}_avg_total_du_trigger_rate.png"
        ),
        "avg_du_heatmap_png": yaml_path.with_name(
            f"{stem}_avg_du_trigger_rate_heatmap.png"
        ),
        "du_topn_png": yaml_path.with_name(f"{stem}_du_trigger_rate_topn.png"),
        "avg_du_topn_png": yaml_path.with_name(
            f"{stem}_avg_du_trigger_rate_topn.png"
        ),
    }


def compute_core_statistics(
    data_for_aggregates: Dict[str, Dict[str, Any]],
    records: List[NamedTuple],
) -> Tuple[
    SecondDateTimeMap,
    List[NamedTuple],
    List[int],
    List[NamedTuple],
    List[NamedTuple],
    Dict[int, Counter[str]],
    List[NamedTuple],
]:
    """Compute all shared derived statistics used by both data branches."""
    second_datetime_map = build_second_datetime_map(records)
    rates = build_rate_per_second(records)
    adjacent_time_deltas = build_adjacent_time_deltas(records)
    adjacent_time_delta_rows = build_adjacent_time_delta_distribution(
        adjacent_time_deltas
    )
    du_rates = build_du_trigger_rate(data_for_aggregates, records)
    per_second_du = build_du_counts_per_second(data_for_aggregates)
    du_second_rows = build_du_rate_per_second_rows(per_second_du)
    return (
        second_datetime_map,
        rates,
        adjacent_time_deltas,
        adjacent_time_delta_rows,
        du_rates,
        per_second_du,
        du_second_rows,
    )


def write_event_csv(
    path: Path,
    records: List[NamedTuple],
    event_du_ids_map: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Write per-event DU-count CSV with event_number as leading key."""
    if event_du_ids_map is None:
        event_du_ids_map = {}

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "event_number",
                "event_second",
                "event_datetime",
                "du_count",
                "du_ids",
            ]
        )
        rows = [
            (
                record.event_number,
                record.event_second,
                record.event_datetime,
                record.du_count,
                serialize_du_ids(
                    event_du_ids_map.get(
                        str(record.event_number),
                        event_du_ids_map.get(str(record.event_key), []),
                    )
                ),
            )
            for record in records
        ]
        writer.writerows(rows)
    logger.debug(f"Wrote event CSV rows: {len(records)} -> {path}")


def read_event_csv(
    path: Path,
) -> List[NamedTuple]:
    """Read per-event DU-count CSV."""
    rows: List[NamedTuple] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        invalid_row_count = 0
        for row in reader:
            try:
                event_number = int(row.get("event_number", 0))
                event_second = int(row.get("event_second", 0))
                du_count = int(row.get("du_count", 0))
                event_datetime = str(row.get("event_datetime", ""))
                rows.append(
                    make_named_row(
                        EVENT_RECORD_FIELDS,
                        event_key=str(event_number),
                        event_second=event_second,
                        du_count=du_count,
                        event_number=event_number,
                        event_datetime=event_datetime,
                    )
                )
            except (ValueError, TypeError):
                invalid_row_count += 1
                continue
        if invalid_row_count > 0:
            logger.warning(
                "Skipped {} invalid rows while reading event cache: {}",
                invalid_row_count,
                path,
            )
    return rows


def read_event_csv_du_ids(path: Path) -> Dict[str, List[str]]:
    """Read event_number -> DU IDs mapping from event CSV cache."""
    event_du_ids_map: Dict[str, List[str]] = {}
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            event_number_key = str(row.get("event_number", 0))
            event_du_ids_map[event_number_key] = deserialize_du_ids(
                str(row.get("du_ids", ""))
            )
    return event_du_ids_map


def plot_du_count_histogram(
    path: Path,
    records: List[NamedTuple],
) -> None:
    """Plot histogram of DU counts per event."""
    du_counts = [int(record.du_count) for record in records]
    if not du_counts:
        logger.warning("No event data; skip DU-count histogram")
        return

    max_du = max(du_counts)
    bins = [value - 0.5 for value in range(1, max_du + 2)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(du_counts, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_xlabel("DU count per event")
    ax.set_ylabel("Event count")
    ax.set_yscale("log")
    ax.set_title("Histogram of Trigger DU Count")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    tick_positions = list(range(1, max_du + 1, 5))
    if tick_positions and tick_positions[-1] != max_du:
        tick_positions.append(max_du)
    ax.set_xticks(tick_positions)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_event_du_count_over_time(
    path: Path,
    records: List[NamedTuple],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot DU count per event over time."""
    if not records:
        logger.warning("No event data; skip event DU-count time-series plot")
        return

    seconds = [int(record.event_second) for record in records]
    du_counts = [int(record.du_count) for record in records]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(seconds, du_counts, marker=".", s=8, linewidth=1)
    ax.set_xlabel("Event second (GPS)")
    ax.set_ylabel("DU count per event")
    ax.set_title("Event DU Count Over Time")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, seconds, second_datetime_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adjacent_time_delta_histogram(path: Path, deltas: List[int]) -> None:
    """Plot histogram of adjacent event time deltas."""
    if not deltas:
        logger.warning("No adjacent-event delta data; skip delta histogram")
        return

    min_delta = min(deltas)
    max_delta = max(deltas)
    if min_delta == max_delta:
        bins = [min_delta - 0.5, max_delta + 0.5]
    else:
        bins = np.arange(min_delta - 0.5, max_delta + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(deltas, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Adjacent event time delta (s)")
    ax.set_ylabel("Event-pair count")
    ax.set_yscale("log")
    ax.set_title("Histogram of Adjacent Event Time Delta")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    if max_delta - min_delta <= 30:
        ax.set_xticks(list(range(min_delta, max_delta + 1)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rate_line(
    path: Path,
    rates: List[NamedTuple],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of event rate per second."""
    if not rates:
        logger.warning("No event-rate data; skip per-second event-rate plot")
        return

    seconds = [int(row.event_second) for row in rates]
    event_rates = [float(row.event_rate_hz) for row in rates]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(seconds, event_rates, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("Event second (GPS)")
    ax.set_ylabel("Event rate (Hz)")
    ax.set_title("Event Rate Per Second")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, seconds, second_datetime_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_du_trigger_rate(path: Path, du_rates: List[NamedTuple]) -> None:
    """Plot bar chart of trigger rate for each DU."""
    if not du_rates:
        logger.warning("No DU trigger-rate data; skip DU trigger-rate plot")
        return

    du_ids = [str(row.du_id) for row in du_rates]
    trigger_rates = [float(row.trigger_rate_hz) for row in du_rates]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(du_ids, trigger_rates, color="tab:blue", alpha=0.85)
    ax.set_xlabel("DU ID")
    ax.set_ylabel("Trigger rate (Hz)")
    ax.set_title("Trigger Rate per DU")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_total_du_trigger_per_second(
    path: Path,
    rows: List[NamedTuple],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of total DU trigger count per second."""
    if not rows:
        logger.warning(
            "No per-second total DU trigger data; skip total DU trend plot"
        )
        return

    seconds = [int(row.event_second) for row in rows]
    total_counts = [float(row.total_du_trigger_count) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(seconds, total_counts, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("Event second (GPS)")
    ax.set_ylabel("Total DU trigger count per second")
    ax.set_title("Total DU Trigger Count Per Second")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, seconds, second_datetime_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_du_rate_per_second_heatmap(
    path: Path,
    per_second_du: Dict[int, Counter[str]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
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

    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Trigger rate (Hz)")

    ax.set_xlabel("Event second (index / GPS+datetime)")
    ax.set_ylabel("DU ID")
    ax.set_title("Per-second Trigger Rate per DU")

    if len(du_ids) <= 40:
        ax.set_yticks(range(len(du_ids)))
        ax.set_yticklabels(du_ids)

    if second_datetime_map is not None and len(seconds) > 0:
        tick_count = min(8, len(seconds))
        if tick_count == 1:
            tick_positions = [0]
        else:
            tick_positions = np.linspace(0, len(seconds) - 1, tick_count, dtype=int).tolist()
        tick_labels: List[str] = []
        for pos in tick_positions:
            second = seconds[pos]
            datetime_str = get_second_datetime(second, second_datetime_map)
            if datetime_str:
                tick_labels.append(f"{second}\n{datetime_str}")
            else:
                tick_labels.append(str(second))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_avg_event_rate(
    path: Path,
    rows: List[NamedTuple],
    window_seconds: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of window-averaged event rate."""
    if not rows:
        logger.warning("No window-averaged event-rate data; skip plotting")
        return

    x_values = [int(row.window_start_second) for row in rows]
    y_values = [float(row.avg_event_rate_hz) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, marker="o", linewidth=1)
    ax.set_xlabel("Window start second (GPS)")
    ax.set_ylabel("Average event rate (Hz)")
    ax.set_title(f"Average Event Rate (window={window_seconds}s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, x_values, second_datetime_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_avg_total_du_trigger_rate(
    path: Path,
    rows: List[NamedTuple],
    window_seconds: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of window-averaged total DU trigger rate."""
    if not rows:
        logger.warning("No window-averaged total-DU data; skip plotting")
        return

    x_values = [int(row.window_start_second) for row in rows]
    y_values = [float(row.avg_total_du_trigger_rate_hz) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_values, y_values, marker="o", linewidth=1)
    ax.set_xlabel("Window start second (GPS)")
    ax.set_ylabel("Average total DU trigger rate (Hz)")
    ax.set_title(f"Average Total DU Trigger Rate (window={window_seconds}s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, x_values, second_datetime_map)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_avg_du_rate_heatmap(
    path: Path,
    rows: List[NamedTuple],
    window_seconds: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot heatmap of window-averaged trigger rate for each DU."""
    if not rows:
        logger.warning("No window-averaged DU trigger-rate data; skip plotting")
        return

    windows = sorted(
        {
            (
                int(row.window_start_second),
                int(row.window_end_second_exclusive),
            )
            for row in rows
        }
    )
    du_ids = sorted(
        {str(row.du_id) for row in rows},
        key=lambda item: int(item) if item.isdigit() else item,
    )

    window_index = {window: idx for idx, window in enumerate(windows)}
    du_index = {du_id: idx for idx, du_id in enumerate(du_ids)}

    matrix = np.zeros((len(du_ids), len(windows)), dtype=float)
    for row in rows:
        win_key = (
            int(row.window_start_second),
            int(row.window_end_second_exclusive),
        )
        du_id = str(row.du_id)
        matrix[du_index[du_id], window_index[win_key]] = float(
            row.avg_trigger_rate_hz
        )

    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average trigger rate (Hz)")

    ax.set_xlabel(f"Window index ({window_seconds}s)")
    ax.set_ylabel("DU ID")
    ax.set_title(f"Average Trigger Rate per DU (window={window_seconds}s)")

    if len(du_ids) <= 40:
        ax.set_yticks(range(len(du_ids)))
        ax.set_yticklabels(du_ids)

    if second_datetime_map is not None and len(windows) > 0:
        tick_count = min(8, len(windows))
        if tick_count == 1:
            tick_positions = [0]
        else:
            tick_positions = np.linspace(0, len(windows) - 1, tick_count, dtype=int).tolist()
        tick_labels: List[str] = []
        for pos in tick_positions:
            start_second = windows[pos][0]
            datetime_str = get_second_datetime(start_second, second_datetime_map)
            if datetime_str:
                tick_labels.append(f"{start_second}\n{datetime_str}")
            else:
                tick_labels.append(str(start_second))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_avg_du_rate_topn_lines(
    path: Path,
    rows: List[NamedTuple],
    window_seconds: int,
    top_n: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
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
        du_total_counts[str(row.du_id)] += int(row.trigger_count)

    top_du_ids = [du_id for du_id, _ in du_total_counts.most_common(top_n)]
    if not top_du_ids:
        logger.warning("No Top-N DU selected; skip line plot")
        return

    window_starts = sorted({int(row.window_start_second) for row in rows})
    rate_map: Dict[str, Dict[int, float]] = {du_id: {} for du_id in top_du_ids}
    for row in rows:
        window_start = int(row.window_start_second)
        du_id = str(row.du_id)
        avg_rate = float(row.avg_trigger_rate_hz)
        if du_id in rate_map:
            rate_map[du_id][window_start] = avg_rate

    fig, ax = plt.subplots(figsize=(14, 6))
    for du_id in top_du_ids:
        y_values = [rate_map[du_id].get(window_start, 0.0) for window_start in window_starts]
        ax.plot(window_starts, y_values, marker="o", linewidth=1, label=f"DU {du_id}")

    ax.set_xlabel("Window start second (GPS)")
    ax.set_ylabel("Average trigger rate (Hz)")
    ax.set_title(f"Top-{len(top_du_ids)} DU Average Trigger Rate (window={window_seconds}s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    apply_gps_datetime_ticks(ax, window_starts, second_datetime_map)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_du_rate_topn_lines(

    path: Path,
    rows: List[NamedTuple],
    top_n: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """
    Plot trigger-rate bar chart for Top-N active DUs.
    """
    if not rows:
        logger.warning("No DU trigger-rate data; skip Top-N line plot")
        return
    if top_n <= 0:
        logger.warning("Top-N <= 0; skip Top-N line plot")
        return

    du_total_counts: Counter[str] = Counter()
    for row in rows:
        du_total_counts[str(row.du_id)] += int(row.trigger_count)

    top_du_ids = [du_id for du_id, _ in du_total_counts.most_common(top_n)]
    if not top_du_ids:
        logger.warning("No Top-N DU selected; skip line plot")
        return

    # 构建 Top-N DU 的 trigger_rate_hz
    du_rates = {str(row.du_id): float(row.trigger_rate_hz) for row in rows}
    du_ids = top_du_ids
    trigger_rates = [du_rates.get(du_id, 0.0) for du_id in du_ids]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(du_ids, trigger_rates, color="tab:blue", alpha=0.85)
    ax.set_xlabel("DU ID")
    ax.set_ylabel("Trigger rate (Hz)")
    ax.set_title(f"Trigger Rate per DU")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



def build_second_level_topn_rows(
    rows: List[NamedTuple],
) -> List[NamedTuple]:
    """Convert per-second DU rows to 1-second window rows for Top-N plotting."""
    converted_rows: List[NamedTuple] = []
    for row in rows:
        converted_rows.append(
            make_named_row(
                DU_RATE_PER_SECOND_FIELDS,
                event_second=int(row.event_second),
                du_id=str(row.du_id),
                trigger_count=int(row.trigger_count),
                trigger_rate_hz=float(row.trigger_rate_hz),
            )
        )
    return converted_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute DU counts and per-second event rates from Trigger YAML"
    )
    parser.add_argument("yaml_file", help="Input YAML file path")
    parser.add_argument(
        "--avg-window",
        type=int,
        default=0,
        help="Averaging window length in seconds; output window averages when >0 (e.g., 60 means 60-second average)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of most active DUs shown in Top-N line plots (default: 10, set to 0 to disable)",
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Export CSV only, do not generate plots",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing CSV cache and recompute from YAML",
    )
    return parser.parse_args()


def main() -> int:
    """Main workflow."""
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

    logger.info(f"Start stats workflow for YAML: {yaml_path}")
    output_paths = build_output_paths(yaml_path)
    event_out = output_paths["event_csv"]
    event_meta_out = event_cache_meta_path(event_out)

    use_csv_cache = event_out.exists() and not args.force_recompute

    records: List[NamedTuple] = []
    rates: List[NamedTuple] = []
    adjacent_time_deltas: List[int] = []
    adjacent_time_delta_rows: List[NamedTuple] = []
    du_rates: List[NamedTuple] = []
    per_second_du: Dict[int, Counter[str]] = {}
    du_second_rows: List[NamedTuple] = []
    total_du_rows: List[NamedTuple] = []
    data_for_aggregates: Dict[str, Dict[str, Any]] = {}
    event_du_ids_map: Dict[str, List[str]] = {}
    expected_event_meta = build_event_cache_meta(
        yaml_path,
        start_datetime,
        end_datetime,
    )

    if use_csv_cache:
        logger.info("Found event CSV cache, load directly without YAML recompute")
        cached_records: Optional[List[NamedTuple]] = None
        cached_event_du_ids_map: Dict[str, List[str]] = {}

        if not event_meta_out.exists():
            logger.warning(
                "Event CSV cache meta missing for {}, fallback to recompute",
                event_out,
            )
        else:
            try:
                cached_meta = read_event_cache_meta(event_meta_out)
                is_valid, reason = event_cache_meta_is_valid(
                    cached_meta,
                    expected_event_meta,
                )
                if not is_valid:
                    logger.warning(
                        "Event CSV cache meta mismatch for {}: {}, "
                        "fallback to recompute",
                        event_out,
                        reason,
                    )
                else:
                    cached_records = read_event_csv(event_out)
                    cached_event_du_ids_map = read_event_csv_du_ids(event_out)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Invalid event CSV cache or meta: {} / {} ({}). "
                    "Fallback to recompute",
                    event_out,
                    event_meta_out,
                    exc,
                )

        if cached_records is None:
            use_csv_cache = False
            logger.info("Skip invalid event CSV cache and continue recompute path")
        else:
            records = cached_records
            event_du_ids_map = cached_event_du_ids_map
            data_for_aggregates = build_data_from_cached_events(
                records,
                event_du_ids_map,
            )

    if not use_csv_cache:
        if args.force_recompute and event_out.exists():
            logger.info("Force recompute enabled, ignore existing event CSV cache")

        if not yaml_path.exists():
            logger.error(f"Input file does not exist: {yaml_path}")
            return 2

        with yaml_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        if not isinstance(data, dict):
            logger.error(
                "YAML top-level structure is not a dict; cannot parse by event key"
            )
            return 2

        logger.info(f"Loaded YAML entries: {len(data)}")

        data = filter_data_by_datetime_range(data, start_datetime, end_datetime)
        event_du_ids_map = build_event_du_ids_map(data)
        data_for_aggregates = data

        records = parse_event_records(data)

    (
        second_datetime_map,
        rates,
        adjacent_time_deltas,
        adjacent_time_delta_rows,
        du_rates,
        per_second_du,
        du_second_rows,
    ) = compute_core_statistics(data_for_aggregates, records)
    total_du_rows = build_total_du_trigger_per_second_rows(per_second_du)

    avg_event_rows: List[NamedTuple] = []
    avg_total_du_rows: List[NamedTuple] = []
    avg_du_rows: List[NamedTuple] = []

    effective_avg_du_window = args.avg_window
    logger.info(
        "Window settings: avg_window={}, effective_avg_du_window={}",
        args.avg_window,
        effective_avg_du_window,
    )

    if args.avg_window > 0:
        avg_event_rows, _ = build_window_averages(
            records,
            per_second_du,
            args.avg_window,
        )
        avg_total_du_rows = build_avg_total_du_trigger_rows(
            total_du_rows,
            args.avg_window,
        )

    if effective_avg_du_window > 0:
        _, avg_du_rows = build_window_averages(
            records,
            per_second_du,
            effective_avg_du_window,
        )

    if not use_csv_cache:
        write_event_csv(event_out, records, event_du_ids_map)
        write_event_cache_meta(event_meta_out, expected_event_meta)

    if not args.no_plot:
        logger.info("Plotting is enabled; generating figures")
        plot_du_count_histogram(output_paths["du_hist_png"], records)
        plot_event_du_count_over_time(
            output_paths["event_du_time_png"],
            records,
            second_datetime_map,
        )
        plot_rate_line(output_paths["rate_png"], rates, second_datetime_map)
        plot_du_trigger_rate(output_paths["du_rate_png"], du_rates)
        plot_total_du_trigger_per_second(
            output_paths["total_du_rate_png"],
            total_du_rows,
            second_datetime_map,
        )
        plot_adjacent_time_delta_histogram(
            output_paths["delta_hist_png"],
            adjacent_time_deltas,
        )
        plot_du_rate_per_second_heatmap(
            output_paths["du_second_heatmap_png"],
            per_second_du,
            second_datetime_map,
        )
        if args.topn > 0:
            du_topn_rows = build_second_level_topn_rows(du_second_rows)
            plot_du_rate_topn_lines(
                output_paths["du_topn_png"],
                du_topn_rows,
                args.topn,
                second_datetime_map,
            )
        if args.avg_window > 0:
            plot_avg_event_rate(
                output_paths["avg_event_png"],
                avg_event_rows,
                args.avg_window,
                second_datetime_map,
            )
            plot_avg_total_du_trigger_rate(
                output_paths["avg_total_du_png"],
                avg_total_du_rows,
                args.avg_window,
                second_datetime_map,
            )
        if effective_avg_du_window > 0:
            plot_avg_du_rate_heatmap(
                output_paths["avg_du_heatmap_png"],
                avg_du_rows,
                effective_avg_du_window,
                second_datetime_map,
            )
            if args.topn > 0:
                plot_avg_du_rate_topn_lines(
                    output_paths["avg_du_topn_png"],
                    avg_du_rows,
                    effective_avg_du_window,
                    args.topn,
                    second_datetime_map,
                )
    else:
        logger.info("Plotting is disabled by --no-plot")

    logger.info(f"Total events: {len(records)}")
    logger.info(f"Per-event DU-count stats written to: {event_out}")
    logger.info("Derived statistics are computed in-memory from the event CSV cache")
    if not args.no_plot:
        logger.info(f"DU-count histogram written to: {output_paths['du_hist_png']}")
        logger.info(
            "Event DU-count time-series plot written to: "
            f"{output_paths['event_du_time_png']}"
        )
        logger.info(
            "Per-second event-rate line plot written to: "
            f"{output_paths['rate_png']}"
        )
        logger.info(
            "Per-DU trigger-rate bar plot written to: "
            f"{output_paths['du_rate_png']}"
        )
        logger.info(
            "Per-second total DU trigger-count line plot written to: "
            f"{output_paths['total_du_rate_png']}"
        )
        logger.info(
            "Adjacent time-delta histogram written to: "
            f"{output_paths['delta_hist_png']}"
        )
        logger.info(
            "Per-second per-DU trigger-rate heatmap written to: "
            f"{output_paths['du_second_heatmap_png']}"
        )
        if args.topn > 0:
            logger.info(
                "Top-N per-second per-DU trigger-rate line plot "
                f"written to: {output_paths['du_topn_png']}"
            )
        if args.avg_window > 0:
            logger.info(
                "Window-averaged event-rate line plot written to: "
                f"{output_paths['avg_event_png']}"
            )
            logger.info(
                "Window-averaged total-DU trigger-rate line plot written to: "
                f"{output_paths['avg_total_du_png']}"
            )
        if effective_avg_du_window > 0:
            logger.info(
                "Window-averaged per-DU trigger-rate heatmap written to: "
                f"{output_paths['avg_du_heatmap_png']}"
            )
            if args.topn > 0:
                logger.info(
                    "Top-N window-averaged per-DU trigger-rate line plot "
                    f"written to: {output_paths['avg_du_topn_png']}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
