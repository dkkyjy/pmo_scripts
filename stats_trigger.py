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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import yaml

from logger_config import logger

plt.style.use(["science", "grid", "notebook"])


SecondDateTimeMap = Dict[int, Tuple[str, str, str]]
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATETIME_HELP_FORMAT = DATETIME_FORMAT.replace("%", "%%")


def format_event_datetime(date_str: str, time_str: str) -> str:
    """Format event date/time into a human-readable datetime string."""
    if not date_str or not time_str:
        return ""

    try:
        dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        return dt_obj.strftime(DATETIME_FORMAT)
    except ValueError:
        return f"{date_str} {time_str}".strip()


def build_second_datetime_map(
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> SecondDateTimeMap:
    """Build mapping: gps second -> (date, time, datetime string)."""
    second_map: SecondDateTimeMap = {}
    for _, second, _, _, _, date_str, time_str in records:
        if second in second_map:
            continue
        second_map[second] = (
            date_str,
            time_str,
            format_event_datetime(date_str, time_str),
        )
    logger.debug(f"Built second-datetime map with {len(second_map)} unique seconds")
    return second_map


def get_second_datetime_tuple(
    second: int,
    second_datetime_map: Optional[SecondDateTimeMap],
) -> Tuple[str, str, str]:
    """Get datetime tuple for a gps second from mapping."""
    if second_datetime_map is None:
        return "", "", ""
    return second_datetime_map.get(second, ("", "", ""))


def apply_gps_datetime_ticks(
    ax: plt.Axes,
    seconds: List[int],
    second_datetime_map: Optional[SecondDateTimeMap],
    max_ticks: int = 8,
) -> None:
    """Apply compact x ticks showing both GPS second and date/time."""
    if second_datetime_map is None or not seconds:
        return

    unique_seconds = sorted(set(seconds))
    if not unique_seconds:
        return

    tick_count = min(max_ticks, len(unique_seconds))
    if tick_count <= 0:
        return

    if tick_count == 1:
        tick_positions = [unique_seconds[0]]
    else:
        indices = np.linspace(0, len(unique_seconds) - 1, tick_count, dtype=int)
        tick_positions = [unique_seconds[index] for index in indices]

    tick_labels: List[str] = []
    for second in tick_positions:
        _, _, datetime_str = get_second_datetime_tuple(second, second_datetime_map)
        if datetime_str:
            tick_labels.append(f"{second}\n{datetime_str}")
        else:
            tick_labels.append(str(second))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")

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
    time_value = payload.get("time", "")
    if isinstance(time_value, dict):
        time_value = ""

    date_str = str(date_value) if date_value is not None else ""
    time_str = str(time_value) if time_value is not None else ""
    if time_str.isdigit() and len(time_str) < 6:
        time_str = time_str.zfill(6)

    return date_str, time_str


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime argument as ``YYYY-MM-DDTHH:MM:SS``."""
    try:
        return datetime.strptime(value, DATETIME_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid datetime format: "
            f"{value!r}. Expected format: {DATETIME_FORMAT}"
        ) from exc


def parse_payload_datetime(payload: Dict[str, Any]) -> Optional[datetime]:
    """Parse event datetime from payload ``date`` and ``time`` fields."""
    date_str, time_str = parse_event_date_time(payload)
    if not date_str or not time_str:
        return None

    try:
        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def in_datetime_range(
    event_dt: datetime,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> bool:
    """Check whether event datetime is inside an inclusive range."""
    if start_dt is not None and event_dt < start_dt:
        return False
    if end_dt is not None and event_dt > end_dt:
        return False
    return True


def filter_data_by_datetime_range(
    data: Dict[str, Dict[str, Any]],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[str, Dict[str, Any]]:
    """Filter YAML event payloads by datetime range from ``date``/``time``."""
    if start_dt is None and end_dt is None:
        return data

    filtered_data: Dict[str, Dict[str, Any]] = {}
    invalid_datetime_count = 0

    for event_key, payload in data.items():
        event_dt = parse_payload_datetime(payload)
        if event_dt is None:
            invalid_datetime_count += 1
            continue

        if in_datetime_range(event_dt, start_dt, end_dt):
            filtered_data[event_key] = payload

    logger.info(
        "Datetime range filter on YAML events: {} -> {}",
        len(data),
        len(filtered_data),
    )
    if invalid_datetime_count > 0:
        logger.warning(
            "Skipped {} events with invalid/missing date/time while filtering",
            invalid_datetime_count,
        )
    return filtered_data


def get_du_ns_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get DU nanosecond map from ``payload['du_ns']``."""
    du_ns = payload.get("du_ns")
    if isinstance(du_ns, dict):
        return du_ns

    return {}


def get_event_du_ids(payload: Dict[str, Any]) -> List[str]:
    """Get event DU IDs, preferring ``du_id`` and falling back to ``du_ns``."""
    du_ids = payload.get("du_id", [])
    if isinstance(du_ids, list):
        return [str(du_id) for du_id in du_ids]

    return [str(du_id) for du_id in get_du_ns_map(payload).keys()]


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
        du_count = len(get_event_du_ids(payload))

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
    if records:
        logger.info(
            "Parsed event records: total={}, second_range=[{}, {}]",
            len(records),
            records[0][1],
            records[-1][1],
        )
    else:
        logger.info("Parsed event records: total=0")
    return records


def build_rate_per_second(
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> List[Tuple[int, int, float]]:
    """Compute event rates per second (Hz)."""
    second_counter = Counter(record[1] for record in records)
    rates = [(sec, count, float(count)) for sec, count in sorted(second_counter.items())]
    logger.debug(f"Built per-second event rates for {len(rates)} seconds")
    return rates


def build_adjacent_time_deltas(
    records: List[Tuple[str, int, int, int, int, str, str]],
) -> List[int]:
    """Compute second-level deltas between adjacent events."""
    if len(records) < 2:
        return []

    deltas: List[int] = []
    for index in range(1, len(records)):
        delta_second = records[index][1] - records[index - 1][1]
        deltas.append(delta_second)

    logger.debug(f"Built adjacent event time deltas: {len(deltas)}")
    return deltas


def build_adjacent_time_delta_distribution(
    deltas: List[int],
) -> List[Tuple[int, int]]:
    """Build distribution rows as (delta_second, event_pair_count)."""
    counter = Counter(deltas)
    rows = [(delta_second, count) for delta_second, count in sorted(counter.items())]
    logger.debug(f"Built adjacent time-delta distribution rows: {len(rows)}")
    return rows


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
        for du_id in get_event_du_ids(payload):
            du_counter[du_id] += 1

    rows: List[Tuple[str, int, float]] = []
    for du_id, count in du_counter.items():
        rows.append((du_id, count, count / float(observation_seconds)))

    rows.sort(key=lambda row: int(row[0]) if row[0].isdigit() else row[0])
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
    logger.debug(f"Expanded per-second DU rate rows: {len(rows)}")
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
    records: List[Tuple[str, int, int, int, int, str, str]],
    event_du_ids_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Rebuild minimal payloads from cache for DU-based aggregations."""
    cached_data: Dict[str, Dict[str, Any]] = {}
    for _, event_second, _, event_number, *_ in records:
        event_number_key = str(event_number)
        cached_data[event_number_key] = {
            "gps_time": event_second,
            "du_id": event_du_ids_map.get(event_number_key, []),
        }
    return cached_data


def write_event_csv(
    path: Path,
    records: List[Tuple[str, int, int, int, int, str, str]],
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
                "event_date",
                "event_time",
                "du_count",
                "du_ids",
            ]
        )
        rows = [
            (
                record[3],
                record[1],
                record[5],
                record[6],
                record[2],
                serialize_du_ids(
                    event_du_ids_map.get(str(record[3]), event_du_ids_map.get(record[0], []))
                ),
            )
            for record in records
        ]
        writer.writerows(rows)
    logger.debug(f"Wrote event CSV rows: {len(records)} -> {path}")


def write_rate_csv(
    path: Path,
    rates: List[Tuple[int, int, float]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Write per-second event-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "event_second",
                "event_date",
                "event_time",
                "event_datetime",
                "event_count",
                "event_rate_hz",
            ]
        )
        rows: List[Tuple[int, str, str, str, int, float]] = []
        for second, count, rate in rates:
            date_str, time_str, datetime_str = get_second_datetime_tuple(
                second,
                second_datetime_map,
            )
            rows.append((second, date_str, time_str, datetime_str, count, rate))
        writer.writerows(rows)
    logger.debug(f"Wrote rate CSV rows: {len(rates)} -> {path}")


def write_du_rate_csv(path: Path, du_rates: List[Tuple[str, int, float]]) -> None:
    """Write per-DU trigger count/rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["du_id", "trigger_count", "trigger_rate_hz"])
        writer.writerows(du_rates)
    logger.debug(f"Wrote DU-rate CSV rows: {len(du_rates)} -> {path}")


def write_adjacent_time_delta_csv(
    path: Path,
    rows: List[Tuple[int, int]],
) -> None:
    """Write adjacent event time-delta distribution CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["delta_second", "event_pair_count"])
        writer.writerows(rows)
    logger.debug(f"Wrote adjacent time-delta CSV rows: {len(rows)} -> {path}")


def write_du_per_second_csv(
    path: Path,
    rows: List[Tuple[int, str, int, float]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Write per-second per-DU trigger-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "event_second",
                "event_date",
                "event_time",
                "event_datetime",
                "du_id",
                "trigger_count",
                "trigger_rate_hz",
            ]
        )
        output_rows: List[Tuple[int, str, str, str, str, int, float]] = []
        for second, du_id, count, rate in rows:
            date_str, time_str, datetime_str = get_second_datetime_tuple(
                second,
                second_datetime_map,
            )
            output_rows.append(
                (second, date_str, time_str, datetime_str, du_id, count, rate)
            )
        writer.writerows(output_rows)
    logger.debug(f"Wrote per-second DU CSV rows: {len(rows)} -> {path}")


def write_avg_event_csv(
    path: Path,
    rows: List[Tuple[int, int, int, float]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Write window-averaged event-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "window_start_second",
                "window_end_second_exclusive",
                "window_start_datetime",
                "window_end_datetime",
                "event_count",
                "avg_event_rate_hz",
            ]
        )
        output_rows: List[Tuple[int, int, str, str, int, float]] = []
        for start_second, end_second, event_count, avg_rate in rows:
            _, _, start_datetime = get_second_datetime_tuple(
                start_second,
                second_datetime_map,
            )
            _, _, end_datetime = get_second_datetime_tuple(
                end_second - 1,
                second_datetime_map,
            )
            output_rows.append(
                (
                    start_second,
                    end_second,
                    start_datetime,
                    end_datetime,
                    event_count,
                    avg_rate,
                )
            )
        writer.writerows(output_rows)
    logger.debug(f"Wrote avg-event CSV rows: {len(rows)} -> {path}")


def write_avg_du_csv(
    path: Path,
    rows: List[Tuple[int, int, str, int, float]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Write window-averaged per-DU trigger-rate CSV."""
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "window_start_second",
                "window_end_second_exclusive",
                "window_start_datetime",
                "window_end_datetime",
                "du_id",
                "trigger_count",
                "avg_trigger_rate_hz",
            ]
        )
        output_rows: List[Tuple[int, int, str, str, str, int, float]] = []
        for start_second, end_second, du_id, trigger_count, avg_rate in rows:
            _, _, start_datetime = get_second_datetime_tuple(
                start_second,
                second_datetime_map,
            )
            _, _, end_datetime = get_second_datetime_tuple(
                end_second - 1,
                second_datetime_map,
            )
            output_rows.append(
                (
                    start_second,
                    end_second,
                    start_datetime,
                    end_datetime,
                    du_id,
                    trigger_count,
                    avg_rate,
                )
            )
        writer.writerows(output_rows)
    logger.debug(f"Wrote avg-DU CSV rows: {len(rows)} -> {path}")


def read_event_csv(
    path: Path,
) -> List[Tuple[str, int, int, int, int, str, str]]:
    """Read per-event DU-count CSV."""
    rows: List[Tuple[str, int, int, int, int, str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row_index, row in enumerate(reader):
            event_number = int(row["event_number"])
            rows.append(
                (
                    str(event_number),
                    int(row["event_second"]),
                    int(row["du_count"]),
                    event_number,
                    row_index,
                    str(row.get("event_date", "")),
                    str(row.get("event_time", "")),
                )
            )
    return rows


def read_event_csv_du_ids(path: Path) -> Dict[str, List[str]]:
    """Read event_number -> DU IDs mapping from event CSV cache."""
    event_du_ids_map: Dict[str, List[str]] = {}
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            event_number_key = str(row["event_number"])
            event_du_ids_map[event_number_key] = deserialize_du_ids(
                str(row.get("du_ids", ""))
            )
    return event_du_ids_map


def read_rate_csv(path: Path) -> List[Tuple[int, int, float]]:
    """Read per-second event-rate CSV."""
    rows: List[Tuple[int, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    int(row["event_second"]),
                    int(row["event_count"]),
                    float(row["event_rate_hz"]),
                )
            )
    return rows


def read_du_rate_csv(path: Path) -> List[Tuple[str, int, float]]:
    """Read per-DU trigger-rate CSV."""
    rows: List[Tuple[str, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    str(row["du_id"]),
                    int(row["trigger_count"]),
                    float(row["trigger_rate_hz"]),
                )
            )
    return rows


def read_adjacent_time_delta_csv(path: Path) -> List[Tuple[int, int]]:
    """Read adjacent event time-delta distribution CSV."""
    rows: List[Tuple[int, int]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append((int(row["delta_second"]), int(row["event_pair_count"])))
    return rows


def read_du_per_second_csv(path: Path) -> List[Tuple[int, str, int, float]]:
    """Read per-second per-DU trigger-rate CSV."""
    rows: List[Tuple[int, str, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    int(row["event_second"]),
                    str(row["du_id"]),
                    int(row["trigger_count"]),
                    float(row["trigger_rate_hz"]),
                )
            )
    return rows


def read_avg_event_csv(path: Path) -> List[Tuple[int, int, int, float]]:
    """Read window-averaged event-rate CSV."""
    rows: List[Tuple[int, int, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    int(row["window_start_second"]),
                    int(row["window_end_second_exclusive"]),
                    int(row["event_count"]),
                    float(row["avg_event_rate_hz"]),
                )
            )
    return rows


def read_avg_du_csv(path: Path) -> List[Tuple[int, int, str, int, float]]:
    """Read window-averaged per-DU trigger-rate CSV."""
    rows: List[Tuple[int, int, str, int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    int(row["window_start_second"]),
                    int(row["window_end_second_exclusive"]),
                    str(row["du_id"]),
                    int(row["trigger_count"]),
                    float(row["avg_trigger_rate_hz"]),
                )
            )
    return rows


def expand_adjacent_time_deltas_from_distribution(
    rows: List[Tuple[int, int]],
) -> List[int]:
    """Expand (delta_second, event_pair_count) rows back to delta samples."""
    deltas: List[int] = []
    for delta_second, pair_count in rows:
        deltas.extend([delta_second] * max(0, pair_count))
    return deltas


def build_per_second_du_from_rows(
    rows: List[Tuple[int, str, int, float]],
) -> Dict[int, Counter[str]]:
    """Rebuild per-second DU counters from flattened CSV rows."""
    per_second_du: Dict[int, Counter[str]] = {}
    for second, du_id, trigger_count, _ in rows:
        if second not in per_second_du:
            per_second_du[second] = Counter()
        per_second_du[second][du_id] = trigger_count
    return per_second_du


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
    records: List[Tuple[str, int, int, int, int, str, str]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot DU count per event over time."""
    if not records:
        logger.warning("No event data; skip event DU-count time-series plot")
        return

    seconds = [record[1] for record in records]
    du_counts = [record[2] for record in records]

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
    rates: List[Tuple[int, int, float]],
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of event rate per second."""
    if not rates:
        logger.warning("No event-rate data; skip per-second event-rate plot")
        return

    seconds = [row[0] for row in rates]
    event_rates = [row[2] for row in rates]

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
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=75)
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
            _, _, datetime_str = get_second_datetime_tuple(second, second_datetime_map)
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
    rows: List[Tuple[int, int, int, float]],
    window_seconds: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot line chart of window-averaged event rate."""
    if not rows:
        logger.warning("No window-averaged event-rate data; skip plotting")
        return

    x_values = [row[0] for row in rows]
    y_values = [row[3] for row in rows]

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


def plot_avg_du_rate_heatmap(
    path: Path,
    rows: List[Tuple[int, int, str, int, float]],
    window_seconds: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
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
            _, _, datetime_str = get_second_datetime_tuple(start_second, second_datetime_map)
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
    rows: List[Tuple[int, int, str, int, float]],
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
    rows: List[Tuple[int, int, str, int, float]],
    window_seconds: int,
    top_n: int,
    second_datetime_map: Optional[SecondDateTimeMap] = None,
) -> None:
    """Plot trigger-rate lines for Top-N active DUs."""
    plot_avg_du_rate_topn_lines(
        path,
        rows,
        window_seconds,
        top_n,
        second_datetime_map,
    )


def build_second_level_topn_rows(
    rows: List[Tuple[int, str, int, float]],
) -> List[Tuple[int, int, str, int, float]]:
    """Convert per-second DU rows to 1-second window rows for Top-N plotting."""
    converted_rows: List[Tuple[int, int, str, int, float]] = []
    for second, du_id, trigger_count, rate in rows:
        converted_rows.append((second, second + 1, du_id, trigger_count, rate))
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

    event_out = yaml_path.with_name(f"{yaml_path.stem}_event_du_count.csv")
    du_hist_out = yaml_path.with_name(f"{yaml_path.stem}_du_count_hist.png")
    event_du_count_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_event_du_count_over_time.png"
    )
    rate_plot_out = yaml_path.with_name(f"{yaml_path.stem}_rate_per_second.png")
    du_rate_plot_out = yaml_path.with_name(f"{yaml_path.stem}_du_trigger_rate.png")
    time_delta_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_adjacent_time_delta_hist.png"
    )
    du_second_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_du_rate_per_second_heatmap.png"
    )
    avg_event_plot_out = yaml_path.with_name(f"{yaml_path.stem}_avg_event_rate.png")
    avg_du_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_avg_du_trigger_rate_heatmap.png"
    )
    du_topn_plot_out = yaml_path.with_name(f"{yaml_path.stem}_du_trigger_rate_topn.png")
    avg_du_topn_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_avg_du_trigger_rate_topn.png"
    )

    use_csv_cache = event_out.exists() and not args.force_recompute
    if start_datetime is not None or end_datetime is not None:
        use_csv_cache = False

    records: List[Tuple[str, int, int, int, int, str, str]] = []
    rates: List[Tuple[int, int, float]] = []
    adjacent_time_deltas: List[int] = []
    adjacent_time_delta_rows: List[Tuple[int, int]] = []
    du_rates: List[Tuple[str, int, float]] = []
    per_second_du: Dict[int, Counter[str]] = {}
    du_second_rows: List[Tuple[int, str, int, float]] = []
    data_for_aggregates: Dict[str, Dict[str, Any]] = {}
    event_du_ids_map: Dict[str, List[str]] = {}

    if use_csv_cache:
        logger.info("Found event CSV cache, load directly without YAML recompute")
        records = read_event_csv(event_out)
        event_du_ids_map = read_event_csv_du_ids(event_out)
        data_for_aggregates = build_data_from_cached_events(records, event_du_ids_map)
        second_datetime_map = build_second_datetime_map(records)
        rates = build_rate_per_second(records)
        adjacent_time_deltas = build_adjacent_time_deltas(records)
        adjacent_time_delta_rows = build_adjacent_time_delta_distribution(
            adjacent_time_deltas
        )
        du_rates = build_du_trigger_rate(data_for_aggregates, records)
        per_second_du = build_du_counts_per_second(data_for_aggregates)
        du_second_rows = build_du_rate_per_second_rows(per_second_du)
    else:
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
        second_datetime_map = build_second_datetime_map(records)
        rates = build_rate_per_second(records)
        adjacent_time_deltas = build_adjacent_time_deltas(records)
        adjacent_time_delta_rows = build_adjacent_time_delta_distribution(
            adjacent_time_deltas
        )
        du_rates = build_du_trigger_rate(data_for_aggregates, records)
        per_second_du = build_du_counts_per_second(data_for_aggregates)
        du_second_rows = build_du_rate_per_second_rows(per_second_du)

    avg_event_rows: List[Tuple[int, int, int, float]] = []
    avg_du_rows: List[Tuple[int, int, str, int, float]] = []

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

    if effective_avg_du_window > 0:
        _, avg_du_rows = build_window_averages(
            records,
            per_second_du,
            effective_avg_du_window,
        )

    if not use_csv_cache:
        write_event_csv(event_out, records, event_du_ids_map)

    if not args.no_plot:
        logger.info("Plotting is enabled; generating figures")
        plot_du_count_histogram(du_hist_out, records)
        plot_event_du_count_over_time(event_du_count_plot_out, records, second_datetime_map)
        plot_rate_line(rate_plot_out, rates, second_datetime_map)
        plot_du_trigger_rate(du_rate_plot_out, du_rates)
        plot_adjacent_time_delta_histogram(time_delta_plot_out, adjacent_time_deltas)
        plot_du_rate_per_second_heatmap(du_second_plot_out, per_second_du, second_datetime_map)
        if args.topn > 0:
            du_topn_rows = build_second_level_topn_rows(du_second_rows)
            plot_du_rate_topn_lines(
                du_topn_plot_out,
                du_topn_rows,
                1,
                args.topn,
                second_datetime_map,
            )
        if args.avg_window > 0:
            plot_avg_event_rate(
                avg_event_plot_out,
                avg_event_rows,
                args.avg_window,
                second_datetime_map,
            )
        if effective_avg_du_window > 0:
            plot_avg_du_rate_heatmap(
                avg_du_plot_out,
                avg_du_rows,
                effective_avg_du_window,
                second_datetime_map,
            )
            if args.topn > 0:
                plot_avg_du_rate_topn_lines(
                    avg_du_topn_plot_out,
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
        logger.info(f"DU-count histogram written to: {du_hist_out}")
        logger.info(f"Event DU-count time-series plot written to: {event_du_count_plot_out}")
        logger.info(f"Per-second event-rate line plot written to: {rate_plot_out}")
        logger.info(f"Per-DU trigger-rate bar plot written to: {du_rate_plot_out}")
        logger.info(f"Adjacent time-delta histogram written to: {time_delta_plot_out}")
        logger.info(f"Per-second per-DU trigger-rate heatmap written to: {du_second_plot_out}")
        if args.topn > 0:
            logger.info(
                "Top-N per-second per-DU trigger-rate line plot "
                f"written to: {du_topn_plot_out}"
            )
        if args.avg_window > 0:
            logger.info(f"Window-averaged event-rate line plot written to: {avg_event_plot_out}")
        if effective_avg_du_window > 0:
            logger.info(f"Window-averaged per-DU trigger-rate heatmap written to: {avg_du_plot_out}")
            if args.topn > 0:
                logger.info(
                    "Top-N window-averaged per-DU trigger-rate line plot "
                    f"written to: {avg_du_topn_plot_out}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
