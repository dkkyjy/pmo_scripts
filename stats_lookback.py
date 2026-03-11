#!/usr/bin/env python3
"""Statistics for previous-N-event common DU-pair delta differences.

For each event in time order, this script compares it with up to ``N``
previous events and only keeps DU pairs that are present in both events.
For every shared DU pair ``(du_a, du_b)``, it computes:

``delta_diff = (time_b - time_a)_curr - (time_b - time_a)_prev``

It exports per-sample rows to CSV and optionally plots a histogram.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import scienceplots
import yaml

from logger_config import logger

plt.style.use(["science", "grid", "notebook"])

DELTA_PLOT_MIN_NS = -5e4
DELTA_PLOT_MAX_NS = 5e4


class NamedDefaultDict(defaultdict):
    """Dict row with stable field order for CSV serialization."""

    def __init__(self, fields: Iterable[str], **values: Any) -> None:
        super().__init__(lambda: None)
        self.fields = tuple(fields)
        for field in self.fields:
            super().__setitem__(field, values.get(field))

    def __iter__(self):
        for field in self.fields:
            yield super().__getitem__(field)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            raise TypeError("NamedDefaultDict only supports field-name access")
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            raise TypeError("NamedDefaultDict only supports field-name access")
        super().__setitem__(key, value)

    def __eq__(self, other: object) -> bool:
        return dict(self) == other


def make_named_row(fields: Iterable[str], **values: Any) -> NamedDefaultDict:
    """Create one named row based on a predefined field schema."""
    return NamedDefaultDict(fields, **values)


EVENT_RECORD_FIELDS = (
    "event_key",
    "event_second",
    "index",
    "event_number",
    "du_ns_map",
    "event_datetime",
)
ADJACENT_PAIR_FIELDS = (
    "prev_event_number",
    "curr_event_number",
    "prev_second",
    "curr_second",
    "du_a",
    "du_b",
    "prev_delta_ns",
    "curr_delta_ns",
    "adjacent_delta_ns",
    "abs_adjacent_delta_ns",
    "curr_event_datetime",
)
SHARED_PAIR_COUNT_FIELDS = (
    "prev_event_number",
    "curr_event_number",
    "prev_second",
    "curr_second",
    "shared_pair_count",
    "curr_event_datetime",
)
SHARED_DU_COUNT_FIELDS = (
    "prev_event_number",
    "curr_event_number",
    "prev_second",
    "curr_second",
    "shared_du_count",
    "curr_event_datetime",
)

EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]
EventRecord = NamedDefaultDict
PairKey = Tuple[str, str]
PairDeltaMap = Dict[PairKey, float]
OffsetMap = Dict[str, float]
AdjacentPairRow = NamedDefaultDict
SharedPairCountRow = NamedDefaultDict
SharedDuCountRow = NamedDefaultDict

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATETIME_HELP_FORMAT = DATETIME_FORMAT.replace("%", "%%")


def normalize_cli_datetime_text(value: datetime | None) -> str:
    """Normalize optional CLI datetime to deterministic text."""
    if value is None:
        return ""
    return value.strftime(DATETIME_FORMAT)


def cache_file_state(path: Path) -> Dict[str, Any]:
    """Build lightweight file signature for cache validation."""
    resolved_path = str(path.resolve())
    if not path.exists():
        return {
            "path": resolved_path,
            "exists": False,
            "size": 0,
            "mtime_ns": 0,
        }

    stat_info = path.stat()
    return {
        "path": resolved_path,
        "exists": True,
        "size": int(stat_info.st_size),
        "mtime_ns": int(stat_info.st_mtime_ns),
    }


def cache_meta_path(csv_path: Path) -> Path:
    """Return cache meta path for one lookback CSV."""
    return csv_path.with_name(f"{csv_path.stem}.meta.json")


def build_cache_meta(
    yaml_path: Path,
    offset_path: Path,
    lookback: int,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> Dict[str, Any]:
    """Build cache metadata signature from runtime inputs."""
    return {
        "schema_version": 1,
        "yaml": cache_file_state(yaml_path),
        "offset": cache_file_state(offset_path),
        "lookback": int(lookback),
        "start_datetime": normalize_cli_datetime_text(start_dt),
        "end_datetime": normalize_cli_datetime_text(end_dt),
    }


def write_cache_meta(path: Path, meta: Dict[str, Any]) -> None:
    """Write cache metadata JSON file."""
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(meta, file_obj, ensure_ascii=True, sort_keys=True)


def read_cache_meta(path: Path) -> Dict[str, Any]:
    """Read cache metadata JSON file."""
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError("Lookback cache meta must be a JSON object")
    return loaded


def cache_meta_is_valid(
    cached_meta: Dict[str, Any],
    expected_meta: Dict[str, Any],
) -> tuple[bool, str]:
    """Check whether cached meta matches expected runtime signature."""
    if int(cached_meta.get("schema_version", -1)) != int(
        expected_meta.get("schema_version", -2)
    ):
        return False, "schema_version mismatch"

    for key in ("yaml", "offset"):
        if cached_meta.get(key) != expected_meta.get(key):
            return False, f"{key} signature mismatch"

    if int(cached_meta.get("lookback", -1)) != int(expected_meta.get("lookback", -2)):
        return False, "lookback mismatch"

    for key in ("start_datetime", "end_datetime"):
        if str(cached_meta.get(key, "")) != str(expected_meta.get(key, "")):
            return False, f"{key} mismatch"

    return True, ""


def sort_du_id_key(du_id: str) -> tuple[int, str]:
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
    """Parse numeric ``time`` map from one event payload.

    Supports both scalar values and list values; for list values, the last
    numeric sample is used to keep backward-compatible per-event scalar math.
    """
    time_map = payload.get("time")
    if not isinstance(time_map, dict):
        return {}

    parsed: Dict[str, float] = {}
    for key, value in time_map.items():
        if isinstance(value, list):
            numeric_values: List[float] = []
            for item in value:
                try:
                    numeric_values.append(float(item))
                except (TypeError, ValueError):
                    continue
            if not numeric_values:
                logger.debug("Skip non-numeric time list for DU {}: {}", key, value)
                continue
            parsed[str(key)] = numeric_values[-1]
            continue

        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            logger.debug("Skip non-numeric time value for DU {}: {}", key, value)
    return parsed


def parse_event_datetime(payload: EventPayload) -> str:
    """Parse and format event datetime from payload ``datetime`` field."""
    datetime_value = payload.get("datetime", "")
    datetime_text = str(datetime_value) if datetime_value is not None else ""
    if not datetime_text:
        return ""

    try:
        dt_obj = datetime.strptime(datetime_text, DATETIME_FORMAT)
        return dt_obj.strftime(DATETIME_FORMAT)
    except ValueError:
        return ""


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime argument as ``YYYY-MM-DDTHH:MM:SS``."""
    try:
        return datetime.strptime(value, DATETIME_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid datetime format: "
            f"{value!r}. Expected format: {DATETIME_FORMAT}"
        ) from exc


def parse_payload_datetime(payload: EventPayload) -> datetime | None:
    """Parse payload ``datetime`` field for range filtering."""
    datetime_value = payload.get("datetime", "")
    datetime_text = str(datetime_value) if datetime_value is not None else ""
    if not datetime_text:
        return None

    try:
        return datetime.strptime(datetime_text, DATETIME_FORMAT)
    except ValueError:
        return None


def in_datetime_range(
    event_dt: datetime,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> bool:
    """Check whether event datetime is inside an inclusive range."""
    if start_dt is not None and event_dt < start_dt:
        return False
    if end_dt is not None and event_dt > end_dt:
        return False
    return True


def filter_data_by_datetime_range(
    data: YamlData,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> YamlData:
    """Filter YAML events by datetime range from payload datetime field."""
    if start_dt is None and end_dt is None:
        return data

    filtered_data: YamlData = {}
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
            "Skipped {} events with invalid/missing datetime while filtering",
            invalid_datetime_count,
        )

    return filtered_data


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
            make_named_row(
                EVENT_RECORD_FIELDS,
                event_key=event_key,
                event_second=event_second,
                index=index,
                event_number=event_number,
                du_ns_map=du_ns_map,
                event_datetime=event_datetime,
            )
        )

    records.sort(
        key=lambda row: (
            int(row["event_second"]),
            int(row["index"]),
            int(row["event_number"]),
            str(row["event_key"]),
        )
    )
    logger.info("Built event records: {}", len(records))
    return records


def build_pair_delta_map(
    du_ns_map: Dict[str, float],
    offset_map: OffsetMap | None = None,
) -> PairDeltaMap:
    """Build DU-pair in-event delta map: ``time_b - time_a``."""
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
    pair_rows, _ = _build_common_pair_and_count_rows(records, lookback, offset_map)
    return pair_rows


def _build_common_pair_and_count_rows(
    records: Sequence[EventRecord],
    lookback: int,
    offset_map: OffsetMap | None,
) -> tuple[List[AdjacentPairRow], List[SharedPairCountRow]]:
    """Build both pair-delta rows and shared-pair-count rows in one pass."""
    if len(records) < 2:
        return [], []

    lookback = max(1, int(lookback))
    if offset_map is None:
        offset_map = {}

    rows: List[AdjacentPairRow] = []
    count_rows: List[SharedPairCountRow] = []
    for row_index in range(1, len(records)):
        curr_row = records[row_index]
        curr_second = int(curr_row["event_second"])
        curr_index = int(curr_row["index"])
        curr_event_number = int(curr_row["event_number"])
        curr_du_ns = curr_row["du_ns_map"]
        curr_event_datetime = str(curr_row["event_datetime"])
        curr_pair_map = build_pair_delta_map(curr_du_ns, offset_map)

        start_index = max(0, row_index - lookback)
        for prev_row_index in range(start_index, row_index):
            prev_row = records[prev_row_index]
            prev_second = int(prev_row["event_second"])
            prev_index = int(prev_row["index"])
            prev_event_number = int(prev_row["event_number"])
            prev_du_ns = prev_row["du_ns_map"]
            prev_pair_map = build_pair_delta_map(prev_du_ns, offset_map)
            common_pairs = sorted(set(prev_pair_map) & set(curr_pair_map))

            count_rows.append(
                make_named_row(
                    SHARED_PAIR_COUNT_FIELDS,
                    prev_event_number=prev_event_number,
                    curr_event_number=curr_event_number,
                    prev_second=prev_second,
                    curr_second=curr_second,
                    shared_pair_count=len(common_pairs),
                    curr_event_datetime=curr_event_datetime,
                )
            )

            for du_a, du_b in common_pairs:
                prev_delta = prev_pair_map[(du_a, du_b)]
                curr_delta = curr_pair_map[(du_a, du_b)]
                adjacent_delta = curr_delta - prev_delta
                rows.append(
                    make_named_row(
                        ADJACENT_PAIR_FIELDS,
                        prev_event_number=prev_event_number,
                        curr_event_number=curr_event_number,
                        prev_second=prev_second,
                        curr_second=curr_second,
                        du_a=du_a,
                        du_b=du_b,
                        prev_delta_ns=prev_delta,
                        curr_delta_ns=curr_delta,
                        adjacent_delta_ns=adjacent_delta,
                        abs_adjacent_delta_ns=abs(adjacent_delta),
                        curr_event_datetime=curr_event_datetime,
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
    logger.info(
        "Built shared DU-pair count rows: {} (lookback={})",
        len(count_rows),
        lookback,
    )
    return rows, count_rows


def build_shared_pair_count_rows(
    records: Sequence[EventRecord],
    lookback: int = 10,
    offset_map: OffsetMap | None = None,
) -> List[SharedPairCountRow]:
    """Build shared DU-pair count rows between current and previous events."""
    _, count_rows = _build_common_pair_and_count_rows(records, lookback, offset_map)
    return count_rows


def build_shared_du_count_rows(
    records: Sequence[EventRecord],
    lookback: int = 10,
) -> List[SharedDuCountRow]:
    """Build shared DU count rows between current and previous events."""
    if len(records) < 2:
        return []

    lookback = max(1, int(lookback))
    rows: List[SharedDuCountRow] = []
    for row_index in range(1, len(records)):
        curr_row = records[row_index]
        curr_second = int(curr_row["event_second"])
        curr_event_number = int(curr_row["event_number"])
        curr_du_ns = curr_row["du_ns_map"]
        curr_event_datetime = str(curr_row["event_datetime"])
        curr_du_ids = set(curr_du_ns.keys())

        start_index = max(0, row_index - lookback)
        for prev_row_index in range(start_index, row_index):
            prev_row = records[prev_row_index]
            prev_second = int(prev_row["event_second"])
            prev_event_number = int(prev_row["event_number"])
            prev_du_ns = prev_row["du_ns_map"]
            prev_du_ids = set(prev_du_ns.keys())
            shared_du_count = len(curr_du_ids & prev_du_ids)
            rows.append(
                make_named_row(
                    SHARED_DU_COUNT_FIELDS,
                    prev_event_number=prev_event_number,
                    curr_event_number=curr_event_number,
                    prev_second=prev_second,
                    curr_second=curr_second,
                    shared_du_count=shared_du_count,
                    curr_event_datetime=curr_event_datetime,
                )
            )

    logger.info(
        "Built shared DU count rows: {} (lookback={})",
        len(rows),
        lookback,
    )
    return rows


def derive_shared_pair_count_rows_from_adjacent_rows(
    rows: Sequence[AdjacentPairRow],
) -> List[SharedPairCountRow]:
    """Derive shared-pair counts from pair rows (cache-compat fallback)."""
    grouped: Dict[tuple[int, int, int, int, str], int] = {}
    for row in rows:
        key = (
            int(row["prev_event_number"]),
            int(row["curr_event_number"]),
            int(row["prev_second"]),
            int(row["curr_second"]),
            str(row["curr_event_datetime"]),
        )
        grouped[key] = grouped.get(key, 0) + 1

    derived_rows: List[SharedPairCountRow] = []
    for key in sorted(grouped.keys()):
        prev_event_number, curr_event_number, prev_second, curr_second, curr_datetime = (
            key
        )
        derived_rows.append(
            make_named_row(
                SHARED_PAIR_COUNT_FIELDS,
                prev_event_number=prev_event_number,
                curr_event_number=curr_event_number,
                prev_second=prev_second,
                curr_second=curr_second,
                shared_pair_count=grouped[key],
                curr_event_datetime=curr_datetime,
            )
        )
    return derived_rows


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
        writer.writerows(tuple(row) for row in rows)


def read_adjacent_pair_csv(path: Path) -> List[AdjacentPairRow]:
    """Read adjacent/common DU-pair rows from CSV cache."""
    rows: List[AdjacentPairRow] = []
    invalid_row_count = 0
    required_fields = {
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
    }

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        header_fields = set(reader.fieldnames or [])
        if not required_fields.issubset(header_fields):
            missing_fields = sorted(required_fields - header_fields)
            raise ValueError(
                "Lookback cache missing required columns: "
                f"{','.join(missing_fields)}"
            )

        for row_index, row in enumerate(reader, start=2):
            try:
                rows.append(
                    make_named_row(
                        ADJACENT_PAIR_FIELDS,
                        prev_event_number=int(row["prev_event_number"]),
                        curr_event_number=int(row["curr_event_number"]),
                        prev_second=int(row["prev_gps_time"]),
                        curr_second=int(row["curr_gps_time"]),
                        du_a=str(row["du_a"]),
                        du_b=str(row["du_b"]),
                        prev_delta_ns=float(row["prev_pair_delta_ns"]),
                        curr_delta_ns=float(row["curr_pair_delta_ns"]),
                        adjacent_delta_ns=float(row["adjacent_pair_delta_ns"]),
                        abs_adjacent_delta_ns=float(row["abs_adjacent_pair_delta_ns"]),
                        curr_event_datetime=str(row.get("curr_event_datetime", "")),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                invalid_row_count += 1
                logger.warning(
                    "Skip invalid lookback cache row at line {}: {}",
                    row_index,
                    exc,
                )
                continue

    if invalid_row_count > 0:
        logger.warning(
            "Skipped {} invalid rows while reading lookback cache: {}",
            invalid_row_count,
            path,
        )

    return rows


def write_shared_pair_count_csv(
    path: Path,
    rows: Sequence[SharedPairCountRow],
) -> None:
    """Write shared DU-pair count rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "prev_event_number",
                "curr_event_number",
                "prev_gps_time",
                "curr_gps_time",
                "shared_du_pair_count",
                "curr_event_datetime",
            ]
        )
        writer.writerows(tuple(row) for row in rows)


def read_shared_pair_count_csv(path: Path) -> List[SharedPairCountRow]:
    """Read shared DU-pair count rows from CSV cache."""
    rows: List[SharedPairCountRow] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                make_named_row(
                    SHARED_PAIR_COUNT_FIELDS,
                    prev_event_number=int(row["prev_event_number"]),
                    curr_event_number=int(row["curr_event_number"]),
                    prev_second=int(row["prev_gps_time"]),
                    curr_second=int(row["curr_gps_time"]),
                    shared_pair_count=int(row["shared_du_pair_count"]),
                    curr_event_datetime=str(row.get("curr_event_datetime", "")),
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

    gps_times = [int(row["curr_second"]) for row in rows]
    delta_values = [float(row["adjacent_delta_ns"]) for row in rows]
    datetime_labels = [str(row["curr_event_datetime"]) for row in rows]

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

    deltas = [float(row["adjacent_delta_ns"]) for row in rows]

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


def plot_shared_pair_count_histogram(
    path: Path,
    rows: Sequence[SharedPairCountRow],
    bins: int,
) -> None:
    """Plot histogram for shared DU-pair counts across event comparisons."""
    if not rows:
        logger.warning("No shared DU-pair count rows; skip plotting")
        return

    counts = [int(row["shared_pair_count"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=max(1, bins), edgecolor="black")
    ax.set_xlabel("Shared DU-pair count")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("Distribution of shared DU-pair count in lookback comparisons")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shared_pair_count_vs_time(
    path: Path,
    rows: Sequence[SharedPairCountRow],
) -> None:
    """Plot shared DU-pair counts versus current event time."""
    if not rows:
        logger.warning("No shared DU-pair count rows; skip time scatter plotting")
        return

    gps_times = [int(row["curr_second"]) for row in rows]
    count_values = [int(row["shared_pair_count"]) for row in rows]
    datetime_labels = [str(row["curr_event_datetime"]) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(gps_times, count_values, s=8, alpha=0.6)
    ax.set_xlabel("Curr event time (gps_time)")
    ax.set_ylabel("Shared DU-pair count")
    ax.set_title("Shared DU-pair count vs curr event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, gps_times, datetime_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shared_du_count_histogram(
    path: Path,
    rows: Sequence[SharedDuCountRow],
    bins: int,
) -> None:
    """Plot histogram for shared DU counts across event comparisons."""
    if not rows:
        logger.warning("No shared DU count rows; skip plotting")
        return

    counts = [int(row["shared_du_count"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=max(1, bins), edgecolor="black")
    ax.set_xlabel("Shared DU count")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("Distribution of shared DU count in lookback comparisons")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shared_du_count_vs_time(
    path: Path,
    rows: Sequence[SharedDuCountRow],
) -> None:
    """Plot shared DU counts versus current event time."""
    if not rows:
        logger.warning("No shared DU count rows; skip time scatter plotting")
        return

    gps_times = [int(row["curr_second"]) for row in rows]
    count_values = [int(row["shared_du_count"]) for row in rows]
    datetime_labels = [str(row["curr_event_datetime"]) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(gps_times, count_values, s=8, alpha=0.6)
    ax.set_xlabel("Curr event time (gps_time)")
    ax.set_ylabel("Shared DU count")
    ax.set_title("Shared DU count vs curr event time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    apply_time_ticks(ax, gps_times, datetime_labels)
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
        "--offset-file",
        default="2025-10-28_beacon_25Hz_offset.txt",
        help="DU time offset file path. Format: du_id, offset, sigma",
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
    offset_path = Path(args.offset_file)
    normalized_lookback = max(1, args.lookback)

    csv_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_distribution.csv"
    )
    meta_out = cache_meta_path(csv_out)
    plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_hist.png"
    )
    time_plot_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_common_du_pair_delta_vs_time.png"
    )
    count_hist_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_shared_du_pair_count_hist.png"
    )
    count_time_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_shared_du_pair_count_vs_time.png"
    )
    du_count_hist_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_shared_du_count_hist.png"
    )
    du_count_time_out = yaml_path.with_name(
        f"{yaml_path.stem}_lookback{normalized_lookback}_shared_du_count_vs_time.png"
    )

    use_csv_cache = csv_out.exists() and not args.force_recompute
    if start_datetime is not None or end_datetime is not None:
        use_csv_cache = False

    if use_csv_cache:
        logger.info("Found CSV cache: {}", csv_out)
        expected_meta = build_cache_meta(
            yaml_path,
            offset_path,
            normalized_lookback,
            start_datetime,
            end_datetime,
        )
        rows: Optional[List[AdjacentPairRow]] = None

        if not meta_out.exists():
            if not yaml_path.exists():
                logger.warning(
                    "Cache meta missing but YAML missing; using legacy cache: {}",
                    csv_out,
                )
                try:
                    rows = read_adjacent_pair_csv(csv_out)
                except ValueError as exc:
                    logger.error(
                        "Invalid legacy cache without YAML fallback: {} ({})",
                        csv_out,
                        exc,
                    )
                    return 2
            else:
                logger.info(
                    "Cache meta missing for {}, fallback to recompute",
                    csv_out,
                )
        else:
            try:
                cached_meta = read_cache_meta(meta_out)
                is_valid, reason = cache_meta_is_valid(cached_meta, expected_meta)
                if not is_valid:
                    logger.info(
                        "Cache meta mismatch for {}: {}, fallback to recompute",
                        csv_out,
                        reason,
                    )
                else:
                    rows = read_adjacent_pair_csv(csv_out)
            except (ValueError, json.JSONDecodeError) as exc:
                if not yaml_path.exists():
                    logger.warning(
                        "Invalid cache/meta {} / {} but YAML missing; using legacy cache ({})",
                        csv_out,
                        meta_out,
                        exc,
                    )
                    rows = read_adjacent_pair_csv(csv_out)
                else:
                    logger.warning(
                        "Invalid lookback cache/meta: {} / {} ({}). Falling back to recompute.",
                        csv_out,
                        meta_out,
                        exc,
                    )

        if rows is None:
            logger.info("Skip invalid cache and continue recompute path")
        else:
            logger.info("Loaded rows from CSV cache: {}", len(rows))
            count_rows = derive_shared_pair_count_rows_from_adjacent_rows(rows)
            logger.info("Derived shared-count rows from pair-row cache: {}", len(count_rows))
            du_count_rows: List[SharedDuCountRow] = []

            if yaml_path.exists():
                try:
                    data = read_yaml_events(yaml_path)
                    data = filter_data_by_datetime_range(data, start_datetime, end_datetime)
                    du_count_rows = build_shared_du_count_rows(
                        build_event_records(data),
                        lookback=args.lookback,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to build shared DU count rows from YAML in cache path: {}",
                        exc,
                    )

            if not args.no_plot:
                plot_adjacent_pair_delta_histogram(plot_out, rows, args.bins)
                plot_adjacent_pair_delta_vs_time(time_plot_out, rows)
                plot_shared_pair_count_histogram(count_hist_out, count_rows, args.bins)
                plot_shared_pair_count_vs_time(count_time_out, count_rows)
                plot_shared_du_count_histogram(
                    du_count_hist_out,
                    du_count_rows,
                    args.bins,
                )
                plot_shared_du_count_vs_time(
                    du_count_time_out,
                    du_count_rows,
                )
                if rows:
                    logger.info("Plot written: {}", plot_out)
                    logger.info("Plot written: {}", time_plot_out)
                if count_rows:
                    logger.info("Plot written: {}", count_hist_out)
                    logger.info("Plot written: {}", count_time_out)
                if du_count_rows:
                    logger.info("Plot written: {}", du_count_hist_out)
                    logger.info("Plot written: {}", du_count_time_out)
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

    data = filter_data_by_datetime_range(data, start_datetime, end_datetime)

    offset_map = load_du_time_offsets(offset_path)
    records = build_event_records(data)
    rows, count_rows = _build_common_pair_and_count_rows(
        records,
        lookback=args.lookback,
        offset_map=offset_map,
    )
    du_count_rows = build_shared_du_count_rows(records, lookback=args.lookback)

    write_adjacent_pair_csv(csv_out, rows)
    write_cache_meta(
        meta_out,
        build_cache_meta(
            yaml_path,
            offset_path,
            normalized_lookback,
            start_datetime,
            end_datetime,
        ),
    )
    logger.info("CSV written: {}", csv_out)

    if not args.no_plot:
        plot_adjacent_pair_delta_histogram(plot_out, rows, args.bins)
        plot_adjacent_pair_delta_vs_time(time_plot_out, rows)
        plot_shared_pair_count_histogram(count_hist_out, count_rows, args.bins)
        plot_shared_pair_count_vs_time(count_time_out, count_rows)
        plot_shared_du_count_histogram(
            du_count_hist_out,
            du_count_rows,
            args.bins,
        )
        plot_shared_du_count_vs_time(
            du_count_time_out,
            du_count_rows,
        )
        if rows:
            logger.info("Plot written: {}", plot_out)
            logger.info("Plot written: {}", time_plot_out)
        if count_rows:
            logger.info("Plot written: {}", count_hist_out)
            logger.info("Plot written: {}", count_time_out)
        if du_count_rows:
            logger.info("Plot written: {}", du_count_hist_out)
            logger.info("Plot written: {}", du_count_time_out)
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