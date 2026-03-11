"""Shared helpers for stats scripts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import yaml

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATETIME_HELP_FORMAT = DATETIME_FORMAT.replace("%", "%%")

TData = TypeVar("TData", bound=MutableMapping[str, Dict[str, Any]])


class NamedDefaultDict(defaultdict):
    """Dict row with stable field order and strict field-name access."""

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


def normalize_cli_datetime_text(value: Optional[datetime]) -> str:
    """Normalize optional CLI datetime to deterministic text."""
    if value is None:
        return ""
    return value.strftime(DATETIME_FORMAT)


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
    """Parse payload datetime from ``datetime`` field."""
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
    data: TData,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    *,
    logger: Any,
    parse_payload_datetime_fn: Callable[[Dict[str, Any]], Optional[datetime]] = parse_payload_datetime,
) -> TData:
    """Filter YAML-like event dict by inclusive datetime range."""
    if start_dt is None and end_dt is None:
        return data

    filtered_data: TData = data.__class__()  # type: ignore[call-arg]
    invalid_datetime_count = 0

    for event_key, payload in data.items():
        event_dt = parse_payload_datetime_fn(payload)
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


def sort_du_id_key(du_id: str) -> tuple[int, str]:
    """Sort DU IDs numerically when possible, else lexicographically."""
    return (0, f"{int(du_id):012d}") if du_id.isdigit() else (1, du_id)


def read_yaml_events(yaml_path: Path, logger: Any) -> Dict[str, Dict[str, Any]]:
    """Read Trigger YAML and validate top-level structure."""
    with yaml_path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj)

    if loaded is None:
        logger.warning("Input YAML is empty: {}", yaml_path)
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Input YAML top-level must be a dict")
    return loaded


def parse_du_ns_map(payload: Dict[str, Any], logger: Any) -> Dict[str, float]:
    """Parse ``time`` map from one event payload."""
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


def load_du_time_offsets(offset_file: Path, logger: Any) -> Dict[str, float]:
    """Load DU time offsets from file, format: du_id, offset, sigma."""
    offsets: Dict[str, float] = {}
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
                logger.warning(
                    "Skip invalid offset value for DU {}: {}",
                    du_id,
                    parts[1],
                )
                continue

            offsets[du_id] = offset_ns

    logger.info("Loaded DU offsets: {} from {}", len(offsets), offset_file)
    return offsets


def apply_gps_datetime_ticks(
    ax: plt.Axes,
    gps_times: Sequence[int],
    datetime_map: Optional[Mapping[int, str]],
    max_ticks: int = 8,
) -> None:
    """Apply compact x ticks showing both gps_time and datetime."""
    if datetime_map is None or not gps_times:
        return

    unique_gps_times = sorted(set(gps_times))
    if not unique_gps_times:
        return

    tick_count = min(max_ticks, len(unique_gps_times))
    if tick_count <= 0:
        return

    if tick_count == 1:
        tick_positions = [unique_gps_times[0]]
    else:
        indices = np.linspace(0, len(unique_gps_times) - 1, tick_count, dtype=int)
        tick_positions = [unique_gps_times[index] for index in indices]

    tick_labels: List[str] = []
    for gps_time in tick_positions:
        datetime_label = datetime_map.get(gps_time, "")
        if datetime_label:
            tick_labels.append(f"{gps_time}\n{datetime_label}")
        else:
            tick_labels.append(str(gps_time))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")


def apply_gps_datetime_ticks_from_labels(
    ax: plt.Axes,
    gps_times: Sequence[int],
    datetime_labels: Sequence[str],
    max_ticks: int = 8,
) -> None:
    """Apply compact x ticks using parallel gps-time/datetime-label sequences."""
    if not gps_times:
        return

    label_map: Dict[int, str] = {}
    for gps_time, datetime_label in zip(gps_times, datetime_labels):
        if gps_time not in label_map and datetime_label:
            label_map[gps_time] = datetime_label

    apply_gps_datetime_ticks(ax, gps_times, label_map, max_ticks=max_ticks)
