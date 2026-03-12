"""Shared helpers for read_header and read_trace workflows."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import yaml

RunNumberList = List[int]
EventNumberList = List[int]
DuIdList = List[List[int]]
GpsTimeList = List[List[int]]
DuNanosecondList = List[List[int]]
EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def find_teventadc_key(keys: List[str], file_name: str) -> str:
    """Find and return the key containing the teventadc tree name."""
    for key in keys:
        if "teventadc" in str(key):
            return key
    raise KeyError(f"teventadc TTree not found in file: {file_name}")


def format_event_datetime(gps_times: List[int]) -> str:
    """Format datetime text from GPS date and hhmmss fields."""
    date = gps_times[0]
    hhmmss_time = gps_times[1]
    time_str = f"{hhmmss_time:0>6}"
    datetime_obj = datetime.strptime(f"{date}T{time_str}", "%Y%m%dT%H%M%S")
    return datetime_obj.strftime(DATETIME_FORMAT)


def build_time_map_and_du_ids(
    du_ids: List[int],
    du_nanoseconds: List[int],
) -> Tuple[Dict[str, List[int]], List[str]]:
    """Build DU->time map and ordered DU id list from event samples."""
    time_map: Dict[str, List[int]] = {}
    list_du_id: List[str] = []

    for du_id, du_ns in zip(du_ids, du_nanoseconds):
        du_id_str = str(du_id)
        list_du_id.append(du_id_str)
        time_map.setdefault(du_id_str, []).append(int(du_ns))

    return time_map, list_du_id


def mkdir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def write_yaml(file_path: str, data: YamlData) -> None:
    """Write one dictionary payload to a YAML file."""
    with open(file_path, "w") as file_obj:
        yaml.dump(data, file_obj)
