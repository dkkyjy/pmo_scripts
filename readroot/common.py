"""Shared helpers for read_header and read_trace workflows."""


from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

RunNumberList = List[int]
EventNumberList = List[int]
DuIdList = List[List[int]]
GpsTimeList = List[List[int]]
DuNanosecondList = List[List[int]]
EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Real GRAND/PMO trigger data uses the ``teventadc`` tree (with a populated
# ``gps_time`` vector). CoREAS/Dunhuang simulation files use a ``tadc`` tree
# instead, which carries per-DU ``du_nanoseconds`` but no ``gps_time`` vector.
TEVENTADC_TREE = "teventadc"
TADC_TREE = "tadc"

# ``sim_Dunhuang_YYYYMMDD_HHMMSS_...`` embed the shower reference time in the
# directory name; used to synthesise a monotonic GPS time axis for simulation.
_SIM_TIME_PATTERN = re.compile(r"(\d{8})_(\d{6})")


def find_event_tree_key(keys: List[str], file_name: str) -> Tuple[str, str]:
    """Return ``(key, tree_kind)`` for the first recognised event tree.

    Prefers ``teventadc`` (real trigger data) and falls back to ``tadc``
    (CoREAS/Dunhuang simulation). ``tree_kind`` is one of ``"teventadc"`` or
    ``"tadc"`` so callers can branch on the concrete schema.
    """
    for key in keys:
        if TEVENTADC_TREE in str(key):
            return key, TEVENTADC_TREE
    for key in keys:
        if TADC_TREE in str(key):
            return key, TADC_TREE
    raise KeyError(
        f"neither {TEVENTADC_TREE} nor {TADC_TREE} TTree found in file: {file_name}"
    )


def find_teventadc_key(keys: List[str], file_name: str) -> str:
    """Find and return the key containing the teventadc tree name."""
    for key in keys:
        if "teventadc" in str(key):
            return key
    raise KeyError(f"teventadc TTree not found in file: {file_name}")


def parse_sim_base_time(file_path: str) -> Optional[Tuple[int, int, int]]:
    """Parse the CoREAS reference time from a simulation file path.

    Looks for a ``YYYYMMDD_HHMMSS`` fragment (as found in the
    ``sim_Dunhuang_<YYYYMMDD>_<HHMMSS>_...`` run directories) and returns
    ``(date_int, hhmmss_int, epoch_utc)`` where ``date_int``/``hhmmss_int``
    mirror the ``[date, hhmmss, epoch]`` fields of the real ``gps_time``
    vector, and ``epoch_utc`` is the corresponding UTC epoch in seconds.
    Returns ``None`` when no such fragment is present.
    """
    for part in reversed(str(file_path).replace("\\", "/").split("/")):
        match = _SIM_TIME_PATTERN.search(part)
        if not match:
            continue
        date_str, time_str = match.group(1), match.group(2)
        try:
            dt = datetime(
                int(date_str[0:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
                int(time_str[0:2]),
                int(time_str[2:4]),
                int(time_str[4:6]),
            )
        except ValueError:
            continue
        epoch = calendar.timegm(dt.timetuple())
        return int(date_str), int(time_str), int(epoch)
    return None


def build_synthetic_gps_times(
    base_time: Tuple[int, int, int],
    event_numbers: List[int],
) -> GpsTimeList:
    """Synthesise a monotonic ``gps_time`` vector for simulation events.

    ``base_time`` is ``(date_int, hhmmss_int, epoch_utc)`` from
    :func:`parse_sim_base_time`. Each event gets ``epoch_utc + event_number``
    so the axis is monotonic within a run while ``date``/``hhmmss`` keep the
    shower reference wall-clock (only used for display/sorting).
    """
    date_int, hhmmss_int, epoch = base_time
    return [[int(date_int), int(hhmmss_int), int(epoch) + int(ev)] for ev in event_numbers]


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
