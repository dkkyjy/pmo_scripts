"""Shared helpers for read_header and read_trace workflows."""


from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# StarShape (ZHAireS) simulations use a collection layout instead: the shower
# scalars live in ``SimuCollection``, the per-antenna field samples in
# ``EfieldCollection`` and the per-antenna arrival times/amplitudes in
# ``SignalCollection``. All three carry one entry per simulated event.
STARSHAPE_TREE = "starshape"
SIMU_COLLECTION_TREE = "SimuCollection"
EFIELD_COLLECTION_TREE = "EfieldCollection"
SIGNAL_COLLECTION_TREE = "SignalCollection"

# CoREAS/Dunhuang simulation files spread one event over several per-event
# trees, all sharing the same ``<lo>-<hi>`` event range in their file names:
#
# * ``adc_<lo>-<hi>_L1_<k>.root``        -> ``tadc``        (trigger inputs)
# * ``shower_<lo>-<hi>_L0_<k>.root``     -> ``tshower``     (shower truth)
# * ``showersim_<lo>-<hi>_L0_<k>.root``  -> ``tshowersim``  (shower-sim truth)
#
# Two run-level trees (one entry per run) live alongside:
#
# * ``run_<n>_L<L>_<k>.root``            -> ``trun``           (DU layout / site)
# * ``runefieldsim_<n>_L<L>_<k>.root``   -> ``trunefieldsim``  (efield config)
SHOWER_TREE = "tshower"
SHOWERSIM_TREE = "tshowersim"
RUN_TREE = "trun"
RUNEFIELDSIM_TREE = "trunefieldsim"

# ``adc_106900-106999_L1_0000.root`` -> the ``106900-106999`` event range.
_EVENT_INTERVAL_PATTERN = re.compile(r"_(\d+)-(\d+)_")

# CoREAS per-event companion files, paired to an ``adc_*`` file by event range.
_COREAS_INTERVAL_COMPANIONS = {"shower_": "shower", "showersim_": "showersim"}

# ``sim_Dunhuang_YYYYMMDD_HHMMSS_...`` embed the shower reference time in the
# directory name; used to synthesise a monotonic GPS time axis for simulation.
_SIM_TIME_PATTERN = re.compile(r"(\d{8})_(\d{6})")


def find_event_tree_key(keys: List[str], file_name: str) -> Tuple[str, str]:
    """Return ``(key, tree_kind)`` for the first recognised event tree.

    Prefers ``teventadc`` (real trigger data), then ``tadc`` (CoREAS/Dunhuang
    simulation), then ``SimuCollection`` (StarShape/ZHAireS simulation).
    ``tree_kind`` is one of ``"teventadc"``, ``"tadc"`` or ``"starshape"`` so
    callers can branch on the concrete schema.
    """
    for key in keys:
        if TEVENTADC_TREE in str(key):
            return key, TEVENTADC_TREE
    for key in keys:
        if TADC_TREE in str(key):
            return key, TADC_TREE
    for key in keys:
        if SIMU_COLLECTION_TREE in str(key):
            return key, STARSHAPE_TREE
    raise KeyError(
        f"neither {TEVENTADC_TREE}, {TADC_TREE} nor {SIMU_COLLECTION_TREE} "
        f"TTree found in file: {file_name}"
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
) -> Tuple[Dict[str, int], List[str]]:
    """Build DU->time map and ordered DU id list from event samples.

    The returned ``time_map`` maps each DU id (as a string) to its trigger
    time in nanoseconds, i.e. ``{du: t}``. A DU fires at most once per
    event, so each id appears exactly once and the mapping is one-to-one.
    """
    time_map: Dict[str, int] = {}
    list_du_id: List[str] = []

    for du_id, du_ns in zip(du_ids, du_nanoseconds):
        du_id_str = str(du_id)
        list_du_id.append(du_id_str)
        time_map[du_id_str] = int(du_ns)

    return time_map, list_du_id


def mkdir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def write_yaml(file_path: str, data: YamlData) -> None:
    """Write one dictionary payload to a YAML file."""
    with open(file_path, "w") as file_obj:
        yaml.dump(data, file_obj)


# --------------------------------------------------------------------------- #
# Shared coercion helpers (used by both the StarShape and CoREAS readers)
# --------------------------------------------------------------------------- #
def as_vector(value) -> List[float]:
    """Coerce one uproot scalar/array cell into a plain ``[x, y, z]`` list."""
    return [float(c) for c in np.asarray(value, dtype=float).reshape(-1)]


def as_str_list(value) -> List[str]:
    """Coerce one uproot cell into a plain list of strings.

    Handles both a bare string cell and a ``STLVector`` of strings (e.g.
    ``prim_type`` -> ``["2212"]``, ``refractivity_model`` -> ``["..."]``).
    """
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


def as_float(value) -> float:
    """Coerce one uproot cell into a float (first element for size-1 cells)."""
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def build_yaml(payloads: List[EventPayload], base_name: str) -> YamlData:
    """Key per-event payloads by ``event_number`` and stamp the source file."""
    data: YamlData = {}
    for payload in payloads:
        payload["file"] = base_name
        data[str(payload["event_number"])] = payload
    return data


def find_tree_key(root_file, tree_name: str) -> Optional[str]:
    """Return the uproot key whose tree name equals ``tree_name`` (sans cycle).

    uproot exposes trees as ``<name>;N``; iterate the keys rather than
    hard-coding the cycle number. Returns ``None`` when absent.
    """
    for key in root_file.keys():
        if str(key).split(";")[0] == tree_name:
            return key
    return None


# --------------------------------------------------------------------------- #
# CoREAS multi-file layout helpers
# --------------------------------------------------------------------------- #
def parse_event_interval(file_path: str) -> Optional[Tuple[int, int]]:
    """Parse the ``<lo>-<hi>`` event range from a CoREAS file name.

    ``adc_106900-106999_L1_0000.root`` -> ``(106900, 106999)``; returns
    ``None`` when the file name carries no such range.
    """
    match = _EVENT_INTERVAL_PATTERN.search(os.path.basename(str(file_path)))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def find_coreas_companion(adc_path: str, prefix: str) -> Optional[str]:
    """Find a CoREAS companion file next to ``adc_path`` sharing its layout.

    ``prefix`` is one of ``"shower_"`` / ``"showersim_"`` (matched by the same
    ``<lo>-<hi>`` event range) or ``"run_"`` / ``"runefieldsim_"`` (run-level;
    matched by prefix only, since they carry a run index instead of a range).
    Returns the first match (sorted for determinism) or ``None``.
    """
    directory = os.path.dirname(str(adc_path))
    if not os.path.isdir(directory):
        return None
    interval = parse_event_interval(adc_path)
    interval_tag = f"_{interval[0]}-{interval[1]}_" if interval else None

    matches: List[str] = []
    for name in os.listdir(directory):
        if not name.endswith(".root") or not name.startswith(prefix):
            continue
        if interval_tag is not None and prefix in _COREAS_INTERVAL_COMPANIONS:
            if interval_tag not in name:
                continue
        matches.append(os.path.join(directory, name))
    matches.sort()
    return matches[0] if matches else None

