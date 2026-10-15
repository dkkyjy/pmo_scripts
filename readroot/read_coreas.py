#!/usr/bin/env python3
"""CoREAS (Dunhuang) ROOT reader -> matching YAML.

CoREAS/Dunhuang simulation output is spread over several ROOT files that share
one ``<lo>-<hi>`` event range per run directory::

    adc_<lo>-<hi>_L1_<k>.root        -> tree ``tadc``        (trigger inputs)
    shower_<lo>-<hi>_L0_<k>.root     -> tree ``tshower``     (shower truth)
    showersim_<lo>-<hi>_L0_<k>.root  -> tree ``tshowersim``  (shower-sim truth)
    run_<n>_L<L>_<k>.root            -> tree ``trun``        (DU layout / site)
    runefieldsim_<n>_L<L>_<k>.root   -> tree ``trunefieldsim`` (efield config)

This reader takes one ``adc_*`` file and joins its companions (found next to it
by event range / run prefix) so a single matching YAML carries the same shape
as the StarShape reader: the trigger inputs, an embedded ``det_pos`` layout, a
``truth`` block and an ``efield_config`` block. ``shower`` and ``runefieldsim``
are optional: when a companion is missing the corresponding block is skipped
with a warning rather than failing the whole file.

The per-event trees (``tadc``/``tshower``/``tshowersim``) share an identical
``event_number`` set and row order (verified against the sample data), so rows
are aligned by ``event_number``.

Output schema (one YAML file per input ROOT file, keyed by ``event_number``)::

    <event_number>:
        run_number: int
        event_number: int
        datetime: str              # synthesised, from the run-directory time
        gps_time: int              # synthesised monotonic epoch (seconds)
        index: int                 # 0-based entry index inside the ROOT file
        file: str                  # input ROOT file name
        du_id: [str, ...]          # detector ids as strings (schema parity)
        time: {du_id: int}         # det_id -> du_nanoseconds (int)
        det_pos: {du_id: [x,y,z]}  # per-event antenna positions (xyz, metres)
        truth: {...}               # shower truth scalars for this event
        efield_config: {...}       # runefieldsim config scalars
        signal: {du_id: int}       # det_id -> trace max (--with-signal only)

``det_pos`` is built from the run tree's ``du_xyz`` (indexed by ``du_id``); it is
identical to the static ``du_coords_CoreasDC2.txt`` layout, but carrying it
per-event lets main.py use one code path for both simulation readers. Unlike
StarShape, the ``efield`` block is never emitted (``tefield.p2p`` is empty in the
sample data) and ``signal`` is derived from the ``tadc.trace_ch`` waveforms, so
it is only produced with ``--with-signal`` (the output file is then named
``<file>_<channel>.yaml``, mirroring the trigger/StarShape convention).

Usage::

    python readroot/read_coreas.py <adc_file> --date <YYYY/MM/DD> \\
        --out_dir_base <dir> [--max-events N] [--with-signal --channel XY]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import uproot

from logger_config import logger
from readroot.common import (
    RUNEFIELDSIM_TREE,
    RUN_TREE,
    SHOWERSIM_TREE,
    SHOWER_TREE,
    TADC_TREE,
    as_float,
    as_str_list,
    as_vector,
    build_synthetic_gps_times,
    build_time_map_and_du_ids,
    build_yaml,
    find_coreas_companion,
    find_event_tree_key,
    find_tree_key,
    format_event_datetime,
    mkdir as shared_mkdir,
    parse_sim_base_time,
    write_yaml,
)
from readroot.read_trace import (
    calculate_trace_max_values,
    calculate_xy_combined_max_values,
)

EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]

# ``tshower`` truth fields: output_key -> (source_branch, kind). ``kind`` is one
# of ``"float"`` / ``"vector"`` / ``"strlist"``; blocks are aligned with the
# StarShape reader where a counterpart exists.
_TSHOWER_FIELDS = {
    "primary_type": ("primary_type", "strlist"),
    "energy_primary": ("energy_primary", "float"),
    "zenith": ("zenith", "float"),
    "azimuth": ("azimuth", "float"),
    "magnetic_field": ("magnetic_field", "vector"),
    "xmax_grams": ("xmax_grams", "float"),
    "xmax_pos_shc": ("xmax_pos_shc", "vector"),
    "atmos_model": ("atmos_model", "strlist"),
    "atmos_model_param": ("atmos_model_param", "vector"),
    "shower_core_pos": ("shower_core_pos", "vector"),
    # CoREAS-only truth fields (no StarShape counterpart).
    "direction": ("direction", "vector"),
    "energy_em": ("energy_em", "float"),
    "core_alt": ("core_alt", "float"),
    "core_time_s": ("core_time_s", "float"),
    "core_time_ns": ("core_time_ns", "float"),
    "xmax_pos": ("xmax_pos", "vector"),
}

# ``tshowersim`` truth fields (shower-simulation metadata), aligned with the
# StarShape reader where a counterpart exists.
_TSHOWERSIM_FIELDS = {
    "rnd_seed": ("rnd_seed", "float"),
    "prim_injpoint_shc": ("primary_inj_point_shc", "vector"),
    "prim_inj_alt_shc": ("primary_inj_alt_shc", "float"),
    "prim_inj_dir_shc": ("primary_inj_dir_shc", "vector"),
    "hadronic_model": ("hadronic_model", "strlist"),
    "low_energy_model": ("low_energy_model", "strlist"),
    "cpu_time": ("cpu_time", "float"),
}

# Run-level ``trun`` scalars copied onto every event (per the request to carry
# more of the run metadata). Per-DU arrays (du_xyz/du_geoid/du_tilt/...) are not
# copied here: the DU layout is exported as ``det_pos`` instead.
_TRUN_SCALAR_FIELDS = (
    "run_mode",
    "first_event",
    "first_event_time",
    "last_event",
    "last_event_time",
    "data_source",
    "data_generator",
    "data_generator_version",
    "event_type",
    "event_version",
    "site_layout",
)

# ``trunefieldsim`` -> ``efield_config`` (StarShape counterpart uses two scalar
# refractivity params; CoREAS keeps its native three-element vector, per the
# "align where possible, otherwise keep native" decision).
_EFIELD_CONFIG_FIELDS = (
    "refractivity_model",
    "refractivity_model_parameters",
    "t_pre",
    "t_post",
)

_COERCERS = {"float": as_float, "vector": as_vector, "strlist": as_str_list}

# Signal uses the exact same channel/max logic as read_trace.py so the two
# paths stay value-for-value identical.
_TRACE_CHANNEL_INDEX = {"F": 0, "X": 0, "Y": 1, "Z": 2}


def _collection_or_tree(root_file, tree_name: str):
    """Return ``root_file[key]`` for ``tree_name`` or ``None`` when absent."""
    key = find_tree_key(root_file, tree_name)
    if key is None:
        return None
    return root_file[key]


def _read_event_number_map(tree, branches: List[str], max_events: int = 0):
    """Read ``branches`` from a per-event tree and index rows by ``event_number``.

    Returns ``(arrays, event_numbers)`` where ``arrays`` is the awkward array
    dict and ``event_numbers`` a plain list, so callers can align companion
    rows to the ``tadc`` file by ``event_number``.
    """
    import awkward as ak  # noqa: WPS433

    arrays = tree.arrays(
        ["event_number", *branches],
        library="ak",
        entry_stop=max_events if (max_events and max_events > 0) else None,
    )
    event_numbers = [int(ev) for ev in ak.to_list(arrays["event_number"])]
    return arrays, event_numbers


def _extract_truth(arrays, event_index: int, fields: dict) -> Dict[str, Any]:
    """Build a truth sub-dict for one row using the field->(branch, kind) map."""
    out: Dict[str, Any] = {}
    for out_key, (branch_name, kind) in fields.items():
        out[out_key] = _COERCERS[kind](arrays[branch_name][event_index])
    return out


def _row_lookup(event_numbers: List[int]) -> Dict[int, int]:
    """Map ``event_number -> row index`` for a per-event companion tree."""
    return {ev: i for i, ev in enumerate(event_numbers)}


def _build_det_pos(du_ids: List[int], du_xyz) -> Dict[str, List[float]]:
    """Build a ``{str(du): [x, y, z]}`` map from the run tree vectors."""
    positions: Dict[str, List[float]] = {}
    for du_id, xyz in zip(du_ids, du_xyz):
        positions[str(int(du_id))] = as_vector(xyz)
    return positions


def read_coreas_file(
    file_path: str,
    max_events: int = 0,
    with_signal: bool = False,
    left: int = 0,
    right: int = 512,
    channel: str = "XY",
) -> Tuple[List[dict], str]:
    """Read one CoREAS ``adc_*`` file (plus companions) into per-event payloads.

    Returns ``(payloads, base_name)``; the caller keys the YAML by each
    payload's ``event_number``. ``max_events > 0`` truncates to the first
    entries. Companions are optional and skipped with a warning when missing.
    """
    root_file = uproot.open(file_path)
    key_name, tree_kind = find_event_tree_key(list(root_file.keys()), file_path)
    if tree_kind != TADC_TREE:
        raise ValueError(
            f"expected a CoREAS '{TADC_TREE}' tree in {file_path}, got '{tree_kind}'"
        )
    adc = root_file[key_name]

    companion_branches = ["run_number", "event_number", "du_id", "du_nanoseconds"]
    if with_signal:
        companion_branches.append("trace_ch")
    adc_arrays = adc.arrays(
        companion_branches,
        library="np",
        entry_stop=max_events if (max_events and max_events > 0) else None,
    )

    run_number_list = adc_arrays["run_number"].tolist()
    event_number_list = adc_arrays["event_number"].tolist()
    du_id_list = adc_arrays["du_id"].tolist()
    du_nanosecond_list = adc_arrays["du_nanoseconds"].tolist()

    base_time = parse_sim_base_time(file_path)
    if base_time is None:
        raise ValueError(
            f"cannot derive a reference time for simulation file: {file_path}"
        )
    gps_time_list = build_synthetic_gps_times(base_time, event_number_list)

    # --- Per-event truth companions (aligned by event_number) ---
    shower_arrays = shower_numbers = None
    shower_path = find_coreas_companion(file_path, "shower_")
    if shower_path is not None:
        shower_arrays, shower_numbers = _read_event_number_map(
            uproot.open(shower_path)[SHOWER_TREE],
            list({src for src, _ in _TSHOWER_FIELDS.values()}),
        )
    else:
        logger.warning(f"CoREAS shower companion not found for {file_path}")

    showersim_arrays = showersim_numbers = None
    showersim_path = find_coreas_companion(file_path, "showersim_")
    if showersim_path is not None:
        showersim_arrays, showersim_numbers = _read_event_number_map(
            uproot.open(showersim_path)[SHOWERSIM_TREE],
            list({src for src, _ in _TSHOWERSIM_FIELDS.values()}) + ["event_date"],
        )
    else:
        logger.warning(f"CoREAS showersim companion not found for {file_path}")

    shower_row = _row_lookup(shower_numbers) if shower_numbers is not None else {}
    showersim_row = (
        _row_lookup(showersim_numbers) if showersim_numbers is not None else {}
    )

    # --- Run-level companions (single entry): DU layout / site / efield config ---
    run_tree = _collection_or_tree(root_file, RUN_TREE)
    run_path = find_coreas_companion(file_path, "run_")
    if run_tree is None and run_path is not None:
        run_tree = _collection_or_tree(uproot.open(run_path), RUN_TREE)

    det_pos_static: Dict[str, List[float]] = {}
    run_truth: Dict[str, Any] = {}
    site_config: Dict[str, Any] = {}
    t_bin_size = 0.0
    if run_tree is not None:
        import awkward as ak  # noqa: WPS433

        run_branches = ["du_id", "du_xyz", "site", "origin_geoid", "t_bin_size"]
        run_arrays = run_tree.arrays(
            [*run_branches, *_TRUN_SCALAR_FIELDS], library="ak", entry_stop=1
        )
        det_pos_static = _build_det_pos(
            ak.to_list(run_arrays["du_id"])[0], ak.to_list(run_arrays["du_xyz"])[0]
        )
        origin_geoid = as_vector(run_arrays["origin_geoid"][0])
        site_config = {
            "site": as_str_list(run_arrays["site"][0]),
            "site_lat_long": origin_geoid[:2],
            "ground_alt": origin_geoid[2],
        }
        t_bin_values = as_vector(run_arrays["t_bin_size"][0])
        t_bin_size = t_bin_values[0] if t_bin_values else 0.0
        for field in _TRUN_SCALAR_FIELDS:
            run_truth[field] = _scalar_or_str(run_arrays[field][0])
    else:
        logger.warning(f"CoREAS run companion not found for {file_path}")

    efield_config: Dict[str, Any] = {}
    efieldsim_path = find_coreas_companion(file_path, "runefieldsim_")
    if efieldsim_path is not None:
        efieldsim_tree = _collection_or_tree(
            uproot.open(efieldsim_path), RUNEFIELDSIM_TREE
        )
        arrays = efieldsim_tree.arrays(
            list(_EFIELD_CONFIG_FIELDS), library="ak", entry_stop=1
        )
        efield_config["refractivity_model"] = as_str_list(
            arrays["refractivity_model"][0]
        )
        efield_config["refractivity_model_parameters"] = as_vector(
            arrays["refractivity_model_parameters"][0]
        )
        efield_config["t_pre"] = as_float(arrays["t_pre"][0])
        efield_config["t_post"] = as_float(arrays["t_post"][0])
        if t_bin_size:
            efield_config["t_bin_size"] = t_bin_size
    else:
        logger.warning(f"CoREAS runefieldsim companion not found for {file_path}")

    # --- Optional per-event signal amplitudes (from tadc.trace_ch) ---
    signal_maps: List[Dict[str, int]] = []
    if with_signal:
        trace_ch = adc_arrays["trace_ch"].tolist()
        if channel == "XY":
            signal_lists = calculate_xy_combined_max_values(
                [[du[0] for du in event] for event in trace_ch],
                [[du[1] for du in event] for event in trace_ch],
                left,
                right,
            )
        else:
            idx = _TRACE_CHANNEL_INDEX[channel]
            signal_lists = calculate_trace_max_values(
                [[du[idx] for du in event] for event in trace_ch], left, right
            )
        for du_ids, values in zip(du_id_list, signal_lists):
            signal_maps.append(
                {str(int(du)): int(v) for du, v in zip(du_ids, values)}
            )

    payloads: List[dict] = []
    for index, du_ids in enumerate(du_id_list):
        if len(du_ids) == 0:
            continue

        event_number = int(event_number_list[index])
        time_map, list_du_id = build_time_map_and_du_ids(
            du_ids, du_nanosecond_list[index]
        )

        det_pos = {
            du_id: det_pos_static[du_id]
            for du_id in list_du_id
            if du_id in det_pos_static
        }

        truth: Dict[str, Any] = {}
        if shower_arrays is not None and event_number in shower_row:
            truth.update(
                _extract_truth(shower_arrays, shower_row[event_number], _TSHOWER_FIELDS)
            )
        if showersim_arrays is not None and event_number in showersim_row:
            truth.update(
                _extract_truth(
                    showersim_arrays, showersim_row[event_number], _TSHOWERSIM_FIELDS
                )
            )
            truth["event_date"] = int(
                showersim_arrays["event_date"][showersim_row[event_number]]
            )
        truth.update(run_truth)
        truth.update(site_config)

        payload: dict = {
            "run_number": int(run_number_list[index]),
            "event_number": event_number,
            "datetime": format_event_datetime(gps_time_list[index]),
            "gps_time": int(gps_time_list[index][2]),
            "index": index,
            "du_id": list_du_id,
            "time": time_map,
            "det_pos": det_pos,
            "truth": truth,
            "efield_config": efield_config,
        }
        if with_signal:
            payload["signal"] = signal_maps[index]

        payloads.append(payload)

    return payloads, os.path.basename(file_path)


def _scalar_or_str(value) -> Any:
    """Coerce one uproot cell into a JSON-friendly scalar (str or number)."""
    if isinstance(value, str):
        return value
    array = np.asarray(value).reshape(-1)
    if len(array) == 1 and isinstance(array[0], (str, bytes)):
        return str(array[0])
    try:
        return float(np.asarray(value, dtype=float).reshape(-1)[0])
    except (TypeError, ValueError):
        return as_str_list(value)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one CoREAS ROOT file and write a matching YAML file"
    )
    parser.add_argument("file_path", help="Path to the CoREAS adc_*.root file")
    parser.add_argument("--date", default="", help="Output date sub-directory")
    parser.add_argument(
        "--out_dir_base",
        "--out-dir-base",
        "-o",
        dest="out_dir_base",
        default="../Reco_Dir",
        help="Output directory base path (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--max-events",
        dest="max_events",
        type=int,
        default=0,
        help="Only read the first N events (0 = all)",
    )
    parser.add_argument(
        "--with-signal",
        dest="with_signal",
        action="store_true",
        help="Also emit a per-antenna ``signal`` block derived from trace_ch",
    )
    parser.add_argument("--left", type=int, default=0, help="Trace window left")
    parser.add_argument(
        "--right", type=int, default=512, help="Trace window right (default 512)"
    )
    parser.add_argument(
        "--channel",
        choices=["F", "X", "Y", "Z", "XY"],
        default="XY",
        help="Channel to fold the trace into for the signal block",
    )
    return parser.parse_args(argv[1:])


def mkdir(path: str) -> None:
    already_exists = os.path.exists(path)
    shared_mkdir(path)
    if already_exists:
        logger.debug(f"Directory already exists: {path}")


def process_root_file(
    file_path: str,
    date: str,
    out_dir_base: str,
    max_events: int = 0,
    with_signal: bool = False,
    left: int = 0,
    right: int = 512,
    channel: str = "XY",
) -> bool:
    """Read one CoREAS ROOT file and write one YAML output file."""
    try:
        out_dir = os.path.join(out_dir_base, date) if date else out_dir_base
        mkdir(out_dir)

        logger.info(f"Processing CoREAS file: {file_path}")
        payloads, base_name = read_coreas_file(
            file_path,
            max_events=max_events,
            with_signal=with_signal,
            left=left,
            right=right,
            channel=channel,
        )
        data = build_yaml(payloads, base_name)
        logger.info(f"Processed events: {len(data)}")

        output_filename = base_name.replace(".root", ".yaml")
        if with_signal:
            output_filename = output_filename.replace(".yaml", f"_{channel}.yaml")
        out_path = os.path.join(out_dir, output_filename)
        write_yaml(out_path, data)
        logger.info(f"Written to: {out_path}")
        return True
    except Exception as exc:
        logger.error(f"Processing failed: {exc}")
        return False


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    success = process_root_file(
        args.file_path,
        args.date,
        args.out_dir_base,
        max_events=args.max_events,
        with_signal=args.with_signal,
        left=args.left,
        right=args.right,
        channel=args.channel,
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
