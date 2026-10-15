#!/usr/bin/env python3
"""StarShape (ZHAireS) ROOT reader -> matching YAML.

StarShape simulation files (e.g. GP300 ``*_stshp_*Events.root``) are laid out
as three collections, each holding one entry per simulated event:

* ``SimuCollection/SimShowerBranch``   -- shower truth scalars
  (``prim_energy``, ``shower_zenith``, ``shower_azimuth``, ``xmax_pos_shc`` ...);
* ``EfieldCollection/SimEfieldBranch`` -- per-antenna layout and arrival times
  (``Detectors_det_id``, ``Detectors_det_pos_shc``, ``Detectors_t_0`` ...);
* ``SignalCollection/SimSignalBranch`` -- the same antennas as voltages.

Unlike the real ``teventadc`` tree and the CoREAS ``tadc`` tree, these files
carry **no** ``gps_time``, ``du_id`` or ``du_nanoseconds`` branches, and the
file name embeds no date. Consequences that shape this reader:

1. the per-antenna arrival times come from ``Detectors_t_0`` (a *float*
   nanosecond offset, possibly negative) and are preserved as floats;
2. the antenna coordinates change event to event, so each event embeds its own
   ``det_pos`` mapping (see the per-event schema below).

Antenna data source
-------------------
``SignalCollection``'s object-typed branches (``Detectors_det_id``,
``Detectors_det_pos_shc``) are unreadable with uproot 5.x (they decode to empty
arrays), whereas the *same* branches under ``EfieldCollection`` decode cleanly
and its ``Detectors_t_0`` matches ``SignalCollection``'s. Since signal
amplitudes do not participate in the PWM/SWM fits, this reader takes **all**
antenna data from ``EfieldCollection``.

Output schema (one YAML file per input ROOT file, keyed by ``event_number``)::

    <event_number>:
        run_number: int
        event_number: int          # == SimShower evt_id
        datetime: str              # synthesised, from the file's reference time
        gps_time: int              # synthesised monotonic epoch (seconds)
        index: int                 # 0-based entry index inside the ROOT file
        file: str                  # input ROOT file name
        du_id: [str, ...]          # detector ids as strings (schema parity)
        time: {du_id: float}       # det_id -> t_0 (ns), float
        det_pos: {du_id: [x,y,z]}  # per-event antenna positions (shc, metres)
        truth: {...}               # shower truth scalars for this event
        efield_config: {...}       # per-event SimEfield config scalars
        efield: {du_id: float}     # det_id -> peak-to-peak amplitude (--with-signal only)
        signal: {du_id: float}     # det_id -> peak-to-peak amplitude (--with-signal only)

The optional ``efield``/``signal`` amplitude blocks are only written when
``--with-signal`` is given; the default matching YAML carries just the
reconstruction inputs and truth. Their keys mirror the per-antenna quantities
the legacy PyROOT ``readRoot.py`` reads out of ``SimEfield``/``SimSignal``
(minus the bulky ``Detectors_trace_*`` waveform samples, which are omitted),
keyed by ``du_id`` in the same ordering as ``time``/``det_pos``. When
``--with-signal`` is set the output file is named ``<file>_<channel>.yaml`` so
it mirrors the CoREAS/trigger signal-matching convention.

The ``truth`` block names are aligned with the CoREAS reader where a CoREAS
counterpart exists (e.g. ``prim_type`` -> ``primary_type``,
``shower_zenith`` -> ``zenith``, ``ground_alt`` -> ``ground_alt`` from
``trun``); StarShape-only scalars keep their native names::

    truth:
        primary_type: [str, ...]   # prim_type
        energy_primary: float      # prim_energy
        zenith: float              # shower_zenith
        azimuth: float             # shower_azimuth + 180 (arrival direction)
        magnetic_field: [x,y,z]
        xmax_grams: float
        xmax_pos_shc: [x,y,z]
        atmos_model: [str, ...]
        atmos_model_param: [...]
        shower_core_pos: [x,y,z]
        rnd_seed: float            # StarShape-only (CoREAS: tshowersim.rnd_seed)
        prim_injpoint_shc: [...]   # StarShape-only
        prim_inj_alt_shc: float    # StarShape-only
        prim_inj_dir_shc: [...]    # StarShape-only
        hadronic_model: [str, ...] # StarShape-only
        low_energy_model: [...]    # StarShape-only
        cpu_time: float            # StarShape-only
        event_date: [str, ...]     # date
        site: [str, ...]
        site_lat_long: [lat, long]
        ground_alt: float          # site ground altitude (== trun origin_geoid[2])
        shower_type: [str, ...]    # StarShape-only
        shower_energy: float       # StarShape-only
        energy_in_neutrinos: float # StarShape-only
        xmax_alt: float            # StarShape-only
        gh_fit_param: [...]        # StarShape-only

    efield_config:
        refractivity_model: [str, ...]
        refractivity_param1: float
        refractivity_param2: float
        t_pre: float               # trace window (ns before/after t_0)
        t_post: float
        t_bin_size: float          # sampling step (ns)

Signal ``Detectors_t_0`` is identical to the efield ``Detectors_t_0`` already
exported as ``time``, so it is not duplicated; only the signal amplitudes are
kept.

``signal``'s ``Detectors_det_id``/``Detectors_det_pos_shc``/``Detectors_det_type``
object branches decode to empty arrays under both uproot 5.x and PyROOT, so
they are not read (the legacy script only used the signal amplitudes/times).

Usage::

    python readroot/read_starshape.py <file_path> --date <YYYY/MM/DD> \\
        --out_dir_base <dir> [--with-signal --channel XY]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import uproot

from logger_config import logger
from readroot.common import (
    EFIELD_COLLECTION_TREE,
    SIGNAL_COLLECTION_TREE,
    SIMU_COLLECTION_TREE,
    as_float,
    as_str_list,
    as_vector,
    build_yaml as shared_build_yaml,
    mkdir as shared_mkdir,
    write_yaml,
)

EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]

# SimShowerBranch truth fields worth exporting alongside the reconstruction
# inputs. Keys are the *output* names (aligned with the CoREAS reader where a
# CoREAS counterpart exists); values are ``(source_branch, kind)`` with kind in
# ``{"float", "vector", "strlist"}``. ``xmax_pos_shc`` is a 3-vector, the string
# fields are ``vector<string>``; ``prim_inj_alt_shc`` is a bare float.
_SIMSHOWER_FIELDS = {
    # output_key: (source_branch, kind)
    "primary_type": ("prim_type", "strlist"),
    "energy_primary": ("prim_energy", "float"),
    "zenith": ("shower_zenith", "float"),
    "azimuth": ("shower_azimuth", "float"),
    "magnetic_field": ("magnetic_field", "vector"),
    "xmax_grams": ("xmax_grams", "float"),
    "xmax_pos_shc": ("xmax_pos_shc", "vector"),
    "atmos_model": ("atmos_model", "strlist"),
    "atmos_model_param": ("atmos_model_param", "vector"),
    "shower_core_pos": ("shower_core_pos", "vector"),
    "rnd_seed": ("rnd_seed", "float"),
    "prim_injpoint_shc": ("prim_injpoint_shc", "vector"),
    "prim_inj_alt_shc": ("prim_inj_alt_shc", "float"),
    "prim_inj_dir_shc": ("prim_inj_dir_shc", "vector"),
    "hadronic_model": ("hadronic_model", "strlist"),
    "low_energy_model": ("low_energy_model", "strlist"),
    "cpu_time": ("cpu_time", "float"),
    "event_date": ("date", "strlist"),
    # Site block (aligned with the CoREAS ``trun`` run tree).
    "site": ("site", "strlist"),
    "site_lat_long": ("site_lat_long", "vector"),
    "ground_alt": ("ground_alt", "float"),
    # StarShape-only truth fields (no CoREAS counterpart); kept as-is.
    "shower_type": ("shower_type", "strlist"),
    "shower_energy": ("shower_energy", "float"),
    "energy_in_neutrinos": ("energy_in_neutrinos", "float"),
    "xmax_alt": ("xmax_alt", "float"),
    "gh_fit_param": ("gh_fit_param", "vector"),
}

_COERCERS = {"float": as_float, "vector": as_vector, "strlist": as_str_list}
_SIMSHOWER_SOURCE_BRANCHES = tuple(src for src, _ in _SIMSHOWER_FIELDS.values())

# SimEfieldBranch per-event config scalars (beyond the per-antenna arrays).
# These live at the event level under an ``efield_config`` block; ``field_sim``
# is intentionally omitted.
_EFIELD_CONFIG_STR_BRANCHES = ("refractivity_model",)
_EFIELD_CONFIG_FLOAT_BRANCHES = (
    "refractivity_param1",
    "refractivity_param2",
    "t_pre",
    "t_post",
    "t_bin_size",
)
_EFIELD_CONFIG_BRANCHES = _EFIELD_CONFIG_STR_BRANCHES + _EFIELD_CONFIG_FLOAT_BRANCHES

# Synthetic GPS epoch for StarShape files. These simulations embed no wall-clock
# date, so we anchor every file at a fixed epoch and offset each event by its
# zero-based index to keep the axis monotonic (one second apart).
_STARSHAPE_EPOCH = 1577836800  # 2020-01-01T00:00:00Z


def _collection_branch(root_file, collection_name: str, branch_name: str):
    """Return ``root_file[collection][branch]`` for a StarShape collection.

    uproot exposes the tree as ``<collection>;N``; iterate the keys to find the
    matching collection instead of hard-coding the cycle number.
    """
    for key in root_file.keys():
        if str(key).split(";")[0] == collection_name:
            return root_file[key][branch_name]
    raise KeyError(f"collection {collection_name} not found in ROOT file")


def read_starshape_file(
    file_path: str, max_events: int = 0, with_signal: bool = False
) -> Tuple[List[dict], str]:
    """Read one StarShape ROOT file into a list of per-event payloads.

    Returns ``(payloads, base_name)`` where ``base_name`` is the ROOT file's
    basename; the caller keys the YAML by each payload's ``event_number``.
    ``max_events > 0`` truncates the read to the first ``max_events`` entries.
    """
    import awkward as ak  # noqa: WPS433 (local import keeps the module light)

    root_file = uproot.open(file_path)

    shower = _collection_branch(root_file, SIMU_COLLECTION_TREE, "SimShowerBranch")
    efield = _collection_branch(root_file, EFIELD_COLLECTION_TREE, "SimEfieldBranch")

    # --- Shower truth scalars + config vectors (one per event) ---
    shower_arrays = shower.arrays(
        ["evt_id", *_SIMSHOWER_SOURCE_BRANCHES],
        library="ak",
        entry_stop=max_events if (max_events and max_events > 0) else None,
    )
    n_events = len(shower_arrays["evt_id"])
    if max_events and max_events > 0:
        n_events = min(n_events, max_events)

    # --- Per-antenna layout + arrival times (EfieldCollection) ---
    # ``Detectors_det_pos_shc`` is an object branch, so read this collection with
    # awkward and convert to python lists; det_id / det_pos_shc / t_0 share the
    # same per-event ordering and length.
    efield_arrays = efield.arrays(
        [
            "Detectors_det_id",
            "Detectors_det_pos_shc",
            "Detectors_t_0",
            *_EFIELD_CONFIG_BRANCHES,
        ],
        library="ak",
        entry_stop=n_events if (max_events and max_events > 0) else None,
    )
    det_ids_all = ak.to_list(efield_arrays["Detectors_det_id"])
    positions_all = ak.to_list(efield_arrays["Detectors_det_pos_shc"])
    t0_all = ak.to_list(efield_arrays["Detectors_t_0"])
    if not (len(det_ids_all) == len(positions_all) == len(t0_all) == n_events):
        raise ValueError(
            "StarShape collection length mismatch: SimShower vs SimEfield "
            f"({n_events} vs {len(det_ids_all)}/{len(positions_all)}/{len(t0_all)})"
        )

    # --- Per-antenna amplitudes (optional, only with --with-signal) ---
    # ``efield``/``signal`` mirror the per-antenna peak-to-peak amplitudes and
    # are only produced when requested, so the default matching YAML stays lean.
    # Only ``Detectors_p2p`` decodes reliably for the SignalCollection (its
    # object branches det_id/det_pos_shc are empty under uproot 5.x and PyROOT),
    # so both amplitude maps are aligned positionally to the efield antennas.
    efield_p2p_all: List[Any] = []
    signal_p2p_all: List[Any] = []
    if with_signal:
        efield_p2p_all = ak.to_list(
            efield.arrays(
                ["Detectors_p2p"],
                library="ak",
                entry_stop=n_events if (max_events and max_events > 0) else None,
            )["Detectors_p2p"]
        )
        signal = _collection_branch(
            root_file, SIGNAL_COLLECTION_TREE, "SimSignalBranch"
        )
        signal_p2p_all = ak.to_list(
            signal.arrays(
                ["Detectors_p2p"],
                library="ak",
                entry_stop=n_events if (max_events and max_events > 0) else None,
            )["Detectors_p2p"]
        )

    payloads: List[dict] = []
    for index in range(n_events):
        shower_evt_id = int(shower_arrays["evt_id"][index])

        det_ids = np.asarray(det_ids_all[index]).reshape(-1)
        t0_values = np.asarray(t0_all[index], dtype=float).reshape(-1)
        positions = np.asarray(positions_all[index], dtype=float)
        if not (len(det_ids) == len(t0_values) == len(positions)):
            raise ValueError(
                "SimEfield det_id/det_pos_shc/t_0 length mismatch at event "
                f"{index} ({len(det_ids)}/{len(positions)}/{len(t0_values)})"
            )

        du_ids: List[str] = []
        time_map: Dict[str, float] = {}
        det_pos: Dict[str, List[float]] = {}
        for det_id, t0, pos in zip(det_ids, t0_values, positions):
            du_id_str = str(int(det_id))
            du_ids.append(du_id_str)
            time_map[du_id_str] = float(t0)
            det_pos[du_id_str] = as_vector(pos)

        # Shower truth block (names aligned with the CoREAS reader).
        truth: Dict[str, Any] = {}
        for out_key, (branch_name, kind) in _SIMSHOWER_FIELDS.items():
            truth[out_key] = _COERCERS[kind](shower_arrays[branch_name][index])

        # ZHAireS ``shower_azimuth`` describes the *propagation* direction
        # (away from the source), whereas the CoREAS truth ``azimuth`` and the
        # PWM/SWM exported arrival direction (detector -> source) share the
        # opposite convention, differing by 180 deg. Rotate it so the exported
        # truth azimuth matches the reconstruction convention. ``zenith`` needs
        # no flip: the arrival/propagation directions only differ in their
        # horizontal components.
        if truth.get("azimuth") is not None:
            truth["azimuth"] = (float(truth["azimuth"]) + 180.0) % 360.0

        # Event-level SimEfield config scalars (``field_sim`` omitted).
        efield_config: Dict[str, Any] = {}
        for branch_name in _EFIELD_CONFIG_BRANCHES:
            value = efield_arrays[branch_name][index]
            if branch_name in _EFIELD_CONFIG_STR_BRANCHES:
                efield_config[branch_name] = as_str_list(value)
            else:
                efield_config[branch_name] = as_float(value)

        gps_epoch = _STARSHAPE_EPOCH + index
        # StarShape embeds no wall-clock date. Anchor every event on the synthetic
        # epoch plus its zero-based index (one second apart) so that the datetime
        # axis is monotonic and plotting stays meaningful.
        event_datetime = datetime.datetime.utcfromtimestamp(gps_epoch).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        payload: dict = {
            "run_number": 0,
            "event_number": shower_evt_id,
            "datetime": event_datetime,
            "gps_time": int(gps_epoch),
            "index": index,
            "du_id": du_ids,
            "time": time_map,
            "det_pos": det_pos,
            "truth": truth,
            "efield_config": efield_config,
        }

        # Optional per-antenna amplitude blocks (only with --with-signal).
        if with_signal:
            efield_p2p = np.asarray(efield_p2p_all[index], dtype=float).reshape(-1)
            signal_p2p = np.asarray(signal_p2p_all[index], dtype=float).reshape(-1)
            if len(efield_p2p) != len(du_ids):
                raise ValueError(
                    "SimEfield p2p length mismatch at event "
                    f"{index} ({len(efield_p2p)} vs {len(du_ids)})"
                )
            if len(signal_p2p) != len(du_ids):
                raise ValueError(
                    "SimSignal p2p length mismatch at event "
                    f"{index} ({len(signal_p2p)} vs {len(du_ids)})"
                )
            payload["efield"] = {
                du_id: float(v) for du_id, v in zip(du_ids, efield_p2p)
            }
            payload["signal"] = {
                du_id: float(v) for du_id, v in zip(du_ids, signal_p2p)
            }

        payloads.append(payload)

    return payloads, os.path.basename(file_path)


build_yaml = shared_build_yaml


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a StarShape ROOT file and write a matching YAML file"
    )
    parser.add_argument("file_path", help="Path to the StarShape ROOT file")
    parser.add_argument(
        "--date",
        default="",
        help="Output date sub-directory (default: empty)",
    )
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
        help="Also emit per-antenna ``efield``/``signal`` amplitude blocks",
    )
    parser.add_argument(
        "--channel",
        choices=["F", "X", "Y", "Z", "XY"],
        default="XY",
        help="Channel suffix for the output file name when --with-signal is set",
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
    channel: str = "XY",
) -> bool:
    """Read one StarShape ROOT file and write one YAML output file."""
    try:
        out_dir = os.path.join(out_dir_base, date) if date else out_dir_base
        mkdir(out_dir)

        logger.info(f"Processing StarShape file: {file_path}")
        payloads, base_name = read_starshape_file(
            file_path, max_events=max_events, with_signal=with_signal
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
        channel=args.channel,
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
