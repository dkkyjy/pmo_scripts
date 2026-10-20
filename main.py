"""
Main program entry module

This module is responsible for:
1. Parsing command line arguments
2. Coordinating various submodules to execute the data processing workflow
3. Generating final analysis results and visualization charts
"""

# Standard libraries
import os
import sys
import json
import yaml
import argparse
import numpy as np
from logger_config import logger

# Local modules (find_event package)
from find_event.io import load_data_from_file
from find_event import plotting as fe_plot
from find_event import estimation as fe_est
from find_event import matching_times_graph as fe_mt
from find_event import plane_wave_model_gradient as fe_pwm
from find_event import spherical_wave_model_nopenal as fe_swm
from find_event import time_difference_fingerprint as fe_tdf


def _load_event_metadata_map(matching_file):
    """Load source event metadata map keyed by top-level event key."""
    if not os.path.exists(matching_file):
        return {}

    with open(matching_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(data, dict):
        return {}

    metadata = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        metadata[key] = {
            "du_id": payload.get("du_id", None),
            "time": payload.get("time", None),
            "signal": payload.get("signal", None),
            "run_number": payload.get("run_number", None),
            "event_number": payload.get("event_number", None),
            "datetime": payload.get("datetime", None),
            "gps_time": payload.get("gps_time", None),
            "file": payload.get("file", None),
            "index": payload.get("index", None),
        }
    return metadata


def _load_det_pos_by_event(matching_file):
    """Load per-event detector coordinates embedded in a matching YAML file.

    StarShape files carry no static detector-coordinate file: each event embeds
    its own ``det_pos`` mapping. Returns ``{event_key: {det_id(int): np.array}}``
    when at least one event has an embedded ``det_pos``, otherwise ``None`` so
    callers fall back to the static ``det_pos_file`` layout.
    """
    if not os.path.exists(matching_file):
        return None

    with open(matching_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(data, dict):
        return None

    by_event = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        det_pos = payload.get("det_pos", None)
        if not isinstance(det_pos, dict) or not det_pos:
            continue
        by_event[key] = {
            int(det_id): np.asarray(position, dtype=float)
            for det_id, position in det_pos.items()
        }

    return by_event or None


# =========================
# Command line argument parsing (modular entry)
# =========================
def parse_args(argv):
    """
    Parse command line arguments:
      positional:
        matching_file: Input matching file path
      optional:
        --fig_name: Image figure name and output prefix
        --with-signal: Whether to process signal amplitude related processes
        --det-pos: Detector position file path (if not provided, use the default path in the script)
    """
    parser = argparse.ArgumentParser(
        description="Find events and reconstruct directions"
    )
    parser.add_argument("matching_file", help="Path to matching file")
    parser.add_argument("--fig_name", help="Image figure name and output prefix")
    parser.add_argument(
        "--det-pos",
        help="Detector position file (optional), if not specified, use the default path in the script",
        default="position/_gp65_rtksort_DU7_1029.txt",
    )
    parser.add_argument(
        "--with-signal",
        dest="with_signal",
        action="store_true",
        help="Whether to process signal amplitude related processes (such as signal amplitude fitting, signal related plotting, etc.). Adding this parameter will enable signal-related reading and analysis logic, disabled by default.",
    )
    parser.add_argument(
        "--force-recompute",
        dest="force_recompute",
        action="store_true",
        help="Ignore existing _matched/_PWM/_SWM cache files and recompute.",
    )
    parser.add_argument(
        "--skip-matching",
        dest="skip_matching",
        default=False,
        action="store_true",
        help="Skip the matching stage, assuming precomputed _matched cache files exist.",
    )
    parser.add_argument(
        "--skip-fingerprint",
        dest="skip_fingerprint",
        default=False,
        action="store_true",
        help="Skip the fingerprint stage and load source YAML payload directly.",
    )
    parser.add_argument(
        "--min-detectors",
        dest="min_detectors",
        type=int,
        default=5,
        help="Minimum number of detectors required in matching stage (default: 5).",
    )
    parser.add_argument(
        "--skip-footprint",
        dest="skip_footprint",
        default=False,
        action="store_true",
        help="Skip the customized shower-footprint shape filter between PWM and SWM (enabled by default).",
    )
    parser.add_argument(
        "--skip-ringcut",
        dest="skip_ringcut",
        default=False,
        action="store_true",
        help="Skip the customized d_max ring cut between PWM and SWM (enabled by default).",
    )
    parser.add_argument(
        "--skip-offset",
        dest="skip_offset",
        default=False,
        action="store_true",
        help="Skip the time/position offset calibration from the calibration CSV (enabled by default).",
    )
    parser.add_argument(
        "--offset-csv",
        dest="offset_csv",
        default=None,
        help="Path to the offset calibration CSV (default: ./calibration/jul24_locked_xyz_additive_utc.csv).",
    )
    return parser.parse_args(argv[1:])


def _new_stage_state():
    """Create an empty mutable state container shared across stages."""
    return {
        "times": {},
        "signals": {},
        "du_ids": {},
        "run_numbers": {},
        "event_numbers": {},
        "files": {},
        "index": {},
        "datetimes": {},
        "gps_times": {},
        "azimuths": {},
        "zeniths": {},
        "directions": {},
        "chi_squares": {},
    }


MATCHING_STATE_FIELDS = (
    "times",
    "signals",
    "du_ids",
    "run_numbers",
    "event_numbers",
    "files",
    "index",
    "datetimes",
    "gps_times",
)

PWM_STATE_FIELDS = MATCHING_STATE_FIELDS + (
    "azimuths",
    "zeniths",
    "directions",
    "chi_squares",
)

SWM_STATE_FIELDS = PWM_STATE_FIELDS


def _reset_state_fields(state, fields):
    """Reset selected state fields to empty dictionaries."""
    for field in fields:
        state[field] = {}


def _canonicalize_state_keys(state, canonical_keys, fields, stage_name):
    """Align selected state fields to one canonical key set."""
    ordered_keys = list(canonical_keys)
    key_set = set(ordered_keys)

    for field in fields:
        current = state.get(field, {})
        if not isinstance(current, dict):
            current = {}

        orphan_keys = sorted(set(current.keys()) - key_set)
        if orphan_keys:
            logger.debug(
                "Canonicalization dropped {} orphan keys in stage '{}' field '{}': {}",
                len(orphan_keys),
                stage_name,
                field,
                orphan_keys,
            )

        state[field] = {key: current.get(key, None) for key in ordered_keys}

    return state


def _read_yaml_dict(file_path):
    """Load YAML mapping from disk, returning empty dict for null payloads."""
    with open(file_path, "r") as file_obj:
        return yaml.load(file_obj, Loader=yaml.FullLoader) or {}


def _write_yaml_dict(file_path, payload):
    """Write mapping payload to YAML file."""
    with open(file_path, "w") as file_obj:
        yaml.dump(payload, file_obj)


def _meta_file_for(cache_file):
    """Return the sidecar meta file path for a cache file."""
    return f"{cache_file}.meta.json"


def _build_file_signature(file_path):
    """Build a basic file signature used for cache validation."""
    absolute_path = os.path.abspath(file_path)
    exists = os.path.exists(absolute_path)
    if not exists:
        return {
            "path": absolute_path,
            "exists": False,
        }
    file_stat = os.stat(absolute_path)
    return {
        "path": absolute_path,
        "exists": True,
        "size": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
    }


def _build_stage_cache_meta(stage, inputs):
    """Build cache metadata payload for one stage."""
    return {
        "schema_version": 1,
        "stage": stage,
        "inputs": inputs,
    }


def _read_cache_meta(meta_file):
    """Read cache metadata from disk with resilient parsing."""
    if not os.path.exists(meta_file):
        return None
    try:
        with open(meta_file, "r") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            logger.warning(f"Invalid cache meta payload type in {meta_file}, expected object.")
            return None
        return payload
    except Exception as exc:
        logger.warning(f"Failed to read cache meta file {meta_file}: {exc}")
        return None


def _write_cache_meta(meta_file, payload):
    """Write cache metadata sidecar file."""
    with open(meta_file, "w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def _cache_meta_is_valid(meta_file, expected_meta):
    """Check whether cache metadata matches the expected signature."""
    cached_meta = _read_cache_meta(meta_file)
    if cached_meta is None:
        return False, "missing-or-unreadable"
    if cached_meta != expected_meta:
        return False, "signature-mismatch"
    return True, "ok"


def _required_fields_for_stage(stage, with_signal):
    """Return required cache fields for one stage payload row."""
    base_fields = {
        "run_number",
        "event_number",
        "datetime",
        "gps_time",
        "du_id",
        "file",
        "index",
        "time",
    }
    if stage == "matching":
        required = set(base_fields)
    elif stage in ("fingerprint",):
        required = set(base_fields)
    elif stage in ("pwm", "swm"):
        required = base_fields | {
            "chi_square",
            "zenith",
            "azimuth",
            "x",
            "y",
            "z",
        }
    else:
        raise ValueError(f"Unsupported stage for cache validation: {stage}")

    if with_signal:
        required.add("signal")
    return required


def _validate_stage_cache_payload(stage, payload, with_signal):
    """Validate stage cache payload shape and required fields."""
    if not isinstance(payload, dict):
        return False, "top-level payload is not a dict"

    required_fields = _required_fields_for_stage(stage, with_signal)
    for event_key, row in payload.items():
        if not isinstance(row, dict):
            return False, f"row is not a dict for event {event_key}"
        missing = sorted(required_fields - set(row.keys()))
        if missing:
            return False, f"missing fields {missing} for event {event_key}"

    return True, "ok"


def _plot_pwm_if_needed(state, fig_prefix):
    """Plot PWM outputs while isolating plotting failures from cache generation."""
    if len(state["times"]) < 2:
        logger.warning("No events after filtering, skipping PWM plotting.")
        return
    try:
        fe_plot.plot_reconstructed_positions_PWM(
            state["datetimes"],
            state["directions"],
            state["chi_squares"],
            fig_prefix + "_PWM",
        )
        fe_plot.plot_fitting_parameters_PWM(
            state["datetimes"],
            state["directions"],
            state["chi_squares"],
            "PWM",
            fig_prefix + "_PWM",
        )
    except Exception as exc:
        logger.warning(f"PWM plotting failed after cache generation: {exc}")


def _plot_swm_if_needed(state, fig_prefix):
    """Plot SWM outputs while isolating plotting failures from cache generation."""
    if len(state["times"]) < 2:
        logger.warning("No events after filtering, skipping SWM plotting.")
        return
    try:
        fe_plot.plot_reconstructed_positions_SWM(
            state["datetimes"],
            state["directions"],
            state["chi_squares"],
            fig_prefix + "_SWM",
        )
        fe_plot.plot_fitting_parameters_SWM(
            state["datetimes"],
            state["directions"],
            state["chi_squares"],
            "SWM",
            fig_prefix + "_SWM",
        )
    except Exception as exc:
        logger.warning(f"SWM plotting failed after cache generation: {exc}")


def skip_matching_stage(metadata, with_signal, state):
    """Skip the matching stage, assuming precomputed _matched cache files exist."""
    _reset_state_fields(state, MATCHING_STATE_FIELDS)

    for key, data in metadata.items():
        if not isinstance(data, dict):
            continue
        state["times"][key] = data["time"]
        state["signals"][key] = data.get("signal", None) if with_signal else None
        state["du_ids"][key] = data.get("du_id", None)
        state["event_numbers"][key] = data.get("event_number", None)
        state["index"][key] = data.get("index", None)
        state["run_numbers"][key] = data.get("run_number", None)
        state["datetimes"][key] = data.get("datetime", None)
        state["gps_times"][key] = data.get("gps_time", None)
        state["files"][key] = data.get("file", None)

    _canonicalize_state_keys(
        state,
        state["times"].keys(),
        MATCHING_STATE_FIELDS,
        "skip-matching",
    )

    return state, False


def skip_fingerprint_stage(matching_file, with_signal, state):
    """Skip fingerprint stage by loading _matched.yaml or original matching YAML."""
    matched_file = matching_file.replace(".yaml", "_matched.yaml")
    source_file = matched_file if os.path.exists(matched_file) else matching_file
    metadata = _load_event_metadata_map(source_file)

    _reset_state_fields(state, MATCHING_STATE_FIELDS)
    for key, data in metadata.items():
        if not isinstance(data, dict):
            continue
        state["times"][key] = data.get("time", None)
        state["signals"][key] = data.get("signal", None) if with_signal else None
        state["du_ids"][key] = data.get("du_id", None)
        state["event_numbers"][key] = data.get("event_number", None)
        state["index"][key] = data.get("index", None)
        state["run_numbers"][key] = data.get("run_number", None)
        state["datetimes"][key] = data.get("datetime", None)
        state["gps_times"][key] = data.get("gps_time", None)
        state["files"][key] = data.get("file", None)

    _canonicalize_state_keys(
        state,
        state["times"].keys(),
        MATCHING_STATE_FIELDS,
        "skip-fingerprint",
    )

    return state, False


def run_matching_stage(
    matching_file,
    metadata,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
    min_detectors=5,
):
    """Ensure matching-stage data is available in memory and cache."""
    matched_file = matching_file.replace(".yaml", "_matched.yaml")
    matched_meta_file = _meta_file_for(matched_file)
    matching_computed = False
    expected_meta = _build_stage_cache_meta(
        "matching",
        {
            "matching_file": _build_file_signature(matching_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
        },
    )

    use_cache = os.path.exists(matched_file) and not force_recompute
    if use_cache:
        is_valid, reason = _cache_meta_is_valid(matched_meta_file, expected_meta)
        if not is_valid:
            logger.info(
                f"Ignoring matched cache due to {reason}: {matched_file}"
            )
            use_cache = False

    if use_cache:
        logger.info(f"Found cached matched file: {matched_file}, loading directly.")
        try:
            results = _read_yaml_dict(matched_file)
            is_payload_valid, reason = _validate_stage_cache_payload(
                "matching", results, with_signal
            )
        except Exception as exc:
            is_payload_valid = False
            reason = f"unreadable-cache: {exc}"

        if is_payload_valid:
            _reset_state_fields(state, MATCHING_STATE_FIELDS)
            for key, result in results.items():
                state["times"][key] = result["time"]
                state["signals"][key] = result.get("signal", None) if with_signal else None
                state["du_ids"][key] = result.get("du_id", None)
                state["event_numbers"][key] = result.get("event_number", None)
                state["index"][key] = result.get("index", None)
                state["run_numbers"][key] = result.get("run_number", None)
                state["datetimes"][key] = result.get("datetime", None)
                state["gps_times"][key] = result.get("gps_time", None)
                state["files"][key] = result.get("file", None)

            _canonicalize_state_keys(
                state,
                results.keys(),
                MATCHING_STATE_FIELDS,
                "run-matching-cache",
            )
            matching_computed = True
        else:
            logger.warning(
                f"Invalid matching cache payload in {matched_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(matched_file):
            logger.info(f"Force recompute enabled, ignoring cache: {matched_file}")

        logger.info('Running matching stage with matching_file={}, det_pos_file={}, with_signal={}', matching_file, det_pos_file, with_signal)
        
        times, signals, du_ids = fe_mt.optimized_read_matching_times_graph(
            state["times"],
            state["signals"],
            det_pos_file,
            min_detectors=min_detectors,
            speed_of_light_tolerance=1.05,
            force_recompute=force_recompute,
        )

        if len(times) < 1:
            return state, False

        results = {}
        for key in times.keys():
            meta = metadata[key] if key in metadata else {}
            signal = signals[key] if signals is not None else None
            results[key] = {
                "run_number": meta.get("run_number", None),
                "event_number": meta.get("event_number", None),
                "datetime": meta.get("datetime", None),
                "gps_time": meta.get("gps_time", None),
                "file": meta.get("file", None),
                "index": meta.get("index", None),
                "du_id": du_ids[key],
                "time": times[key],
            }
            if with_signal:
                results[key]["signal"] = signal

        _write_yaml_dict(matched_file, results)
        _write_cache_meta(matched_meta_file, expected_meta)
        matching_computed = True

        state["times"] = {}
        for key, result in results.items():
            state["times"][key] = result["time"]
            state["du_ids"][key] = result.get("du_id", None)
            state["signals"][key] = result.get("signal", None) if with_signal else None
            state["event_numbers"][key] = result.get("event_number", None)
            state["index"][key] = result.get("index", None)
            state["run_numbers"][key] = result.get("run_number", None)
            state["datetimes"][key] = result.get("datetime", None)
            state["gps_times"][key] = result.get("gps_time", None)
            state["files"][key] = result.get("file", None)

        _canonicalize_state_keys(
            state,
            results.keys(),
            MATCHING_STATE_FIELDS,
            "run-matching-recompute",
        )

    logger.info(f"Number of events after reading and filtering: {len(results)}")
    logger.info(f"{matched_file} has been written with matched results.")
    return state, matching_computed


def run_fingerprint_stage(
    matching_file,
    metadata,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
    matching_computed,
):
    """Ensure fingerprint-stage data is available in memory and cache."""
    fingerprint_file = matching_file.replace(".yaml", "_fingerprint.yaml")
    fingerprint_meta_file = _meta_file_for(fingerprint_file)
    expected_meta = _build_stage_cache_meta(
        "fingerprint",
        {
            "matching_file": _build_file_signature(matching_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
        },
    )

    use_cache = (
        os.path.exists(fingerprint_file)
        and not force_recompute
        and not matching_computed
    )
    if use_cache:
        is_valid, reason = _cache_meta_is_valid(fingerprint_meta_file, expected_meta)
        if not is_valid:
            logger.info("Ignoring fingerprint cache due to {}: {}", reason, fingerprint_file)
            use_cache = False

    if use_cache:
        logger.info("Found cached fingerprint file: {}, loading directly.", fingerprint_file)
        try:
            results = _read_yaml_dict(fingerprint_file)
            is_payload_valid, reason = _validate_stage_cache_payload(
                "fingerprint", results, with_signal
            )
        except Exception as exc:
            is_payload_valid = False
            reason = f"unreadable-cache: {exc}"

        if not is_payload_valid:
            logger.warning(
                "Invalid fingerprint cache payload in {}: {}; falling back to recompute.",
                fingerprint_file,
                reason,
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(fingerprint_file):
            logger.info("Force recompute enabled, ignoring cache: {}", fingerprint_file)

        source_payload = _read_yaml_dict(matching_file)
        if not isinstance(source_payload, dict):
            raise ValueError("Top-level matching payload must be a mapping")

        results = fe_tdf.filter_fixed_sources_from_payload(
            source_payload,
        )

        _write_yaml_dict(fingerprint_file, results)
        _write_cache_meta(fingerprint_meta_file, expected_meta)
        fingerprint_computed = True

    _reset_state_fields(state, MATCHING_STATE_FIELDS)
    for key, result in results.items():
        if not isinstance(result, dict):
            continue
        state["times"][key] = result.get("time", None)
        state["signals"][key] = result.get("signal", None) if with_signal else None
        state["du_ids"][key] = result.get("du_id", None)
        state["event_numbers"][key] = result.get("event_number", None)
        state["index"][key] = result.get("index", None)
        state["run_numbers"][key] = result.get("run_number", None)
        state["datetimes"][key] = result.get("datetime", None)
        state["gps_times"][key] = result.get("gps_time", None)
        state["files"][key] = result.get("file", None)

    _canonicalize_state_keys(
        state,
        state["times"].keys(),
        MATCHING_STATE_FIELDS,
        "run-fingerprint",
    )

    fingerprint_computed = True
    logger.info("Number of events after fingerprint filtering: {}", len(state["times"]))
    logger.info("{} has been written with fingerprint results.", fingerprint_file)
    return state, fingerprint_computed


def run_pwm_stage(
    matching_file,
    detector_positions,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
    fig_prefix,
    fingerprint_computed,
    detector_positions_by_event=None,
):
    """Ensure PWM-stage data is available in memory and cache."""
    pwm_fitted_file = matching_file.replace(".yaml", "_PWM.yaml")
    pwm_meta_file = _meta_file_for(pwm_fitted_file)
    pwm_computed = False
    expected_meta = _build_stage_cache_meta(
        "pwm",
        {
            "matching_file": _build_file_signature(matching_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
        },
    )

    use_cache = os.path.exists(pwm_fitted_file) and not force_recompute and not fingerprint_computed
    if use_cache:
        is_valid, reason = _cache_meta_is_valid(pwm_meta_file, expected_meta)
        if not is_valid:
            logger.info(
                f"Ignoring PWM cache due to {reason}: {pwm_fitted_file}"
            )
            use_cache = False

    if use_cache:
        logger.info(f"Found cached PWM fitted file: {pwm_fitted_file}, loading directly.")
        try:
            results = _read_yaml_dict(pwm_fitted_file)
            is_payload_valid, reason = _validate_stage_cache_payload(
                "pwm", results, with_signal
            )
        except Exception as exc:
            is_payload_valid = False
            reason = f"unreadable-cache: {exc}"

        if is_payload_valid:
            _reset_state_fields(state, PWM_STATE_FIELDS)
            for key, result in results.items():
                state["times"][key] = result["time"]
                state["signals"][key] = result.get("signal", None) if with_signal else None
                state["du_ids"][key] = result.get("du_id", None)
                state["event_numbers"][key] = result.get("event_number", None)
                state["index"][key] = result.get("index", None)
                state["run_numbers"][key] = result.get("run_number", state["run_numbers"].get(key, None))
                state["datetimes"][key] = result.get("datetime", state["datetimes"].get(key, None))
                state["gps_times"][key] = result.get("gps_time", state["gps_times"].get(key, None))
                state["files"][key] = result.get("file", state["files"].get(key, matching_file))
                state["azimuths"][key] = result.get("azimuth", None)
                state["zeniths"][key] = result.get("zenith", None)
                state["directions"][key] = np.array([
                    result.get("x", None),
                    result.get("y", None),
                    result.get("z", None),
                ])
                state["chi_squares"][key] = result.get("chi_square", None)

            _canonicalize_state_keys(
                state,
                state["directions"].keys(),
                PWM_STATE_FIELDS,
                "run-pwm-cache",
            )
        else:
            logger.warning(
                f"Invalid PWM cache payload in {pwm_fitted_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(pwm_fitted_file):
            logger.info(f"Force recompute enabled, ignoring cache: {pwm_fitted_file}")

        _reset_state_fields(
            state,
            (
                "azimuths",
                "zeniths",
                "directions",
                "chi_squares",
            ),
        )

        directions, zeniths, azimuths, chi_squares = fe_pwm.plane_wave_model(
            state["times"],
            detector_positions,
            detector_positions_by_event=detector_positions_by_event,
        )

        results = {}
        for key in directions.keys():
            direction = directions[key]
            results[key] = {
                "run_number": state["run_numbers"].get(key, None),
                "event_number": state["event_numbers"].get(key, None),
                "datetime": state["datetimes"].get(key, None),
                "gps_time": state["gps_times"].get(key, None),
                "du_id": state["du_ids"][key],
                "file": state["files"].get(key, None),
                "index": state["index"].get(key, None),
                "time": state["times"][key],
                "chi_square": chi_squares[key],
                "zenith": zeniths[key],
                "azimuth": azimuths[key],
                "x": float(direction[0]),
                "y": float(direction[1]),
                "z": float(direction[2]),
            }
            if with_signal:
                results[key]["signal"] = state["signals"][key]

        logger.info(f"Plane wave fitting completed for {len(results)} events, writing cache to {pwm_fitted_file}")
        _write_yaml_dict(pwm_fitted_file, results)
        _write_cache_meta(pwm_meta_file, expected_meta)
        pwm_computed = True

        for key, result in results.items():
            state["times"][key] = result["time"]
            state["signals"][key] = result.get("signal", None) if with_signal else None
            state["du_ids"][key] = result.get("du_id", None)
            state["event_numbers"][key] = result.get("event_number", None)
            state["index"][key] = result.get("index", None)
            state["run_numbers"][key] = result.get("run_number", None)
            state["datetimes"][key] = result.get("datetime", None)
            state["gps_times"][key] = result.get("gps_time", None)
            state["files"][key] = result.get("file", None)
            state["azimuths"][key] = result.get("azimuth", None)
            state["zeniths"][key] = result.get("zenith", None)
            state["directions"][key] = np.array([result["x"], result["y"], result["z"]])
            state["chi_squares"][key] = result.get("chi_square", None)

        _canonicalize_state_keys(
            state,
            state["directions"].keys(),
            PWM_STATE_FIELDS,
            "run-pwm-recompute",
        )

    _plot_pwm_if_needed(state, fig_prefix)
    return state, pwm_computed


def run_swm_stage(
    matching_file,
    detector_positions,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
    fig_prefix,
    pwm_computed,
    detector_positions_by_event=None,
    extra_cache_inputs=None,
):
    """Ensure SWM-stage data is available in memory and cache."""
    swm_fitted_file = matching_file.replace(".yaml", "_SWM.yaml")
    swm_meta_file = _meta_file_for(swm_fitted_file)
    expected_meta = _build_stage_cache_meta(
        "swm",
        {
            "matching_file": _build_file_signature(matching_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
            **(extra_cache_inputs or {}),
        },
    )

    use_cache = os.path.exists(swm_fitted_file) and not force_recompute and not pwm_computed
    if use_cache:
        is_valid, reason = _cache_meta_is_valid(swm_meta_file, expected_meta)
        if not is_valid:
            logger.info(
                f"Ignoring SWM cache due to {reason}: {swm_fitted_file}"
            )
            use_cache = False

    if use_cache:
        logger.info(f"Found cached SWM fitted file: {swm_fitted_file}, loading directly.")
        try:
            results = _read_yaml_dict(swm_fitted_file)
            is_payload_valid, reason = _validate_stage_cache_payload(
                "swm", results, with_signal
            )
        except Exception as exc:
            is_payload_valid = False
            reason = f"unreadable-cache: {exc}"

        if is_payload_valid:
            _reset_state_fields(state, SWM_STATE_FIELDS)
            for key, result in results.items():
                state["times"][key] = result["time"]
                state["signals"][key] = result.get("signal", None) if with_signal else None
                state["chi_squares"][key] = result.get("chi_square", None)
                state["du_ids"][key] = result.get("du_id", None)
                state["event_numbers"][key] = result.get("event_number", None)
                state["index"][key] = result.get("index", None)
                state["run_numbers"][key] = result.get("run_number", state["run_numbers"].get(key, None))
                state["datetimes"][key] = result.get("datetime", state["datetimes"].get(key, None))
                state["gps_times"][key] = result.get("gps_time", state["gps_times"].get(key, None))
                state["files"][key] = result.get("file", state["files"].get(key, matching_file))
                state["azimuths"][key] = result.get("azimuth", None)
                state["zeniths"][key] = result.get("zenith", None)
                state["directions"][key] = np.array([
                    result.get("x", None),
                    result.get("y", None),
                    result.get("z", None),
                ])

            _canonicalize_state_keys(
                state,
                state["directions"].keys(),
                SWM_STATE_FIELDS,
                "run-swm-cache",
            )
        else:
            logger.warning(
                f"Invalid SWM cache payload in {swm_fitted_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(swm_fitted_file):
            logger.info(f"Force recompute enabled, ignoring cache: {swm_fitted_file}")

        seed_directions = dict(state["directions"])
        _reset_state_fields(state, ("directions", "chi_squares"))
        directions, chi_squares, new_matches = fe_swm.spherical_wave_model(
            state["times"],
            detector_positions,
            seed_directions,
            detector_positions_by_event=detector_positions_by_event,
            prune_du1=True,
            prune_du2=False,
        )

        results = {}
        for key, new_match in new_matches.items():
            results[key] = {
                "run_number": state["run_numbers"].get(key, None),
                "event_number": state["event_numbers"].get(key, None),
                "datetime": state["datetimes"].get(key, None),
                "gps_time": state["gps_times"].get(key, None),
                "du_id": state["du_ids"][key],
                "file": state["files"].get(key, None),
                "index": state["index"].get(key, None),
                "time": new_match,
                "azimuth": float(state["azimuths"][key]),
                "zenith": float(state["zeniths"][key]),
                "x": float(directions[key][0]),
                "y": float(directions[key][1]),
                "z": float(directions[key][2]),
                "chi_square": float(chi_squares[key]),
            }
            if with_signal:
                results[key]["signal"] = state["signals"][key]

        logger.info(f"Spherical wave fitting completed for {len(results)} events, writing cache to {swm_fitted_file}")
        _write_yaml_dict(swm_fitted_file, results)
        _write_cache_meta(swm_meta_file, expected_meta)

        for key, result in results.items():
            state["chi_squares"][key] = result.get("chi_square", None)
            state["directions"][key] = np.array([result["x"], result["y"], result["z"]])

        _canonicalize_state_keys(
            state,
            state["directions"].keys(),
            SWM_STATE_FIELDS,
            "run-swm-recompute",
        )

    _plot_swm_if_needed(state, fig_prefix)
    return state


# =========================
# Main entry point (main)
# =========================
def main(
    matching_file,
    fig_name,
    with_signal,
    det_pos_file,
    force_recompute=False,
    skip_matching=False,
    skip_fingerprint=False,
    skip_footprint=False,
    skip_ringcut=False,
    skip_offset=False,
    offset_csv=None,
):
    """
    Main workflow:
      1. Load detector coordinates (from det_pos_file or script default path)
      2. Perform necessary rotation/offset on coordinates
      3. Read and filter matching data (call optimized_read_matching_times)
      4. Use plane wave model and spherical wave model for fitting and plotting, save results
    """
    # Configure loguru to replace the root logging configuration
    logger.info(f"Start processing: {matching_file}")

    # 数据形态判别：trigger 数据的坐标来自静态 det_pos 文件，其 gps_time 是真实
    # 时间码，需施加时间/位置标定；coreas/starshape 的坐标逐事件内嵌于 matching
    # YAML 的 ``det_pos``，gps_time 为合成值，不做标定（--skip-offset 亦可跳过）。
    detector_positions_by_event = _load_det_pos_by_event(matching_file)
    is_trigger_layout = detector_positions_by_event is None

    if is_trigger_layout:
        if skip_offset:
            logger.info("Offset calibration skipped (--skip-offset)")
        else:
            # Step 0: 时间/探测器位置标定（trigger 数据默认启用；--skip-offset 跳过）
            # 独立子程序见 time_position_offset.py：按每个事例的 gps_time 选最接近的
            # 标定时间窗，正向修正 DU 触发时间(整数 ns)与探测器坐标；产出新的 YAML
            # 与 det-pos 文件（其文件签名自动使既有缓存失效，无需改动任何现有函数）。
            from time_position_offset import apply_calibration, DEFAULT_OFFSET_CSV
            matching_file, det_pos_file, _offset_stats = apply_calibration(
                matching_file, det_pos_file, offset_csv or DEFAULT_OFFSET_CSV)
            logger.info(f"Calibrated inputs: matching_file={matching_file}, det_pos_file={det_pos_file}")
    else:
        logger.info(
            "Loaded per-event detector coordinates from {} for {} events "
            "(simulation layout: offset calibration not applied)",
            matching_file,
            len(detector_positions_by_event),
        )

    if os.path.exists(det_pos_file):
        try:
            detector_positions = load_data_from_file(det_pos_file)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"Failed to load detector position file {det_pos_file}: {exc}")
            return 2
    elif detector_positions_by_event is not None:
        # No static file, but every event carries its own layout: seed an empty
        # fallback map (used only if an event is missing from the by-event map).
        logger.info(
            "Detector position file {} not found; using per-event coordinates only.",
            det_pos_file,
        )
        detector_positions = {}
    else:
        logger.error(f"Detector position file does not exist: {det_pos_file}")
        return 2
    metadata = _load_event_metadata_map(matching_file)

    # Ensure plots are written into the same directory as the input matching file
    # If matching_file is relative, convert to absolute first
    output_dir = os.path.dirname(os.path.abspath(matching_file))
    if fig_name is None:
        fig_name = os.path.basename(matching_file).replace(".yaml", "")
    fig_prefix = os.path.join(output_dir, fig_name)
    state = _new_stage_state()
    matching_computed = False
    fingerprint_computed = False
    pwm_computed = False
    
    state, matching_computed = skip_matching_stage(metadata, with_signal, state)
    if skip_matching:
        logger.info("Skipping matching stage")
        matching_computed=True
    else:
        logger.info('Running matching stage with matching_file {}', matching_file)
        state, matching_computed = run_matching_stage(
            matching_file,
            metadata,
            det_pos_file,
            with_signal,
            force_recompute,
            state,
            min_detectors=args.min_detectors,
        )
        
        matching_file = matching_file.replace(".yaml", "_matched.yaml")
        
        if not matching_computed:
            logger.warning("No events after matching stage, skipping subsequent stages.")
            return 0
        logger.info('Finished matching stage')
        
    state, fingerprint_computed = skip_fingerprint_stage(
        matching_file,
        with_signal,
        state,
    )
    if skip_fingerprint:
        logger.info("Skipping fingerprint stage")
        fingerprint_computed = True
    else:
        logger.info('Running fingerprint stage with matching_file={}', matching_file)
        state, fingerprint_computed = run_fingerprint_stage(
            matching_file,
            metadata,
            det_pos_file,
            with_signal,
            force_recompute,
            state,
            matching_computed,
        )
        
        matching_file = matching_file.replace(".yaml", "_fingerprint.yaml")
        logger.info('Finished fingerprint stage')
        
    logger.info('Running PWM stage with matching_file={}', matching_file)
    state, pwm_computed = run_pwm_stage(
        matching_file,
        detector_positions,
        det_pos_file,
        with_signal,
        force_recompute,
        state,
        fig_prefix,
        fingerprint_computed,
        detector_positions_by_event=detector_positions_by_event,
    )
    logger.info('Finished PWM stage')

    # ===== 定制化筛选子程序（默认启用，可 --skip-footprint / --skip-ringcut 关闭）=====
    # 独立子程序位于 find_event/shower_footprint_filter.py：
    # 仅过滤内存中的 state（PWM 与 SWM 之间），不改任何现有阶段逻辑与缓存文件约定。
    if skip_footprint and skip_ringcut:
        logger.info("Customized cuts skipped (both --skip-footprint and --skip-ringcut)")
        cut_meta = None
    else:
        from find_event.shower_footprint_filter import apply_customized_cuts
        state = apply_customized_cuts(
            state,
            detector_positions,
            fig_prefix,
            run_footprint=not skip_footprint,
            run_ring=not skip_ringcut,
        )
        # 记录判据配置到 SWM 缓存签名，避免不同判据配置间错误复用缓存
        cut_meta = {
            "customized_cuts": {
                "footprint": bool(not skip_footprint),
                "ring": bool(not skip_ringcut),
            }
        }
        if not state["directions"]:
            logger.warning("No events left after customized cuts; skipping subsequent stages.")
            return 0
    
    logger.info('Running SWM stage with matching_file={}', matching_file)
    run_swm_stage(
        matching_file,
        detector_positions,
        det_pos_file,
        with_signal,
        force_recompute,
        state,
        fig_prefix,
        pwm_computed,
        detector_positions_by_event=detector_positions_by_event,
        extra_cache_inputs=cut_meta,
    )
    logger.info('Finished SWM stage')

    logger.info('Running PWM stage with matching_file={}', matching_file)
    state, pwm_computed = run_pwm_stage(
        matching_file,
        detector_positions,
        det_pos_file,
        with_signal,
        force_recompute,
        state,
        fig_prefix,
        fingerprint_computed,
        detector_positions_by_event=detector_positions_by_event,
    )
    logger.info('Finished PWM stage')
    
    return 0


# =========================
# CLI entry: Compatible with original call (matching_file, fig_name, with_signal) and new --det-pos parameter
# =========================
if __name__ == "__main__":
    args = parse_args(sys.argv)
    raise SystemExit(
        main(
            args.matching_file,
            args.fig_name,
            args.with_signal,
            det_pos_file=args.det_pos,
            force_recompute=args.force_recompute,
            skip_matching=args.skip_matching,
            skip_fingerprint=args.skip_fingerprint,
            skip_footprint=args.skip_footprint,
            skip_ringcut=args.skip_ringcut,
            skip_offset=args.skip_offset,
            offset_csv=args.offset_csv,
        )
    )
