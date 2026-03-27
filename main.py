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
from find_event import plane_wave_model as fe_pwm
from find_event import spherical_wave_model as fe_swm


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
            "run_number": payload.get("run_number", None),
            "event_number": payload.get("event_number", None),
            "datetime": payload.get("datetime", None),
            "gps_time": payload.get("gps_time", None),
            "file": payload.get("file", None),
            "index": payload.get("index", None),
        }
    return metadata


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
        default="_gp65_rtksort.txt",
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
        "--run-matching",
        dest="run_matching",
        action="store_true",
        help="Run the matching stage explicitly.",
    )
    parser.add_argument(
        "--run-pwm",
        dest="run_pwm",
        action="store_true",
        help="Run the PWM stage explicitly.",
    )
    parser.add_argument(
        "--run-swm",
        dest="run_swm",
        action="store_true",
        help="Run the SWM stage explicitly.",
    )
    return parser.parse_args(argv[1:])


def _resolve_selected_stages(run_matching, run_pwm, run_swm):
    """Resolve stage selection from three CLI flags."""
    if not any([run_matching, run_pwm, run_swm]):
        return {
            "matching": True,
            "pwm": True,
            "swm": True,
        }
    return {
        "matching": run_matching,
        "pwm": run_pwm,
        "swm": run_swm,
    }


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


def run_matching_stage(
    matching_file,
    metadata,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
):
    """Ensure matching-stage data is available in memory and cache."""
    if state["times"]:
        return state, False

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
        else:
            logger.warning(
                f"Invalid matching cache payload in {matched_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(matched_file):
            logger.info(f"Force recompute enabled, ignoring cache: {matched_file}")

        # Load raw data from matching_file
        raw_data = _read_yaml_dict(matching_file)
        times_input = {}
        signals_input = {}
        for key, data in raw_data.items():
            if not isinstance(data, dict):
                continue
            times_input[key] = data["time"]
            signals_input[key] = data.get("signal", None)

        times, signals, du_ids = fe_mt.optimized_read_matching_times_graph(
            times_input,
            signals_input,
            det_pos_file,
            min_detectors=5,
            speed_of_light_tolerance=1.05,
            force_recompute=force_recompute,
        )

        if len(times) < 1:
            logger.warning("No events after filtering, skipping subsequent stages.")
            return state, True

        results = {}
        for key in times.keys():
            meta = metadata[key] if key in metadata else {}
            signal = signals[key] if signals is not None else None
            results[key] = {
                "run_number": meta.get("run_number", None),
                "event_number": meta.get("event_number", None),
                "datetime": meta.get("datetime", None),
                "gps_time": meta.get("gps_time", None),
                "du_id": du_ids[key],
                "file": meta.get("file", None),
                "index": meta.get("index", None),
                "time": times[key],
            }
            if with_signal:
                results[key]["signal"] = signal

        _write_yaml_dict(matched_file, results)
        _write_cache_meta(matched_meta_file, expected_meta)
        matching_computed = True

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

    logger.info(f"Number of events after reading and filtering: {len(state['times'])}")
    logger.info(f"{matched_file} has been written with matched results.")
    return state, matching_computed


def run_pwm_stage(
    matching_file,
    detector_positions,
    det_pos_file,
    with_signal,
    force_recompute,
    state,
    fig_prefix,
    matching_computed,
):
    """Ensure PWM-stage data is available in memory and cache."""
    pwm_fitted_file = matching_file.replace(".yaml", "_PWM.yaml")
    pwm_meta_file = _meta_file_for(pwm_fitted_file)
    matched_file = matching_file.replace(".yaml", "_matched.yaml")
    pwm_computed = False
    expected_meta = _build_stage_cache_meta(
        "pwm",
        {
            "matched_file": _build_file_signature(matched_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
        },
    )

    use_cache = os.path.exists(pwm_fitted_file) and not force_recompute and not matching_computed
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
                state["directions"][key] = np.array([result["x"], result["y"], result["z"]])
                state["chi_squares"][key] = result.get("chi_square", None)
        else:
            logger.warning(
                f"Invalid PWM cache payload in {pwm_fitted_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(pwm_fitted_file):
            logger.info(f"Force recompute enabled, ignoring cache: {pwm_fitted_file}")

        directions, zeniths, azimuths, chi_squares = fe_pwm.plane_wave_model(
            state["times"],
            state["signals"],
            detector_positions,
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

        _write_yaml_dict(pwm_fitted_file, results)
        _write_cache_meta(pwm_meta_file, expected_meta)
        pwm_computed = True

        for key, result in results.items():
            state["azimuths"][key] = result.get("azimuth", None)
            state["zeniths"][key] = result.get("zenith", None)
            state["directions"][key] = np.array([result["x"], result["y"], result["z"]])
            state["chi_squares"][key] = result.get("chi_square", None)

    logger.info(f"Number of events after plane wave fitting: {len(state['times'])}")
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
):
    """Ensure SWM-stage data is available in memory and cache."""
    swm_fitted_file = matching_file.replace(".yaml", "_SWM.yaml")
    swm_meta_file = _meta_file_for(swm_fitted_file)
    pwm_fitted_file = matching_file.replace(".yaml", "_PWM.yaml")
    expected_meta = _build_stage_cache_meta(
        "swm",
        {
            "pwm_file": _build_file_signature(pwm_fitted_file),
            "det_pos_file": _build_file_signature(det_pos_file),
            "with_signal": bool(with_signal),
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
        else:
            logger.warning(
                f"Invalid SWM cache payload in {swm_fitted_file}: {reason}; falling back to recompute."
            )
            use_cache = False

    if not use_cache:
        if force_recompute and os.path.exists(swm_fitted_file):
            logger.info(f"Force recompute enabled, ignoring cache: {swm_fitted_file}")
        directions, chi_squares = fe_swm.spherical_wave_model(
            state["times"],
            state["signals"],
            detector_positions,
            state["directions"],
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
                "azimuth": float(state["azimuths"][key]),
                "zenith": float(state["zeniths"][key]),
                "x": float(direction[0]),
                "y": float(direction[1]),
                "z": float(direction[2]),
                "chi_square": float(chi_squares[key]),
            }
            if with_signal:
                results[key]["signal"] = state["signals"][key]

        _write_yaml_dict(swm_fitted_file, results)
        _write_cache_meta(swm_meta_file, expected_meta)

        for key, result in results.items():
            state["chi_squares"][key] = result.get("chi_square", None)
            state["directions"][key] = np.array([result["x"], result["y"], result["z"]])

    logger.info(f"Number of events after spherical wave fitting: {len(state['times'])}")
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
    run_matching=False,
    run_pwm=False,
    run_swm=False,
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

    if not os.path.exists(det_pos_file):
        logger.error(f"Detector position file does not exist: {det_pos_file}")
        return 2

    try:
        detector_positions = load_data_from_file(det_pos_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load detector position file {det_pos_file}: {exc}")
        return 2
    metadata = _load_event_metadata_map(matching_file)

    # Ensure plots are written into the same directory as the input matching file
    # If matching_file is relative, convert to absolute first
    output_dir = os.path.dirname(os.path.abspath(matching_file))
    if fig_name is None:
        fig_name = os.path.basename(matching_file).replace(".yaml", "")
    fig_prefix = os.path.join(output_dir, fig_name)
    selected_stages = _resolve_selected_stages(run_matching, run_pwm, run_swm)
    state = _new_stage_state()
    matching_computed = False
    pwm_computed = False

    if selected_stages["matching"] or selected_stages["pwm"] or selected_stages["swm"]:
        state, matching_computed = run_matching_stage(
            matching_file,
            metadata,
            det_pos_file,
            with_signal,
            force_recompute,
            state,
        )

    if not state["times"]:
        return 0

    if selected_stages["pwm"] or selected_stages["swm"]:
        state, pwm_computed = run_pwm_stage(
            matching_file,
            detector_positions,
            det_pos_file,
            with_signal,
            force_recompute,
            state,
            fig_prefix,
            matching_computed,
        )

    if selected_stages["swm"]:
        run_swm_stage(
            matching_file,
            detector_positions,
            det_pos_file,
            with_signal,
            force_recompute,
            state,
            fig_prefix,
            pwm_computed,
        )

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
            run_matching=args.run_matching,
            run_pwm=args.run_pwm,
            run_swm=args.run_swm,
        )
    )
