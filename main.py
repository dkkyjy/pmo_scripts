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
import yaml
import argparse
import numpy as np
from logger_config import logger

# Local modules (find_event package)
from find_event.io import load_data_from_file
from find_event import plotting as fe_plot
from find_event import estimation as fe_est
from find_event import matching_times as fe_mt
from find_event import background_rejection as fe_br


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
    return parser.parse_args(argv[1:])


# =========================
# Main entry point (main)
# =========================
def main(matching_file, fig_name, with_signal, det_pos_file):
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
        logger.warning(
            f"Detector position file {det_pos_file} does not exist, attempting to continue (may cause subsequent function errors)"
        )
    detector_positions = load_data_from_file(det_pos_file)

    # Ensure plots are written into the same directory as the input matching file
    # If matching_file is relative, convert to absolute first
    output_dir = os.path.dirname(os.path.abspath(matching_file))
    if fig_name is None:
        fig_name = os.path.basename(matching_file).replace(".yaml", "")
    fig_prefix = os.path.join(output_dir, fig_name)

    
    # Read and process matching_times (using more robust optimized reading function)
    matched_file = matching_file.replace(".yaml", "_matched.yaml")
    if os.path.exists(matched_file):
        logger.info(f"Found cached matched file: {matched_file}, loading directly.")
        times = {}
        signals = {}
        du_ids = {}
        event_numbers = {}
        index = {}
        with open(matched_file, "r") as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        for key, result in results.items():
            times[key] = result["time"]
            signals[key] = result.get("signal", None)
            du_ids[key] = result.get("du_id", None)
            event_numbers[key] = result.get("event_number", None)
            index[key] = result.get("index", None)
    else:
        (
            times,
            signals,
            du_ids,
            event_numbers,
            index,
        ) = fe_mt.optimized_read_matching_times(
            matching_file,
            detector_positions,
            min_detectors=5,
            speed_of_light_tolerance=1.05,
        )

        if len(times) < 1:
            logger.warning("No events after filtering, exiting.")
            return
        
        results = {}
        for key in times.keys():
            time = times[key]
            signal = signals[key] if signals is not None else None
            du_id = du_ids[key]
            event_number = event_numbers[key]
            index_value = index[key]
            results[key] = {
                "time": time,
                "signal": signal,
                "du_id": du_id,
                "event_number": event_number,
                "index": index_value,
            }
        with open(matched_file, "w") as f:
            yaml.dump(results, f)
    logger.info(f"Number of events after reading and filtering: {len(times)}")

    fe_plot.plot_du_frequencies(
        du_ids,
        fig_prefix,
    )
    
    
    # Plane wave model fitting
    pwm_fitted_file = matching_file.replace(".yaml", "_PWM.yaml")
    if os.path.exists(pwm_fitted_file):
        logger.info(f"Found cached PWM fitted file: {pwm_fitted_file}, loading directly.")
        times = {}
        signals = {}
        du_ids = {}
        gps_times = {}
        event_numbers = {}
        index = {}
        azimuths = {}
        zeniths = {}
        directions = {}
        chi_squares = {}
        with open(pwm_fitted_file, "r") as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        for key, result in results.items():
            times[key] = result["time"]
            signals[key] = result.get("signal", None)
            du_ids[key] = result.get("du_id", None)
            gps_times[key] = result.get("gps_time", None)
            event_numbers[key] = result.get("event_number", None)
            index[key] = result.get("index", None)
            azimuths[key] = result.get("azimuth", None)
            zeniths[key] = result.get("zenith", None)
            directions[key] = np.array([result['x'], result['y'], result['z']])
            chi_squares[key] = result.get("chi_square", None)
    else:
        (   gps_times,
            directions,
            zeniths,
            azimuths,
            chi_squares,
        ) = fe_est.plane_wave_model(times, signals, detector_positions)

        results = {}
        for key in times.keys():
            chi_square = chi_squares[key]
            zenith = zeniths[key]
            azimuth = azimuths[key]
            time = times[key]
            signal = signals[key] if signals is not None else None
            du_id = du_ids[key]
            gps_time = gps_times[key]
            event_number = event_numbers[key]
            index_value = index[key]
            x = directions[key][0]
            y = directions[key][1]
            z = directions[key][2]
            results[key] = {
                "time": time,
                "du_id": du_id,
                "chi_square": chi_square,
                "zenith": zenith,
                "azimuth": azimuth,
                "signal": signal,
                "gps_time": gps_time,
                "event_number": event_number,
                "index": index_value,
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        with open(pwm_fitted_file, "w") as f:
            yaml.dump(results, f)
    
    logger.info(f"Number of events after plane wave fitting: {len(times)}")
    
    if len(times) < 2:
        logger.warning("No events after filtering, skipping plotting.")
    else:
        fe_plot.plot_reconstructed_positions_PWM(
            gps_times, directions, chi_squares, fig_prefix + "_PWM"
        )
        fe_plot.plot_fitting_parameters_PWM(
            gps_times,
            directions,
            chi_squares,
            "PWM",
            fig_prefix + "_PWM",
        )
    
    # Spherical wave model fitting
    swm_fitted_file = matching_file.replace(".yaml", "_SWM.yaml")
    if os.path.exists(swm_fitted_file):
        logger.info(f"Found cached SWM fitted file: {swm_fitted_file}, loading directly.")
        times = {}
        signals = {}
        du_ids = {}
        gps_times = {}
        directions = {}
        chi_squares = {}
        event_numbers = {}
        index = {}
        azimuths = {}
        zeniths = {}
        with open(swm_fitted_file, "r") as f:
            results = yaml.load(f, Loader=yaml.FullLoader)
        for key, result in results.items():
            times[key] = result["time"]
            signals[key] = result.get("signal", None)
            gps_times[key] = result.get("gps_time", None)
            chi_squares[key] = result.get("chi_square", None)
            du_ids[key] = result.get("du_id", None)
            event_numbers[key] = result.get("event_number", None)
            index[key] = result.get("index", None)
            azimuths[key] = result.get("azimuth", None)
            zeniths[key] = result.get("zenith", None)
            directions[key] = np.array(
                [result.get("x", None), result.get("y", None), result.get("z", None)]
            )
    else:
        (gps_times, directions, chi_squares) = fe_est.spherical_wave_model(
            times, signals, detector_positions, directions
        )

        results = {}
        for key in times.keys():
            gps_time = gps_times[key]
            chi_square = chi_squares[key]
            time = times[key]
            signal = signals[key] if signals is not None else None
            du_id = du_ids[key]
            zenith = zeniths[key]
            azimuth = azimuths[key]
            direction = directions[key]
            event_number = event_numbers[key]
            index_value = index[key]
            results[key] = {
                "time": time,
                "signal": signal,
                "gps_time": gps_time,
                "azimuth": float(azimuth),
                "zenith": float(zenith),
                "du_id": du_id,
                "event_number": event_number,
                "index": index_value,
                "x": float(direction[0]),
                "y": float(direction[1]),
                "z": float(direction[2]),
                "chi_square": float(chi_square),
            }
        with open(swm_fitted_file, "w") as f:
            yaml.dump(results, f)
    logger.info(f"Number of events after spherical wave fitting: {len(times)}")

    if len(times) < 2:
        logger.warning("No events after filtering, skipping plotting.")
    else:
        fe_plot.plot_reconstructed_positions_SWM(
            gps_times,
            directions,
            chi_squares,
            fig_prefix + "_SWM",
        )
        fe_plot.plot_fitting_parameters_SWM(
            gps_times,
            directions,
            chi_squares,
            "SWM",
            fig_prefix + "_SWM",
        )
    exit()
    
    (
        du_ids_filtered,
        times_filtered,
        signals_filtered,
        chi_squares_filtered,
        azimuths_filtered,
        zeniths_filtered,
        source_directions_filtered,
        gps_time_filtered,
    ) = fe_br.background_reject(
        detector_positions,
        du_ids,
        times,
        signals,
        chi_squares,
        azimuths,
        zeniths,
        directions,
        gps_times,
        index,
        fig_prefix,
        output_dir,
        with_signal,
    )
    logger.info(f"Number of events after background rejection: {len(times_filtered)}")

    filtered_file = matching_file.replace(".yaml", "_filtered.yaml")
    results = {}
    for key in times_filtered.keys():
        gps_time = gps_time_filtered[key]
        chi_square = chi_squares_filtered[key]
        du_id = du_ids_filtered[key]
        time = times_filtered[key]
        signal = signals_filtered[key]
        azimuth = azimuths_filtered[key]
        zenith = zeniths_filtered[key]
        direction = source_directions_filtered[key]
        results[key] = {
            "time": time,
            "signal": signal,
            "du_id": du_id,
            "gps_time": gps_time,
            "azimuth": float(azimuth),
            "zenith": float(zenith),
            "x": float(direction[0]),
            "y": float(direction[1]),
            "z": float(direction[2]),
            "chi_square": float(chi_square),
        }
    with open(filtered_file, "w") as f:
        yaml.dump(results, f)


# =========================
# CLI entry: Compatible with original call (matching_file, fig_name, with_signal) and new --det-pos parameter
# =========================
if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.matching_file, args.fig_name, args.with_signal, det_pos_file=args.det_pos)
