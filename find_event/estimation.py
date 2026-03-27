"""
Event reconstruction and parameter estimation module

This module contains core algorithms for cosmic ray event reconstruction, including:
1. Plane Wave Model (PWM) fitting
2. Spherical Wave Model (SWM) fitting
3. Direction estimation and parameter optimization
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from itertools import combinations
from .utils import gps_to_utc, calculate_azimuth, calculate_zenith
from .time_offset import time_data
from logger_config import logger

# Speed of light (m/ns), constant used in the script for time/distance conversion
c_val = 299792458.0/1e9/1.000259


def sphericalToCartesian(theta_deg: float, phi_deg: float) -> np.ndarray:
    th, ph = np.deg2rad([theta_deg, phi_deg])
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

def estimate_initial_direction(detector_positions, times):
    """
    Provide an initial estimate of the plane wave incidence direction based on detector positions and trigger times.

    Parameters:
      - detector_positions: dict {det_id: np.array([x,y,z])}
      - times: dict {det_id: time_ns}
      - c_val: Speed of light approximation (m/ns)

    Returns:
      - direction: Unit direction vector numpy.array([x,y,z])
      - t0: Reference time (ns)

    Description:
      This function first uses the earliest and latest triggered detectors to approximately estimate geometric distance and time difference,
      calculates initial zenith and azimuth angles, and then uses least_squares to
      minimize pairwise time residuals for parameters (theta, phi, t0), obtaining a more robust initial value.
    """
    positions = np.array([detector_positions[det_id] for det_id in times.keys()])
    t_ns = np.array([times[det_id] for det_id in times.keys()])
    det_keys = list(times.keys())
    logger.debug(f"Estimating initial direction with detector positions: {positions} and times: {t_ns}")

    # Estimate direction using earliest/latest triggered detectors
    min_index = np.argmin(t_ns)
    max_index = np.argmax(t_ns)
    delta_t = t_ns[max_index] - t_ns[min_index]
    logger.debug(f"Initial delta_t: {delta_t} ns between detectors {det_keys[min_index]} and {det_keys[max_index]}")

    min_position = positions[min_index]
    max_position = positions[max_index]
    distance = np.linalg.norm(max_position - min_position)
    logger.debug(f"Distance between earliest and latest triggered detectors: {distance} m")

    # Initial azimuth angle (degrees)
    dx = max_position[0] - min_position[0]
    dy = max_position[1] - min_position[1]
    azimuth = np.arctan2(dy, dx) * 180 / np.pi
    azimuth = azimuth + 360 if azimuth < 0 else azimuth
    logger.debug(f"Initial azimuth estimate: {azimuth} degrees")

    # Initial zenith angle estimate (degrees) - estimated using geometric relationships and time difference
    if delta_t > distance / c_val:
        zenith = np.arccos((distance / c_val) / delta_t) * 180 / np.pi
    else:
        zenith = 90 - np.arccos(delta_t / (distance / c_val)) * 180 / np.pi
    logger.debug(f"Initial zenith estimate: {zenith} degrees")

    # Set zenith angle to 0 when the earliest and latest triggered detectors are the same (extreme case)
    # det_keys = list(times.keys())
    # if det_keys[min_index] == det_keys[max_index]:
    #     zenith = 0

    t_s = t_ns
    t_s_min = np.median(t_s)

    # Define residual function for least_squares: use pairwise time difference residuals as target
    def residuals(params):
        theta, phi, t0 = params
        direction = np.array([
            np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
            np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
            np.cos(np.deg2rad(theta))
        ])
        predicted_times = t0 - np.dot(positions, direction) / c_val
        residuals = []
        for i in range(len(predicted_times)):
            for j in range(i + 1, len(predicted_times)):
                residuals.append(predicted_times[i] - predicted_times[j] - (t_s[i] - t_s[j]))
        return residuals

    # initial_guess = [zenith, azimuth, t_s_min]
    # logger.debug(f"Initial guess for least_squares: {initial_guess}")
    # result = least_squares(residuals, initial_guess)
    # logger.debug(f"Least_squares result: {result.x}")
    # theta, phi, t0 = result.x
    # direction = np.array([
    #     np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
    #     np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
    #     np.cos(np.deg2rad(theta))
    # ])
    
    theta, phi, t0 = zenith, azimuth, t_s_min
    direction = sphericalToCartesian(theta, phi)

    return direction, t0


def calculate_chi_square(times, detector_positions, direction, t0, c_val=c_val):
    """
    Calculate simplified chi-square value for pairwise time differences based on the plane wave model (returns normalized value).

    Parameters:
      - times: [(det_id, time_ns), ...]
      - detector_positions: Detector position dictionary
      - direction: Incident direction unit vector
      - t0: Reference time

    Description:
      This function constructs chi-square contributions from the difference between theoretical and observed time differences between detector pairs,
      and normalizes by dividing by (len(times) - 2) (simplified treatment, using 6 ns as the error scale).
    """
    total_error = 0.0
    t_ns = np.array(times.values())

    def theoretical_time_difference(pos1, pos2):
        return np.dot(pos2 - pos1, direction) / c_val

    key_pairs = list(combinations(times.keys(), 2))
    for det_id_i, det_id_j in key_pairs:
        t_i = times[det_id_i]
        t_j = times[det_id_j]
        pos_i = detector_positions[det_id_i]
        pos_j = detector_positions[det_id_j]
        measured_time_difference = (t_j - t_i)
        theoretical_time_diff = theoretical_time_difference(pos_i, pos_j)
        chi_square_contribution = (((measured_time_difference - theoretical_time_diff) / 6.0) ** 2)
        total_error += chi_square_contribution
    reduced_total_error = total_error / (len(times) - 2)
    return reduced_total_error


def calculate_chi_square_spherical(times, detector_positions, source_position, t0, c_val=c_val):
    """
    Based on the spherical wave source (source_position), calculate the theoretical reception time for each detector and compare with observed times,
    obtaining the chi-square value for the spherical wave model (simplified normalization treatment).
    """
    total_error = 0.0
    t_ns = np.array([times[det_id] for det_id in times.keys()])
    average_t_value = np.mean(t_ns)
    for detector_id, ns in times.items():
        detector_position = np.array(detector_positions[detector_id])
        distance = np.linalg.norm(detector_position - source_position)
        theoretical_time = t0 + distance / c_val
        observed_time = ns - average_t_value
        residual = ((observed_time - theoretical_time) / 6.0) ** 2
        total_error += residual
    reduced_total_err = total_error / (len(times) - 4)
    return reduced_total_err


def plane_wave_model(matching_times, matching_signals, detector_positions):
    """
    Fit each event using the plane wave model and return directions and related information that meet the criteria.

    Returns:
      gps_times, directions, chi_squares, du_ids, filtered_times

    Main process:
      1. Obtain initial guess for each event using estimate_initial_direction
      2. Optimize sum of squared pairwise time differences in (theta, phi, t0) space using minimize
      3. Calculate chi-square and filter events based on threshold
      4. Optional: Plot and save qualifying events (comment/switch)
    """
    gps_times = {}
    directions = {}
    zeniths = {}
    azimuths = {}
    chi_squares = {}

    for key, times in matching_times.items():
        signals = matching_signals[key] if matching_signals is not None else None
        Tvalues = list(times.values())
        logger.debug(f"Tvalues: {Tvalues}")

        # Define objective function for minimization (sum of squares of theoretical-measured differences for pairwise time differences)
        def objective_function(params):
            theta, phi, t0 = params
            direction = np.array([
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta))
            ])
            total_error = 0.0
            det_ids = list(times.keys())
            num_detectors = len(det_ids)
            for i in range(num_detectors):
                for j in range(i + 1, num_detectors):
                    det_id_i = det_ids[i]
                    det_id_j = det_ids[j]
                    pos_i = detector_positions[det_id_i]
                    pos_j = detector_positions[det_id_j]
                    td_i = time_data.get(int(det_id_i)) if isinstance(time_data, dict) else None
                    if td_i is not None:
                        time_i = times[det_id_i] + td_i.get('mean', 0)
                    else:
                        time_i = times[det_id_i]
                    td_j = time_data.get(int(det_id_j)) if isinstance(time_data, dict) else None
                    if td_j is not None:
                        time_j = times[det_id_j] + td_j.get('mean', 0)
                    else:
                        time_j = times[det_id_j]
                    predicted_time_i = t0 + np.dot(pos_i, direction) / c_val
                    predicted_time_j = t0 + np.dot(pos_j, direction) / c_val
                    relative_time_theoretical = predicted_time_i - predicted_time_j
                    relative_time_measured = (time_i - time_j) / 1e0
                    total_error += ((relative_time_theoretical - relative_time_measured) / 6) ** 2
            return total_error

        logger.debug(f"Estimating initial direction for event {gps_time_str}")
        initial_guess, initial_t0 = estimate_initial_direction(detector_positions, times)
        logger.debug(f"Initial guess for event {gps_time_str}: {initial_guess}, t0: {initial_t0}")
        
        azimuth_init = calculate_azimuth(initial_guess)
        zenith_init = calculate_zenith(initial_guess)
        logger.debug(f"Initial direction for event {gps_time_str}: azimuth={azimuth_init}, zenith={zenith_init}")
        bounds = [(0, 91), (-180, 180), (-np.inf, np.inf)]
        
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)
        theta = result.x[0]
        phi = result.x[1]
        t0 = result.x[2]
        # direction = np.array([
        #     np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
        #     np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
        #     np.cos(np.deg2rad(theta))
        # ])
        direction = sphericalToCartesian(theta, phi)
        azimuth = calculate_azimuth(direction)
        zenith = calculate_zenith(direction)
        chi_square = calculate_chi_square(times, detector_positions, direction, t0, c_val)
        logger.debug(f"Fitted direction for event {gps_time_str}: azimuth={azimuth}, zenith={zenith}")
        logger.debug(f"plane_wave_model event {gps_time_str}: direction={direction}, t0={round(t0,3)} ns")

        gps_times[gps_time_str] = gps_time
        directions[gps_time_str] = direction
        zeniths[gps_time_str] = float(zenith)
        azimuths[gps_time_str] = float(azimuth)
        chi_squares[gps_time_str] = float(chi_square)

    return gps_times, directions, zeniths, azimuths, chi_squares


def spherical_wave_model(matching_times, matching_signals, detector_positions, initial_directions):
    """
    Fit spherical wave sources for each event: parameterize source position with (rho, theta, phi, t0),
    minimize the sum of squares of differences between observed times and theoretical arrival times to obtain optimal source position.
    Return a list of source positions meeting the criteria and corresponding chi-square values.
    """
    source_positions = {}
    chi_squares = {}
    gps_times = {}
    filtered_times = {}
    # logger.info(f"initial_directions length: {len(initial_directions)}")

    for i, (gps_time_str, times) in enumerate(matching_times.items()):
        gps_time = float(gps_time_str.split('_')[0])
        signals = matching_signals[gps_time_str] if matching_signals is not None else None
        t_ns = list(times.values())
        t_mean_ns = np.mean(t_ns)
        min_index = np.argmin(t_ns)
        max_index = np.argmax(t_ns)
        delta_t = t_ns[max_index] - t_ns[min_index]
        initial_t0 = t_ns[min_index] - t_mean_ns - 1.3e3 / c_val
        initial_direction = initial_directions[gps_time_str]

        # At least 5 detectors are needed to perform SWM fitting
        if len(times) < 5:
            continue

        # Initial guess: distance rho, angles theta/phi
        initial_rho = 13e3
        initial_phi = calculate_azimuth(initial_direction)
        initial_theta = calculate_zenith(initial_direction)
        # vector = np.array([initial_direction[0], initial_direction[1], initial_direction[2]])
        # norm = np.linalg.norm(vector)
        # initial_phi = np.arctan2(initial_direction[1], initial_direction[0]) * (180 / np.pi)
        # initial_theta = np.arccos(initial_direction[2] / norm) * (180 / np.pi)

        def objective_function(params):
            rho, theta, phi, t0 = params
            source_position = rho * np.array([
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta))
            ])
            total_error = 0.0
            for detector_id, ns in times.items():
                detector_position = detector_positions[detector_id]
                theoretical_time = np.linalg.norm(detector_position - source_position) / c_val + t0
                residual = (theoretical_time - (ns - t_mean_ns)) / 6e0
                total_error += residual ** 2
            return total_error

        initial_guess = [initial_rho, initial_theta, initial_phi, initial_t0]
        bounds = [(0, None), (0, 91), (0, 360), (-np.inf, np.inf)]
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

        rho = result.x[0]
        theta = result.x[1]
        phi = result.x[2]
        t0 = result.x[3]
        # source_position = rho * np.array([
        #     np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
        #     np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
        #     np.cos(np.deg2rad(theta))
        # ])
        source_position = rho * sphericalToCartesian(theta, phi)
        # Calculate azimuth/zenith for filtering/display
        # epsilon = 1e-12
        # zenith = np.arccos(source_position[2] / (np.linalg.norm(source_position) + epsilon)) * 180 / np.pi
        # azimuth = np.arctan2(source_position[1], source_position[0]) * 180 / np.pi
        # zenith = calculate_zenith(source_position)
        # azimuth = calculate_azimuth(source_position)

        chi_square = calculate_chi_square_spherical(times, detector_positions, source_position, t0, c_val)

        source_positions[gps_time_str] = source_position
        chi_squares[gps_time_str] = float(chi_square)
        gps_times[gps_time_str] = gps_time
        filtered_times[gps_time_str] = times

    return gps_times, source_positions, chi_squares

