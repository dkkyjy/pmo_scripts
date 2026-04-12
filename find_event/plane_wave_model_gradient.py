import numpy as np
from scipy import linalg
from scipy.optimize import minimize

from logger_config import logger


try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259
SIGMA_NS = 6.0
DEG2RAD = np.pi / 180.0


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _objective_numba_core(params, diff_pos, diff_time, c_val, sigma, deg2rad):
        theta = params[0]
        phi = params[1]

        theta_rad = theta * deg2rad
        phi_rad = phi * deg2rad
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        n_vec_0 = sin_theta * cos_phi
        n_vec_1 = sin_theta * sin_phi
        n_vec_2 = cos_theta

        dn_dtheta_0 = cos_theta * cos_phi
        dn_dtheta_1 = cos_theta * sin_phi
        dn_dtheta_2 = -sin_theta

        dn_dphi_0 = -sin_theta * sin_phi
        dn_dphi_1 = sin_theta * cos_phi
        dn_dphi_2 = 0.0

        total_error = 0.0
        jac_theta = 0.0
        jac_phi = 0.0

        for pair_index in range(diff_pos.shape[0]):
            theoretical = (
                diff_pos[pair_index, 0] * n_vec_0
                + diff_pos[pair_index, 1] * n_vec_1
                + diff_pos[pair_index, 2] * n_vec_2
            ) / c_val
            residual = (theoretical - diff_time[pair_index]) / sigma
            total_error += residual * residual

            dtheory_dtheta = (
                diff_pos[pair_index, 0] * dn_dtheta_0
                + diff_pos[pair_index, 1] * dn_dtheta_1
                + diff_pos[pair_index, 2] * dn_dtheta_2
            ) / c_val * deg2rad / sigma
            dtheory_dphi = (
                diff_pos[pair_index, 0] * dn_dphi_0
                + diff_pos[pair_index, 1] * dn_dphi_1
                + diff_pos[pair_index, 2] * dn_dphi_2
            ) / c_val * deg2rad / sigma

            jac_theta += 2.0 * residual * dtheory_dtheta
            jac_phi += 2.0 * residual * dtheory_dphi

        return total_error, jac_theta, jac_phi


def estimate_initial_direction_robust(positions, t_s, c=C_LIGHT_NS):
    """Estimate a robust plane-wave initial direction from positions and trigger times."""
    n_det = len(positions)
    if n_det < 3:
        return 0.0, 0.0, np.median(t_s), False

    positions = np.array(positions)
    t_s = np.array(t_s)

    pos_mean = np.mean(positions, axis=0)
    t_mean = np.mean(t_s)
    matrix_a = positions - pos_mean
    vector_b = (t_s - t_mean) * c

    try:
        vector_v, _residuals, _rank, _singular_values = linalg.lstsq(matrix_a, vector_b)
    except linalg.LinAlgError:
        return 0.0, 0.0, np.median(t_s), False

    vector_norm = np.linalg.norm(vector_v)
    if vector_norm < 1e-6:
        return 0.0, 0.0, t_mean, False

    vector_v = vector_v / vector_norm

    vz = vector_v[2]
    vx = vector_v[0]
    vy = vector_v[1]

    zenith_rad = np.arccos(-np.clip(vz, -1.0, 1.0))
    zenith_deg = np.degrees(zenith_rad)

    azimuth_rad = np.arctan2(-vy, -vx)
    azimuth_deg = np.degrees(azimuth_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    t0_est = t_mean - np.dot(pos_mean, vector_v) / c
    return zenith_deg, azimuth_deg, t0_est, True


def calculate_azimuth(direction_vector):
    """Return the azimuth angle in degrees for a 3D direction vector."""
    x_value, y_value, _z_value = direction_vector
    azimuth = np.degrees(np.arctan2(y_value, x_value))
    if azimuth < 0:
        azimuth += 360
    return azimuth


def calculate_PWM_chi_square(matches, detector_positions, direction, t0, c):
    """Compute reduced chi-square for one fitted event using pairwise time differences."""
    del t0

    det_ids = list(matches.keys())
    if len(det_ids) <= 3:
        return float("inf")

    total_error = 0.0
    for first_index in range(len(det_ids)):
        for second_index in range(first_index + 1, len(det_ids)):
            first_det_id = det_ids[first_index]
            second_det_id = det_ids[second_index]
            measured_time_difference = matches[second_det_id] - matches[first_det_id]
            theoretical_time_difference = np.dot(
                detector_positions[second_det_id] - detector_positions[first_det_id],
                direction,
            ) / c
            total_error += (
                (measured_time_difference - theoretical_time_difference) / SIGMA_NS
            ) ** 2

    return float(total_error / (len(det_ids) - 3))


def _coerce_time_value(value):
    """Convert cache/matching time payloads into a scalar float."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("empty time array")
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("empty time sequence")
        return float(value[0])
    return float(value)


def _normalize_matches(times, detector_positions):
    """Normalize one event's time payload into ordered detector ids and arrays."""
    if not isinstance(times, dict):
        raise TypeError("expected event times to be a dict keyed by detector id")

    normalized_matches = {}
    for raw_det_id, raw_time in times.items():
        det_id = int(raw_det_id)
        if det_id not in detector_positions:
            raise KeyError(f"missing detector position for DU {det_id}")
        normalized_matches[det_id] = _coerce_time_value(raw_time)

    ordered_det_ids = sorted(normalized_matches.keys())
    positions = np.array([detector_positions[det_id] for det_id in ordered_det_ids], dtype=float)
    time_values = np.array([normalized_matches[det_id] for det_id in ordered_det_ids], dtype=float)
    return normalized_matches, ordered_det_ids, positions, time_values


def _build_pairwise_arrays(positions, time_values):
    """Build pairwise position and time differences for one event."""
    pair_count = len(time_values) * (len(time_values) - 1) // 2
    diff_pos = np.zeros((pair_count, 3), dtype=float)
    diff_time = np.zeros(pair_count, dtype=float)

    pair_index = 0
    for first_index in range(len(time_values)):
        for second_index in range(first_index + 1, len(time_values)):
            diff_pos[pair_index] = positions[first_index] - positions[second_index]
            diff_time[pair_index] = time_values[first_index] - time_values[second_index]
            pair_index += 1

    return diff_pos, diff_time


def _make_objective_with_jac(diff_pos, diff_time, c_value):
    """Create a two-parameter objective over theta and phi."""
    if NUMBA_AVAILABLE:
        def objective_with_jac(params):
            error, jac_theta, jac_phi = _objective_numba_core(
                np.array(params, dtype=float),
                diff_pos,
                diff_time,
                c_value,
                SIGMA_NS,
                DEG2RAD,
            )
            return error, np.array([jac_theta, jac_phi], dtype=float)

        return objective_with_jac

    def objective_with_jac(params):
        theta, phi = params
        theta_rad = theta * DEG2RAD
        phi_rad = phi * DEG2RAD

        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)

        direction = np.array([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
        ])
        d_direction_dtheta = np.array([
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta,
        ])
        d_direction_dphi = np.array([
            -sin_theta * sin_phi,
            sin_theta * cos_phi,
            0.0,
        ])

        theoretical = diff_pos.dot(direction) / c_value
        residuals = (theoretical - diff_time) / SIGMA_NS
        total_error = np.sum(residuals ** 2)

        dtheory_dtheta = diff_pos.dot(d_direction_dtheta) / c_value * DEG2RAD / SIGMA_NS
        dtheory_dphi = diff_pos.dot(d_direction_dphi) / c_value * DEG2RAD / SIGMA_NS
        jac_theta = 2.0 * np.sum(residuals * dtheory_dtheta)
        jac_phi = 2.0 * np.sum(residuals * dtheory_dphi)
        return total_error, np.array([jac_theta, jac_phi], dtype=float)

    return objective_with_jac


def _direction_from_angles(theta_deg, phi_deg):
    """Convert zenith/azimuth-like angles into a unit direction vector."""
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    return np.array([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad),
    ])


def _fit_event_direction(positions, time_values, c_value):
    """Fit one event and return the optimizer result or None."""
    zenith_init, azimuth_init, _t0_init, success = estimate_initial_direction_robust(
        positions,
        time_values,
        c=c_value,
    )
    if not success:
        zenith_init = 45.0
        azimuth_init = 34.0

    diff_pos, diff_time = _build_pairwise_arrays(positions, time_values)
    objective_with_jac = _make_objective_with_jac(diff_pos, diff_time, c_value)
    objective_only = lambda params: objective_with_jac(params)[0]

    bounds = [(0, 95), (-180, 180)]
    best_result = None
    min_error = float("inf")

    primary_candidates = [
        (zenith_init, azimuth_init),
        (45.0, azimuth_init),
        (75.0, azimuth_init),
    ]
    for guess in primary_candidates:
        try:
            result = minimize(
                objective_with_jac,
                guess,
                method="L-BFGS-B",
                bounds=bounds,
                jac=True,
                options={"maxiter": 100, "ftol": 1e-8},
            )
        except Exception:
            continue
        if result.success and result.fun < min_error:
            best_result = result
            min_error = float(result.fun)

    if best_result is None or min_error > 100:
        phi_offsets = [0, 90, -90, 180, -180] if success else [0, 45, 90, 135, 180, -135, -90, -45]
        for theta_guess in [30.0, 45.0, 60.0, 75.0, 85.0]:
            for phi_offset in phi_offsets:
                phi_guess = ((azimuth_init + phi_offset + 180) % 360) - 180
                try:
                    result = minimize(
                        objective_with_jac,
                        (theta_guess, phi_guess),
                        method="L-BFGS-B",
                        bounds=bounds,
                        jac=True,
                        options={"maxiter": 100, "ftol": 1e-8},
                    )
                except Exception:
                    continue
                if result.success and result.fun < min_error:
                    best_result = result
                    min_error = float(result.fun)

    if best_result is None:
        for guess in [(45.0, 0.0), (45.0, 90.0), (45.0, -90.0), (45.0, 180.0)]:
            try:
                result = minimize(
                    objective_only,
                    guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-8},
                )
            except Exception:
                continue
            if getattr(result, "success", False) and result.fun < min_error:
                best_result = result
                min_error = float(result.fun)

    return best_result


def plane_wave_model(matching_times, matching_signals, detector_positions):
    """Fit each event with the gradient-based plane-wave model using the main pipeline contract."""
    del matching_signals

    directions = {}
    zeniths = {}
    azimuths = {}
    chi_squares = {}

    for event_index, (event_key, times) in enumerate(matching_times.items(), start=1):
        if event_index % 1000 == 1:
            print(f"PWM EventNo{event_index}/{len(matching_times)}")

        try:
            matches, _det_ids, positions, time_values = _normalize_matches(times, detector_positions)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Skipping PWM event {event_key} during normalization: {exc}")
            continue

        if len(matches) < 3:
            logger.warning(
                f"Skipping PWM event {event_key}: requires at least 3 detectors, got {len(matches)}"
            )
            continue

        result = _fit_event_direction(positions, time_values, C_LIGHT_NS)
        if result is None:
            logger.warning(f"Skipping PWM event {event_key}: gradient fit did not converge")
            continue

        theta = float(result.x[0])
        phi = float(result.x[1])

        direction = _direction_from_angles(theta, phi)
        azimuth = float(calculate_azimuth(direction))
        chi_square = calculate_PWM_chi_square(matches, detector_positions, direction, 0.0, C_LIGHT_NS)

        directions[event_key] = direction
        zeniths[event_key] = theta
        azimuths[event_key] = azimuth
        chi_squares[event_key] = chi_square

    return directions, zeniths, azimuths, chi_squares