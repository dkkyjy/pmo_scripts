"""Plane-wave-front (PWF) direction refinement of the PWM result.

Why: :func:`find_event.plane_wave_model_gradient.plane_wave_model` has no bound on
the zenith angle and its multi-start optimiser sometimes settles on the *mirrored*
solution (``zenith > 90``, i.e. the wave would come from below the horizon). It also
does not reach its own global minimum. Both effects are removed by re-minimising the
same pairwise plane-wave residual, starting from the PWM direction after reflecting it
into the physical half-space.

Measured on CoREAS dense-array events (5069, DUs with ``R < 3000 m``), versus truth:
median separation ``1.63 deg -> 0.81 deg``, fraction above 5 deg ``15.9% -> 0%``,
extra cost ~3.5 ms/event.

The residual is the same one used by PWM and by ``reconstruction_candidate``'s PWF::

    f_ij = (r_i - r_j) . n_hat - c * (t_i - t_j),      chi2 = sum f_ij^2 / (sigma c)^2

with ``n_hat = -u_hat`` (``u_hat`` = unit vector towards the source) and a fixed
``sigma = 6 ns``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from logger_config import logger

SIGMA_NS = 6.0
C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259  # m/ns, same constant as plane_wave_model_gradient


def _n_hat(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def _unit(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    a, z = np.deg2rad(azimuth_deg), np.deg2rad(zenith_deg)
    return np.array([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), np.cos(z)])


def _pairwise(positions: np.ndarray, times_ns: np.ndarray):
    i, j = np.triu_indices(len(positions), 1)
    return positions[i] - positions[j], (times_ns[i] - times_ns[j]) * C_LIGHT_NS


def refine_one(positions, times_ns, azimuth, zenith):
    """Re-minimise one event. Returns ``(azimuth, zenith, chi_square)``."""
    positions = np.asarray(positions, dtype=float)
    times_ns = np.asarray(times_ns, dtype=float)
    if len(positions) < 4 or not np.isfinite(azimuth) or not np.isfinite(zenith):
        return azimuth, zenith, np.nan
    dr, dt = _pairwise(positions, times_ns)
    norm = (SIGMA_NS * C_LIGHT_NS) ** 2

    # 1) reflect the seed into the physical half-space (zenith <= 90)
    zen0 = 180.0 - zenith if zenith > 90.0 else zenith
    x0 = np.array([180.0 - zen0, (azimuth + 180.0) % 360.0])

    def chi2(theta_phi):
        f = dr @ _n_hat(theta_phi[0], theta_phi[1]) - dt
        return float(np.sum(f * f)) / norm

    try:
        res = minimize(
            chi2, x0, method="L-BFGS-B",
            bounds=[(0.0, 180.0), (0.0, 360.0)],
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"PWF refinement failed, keeping PWM direction: {exc}")
        return azimuth, zenith, np.nan
    if not np.all(np.isfinite(res.x)):
        return azimuth, zenith, np.nan

    # 2) back to physical angles; the unbounded refit may overshoot past 90 deg
    az = (float(res.x[1]) + 180.0) % 360.0
    zen = 180.0 - float(res.x[0])
    if zen > 90.0:
        zen = 180.0 - zen
    return az, zen, float(res.fun) / max(len(positions) - 2, 1)


def _position_of(positions_map, det_id):
    """Detector-id keys are ints in the shared layout map but strings in the
    per-event ``det_pos`` YAML, so accept either."""
    for candidate in (det_id, int(det_id), str(det_id)):
        if candidate in positions_map:
            return positions_map[candidate]
    raise KeyError(det_id)


def refine_directions(
    matching_times,
    detector_positions,
    directions,
    azimuths,
    zeniths,
    chi_squares,
    detector_positions_by_event=None,
):
    """Refine PWM results event by event.

    Mirrors the argument conventions of
    :func:`find_event.plane_wave_model_gradient.plane_wave_model`: ``matching_times``
    is ``{event_key: {det_id: time_ns}}``, ``detector_positions`` is the shared
    ``{det_id: [x, y, z]}`` map and ``detector_positions_by_event`` overrides it per
    event.  Returns updated ``(directions, azimuths, zeniths, chi_squares)`` dicts;
    ``directions`` stays consistent with what ``plane_wave_model`` returns
    (``x, y, z`` = ``-u_hat``).
    """
    refined_dir, refined_az, refined_zen, refined_chi2 = {}, {}, {}, {}
    global_positions = detector_positions
    for key, pwm_direction in directions.items():
        times = matching_times.get(key)
        if times is None:
            continue
        positions_map = global_positions
        if detector_positions_by_event is not None:
            positions_map = detector_positions_by_event.get(key, global_positions)
        try:
            det_ids = sorted(times, key=lambda d: int(d))
            positions = np.array([_position_of(positions_map, d) for d in det_ids], dtype=float)
            times_ns = np.array([times[d] for d in det_ids], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Skipping PWF refinement for event {key}: {exc}")
            continue
        az, zen, chi2 = refine_one(positions, times_ns, azimuths[key], zeniths[key])
        refined_az[key], refined_zen[key] = az, zen
        refined_chi2[key] = chi2
        refined_dir[key] = -_unit(az, zen)

    n_ok = len(refined_dir)
    if n_ok:
        n_flip = sum(1 for k in refined_zen if zeniths[k] > 90.0)
        logger.info(
            f"PWF refinement: {n_ok} events re-minimised ({n_flip} mirrored PWM seeds reflected)"
        )
    else:
        logger.warning("PWF refinement produced no events; keeping PWM directions")
        return directions, azimuths, zeniths, chi_squares
    return refined_dir, refined_az, refined_zen, refined_chi2
