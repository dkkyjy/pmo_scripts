"""
纯信号事例生成逻辑
"""
import numpy as np
from .common import C_LIGHT_NS

CONE_ANGLE_DEG = 1.5
THETA_MIN_DEG = 50.0
THETA_MAX_DEG = 88.0
MIN_SOURCE_HEIGHT = 3000.0
MAX_SOURCE_HEIGHT = 8000.0
XY_EXTEND_FACTOR = 1.5
MAX_ATTEMPTS = 100
MIN_DET_PER_EVENT = 5


def compute_xy_bounds(du_coords):
    x_coords = [float(pos[0]) for pos in du_coords.values()]
    y_coords = [float(pos[1]) for pos in du_coords.values()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_min_ext = x_center - (x_center - x_min) * XY_EXTEND_FACTOR
    x_max_ext = x_center + (x_max - x_center) * XY_EXTEND_FACTOR
    y_min_ext = y_center - (y_center - y_min) * XY_EXTEND_FACTOR
    y_max_ext = y_center + (y_max - y_center) * XY_EXTEND_FACTOR
    return x_min_ext, x_max_ext, y_min_ext, y_max_ext


def sample_isotropic_theta_deg():
    theta_min_rad = np.radians(THETA_MIN_DEG)
    theta_max_rad = np.radians(THETA_MAX_DEG)
    cos_min = np.cos(theta_max_rad)
    cos_max = np.cos(theta_min_rad)
    cos_theta = np.random.uniform(cos_max, cos_min)
    return float(np.degrees(np.arccos(cos_theta)))


def generate_signal_like_arrival_delays(
    du_coords,
    n_det,
    xy_bounds=None,
    max_attempts=MAX_ATTEMPTS,
):
    if not du_coords:
        return {}
    if xy_bounds is None:
        xy_bounds = compute_xy_bounds(du_coords)
    x_min, x_max, y_min, y_max = xy_bounds
    minimum_detectors = max(5, MIN_DET_PER_EVENT)
    target_upper = max(minimum_detectors, int(n_det))
    for _ in range(max_attempts):
        impact_x = np.random.uniform(x_min, x_max)
        impact_y = np.random.uniform(y_min, y_max)
        impact_point = np.array([impact_x, impact_y, 0.0])
        theta_deg = sample_isotropic_theta_deg()
        theta_rad = np.radians(theta_deg)
        phi_deg = float(np.random.uniform(0.0, 360.0))
        phi_rad = np.radians(phi_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        direction = np.array([
            sin_theta * np.cos(phi_rad),
            sin_theta * np.sin(phi_rad),
            -cos_theta,
        ])
        source_height = np.random.uniform(MIN_SOURCE_HEIGHT, MAX_SOURCE_HEIGHT)
        source_pos = impact_point + direction * source_height / abs(direction[2])
        du_delays = {}
        for du_id, du_pos in du_coords.items():
            r = np.linalg.norm(du_pos - source_pos)
            du_delays[du_id] = int(r / C_LIGHT_NS)
        if len(du_delays) >= minimum_detectors:
            return du_delays
    return {}


def generate_spherical_wave_arrival(du_coords, n_det):
    return generate_signal_like_arrival_delays(du_coords, n_det)
