"""
纯信号事例生成逻辑
"""
import argparse
from datetime import datetime
import time
from typing import Dict, Optional, Sequence

import numpy as np
from logger_config import logger
import yaml

try:
    from yaml import CSafeDumper as _SafeDumper
except ImportError:
    from yaml import SafeDumper as _SafeDumper

try:
    from .common import C_LIGHT_NS
except ImportError:
    from toymc.common import C_LIGHT_NS

CONE_ANGLE_DEG = 1.5
THETA_MIN_DEG = 50.0
THETA_MAX_DEG = 88.0
MIN_SOURCE_HEIGHT = 3000.0
MAX_SOURCE_HEIGHT = 8000.0
XY_EXTEND_FACTOR = 1.5
MAX_ATTEMPTS = 100
MIN_DET_PER_EVENT = 5
MAX_DET_PER_EVENT = 10
DURATION_SECONDS_DEFAULT = 60
SEED_DEFAULT = 42


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
    """Generate one signal event as a DU->delay_ns mapping."""

    if not du_coords:
        return {}
    if xy_bounds is None:
        xy_bounds = compute_xy_bounds(du_coords)
    x_min, x_max, y_min, y_max = xy_bounds
    minimum_detectors = max(int(n_det), MIN_DET_PER_EVENT)
    if len(du_coords) < minimum_detectors:
        return {}
    cone_angle_rad = np.radians(CONE_ANGLE_DEG)

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
        if abs(direction[2]) < 1e-6:
            continue

        source_height = np.random.uniform(MIN_SOURCE_HEIGHT, MAX_SOURCE_HEIGHT)
        source_pos = impact_point - direction * source_height / abs(direction[2])
        cone_hits = []
        for du_id, du_pos in du_coords.items():
            radius_vector = du_pos - source_pos
            radius_norm = np.linalg.norm(radius_vector)
            if radius_norm < 1.0:
                continue

            radius_unit = radius_vector / radius_norm
            cos_angle = np.clip(np.dot(radius_unit, direction), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle <= cone_angle_rad:
                cone_hits.append((du_id, radius_norm))

        if len(cone_hits) >= minimum_detectors:
            cone_hits.sort(key=lambda item: item[1])
            max_detectors = min(
                len(cone_hits),
                max(minimum_detectors, MAX_DET_PER_EVENT),
            )
            keep_count = int(
                np.random.randint(minimum_detectors, max_detectors + 1)
            )
            selected_hits = cone_hits[:keep_count]
            return {
                int(du_id): int(round(distance_ns / C_LIGHT_NS))
                for du_id, distance_ns in selected_hits
            }
    return {}


def generate_spherical_wave_arrival(du_coords, n_det):
    """Backward-compatible alias for spherical-wave style arrival generation."""

    return generate_signal_like_arrival_delays(du_coords, n_det)


def _resolve_common_deps_signal():
    """Import common helpers for both package and script execution modes."""

    try:
        from .common import COORD_FILE, load_du_coords
    except ImportError:
        from toymc.common import COORD_FILE, load_du_coords
    return COORD_FILE, load_du_coords


def _build_signal_event_payload(
    event_index: int,
    event_window: Dict[int, int],
    run_number: int,
    gps_start: int,
    second_offset: int,
    root_file: str,
) -> Dict[str, object]:
    """Build one signal event payload in toy-MC YAML schema."""

    # Keep second-level GPS semantics aligned with background generator.
    # If multi-events-per-second is introduced later, extend here with a
    # deterministic sub-second offset field.
    gps_time = int(gps_start + second_offset)
    event_time = datetime.utcfromtimestamp(gps_time).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    sorted_hits = sorted(event_window.items(), key=lambda item: item[1])
    du_ids = [int(du_id) for du_id, _ in sorted_hits]
    du_ns = [int(hit_time) for _, hit_time in sorted_hits]
    time_map = {int(du_id): [int(hit_time)] for du_id, hit_time in sorted_hits}

    return {
        "run_number": int(run_number),
        "event_number": int(event_index),
        "datetime": event_time,
        "gps_time": gps_time,
        "time": time_map,
        "du_id": du_ids,
        "du_ns": du_ns,
        "file": root_file,
        "index": int(event_index),
    }


def generate_signal_yaml(
    output_file: str,
    coord_file: str,
    duration_seconds: int,
    run_number: int,
    gps_start: Optional[int],
    root_file: str,
    min_det_per_event: int,
    max_attempts: int,
    seed: Optional[int],
    write_yaml: bool = True,
) -> int:
    """Generate pure signal events and optionally write one YAML mapping."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if min_det_per_event <= 0:
        raise ValueError("min_det_per_event must be > 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    if seed is not None:
        np.random.seed(seed)

    _, load_du_coords = _resolve_common_deps_signal()
    du_coords = load_du_coords(coord_file)
    if not du_coords:
        raise ValueError("No DU coordinates were loaded from coord_file")

    start_gps = int(time.time()) if gps_start is None else int(gps_start)
    event_count = 0
    all_payloads: Dict[str, Dict[str, object]] = {}

    for second_offset in range(int(duration_seconds)):
        event_window = generate_signal_like_arrival_delays(
            du_coords=du_coords,
            n_det=min_det_per_event,
            max_attempts=max_attempts,
        )
        if not event_window:
            raise RuntimeError(
                "No valid signal event found within max_attempts at "
                f"second_offset={second_offset}"
            )

        event_count += 1
        payload = _build_signal_event_payload(
            event_index=event_count,
            event_window=event_window,
            run_number=run_number,
            gps_start=start_gps,
            second_offset=second_offset,
            root_file=root_file,
        )
        all_payloads[str(event_count)] = payload

    if write_yaml:
        with open(output_file, "w", encoding="utf-8") as output_handle:
            yaml.dump(
                all_payloads,
                output_handle,
                Dumper=_SafeDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
    else:
        logger.debug("Skip writing YAML file because write_yaml is False")

    logger.debug(
        "Finished generating signal YAML: output_file={} events={} "
        "duration_seconds={}",
        output_file,
        event_count,
        duration_seconds,
    )
    return event_count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for signal YAML generation."""

    coord_file_default, _ = _resolve_common_deps_signal()
    parser = argparse.ArgumentParser(
        description="Generate pure signal toy-MC events into a YAML file.",
    )
    parser.add_argument("--output", default="toy_mc_signal.yaml")
    parser.add_argument("--coord-file", default=coord_file_default)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DURATION_SECONDS_DEFAULT,
        help="Total duration in seconds for generated YAML data.",
    )
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--gps-start", type=int, default=None)
    parser.add_argument("--root-file", default="toy_mc_signal.root")
    parser.add_argument(
        "--min-det-per-event",
        type=int,
        default=MIN_DET_PER_EVENT,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS,
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--no-write-yaml",
        action="store_true",
        help="Generate events but do not write YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for generating pure signal event YAML."""

    args = parse_args(argv)
    event_count = generate_signal_yaml(
        output_file=args.output,
        coord_file=args.coord_file,
        duration_seconds=args.duration_seconds,
        run_number=args.run_number,
        gps_start=args.gps_start,
        root_file=args.root_file,
        min_det_per_event=args.min_det_per_event,
        max_attempts=args.max_attempts,
        seed=args.seed,
        write_yaml=not args.no_write_yaml,
    )
    if args.no_write_yaml:
        print(
            f"Generated {event_count} pure signal events across "
            f"{args.duration_seconds}s (YAML not written)"
        )
    else:
        print(
            f"Generated {event_count} pure signal events across "
            f"{args.duration_seconds}s -> {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
