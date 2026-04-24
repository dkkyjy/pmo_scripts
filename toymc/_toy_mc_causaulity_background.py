"""
纯背景事例生成逻辑
"""
import argparse
from datetime import datetime
import random
import time

from typing import Dict, Optional, Sequence, Tuple

from logger_config import logger

BG_RATE_HZ_DEFAULT = 1000.0
BG_SIM_WINDOW_NS = int(1e9)
BG_SELECTION_WINDOW_NS = 20_000
BG_MIN_UNIQUE_DU = 5
MAX_BG_RESAMPLE_ATTEMPTS = 2000
SEED_DEFAULT = 42

def sample_background_trigger_stream(du_ids, bg_rate_hz, simulation_window_ns):
    """Sample Poisson trigger times for each DU within the simulation window."""

    logger.debug("Sampling background trigger stream with parameters: du_ids=%s, bg_rate_hz=%f, simulation_window_ns=%d", du_ids, bg_rate_hz, simulation_window_ns)
    triggers = []
    scale_ns = 1e9 / float(bg_rate_hz)
    t = 0.0
    while t < simulation_window_ns:
        for du_id in random.sample(du_ids, k=len(du_ids)):
            t += random.expovariate(1.0 / scale_ns)
            if t < simulation_window_ns:
                triggers.append((t, du_id))
    triggers.sort(key=lambda item: item[0])
    logger.debug("Generated triggers: %s", triggers)
    return triggers


def select_first_background_window(sorted_triggers, window_ns, min_unique_du):
    """Return the first time window containing at least min_unique_du DUs."""

    logger.debug("Selecting first background window with parameters: window_ns=%d, min_unique_du=%d", window_ns, min_unique_du)
    if not sorted_triggers:
        logger.debug("No triggers available for selection.")
        return None
    left = 0
    du_counts = {}
    threshold = int(min_unique_du)
    for right, (right_time, right_du) in enumerate(sorted_triggers):
        left_time, left_du = sorted_triggers[left]
        du_counts[left_du] = du_counts.get(left_du, 0) + 1
        du_counts[right_du] = du_counts.get(right_du, 0) + 1
        logger.debug("DU counts: %s", du_counts)
        logger.debug("Current window duration: %d ns", right_time - left_time)
        while right_time - left_time >= window_ns:
            du_counts[left_du] -= 1
            if du_counts[left_du] == 0:
                del du_counts[left_du]
            left += 1
        logger.debug("Updated window duration: %d ns, DU counts: %s", right_time - left_time, du_counts)
        window = {}
        if len(du_counts) >= threshold:
            for idx in range(left, right + 1):
                du, t = sorted_triggers[idx][1], sorted_triggers[idx][0]
                if du not in window or t < window[du]:
                    window[du] = t
            logger.debug("Selected window: %s", window)
            return window
    logger.debug("No valid window found.")
    return None


def generate_background_event(
    du_coords,
    n_det,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    simulation_window_ns=BG_SIM_WINDOW_NS,
    window_ns=BG_SELECTION_WINDOW_NS,
    min_unique_du=BG_MIN_UNIQUE_DU,
    max_resample_attempts=MAX_BG_RESAMPLE_ATTEMPTS,
):
    """Generate one background-only event as a {du_id: time_ns} mapping."""

    logger.debug("Generating background event with parameters: bg_rate_hz=%f, simulation_window_ns=%d, window_ns=%d, min_unique_du=%d, max_resample_attempts=%d", bg_rate_hz, simulation_window_ns, window_ns, min_unique_du, max_resample_attempts)
    del n_det
    if bg_rate_hz <= 0:
        raise ValueError("bg_rate_hz must be > 0")
    du_ids = list(du_coords.keys())
    for attempt in range(int(max_resample_attempts)):
        logger.debug("Attempt %d to generate background event", attempt + 1)
        triggers = sample_background_trigger_stream(du_ids, bg_rate_hz, simulation_window_ns)
        window = select_first_background_window(triggers, window_ns, min_unique_du)
        if window:
            logger.debug("Successfully generated background event: %s", window)
            return window
    logger.error("No valid background event found within max_resample_attempts")
    raise RuntimeError("No valid background event found within max_resample_attempts")


def _resolve_common_deps():
    """Import common helpers for both package and script execution modes."""

    try:
        from .common import COORD_FILE, IncrementalYamlMapWriter, load_du_coords
    except ImportError:
        from toymc.common import COORD_FILE, IncrementalYamlMapWriter, load_du_coords
    return COORD_FILE, IncrementalYamlMapWriter, load_du_coords


def _build_event_payload(
    event_index: int,
    event_window: Dict[int, float],
    run_number: int,
    gps_start: int,
    root_file: str,
) -> Dict[str, object]:
    """Build one event payload in the same schema as existing toy MC YAML."""

    gps_time = int(gps_start + event_index - 1)
    event_time = datetime.utcfromtimestamp(gps_time).strftime("%Y-%m-%dT%H:%M:%S")
    sorted_hits = sorted(event_window.items(), key=lambda item: item[1])
    du_ids = [int(du_id) for du_id, _ in sorted_hits]
    time_map = {int(du_id): [int(hit_time)] for du_id, hit_time in sorted_hits}
    return {
        "run_number": int(run_number),
        "event_number": int(event_index),
        "datetime": event_time,
        "gps_time": gps_time,
        "time": time_map,
        "du_id": du_ids,
        "file": root_file,
        "index": int(event_index),
    }


def generate_background_yaml(
    output_file: str,
    coord_file: str,
    n_events: int,
    run_number: int,
    gps_start: Optional[int],
    root_file: str,
    bg_rate_hz: float,
    simulation_window_ns: int,
    window_ns: int,
    min_unique_du: int,
    max_resample_attempts: int,
    seed: Optional[int],
) -> Tuple[str, int]:
    """Generate pure background events and write them as one YAML mapping."""

    logger.debug("Starting YAML generation with parameters: output_file=%s, coord_file=%s, n_events=%d, run_number=%d, gps_start=%s, root_file=%s, bg_rate_hz=%f, simulation_window_ns=%d, window_ns=%d, min_unique_du=%d, max_resample_attempts=%d, seed=%s", output_file, coord_file, n_events, run_number, gps_start, root_file, bg_rate_hz, simulation_window_ns, window_ns, min_unique_du, max_resample_attempts, seed)
    if n_events <= 0:
        raise ValueError("n_events must be > 0")
    if min_unique_du <= 0:
        raise ValueError("min_unique_du must be > 0")
    if seed is not None:
        random.seed(seed)

    _, writer_cls, load_du_coords = _resolve_common_deps()
    du_coords = load_du_coords(coord_file)
    if not du_coords:
        raise ValueError("No DU coordinates were loaded from coord_file")

    start_gps = int(time.time()) if gps_start is None else int(gps_start)
    with writer_cls(output_file) as writer:
        for event_index in range(1, int(n_events) + 1):
            logger.debug("Generating event %d/%d", event_index, n_events)
            window = generate_background_event(
                du_coords=du_coords,
                n_det=min_unique_du,
                bg_rate_hz=bg_rate_hz,
                simulation_window_ns=simulation_window_ns,
                window_ns=window_ns,
                min_unique_du=min_unique_du,
                max_resample_attempts=max_resample_attempts,
            )
            payload = _build_event_payload(
                event_index=event_index,
                event_window=window,
                run_number=run_number,
                gps_start=start_gps,
                root_file=root_file,
            )
            writer.write_entry(event_index, payload)

    logger.debug("Finished generating YAML: %s with %d events", output_file, n_events)
    return output_file, n_events


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for background YAML generation."""

    coord_file_default, _, _ = _resolve_common_deps()
    parser = argparse.ArgumentParser(
        description="Generate pure background toy-MC events into a YAML file.",
    )
    parser.add_argument("--output", default="toy_mc_background.yaml")
    parser.add_argument("--coord-file", default=coord_file_default)
    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--gps-start", type=int, default=None)
    parser.add_argument("--root-file", default="toy_mc_background.root")
    parser.add_argument("--bg-rate-hz", type=float, default=BG_RATE_HZ_DEFAULT)
    parser.add_argument(
        "--simulation-window-ns",
        type=int,
        default=BG_SIM_WINDOW_NS,
    )
    parser.add_argument("--window-ns", type=int, default=BG_SELECTION_WINDOW_NS)
    parser.add_argument("--min-unique-du", type=int, default=BG_MIN_UNIQUE_DU)
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=MAX_BG_RESAMPLE_ATTEMPTS,
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for generating pure background event YAML."""

    args = parse_args(argv)
    output_file, event_count = generate_background_yaml(
        output_file=args.output,
        coord_file=args.coord_file,
        n_events=args.n_events,
        run_number=args.run_number,
        gps_start=args.gps_start,
        root_file=args.root_file,
        bg_rate_hz=args.bg_rate_hz,
        simulation_window_ns=args.simulation_window_ns,
        window_ns=args.window_ns,
        min_unique_du=args.min_unique_du,
        max_resample_attempts=args.max_resample_attempts,
        seed=args.seed,
    )
    print(f"Generated {event_count} pure background events -> {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
