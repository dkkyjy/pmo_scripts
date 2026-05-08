"""
纯背景事例生成逻辑
"""
import argparse
from collections import deque
from datetime import datetime
import random
import time

from typing import Dict, Optional, Sequence, Tuple

from logger_config import logger
import yaml

try:
    from yaml import CSafeDumper as _SafeDumper
except ImportError:
    from yaml import SafeDumper as _SafeDumper

BG_RATE_HZ_DEFAULT = 100.0
BG_SIM_WINDOW_NS = int(1e9)
BG_SELECTION_WINDOW_NS = 20_000
BG_MIN_UNIQUE_DU = 5
MAX_BG_RESAMPLE_ATTEMPTS = 2000
SEED_DEFAULT = 42
DURATION_SECONDS_DEFAULT = 60

def sample_background_trigger_stream(du_ids, bg_rate_hz, simulation_window_ns):
    """Sample Poisson trigger times for each DU within the simulation window."""

    logger.debug(
        f"Sampling background trigger stream with parameters: {len(du_ids)} DUs, "
        f"bg_rate_hz={bg_rate_hz}, simulation_window_ns={simulation_window_ns}"
    )
    triggers = []
    scale_ns = 1e9 / float(bg_rate_hz)
    for du_id in du_ids:
        t = 0.0
        while t < simulation_window_ns:
            t += random.expovariate(1.0 / scale_ns)
            if t < simulation_window_ns:
                triggers.append((du_id, t))
    logger.debug(f"Generated triggers: {len(triggers)} before sorting")
    triggers.sort(key=lambda item: item[1])
    logger.debug(f"Generated triggers: {len(triggers)} after sorting")
    return triggers


def select_background_window(sorted_triggers, window_ns, min_unique_du):
    """
    返回所有满足 min_unique_du 的滑动窗口。
    每个窗口为 dict: {"start_ns", "end_ns", "event_times"}
    兼容混合模块 _select_all_background_windows 语义。

    阈值判定规则：窗口内唯一 DU 数 >= min_unique_du 时，窗口有效。
    窗口唯一性：同一起点（start_ns）只记录一次，避免重复。
    输入格式：支持 (du_id, t) 或 (t, du_id)，自动转置。
    """
    logger.debug(
        f"Selecting all background windows with parameters: "
        f"window_ns={window_ns}, min_unique_du={min_unique_du}"
    )
    windows = []
    if not sorted_triggers:
        logger.warning("No triggers provided to select_background_window.")
        return windows

    left = 0
    du_time_queues = {}
    threshold = int(min_unique_du)
    last_recorded_start = None

    # 统一触发格式 (du_id, t) → (t, du_id)
    # 若输入为 (du_id, t)，需转置
    if len(sorted_triggers) > 0 and isinstance(sorted_triggers[0][0], int):
        triggers = [(t, du) for du, t in sorted_triggers]
    else:
        triggers = sorted_triggers

    logger.debug(f"Initial triggers: {triggers[:10]} (showing up to 10)")

    for right, (right_time, right_du) in enumerate(triggers):
        if right_du not in du_time_queues:
            du_time_queues[right_du] = deque()
        du_time_queues[right_du].append(right_time)

        while right_time - triggers[left][0] > window_ns:
            left_time, left_du = triggers[left]
            left_queue = du_time_queues[left_du]
            # 仅移除当前左边界触发，确保队首始终是窗口内该 DU 最早触发。
            if left_queue and left_queue[0] == left_time:
                left_queue.popleft()
            if not left_queue:
                del du_time_queues[left_du]
            left += 1

        if len(du_time_queues) >= threshold:
            start_ns = int(triggers[left][0])
            if start_ns == last_recorded_start:
                continue

            event_times = {
                du_id: du_queue[0]
                for du_id, du_queue in du_time_queues.items()
                if du_queue
            }
            windows.append(
                {
                    "start_ns": start_ns,
                    "end_ns": start_ns + int(window_ns),
                    "event_times": event_times,
                }
            )
            last_recorded_start = start_ns

    logger.debug(f"Total windows selected: {len(windows)}")
    return windows


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

    logger.debug(
        f"Generating background event with parameters: bg_rate_hz={bg_rate_hz}, "
        f"simulation_window_ns={simulation_window_ns}, window_ns={window_ns}, "
        f"min_unique_du={min_unique_du}, "
        f"max_resample_attempts={max_resample_attempts}"
    )
    del n_det
    if bg_rate_hz <= 0:
        raise ValueError("bg_rate_hz must be > 0")
    du_ids = list(du_coords.keys())
    for attempt in range(int(max_resample_attempts)):
        logger.debug(f"Attempt {attempt + 1} to generate background event")
        triggers = sample_background_trigger_stream(du_ids, bg_rate_hz, simulation_window_ns)
        windows = select_background_window(triggers, window_ns, min_unique_du)
        if windows:
            # 兼容旧接口，返回第一个窗口的 event_times
            logger.debug(f"Successfully generated background event: {windows[0]['event_times']}")
            return windows[0]["event_times"]
    logger.error("No valid background event found within max_resample_attempts")
    raise RuntimeError("No valid background event found within max_resample_attempts")


def generate_background_windows(
    du_coords,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    simulation_window_ns=BG_SIM_WINDOW_NS,
    window_ns=BG_SELECTION_WINDOW_NS,
    min_unique_du=BG_MIN_UNIQUE_DU,
    max_resample_attempts=MAX_BG_RESAMPLE_ATTEMPTS,
):
    """Generate all valid background windows from one sampled stream."""

    if bg_rate_hz <= 0:
        raise ValueError("bg_rate_hz must be > 0")

    du_ids = list(du_coords.keys())
    for attempt in range(int(max_resample_attempts)):
        logger.debug(f"Attempt {attempt + 1} to generate background windows")
        triggers = sample_background_trigger_stream(
            du_ids,
            bg_rate_hz,
            simulation_window_ns,
        )
        windows = select_background_window(triggers, window_ns, min_unique_du)
        if windows:
            logger.debug(f"Generated {len(windows)} valid background windows")
            return windows

    logger.error("No valid background windows found within max_resample_attempts")
    raise RuntimeError("No valid background windows found within max_resample_attempts")


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
    second_offset: int,
    root_file: str,
) -> Dict[str, object]:
    """Build one event payload in the same schema as existing toy MC YAML."""

    gps_time = int(gps_start + second_offset)
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
    duration_seconds: int,
    run_number: int,
    gps_start: Optional[int],
    root_file: str,
    bg_rate_hz: float,
    simulation_window_ns: int,
    window_ns: int,
    min_unique_du: int,
    max_resample_attempts: int,
    seed: Optional[int],
    write_yaml: bool = True,
) -> int:
    """Generate pure background windows and optionally write one YAML mapping."""

    logger.debug(
        f"Starting YAML generation with parameters: output_file={output_file}, "
        f"coord_file={coord_file}, duration_seconds={duration_seconds}, "
        f"run_number={run_number}, "
        f"gps_start={gps_start}, root_file={root_file}, bg_rate_hz={bg_rate_hz}, "
        f"simulation_window_ns={simulation_window_ns}, window_ns={window_ns}, "
        f"min_unique_du={min_unique_du}, "
        f"max_resample_attempts={max_resample_attempts}, seed={seed}, "
        f"write_yaml={write_yaml}"
    )
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if min_unique_du <= 0:
        raise ValueError("min_unique_du must be > 0")
    if seed is not None:
        random.seed(seed)

    _, _, load_du_coords = _resolve_common_deps()
    du_coords = load_du_coords(coord_file)
    if not du_coords:
        raise ValueError("No DU coordinates were loaded from coord_file")

    start_gps = int(time.time()) if gps_start is None else int(gps_start)
    event_count = 0
    all_payloads = {}
    for second_offset in range(int(duration_seconds)):
        logger.debug(
            f"Generating windows for second "
            f"{second_offset + 1}/{duration_seconds}"
        )
        windows = generate_background_windows(
            du_coords=du_coords,
            bg_rate_hz=bg_rate_hz,
            simulation_window_ns=simulation_window_ns,
            window_ns=window_ns,
            min_unique_du=min_unique_du,
            max_resample_attempts=max_resample_attempts,
        )
        logger.debug(
            f"Second {second_offset + 1} selected {len(windows)} windows"
        )
        for window in windows:
            event_count += 1
            payload = _build_event_payload(
                event_index=event_count,
                event_window=window["event_times"],
                run_number=run_number,
                gps_start=start_gps,
                second_offset=second_offset,
                root_file=root_file,
            )
            all_payloads[str(event_count)] = payload

    if write_yaml:
        # 批量写出可显著减少重复 yaml.dump 调用开销。
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
        f"Finished generating YAML: {output_file} with {event_count} events "
        f"across {duration_seconds} seconds"
    )
    return event_count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for background YAML generation."""

    coord_file_default, _, _ = _resolve_common_deps()
    parser = argparse.ArgumentParser(
        description="Generate pure background toy-MC events into a YAML file.",
    )
    parser.add_argument("--output", default="toy_mc_background.yaml")
    parser.add_argument("--coord-file", default=coord_file_default)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DURATION_SECONDS_DEFAULT,
        help="Total duration in seconds for generated YAML data.",
    )
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
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--no-write-yaml",
        action="store_true",
        help="Generate events but do not write YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for generating pure background event YAML."""

    args = parse_args(argv)
    event_count = generate_background_yaml(
        output_file=args.output,
        coord_file=args.coord_file,
        duration_seconds=args.duration_seconds,
        run_number=args.run_number,
        gps_start=args.gps_start,
        root_file=args.root_file,
        bg_rate_hz=args.bg_rate_hz,
        simulation_window_ns=args.simulation_window_ns,
        window_ns=args.window_ns,
        min_unique_du=args.min_unique_du,
        max_resample_attempts=args.max_resample_attempts,
        seed=args.seed,
        write_yaml=not args.no_write_yaml,
    )
    if args.no_write_yaml:
        print(
            f"Generated {event_count} pure background events across "
            f"{args.duration_seconds}s (YAML not written)"
        )
    else:
        print(
            f"Generated {event_count} pure background events across "
            f"{args.duration_seconds}s -> {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
