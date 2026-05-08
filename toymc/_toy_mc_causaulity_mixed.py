
# 只保留混合事件生成主流程，所有背景/信号生成相关逻辑均通过 import 新拆分模块
import numpy as np
import random
import argparse
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scienceplots
plt.style.use(["science", "grid", "notebook"])

from .common import load_du_coords, C_LIGHT_NS
from .validation import (
    _build_validation_state,
    run_visual_validation_panel,
)
from ._toy_mc_causaulity_background import generate_background_event, sample_background_trigger_stream, select_background_window
from ._toy_mc_causaulity_signal import generate_signal_like_arrival_delays, generate_spherical_wave_arrival

# 其余常量、全局变量、参数定义等可根据需要保留或精简

# Process-local worker context for Pool.map.
_WORKER_DU_COORDS: Optional[Dict[int, np.ndarray]] = None
_WORKER_BG_RATE_HZ: float = BG_RATE_HZ_DEFAULT
_WORKER_COSMIC_TO_BG_RATIO: float = COSMIC_TO_BG_RATIO_DEFAULT
_WORKER_ENABLE_VALIDATION: bool = False
_WORKER_XY_BOUNDS: Optional[Tuple[float, float, float, float]] = None

def _build_event_specs(
    n_events: int,
    gps_start: float,
    min_det_per_event: int,
    seed: Optional[int],
) -> List[Tuple[int, int, int]]:
    """Build accepted event specs as (event_number, n_det, gps_second)."""
    if n_events <= 0:
        return []

    specs: List[Tuple[int, int, int]] = []
    gps_current = float(gps_start)

    if seed is None:
        while len(specs) < n_events:
            n_det = int(np.random.poisson(min_det_per_event))
            if n_det < 5:
                continue

            event_number = len(specs) + 1
            gps_second = int(gps_current)
            specs.append((event_number, n_det, gps_second))
            gps_current += float(np.random.uniform(0.1, 0.5))
        return specs

    rng = np.random.default_rng(int(seed))
    while len(specs) < n_events:
        n_det = int(rng.poisson(min_det_per_event))
        if n_det < 5:
            continue

        event_number = len(specs) + 1
        gps_second = int(gps_current)
        specs.append((event_number, n_det, gps_second))
        gps_current += float(rng.uniform(0.1, 0.5))

    return specs


def _resolve_pool_chunksize(total_events: int, jobs: int, chunk_size: int) -> int:
    """Resolve safe Pool.map chunksize with small-workload degradation."""
    if total_events <= 0:
        return 1

    minimum = max(1, total_events // max(1, jobs * 4))
    return max(1, min(int(chunk_size), max(1, minimum)))


def _merge_validation_state(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Merge per-event validation stats into one aggregate collector."""
    for key, value in source.items():
        if key not in target:
            continue

        if isinstance(target[key], list):
            target[key].extend(value)
        elif isinstance(target[key], float):
            target[key] += float(value)
        elif isinstance(target[key], int):
            target[key] += int(value)


def _init_event_worker(
    du_coords: Dict[int, np.ndarray],
    bg_rate_hz: float,
    cosmic_to_bg_ratio: float,
    enable_validation_panel: bool,
) -> None:
    """Initialize process-local worker context for event generation."""
    global _WORKER_DU_COORDS
    global _WORKER_BG_RATE_HZ
    global _WORKER_COSMIC_TO_BG_RATIO
    global _WORKER_ENABLE_VALIDATION
    global _WORKER_XY_BOUNDS

    _WORKER_DU_COORDS = du_coords
    _WORKER_BG_RATE_HZ = float(bg_rate_hz)
    _WORKER_COSMIC_TO_BG_RATIO = float(cosmic_to_bg_ratio)
    _WORKER_ENABLE_VALIDATION = bool(enable_validation_panel)
    _WORKER_XY_BOUNDS = _compute_xy_bounds(du_coords)


def _generate_event_payload(
    event_number: int,
    n_det: int,
    gps_second: int,
    du_coords: Dict[int, np.ndarray],
    bg_rate_hz: float,
    cosmic_to_bg_ratio: float,
    enable_validation_panel: bool,
    event_seed: Optional[int],
) -> Dict[str, Any]:
    """Generate one event payload group (signal + background + mixed)."""
    _set_random_seed(event_seed)

    local_plot_stats = {
        "theta_deg": [],
        "phi_deg": [],
        "impact_x": [],
        "impact_y": [],
        "source_x": [],
        "source_y": [],
        "source_z": [],
        "n_det": [],
        "cone_angle": [],
    }
    local_validation = _build_validation_state() if enable_validation_panel else None

    gps_epoch = datetime(1970, 1, 1)
    event_datetime = gps_epoch + timedelta(seconds=int(gps_second))
    datetime_str = event_datetime.strftime("%Y-%m-%dT%H:%M:%S")

    trigger_dict_signal, background_events, mixed_events = generate_mixed_events(
        du_coords=du_coords,
        n_det=n_det,
        bg_rate_hz=bg_rate_hz,
        cosmic_to_bg_ratio=cosmic_to_bg_ratio,
        validation_state=local_validation,
        plot_collector=local_plot_stats,
    )
    # 若 generate_mixed_events 返回三元组，自动忽略 pure_signal_events
    if isinstance(mixed_events, list) and len(mixed_events) > 0 and isinstance(mixed_events[0], dict) and "event_times" in mixed_events[0]:
        # 兼容旧格式
        pass
    elif isinstance(mixed_events, list) and len(mixed_events) > 0 and isinstance(mixed_events[0], list):
        # 新格式，mixed_events 实际为 (background_events, mixed_events, pure_signal_events)
        background_events, mixed_events = background_events, mixed_events

    return {
        "event_number": int(event_number),
        "gps_second": int(gps_second),
        "datetime": datetime_str,
        "signal_event": trigger_dict_signal,
        "background_events": background_events,
        "mixed_events": mixed_events,
        "no_hit": len(mixed_events) == 0,
        "theta_deg": local_plot_stats["theta_deg"],
        "phi_deg": local_plot_stats["phi_deg"],
        "plot_stats": local_plot_stats,
        "validation": local_validation,
    }


def _generate_event_payload_from_worker(
    task: Tuple[int, int, int, Optional[int]],
) -> Dict[str, Any]:
    """Pool.map worker entrypoint for one event spec."""
    if _WORKER_DU_COORDS is None:
        raise RuntimeError("Worker context is not initialized")

    event_number, n_det, gps_second, event_seed = task
    return _generate_event_payload(
        event_number=event_number,
        n_det=n_det,
        gps_second=gps_second,
        du_coords=_WORKER_DU_COORDS,
        bg_rate_hz=_WORKER_BG_RATE_HZ,
        cosmic_to_bg_ratio=_WORKER_COSMIC_TO_BG_RATIO,
        enable_validation_panel=_WORKER_ENABLE_VALIDATION,
        event_seed=event_seed,
    )


def _iter_event_results(
    event_specs: List[Tuple[int, int, int]],
    runtime_seed: int,
    du_coords: Dict[int, np.ndarray],
    bg_rate_hz: float,
    cosmic_to_bg_ratio: float,
    enable_validation_panel: bool,
    jobs: int,
    chunk_size: int,
):
    """Yield event payloads without materializing the full result list."""

    def task_iterator():
        for event_number, n_det, gps_second in event_specs:
            yield (
                event_number,
                n_det,
                gps_second,
                _derive_event_seed(runtime_seed, event_number),
            )

    if jobs == 1:
        for event_number, n_det, gps_second, event_seed in task_iterator():
            yield _generate_event_payload(
                event_number=event_number,
                n_det=n_det,
                gps_second=gps_second,
                du_coords=du_coords,
                bg_rate_hz=bg_rate_hz,
                cosmic_to_bg_ratio=cosmic_to_bg_ratio,
                enable_validation_panel=enable_validation_panel,
                event_seed=event_seed,
            )
        return

    map_chunksize = _resolve_pool_chunksize(
        total_events=len(event_specs),
        jobs=jobs,
        chunk_size=chunk_size,
    )
    with mp.Pool(
        processes=jobs,
        initializer=_init_event_worker,
        initargs=(
            du_coords,
            bg_rate_hz,
            cosmic_to_bg_ratio,
            enable_validation_panel,
        ),
    ) as pool:
        yield from pool.imap(
            _generate_event_payload_from_worker,
            task_iterator(),
            chunksize=map_chunksize,
        )


def _append_plot_sample(
    plot_collector,
    impact_x: float,
    impact_y: float,
    source_pos: np.ndarray,
    n_selected: int,
    max_cone_angle: float,
    theta_deg: float,
    phi_deg: float,
) -> None:
    """Append one generated event sample to global lists or plot collector."""
    if plot_collector is None:
        impact_x_list.append(float(impact_x))
        impact_y_list.append(float(impact_y))
        source_x_list.append(float(source_pos[0]))
        source_y_list.append(float(source_pos[1]))
        source_z_list.append(float(source_pos[2]))
        n_det_list.append(int(n_selected))
        cone_angle_list.append(float(max_cone_angle))
        theta_list_deg.append(float(theta_deg))
        phi_list_deg.append(float(phi_deg))
        return

    plot_collector["impact_x"].append(float(impact_x))
    plot_collector["impact_y"].append(float(impact_y))
    plot_collector["source_x"].append(float(source_pos[0]))
    plot_collector["source_y"].append(float(source_pos[1]))
    plot_collector["source_z"].append(float(source_pos[2]))
    plot_collector["n_det"].append(int(n_selected))
    plot_collector["cone_angle"].append(float(max_cone_angle))
    plot_collector.setdefault("theta_deg", []).append(float(theta_deg))
    plot_collector.setdefault("phi_deg", []).append(float(phi_deg))


def _merge_plot_stats_into_globals(result: Dict[str, Any]) -> None:
    """Merge one event result's angle and plot statistics into global lists."""
    plot_stats = result.get("plot_stats")
    theta_values = result.get("theta_deg")
    phi_values = result.get("phi_deg")
    if theta_values is None and plot_stats:
        theta_values = plot_stats.get("theta_deg", [])
    if phi_values is None and plot_stats:
        phi_values = plot_stats.get("phi_deg", [])

    theta_list_deg.extend(theta_values or [])
    phi_list_deg.extend(phi_values or [])

    if not plot_stats:
        return

    impact_x_list.extend(plot_stats.get("impact_x", []))
    impact_y_list.extend(plot_stats.get("impact_y", []))
    source_x_list.extend(plot_stats.get("source_x", []))
    source_y_list.extend(plot_stats.get("source_y", []))
    source_z_list.extend(plot_stats.get("source_z", []))
    n_det_list.extend(plot_stats.get("n_det", []))
    cone_angle_list.extend(plot_stats.get("cone_angle", []))


def _consume_event_result(
    result: Dict[str, Any],
    signal_writer: IncrementalYamlMapWriter,
    background_writer: IncrementalYamlMapWriter,
    mixed_writer: IncrementalYamlMapWriter,
    counters: Dict[str, int],
    validation_state=None,
) -> None:
    """Consume one event payload and write it immediately to output files."""
    event_number = int(result["event_number"])
    gps_second = int(result["gps_second"])
    datetime_str = result["datetime"]
    trigger_dict_signal = result["signal_event"]

    print(len(trigger_dict_signal))

    time_map_signal = {
        int(du_id): [int(t)] for du_id, t in trigger_dict_signal.items()
    }
    payload_signal = {
        "run_number": 1,
        "event_number": event_number,
        "datetime": datetime_str,
        "gps_time": gps_second,
        "time": time_map_signal,
        "du_id": list(time_map_signal.keys()),
        "file": "toy_mc_signal.root",
        "index": event_number,
    }
    signal_writer.write_entry(event_number, payload_signal)

    for background_event in result["background_events"]:
        counters["background_event_number"] += 1
        time_map_bg = {
            int(du_id): [int(t)] for du_id, t in background_event.items()
        }
        payload_bg = {
            "run_number": 1,
            "event_number": counters["background_event_number"],
            "datetime": datetime_str,
            "gps_time": gps_second,
            "time": time_map_bg,
            "du_id": list(time_map_bg.keys()),
            "file": "toy_mc_background.root",
            "index": counters["background_event_number"],
        }
        background_writer.write_entry(counters["background_event_number"], payload_bg)

    if result["no_hit"]:
        counters["no_hit_seconds"] += 1

    for mixed_event in result["mixed_events"]:
        counters["mixed_event_number"] += 1
        time_map_mixed = {
            int(du_id): [int(t)] for du_id, t in mixed_event.items()
        }
        payload_mixed = {
            "run_number": 1,
            "event_number": counters["mixed_event_number"],
            "datetime": datetime_str,
            "gps_time": gps_second,
            "time": time_map_mixed,
            "du_id": list(time_map_mixed.keys()),
            "file": "toy_mc_mixed.root",
            "index": counters["mixed_event_number"],
        }
        mixed_writer.write_entry(counters["mixed_event_number"], payload_mixed)

    _merge_plot_stats_into_globals(result)
    if validation_state is not None:
        _merge_validation_state(validation_state, result["validation"])



def _compute_xy_bounds(
    du_coords: Dict[int, np.ndarray],
) -> Tuple[float, float, float, float]:
    """Compute expanded XY sampling bounds from detector coordinates."""
    if not du_coords:
        raise ValueError("du_coords must not be empty")

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


def _sample_isotropic_theta_deg() -> float:
    """Sample zenith angle in degrees with isotropic sin(theta) density."""
    theta_min_rad = np.radians(THETA_MIN_DEG)
    theta_max_rad = np.radians(THETA_MAX_DEG)

    cos_min = np.cos(theta_max_rad)
    cos_max = np.cos(theta_min_rad)
    cos_theta = np.random.uniform(cos_max, cos_min)
    return float(np.degrees(np.arccos(cos_theta)))



# 删除本地 _generate_signal_like_arrival_delays，统一用 from ._toy_mc_causaulity_signal import generate_signal_like_arrival_delays



# 删除本地 generate_spherical_wave_arrival，统一用 from ._toy_mc_causaulity_signal import generate_spherical_wave_arrival



# 删除本地 _sample_background_trigger_stream，统一用 from ._toy_mc_causaulity_background import sample_background_trigger_stream



    return None


def _select_all_background_windows(sorted_triggers, window_ns, min_unique_du):
    """Return candidate windows with earliest per-DU times and full window bounds."""
    windows = []
    if not sorted_triggers:
        return windows

    left = 0
    du_counts = {}
    threshold = int(min_unique_du)
    last_recorded_start = None

    for right, (right_time, right_du) in enumerate(sorted_triggers):
        du_counts[right_du] = du_counts.get(right_du, 0) + 1

        while right_time - sorted_triggers[left][0] > window_ns:
            left_du = sorted_triggers[left][1]
            du_counts[left_du] -= 1
            if du_counts[left_du] == 0:
                del du_counts[left_du]
            left += 1

        if len(du_counts) > threshold:
            start_ns = int(sorted_triggers[left][0])
            if start_ns == last_recorded_start:
                continue

            event_times = {}
            for trigger_time, trigger_du in sorted_triggers[left:right + 1]:
                if trigger_du not in event_times:
                    event_times[trigger_du] = trigger_time
            windows.append(
                {
                    "start_ns": start_ns,
                    "end_ns": start_ns + int(window_ns),
                    "event_times": event_times,
                }
            )
            last_recorded_start = start_ns

    return windows


def _iter_background_windows(sorted_triggers, window_ns, min_unique_du):
    """Yield candidate windows without materializing a full candidate list."""
    if not sorted_triggers:
        return

    left = 0
    du_counts = {}
    threshold = int(min_unique_du)
    last_recorded_start = None

    for right, (right_time, right_du) in enumerate(sorted_triggers):
        du_counts[right_du] = du_counts.get(right_du, 0) + 1

        while right_time - sorted_triggers[left][0] > window_ns:
            left_du = sorted_triggers[left][1]
            du_counts[left_du] -= 1
            if du_counts[left_du] == 0:
                del du_counts[left_du]
            left += 1

        if len(du_counts) > threshold:
            start_ns = int(sorted_triggers[left][0])
            if start_ns == last_recorded_start:
                continue

            event_times = {}
            for trigger_time, trigger_du in sorted_triggers[left:right + 1]:
                if trigger_du not in event_times:
                    event_times[trigger_du] = trigger_time
            yield {
                "start_ns": start_ns,
                "end_ns": start_ns + int(window_ns),
                "event_times": event_times,
            }
            last_recorded_start = start_ns


def _build_cosmic_trigger_stream(
    du_coords,
    cosmic_event_times,
    validation_state=None,
    plot_collector=None,
):
    """Expand time-shifted signal samples into a sorted trigger stream."""
    cosmic_triggers = []
    for cosmic_time in cosmic_event_times:
        try:
            arrivals = _generate_cosmic_wave_arrival_at(
                du_coords,
                cosmic_time,
                validation_state=validation_state,
                plot_collector=plot_collector,
            )
        except TypeError:
            # Backward-compatible path for tests monkeypatching old signatures.
            arrivals = _generate_cosmic_wave_arrival_at(
                du_coords,
                cosmic_time,
                validation_state=validation_state,
            )
        for du_id, time_ns in arrivals.items():
            cosmic_triggers.append((int(time_ns), int(du_id)))
    cosmic_triggers.sort(key=lambda item: item[0])
    return cosmic_triggers


def _classify_candidate_windows(bg_windows, cosmic_triggers, cosmic_min_unique_du):
    """Attach final label and merged event times to each candidate window."""
    classified_windows = []
    cosmic_threshold = int(cosmic_min_unique_du)

    for window in bg_windows:
        start_ns = int(window["start_ns"])
        end_ns = int(window["end_ns"])

        cosmic_event_times_map = {}
        for trigger_time, trigger_du in cosmic_triggers:
            if trigger_time < start_ns:
                continue
            if trigger_time > end_ns:
                break
            if trigger_du not in cosmic_event_times_map:
                cosmic_event_times_map[trigger_du] = trigger_time

        merged = dict(window["event_times"])
        for du_id, trigger_time in cosmic_event_times_map.items():
            if du_id not in merged or trigger_time < merged[du_id]:
                merged[du_id] = trigger_time

        label = "mixed" if len(cosmic_event_times_map) >= cosmic_threshold else "background"
        classified_windows.append(
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "event_times": merged,
                "label": label,
            }
        )

    return classified_windows


def _pick_preferred_window(overlapping_windows):
    """Pick one final window from an overlapping region, preferring mixed."""
    return min(
        overlapping_windows,
        key=lambda window: (
            0 if window["label"] == "mixed" else 1,
            int(window["start_ns"]),
            int(window["end_ns"]),
        ),
    )


def _resolve_non_overlapping_windows(classified_windows):
    """Collapse overlapping candidates to one final window per overlapping region."""
    if not classified_windows:
        return []

    sorted_windows = sorted(
        classified_windows,
        key=lambda window: (int(window["start_ns"]), int(window["end_ns"])),
    )
    final_windows = []
    current_group = [sorted_windows[0]]
    current_group_end = int(sorted_windows[0]["end_ns"])

    for window in sorted_windows[1:]:
        start_ns = int(window["start_ns"])
        end_ns = int(window["end_ns"])
        if start_ns < current_group_end:
            current_group.append(window)
            current_group_end = max(current_group_end, end_ns)
            continue

        final_windows.append(_pick_preferred_window(current_group))
        current_group = [window]
        current_group_end = end_ns

    final_windows.append(_pick_preferred_window(current_group))
    return final_windows


def _iter_final_windows_stream(bg_window_iter, cosmic_triggers, cosmic_min_unique_du):
    """Yield final non-overlapping windows via streaming classify+arbitration."""
    cosmic_threshold = int(cosmic_min_unique_du)
    cosmic_left = 0

    current_group = []
    current_group_end = None

    for window in bg_window_iter:
        start_ns = int(window["start_ns"])
        end_ns = int(window["end_ns"])

        while cosmic_left < len(cosmic_triggers) and cosmic_triggers[cosmic_left][0] < start_ns:
            cosmic_left += 1

        cosmic_right = cosmic_left
        cosmic_event_times_map = {}
        while cosmic_right < len(cosmic_triggers):
            trigger_time, trigger_du = cosmic_triggers[cosmic_right]
            if trigger_time > end_ns:
                break
            if trigger_du not in cosmic_event_times_map:
                cosmic_event_times_map[trigger_du] = trigger_time
            cosmic_right += 1

        merged = dict(window["event_times"])
        for du_id, trigger_time in cosmic_event_times_map.items():
            if du_id not in merged or trigger_time < merged[du_id]:
                merged[du_id] = trigger_time

        classified_window = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "event_times": merged,
            "label": (
                "mixed"
                if len(cosmic_event_times_map) >= cosmic_threshold
                else "background"
            ),
        }

        if not current_group:
            current_group = [classified_window]
            current_group_end = end_ns
            continue

        if start_ns < int(current_group_end):
            current_group.append(classified_window)
            current_group_end = max(int(current_group_end), end_ns)
            continue

        yield _pick_preferred_window(current_group)
        current_group = [classified_window]
        current_group_end = end_ns

    if current_group:
        yield _pick_preferred_window(current_group)


def _sample_cosmic_event_times(cosmic_rate_hz, simulation_window_ns):
    """Sample cosmic event center times over one simulation window."""
    event_times = []
    scale_ns = 1e9 / float(cosmic_rate_hz)
    trigger_time = 0.0

    while True:
        trigger_time += float(np.random.exponential(scale=scale_ns))
        if trigger_time > simulation_window_ns:
            break
        event_times.append(int(trigger_time))
    return event_times


def _generate_cosmic_wave_arrival_at(
    du_coords,
    event_time_ns,
    validation_state=None,
    plot_collector=None,
):
    """Generate one time-shifted signal-like arrival map at event_time_ns."""
    try:
        delays = _generate_signal_like_arrival_delays(
            du_coords=du_coords,
            n_det=MAX_DET_PER_EVENT,
            validation_state=validation_state,
            plot_collector=plot_collector,
            sample_source="cosmic",
        )
    except TypeError:
        # Backward-compatible path for tests monkeypatching old signatures.
        delays = _generate_signal_like_arrival_delays(
            du_coords=du_coords,
            n_det=MAX_DET_PER_EVENT,
            validation_state=validation_state,
            sample_source="cosmic",
        )
    return {
        int(du_id): int(delay_ns + int(event_time_ns))
        for du_id, delay_ns in delays.items()
    }



# 删除本地 generate_background_event，统一用 from ._toy_mc_causaulity_background import generate_background_event


def _generate_background_and_mixed_windows(
    du_coords,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    cosmic_to_bg_ratio=COSMIC_TO_BG_RATIO_DEFAULT,
    simulation_window_ns=BG_SIM_WINDOW_NS,
    window_ns=BG_SELECTION_WINDOW_NS,
    min_unique_du=BG_MIN_UNIQUE_DU,
    cosmic_min_unique_du=COSMIC_MIN_UNIQUE_DU,
    validation_state=None,
    plot_collector=None,
):
    """Generate non-overlapping background and mixed events on a shared time axis."""
    if bg_rate_hz <= 0:
        raise ValueError("bg_rate_hz must be > 0")
    if not (0 < cosmic_to_bg_ratio <= COSMIC_TO_BG_RATIO_MAX):
        raise ValueError(
            "cosmic_to_bg_ratio must satisfy 0 < ratio <= "
            f"{COSMIC_TO_BG_RATIO_MAX}"
        )

    du_ids = list(du_coords.keys())
    # 1. 采样背景触发流
    bg_stream = _sample_background_trigger_stream(
        du_ids=du_ids,
        bg_rate_hz=bg_rate_hz,
        simulation_window_ns=simulation_window_ns,
    )
    _record_background_stream_stats(
        validation_state=validation_state,
        bg_stream=bg_stream,
        simulation_window_ns=simulation_window_ns,
        du_count=len(du_ids),
    )
    # 2. 采样信号触发流（以 cosmic_to_bg_ratio 推导信号事件数）
    cosmic_rate_hz = float(bg_rate_hz) * float(cosmic_to_bg_ratio)
    cosmic_event_times = _sample_cosmic_event_times(
        cosmic_rate_hz=cosmic_rate_hz,
        simulation_window_ns=simulation_window_ns,
    )
    _record_cosmic_event_stats(validation_state, cosmic_event_times)
    signal_triggers = []
    for cosmic_time in cosmic_event_times:
        try:
            arrivals = _generate_cosmic_wave_arrival_at(
                du_coords,
                cosmic_time,
                validation_state=validation_state,
                plot_collector=plot_collector,
            )
        except TypeError:
            arrivals = _generate_cosmic_wave_arrival_at(
                du_coords,
                cosmic_time,
                validation_state=validation_state,
            )
        for du_id, time_ns in arrivals.items():
            signal_triggers.append((int(time_ns), int(du_id), "signal"))
    # 3. 背景触发流加标签
    labeled_bg_triggers = [(t, du, "background") for t, du in bg_stream]
    # 4. 合并触发流并排序
    merged_triggers = labeled_bg_triggers + signal_triggers
    merged_triggers.sort(key=lambda item: item[0])
    # 5. 用统一窗口扫描器判窗
    background_events = []
    mixed_events = []
    pure_signal_events = []
    left = 0
    du_source_map = {}
    threshold = int(min_unique_du)
    last_recorded_start = None
    n = len(merged_triggers)
    for right in range(n):
        right_time, right_du, right_src = merged_triggers[right]
        if right_du not in du_source_map:
            du_source_map[right_du] = set()
        du_source_map[right_du].add(right_src)
        while right_time - merged_triggers[left][0] > window_ns:
            left_du = merged_triggers[left][1]
            left_src = merged_triggers[left][2]
            du_source_map[left_du].discard(left_src)
            if not du_source_map[left_du]:
                del du_source_map[left_du]
            left += 1
        if len(du_source_map) > threshold:
            start_ns = int(merged_triggers[left][0])
            if start_ns == last_recorded_start:
                continue
            # 统计窗口内每个 DU 的最早触发时间和来源
            event_times = {}
            event_sources = {}
            for idx in range(left, right + 1):
                t, du, src = merged_triggers[idx]
                if du not in event_times or t < event_times[du]:
                    event_times[du] = t
                    event_sources[du] = src
            # 分类
            src_set = set(event_sources.values())
            if src_set == {"signal"}:
                pure_signal_events.append(event_times)
            elif src_set == {"background"}:
                background_events.append(event_times)
            else:
                mixed_events.append(event_times)
            last_recorded_start = start_ns
    if validation_state is not None:
        validation_state["background_event_count"] += len(background_events)
        validation_state["mixed_event_count"] += len(mixed_events)
        validation_state["pure_signal_event_count"] = len(pure_signal_events)
    return background_events, mixed_events, pure_signal_events


def generate_mixed_events(
    du_coords,
    n_det,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    cosmic_to_bg_ratio=COSMIC_TO_BG_RATIO_DEFAULT,
    simulation_window_ns=BG_SIM_WINDOW_NS,
    window_ns=BG_SELECTION_WINDOW_NS,
    min_unique_du=BG_MIN_UNIQUE_DU,
    cosmic_min_unique_du=COSMIC_MIN_UNIQUE_DU,
    validation_state=None,
    plot_collector=None,
):
    """Generate one pure-signal event plus background and mixed events."""
    # 兼容旧接口，signal_event 仍为单个
    signal_event = _generate_signal_like_arrival_delays(
        du_coords=du_coords,
        n_det=n_det,
        validation_state=validation_state,
        plot_collector=plot_collector,
        sample_source="signal",
    )
    background_events, mixed_events, pure_signal_events = _generate_background_and_mixed_windows(
        du_coords=du_coords,
        bg_rate_hz=bg_rate_hz,
        cosmic_to_bg_ratio=cosmic_to_bg_ratio,
        simulation_window_ns=simulation_window_ns,
        window_ns=window_ns,
        min_unique_du=min_unique_du,
        cosmic_min_unique_du=cosmic_min_unique_du,
        validation_state=validation_state,
        plot_collector=plot_collector,
    )
    # pure_signal_events 目前不输出，兼容旧三元组
    return signal_event, background_events, mixed_events


def generate_mixed_event(du_coords, n_det):
    """Backward-compatible mixed event API returning the first mixed event."""
    _, mixed_events = generate_background_and_mixed_events(
        du_coords=du_coords,
        n_det=n_det,
    )
    if not mixed_events:
        raise RuntimeError("No mixed event found in simulation window")
    return mixed_events[0]


def generate_background_and_mixed_events(
    du_coords,
    n_det=MAX_DET_PER_EVENT,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    cosmic_to_bg_ratio=COSMIC_TO_BG_RATIO_DEFAULT,
    simulation_window_ns=BG_SIM_WINDOW_NS,
    window_ns=BG_SELECTION_WINDOW_NS,
    min_unique_du=BG_MIN_UNIQUE_DU,
    cosmic_min_unique_du=COSMIC_MIN_UNIQUE_DU,
    validation_state=None,
    plot_collector=None,
):
    """Backward-compatible wrapper returning only background and mixed events."""
    del n_det
    return _generate_background_and_mixed_windows(
        du_coords=du_coords,
        bg_rate_hz=bg_rate_hz,
        cosmic_to_bg_ratio=cosmic_to_bg_ratio,
        simulation_window_ns=simulation_window_ns,
        window_ns=window_ns,
        min_unique_du=min_unique_du,
        cosmic_min_unique_du=cosmic_min_unique_du,
        validation_state=validation_state,
        plot_collector=plot_collector,
    )


def format_gps_and_output(
    du_coords,
    bg_rate_hz=BG_RATE_HZ_DEFAULT,
    cosmic_to_bg_ratio=COSMIC_TO_BG_RATIO_DEFAULT,
    enable_validation_panel=False,
    validation_output_dir="toy_mc_validation",
    jobs=1,
    seed=None,
    chunk_size=DEFAULT_POOL_CHUNK_SIZE,
    unsafe_allow_high_load=False,
):
    """Generate final output format GPS time and trigger list, and output as YAML file"""
    _validate_resource_guardrails(
        bg_rate_hz=bg_rate_hz,
        jobs=jobs,
        chunk_size=chunk_size,
        unsafe_allow_high_load=bool(unsafe_allow_high_load),
    )
    jobs = int(jobs)
    chunk_size = int(chunk_size)

    counters = {
        "background_event_number": 0,
        "mixed_event_number": 0,
        "no_hit_seconds": 0,
    }
    validation_state = _build_validation_state() if enable_validation_panel else None

    event_specs = _build_event_specs(
        n_events=N_EVENTS,
        gps_start=GPS_START,
        min_det_per_event=MIN_DET_PER_EVENT,
        seed=seed,
    )

    runtime_seed = _resolve_runtime_seed(seed)
    output_filename_signal = "toy_mc_signal.yaml"
    output_filename_bg = "toy_mc_background.yaml"
    output_filename_mixed = "toy_mc_mixed.yaml"

    with IncrementalYamlMapWriter(output_filename_signal) as signal_writer, \
            IncrementalYamlMapWriter(output_filename_bg) as background_writer, \
            IncrementalYamlMapWriter(output_filename_mixed) as mixed_writer:
        for result in _iter_event_results(
            event_specs=event_specs,
            runtime_seed=runtime_seed,
            du_coords=du_coords,
            bg_rate_hz=bg_rate_hz,
            cosmic_to_bg_ratio=cosmic_to_bg_ratio,
            enable_validation_panel=enable_validation_panel,
            jobs=jobs,
            chunk_size=chunk_size,
        ):
            _consume_event_result(
                result=result,
                signal_writer=signal_writer,
                background_writer=background_writer,
                mixed_writer=mixed_writer,
                counters=counters,
                validation_state=validation_state,
            )

    print(f"Successfully generated {N_EVENTS} signal events and saved to {output_filename_signal}")
    print(
        f"Successfully generated {counters['background_event_number']} background events and saved to "
        f"{output_filename_bg}"
    )

    print(
        f"Successfully generated {counters['mixed_event_number']} mixed events and "
        f"saved to {output_filename_mixed}. no_hit_seconds={counters['no_hit_seconds']}"
    )

    if enable_validation_panel and validation_state is not None:
        safe_validation_output_dir = _resolve_validation_output_dir(validation_output_dir)
        run_visual_validation_panel(
            validation_state=validation_state,
            output_dir=safe_validation_output_dir,
            target_ratio=cosmic_to_bg_ratio,
            du_count=len(du_coords),
        )



def plot_theta_phi_distribution(du_coords=None):
    """Plot detector-hit, direction, height, multiplicity, and cone-angle distributions."""
    if du_coords is None:
        try:
            du_coords = load_du_coords(COORD_FILE)
        except OSError:
            du_coords = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axes[0, 0]
    if du_coords:
        du_x = [float(pos[0]) for pos in du_coords.values()]
        du_y = [float(pos[1]) for pos in du_coords.values()]
        ax1.scatter(
            du_x,
            du_y,
            c='red',
            s=30,
            marker='s',
            label='Detectors',
            zorder=5,
            edgecolors='black',
        )
    if impact_x_list:
        ax1.scatter(
            impact_x_list,
            impact_y_list,
            c='blue',
            s=5,
            alpha=0.5,
            label='Impact points',
            zorder=1,
        )
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Impact Points (XY)\n{len(impact_x_list)} events', fontsize=13)
    if du_coords or impact_x_list:
        ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    ax2 = axes[0, 1]
    if theta_list_deg:
        ax2.hist(
            theta_list_deg,
            bins=20,
            color='skyblue',
            edgecolor='black',
            alpha=0.7,
            density=True,
        )
        ax2.axvline(
            np.mean(theta_list_deg),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean={np.mean(theta_list_deg):.1f}°',
        )
    ax2.set_xlabel('Theta (Zenith Angle) / °', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Zenith ({THETA_MIN_DEG}°-{THETA_MAX_DEG}°, sinθ)', fontsize=13)
    if theta_list_deg:
        ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    if phi_list_deg:
        ax3.hist(phi_list_deg, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.axvline(
            np.mean(phi_list_deg),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean={np.mean(phi_list_deg):.1f}°',
        )
    ax3.set_xlabel('Phi (Azimuth) / °', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Azimuth (Uniform)', fontsize=13)
    if phi_list_deg:
        ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    if source_z_list:
        source_height_km = np.array(source_z_list, dtype=float) / 1000.0
        ax4.hist(source_height_km, bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax4.axvline(
            MIN_SOURCE_HEIGHT / 1000.0,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Min={MIN_SOURCE_HEIGHT / 1000.0:.1f} km',
        )
    ax4.set_xlabel('Source Height (km)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Source Height', fontsize=13)
    if source_z_list:
        ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    if n_det_list:
        unique_counts = sorted(set(n_det_list))
        counts = [n_det_list.count(val) for val in unique_counts]
        ax5.bar(unique_counts, counts, color='coral', edgecolor='black', alpha=0.7)
        ax5.set_xticks(unique_counts)
    ax5.set_xlabel('Detectors', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title(f'Detectors per Event ({CONE_ANGLE_DEG}° cone)', fontsize=13)
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = axes[1, 2]
    if cone_angle_list:
        ax6.hist(cone_angle_list, bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax6.axvline(
            np.mean(cone_angle_list),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean={np.mean(cone_angle_list):.2f}°',
        )
        ax6.axvline(
            CONE_ANGLE_DEG,
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'Limit={CONE_ANGLE_DEG}°',
        )
    ax6.set_xlabel('Max Angle / °', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Actual Cone Angle', fontsize=13)
    if cone_angle_list:
        ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('theta_phi_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def _parse_args():
    """Parse CLI arguments for toy background configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bg-rate-hz",
        type=float,
        default=BG_RATE_HZ_DEFAULT,
        help="Background trigger rate per DU in Hz (must be > 0).",
    )
    parser.add_argument(
        "--cosmic-to-bg-ratio",
        type=float,
        default=COSMIC_TO_BG_RATIO_DEFAULT,
        help=(
            "Cosmic to background rate ratio (must satisfy "
            f"0 < ratio <= {COSMIC_TO_BG_RATIO_MAX})."
        ),
    )
    parser.add_argument(
        "--enable-validation-panel",
        action="store_true",
        help="Enable visual/statistical validation panel output.",
    )
    parser.add_argument(
        "--validation-output-dir",
        type=str,
        default="toy_mc_validation",
        help="Directory for validation PNG artifacts.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for event generation (>= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base random seed. If omitted, one runtime seed is auto-generated "
            "and reused by both serial and parallel paths."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_POOL_CHUNK_SIZE,
        help="Pool.map chunk size for parallel event tasks (>= 1).",
    )
    parser.add_argument(
        "--unsafe-allow-high-load",
        action="store_true",
        help="Allow parameters above safe default caps for stress experiments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Load detector coordinates
    du_coords = load_du_coords(COORD_FILE)
    if not du_coords:
        print("No valid detector coordinates loaded!")
    else:
        print(f"Successfully loaded {len(du_coords)} detector coordinates, starting simulation (upper hemisphere only)...\n")
        # Generate and output simulation data
        format_gps_and_output(
            du_coords,
            bg_rate_hz=args.bg_rate_hz,
            cosmic_to_bg_ratio=args.cosmic_to_bg_ratio,
            enable_validation_panel=args.enable_validation_panel,
            validation_output_dir=args.validation_output_dir,
            jobs=args.jobs,
            seed=args.seed,
            chunk_size=args.chunk_size,
            unsafe_allow_high_load=args.unsafe_allow_high_load,
        )
        # Plot theta and phi distribution histograms
        print("\n=== Plotting Theta/Phi distribution histograms ===")
        plot_theta_phi_distribution(du_coords)
        
        # Print key statistics
        print("\n=== Theta (Zenith Angle) Statistics ===")
        print(f"  Mean: {np.mean(theta_list_deg):.2f}°")
        print(f"  Median: {np.median(theta_list_deg):.2f}°")
        print(f"  Min: {np.min(theta_list_deg):.2f}°")
        print(f"  Max: {np.max(theta_list_deg):.2f}°")
        print(f"  90th Percentile: {np.percentile(theta_list_deg, 90):.2f}°")
        
        print("\n=== Phi (Azimuth Angle) Statistics ===")
        print(f"  Mean: {np.mean(phi_list_deg):.2f}°")
        print(f"  Median: {np.median(phi_list_deg):.2f}°")
        print(f"  Min: {np.min(phi_list_deg):.2f}°")
        print(f"  Max: {np.max(phi_list_deg):.2f}°")
        print(f"  90th Percentile: {np.percentile(phi_list_deg, 90):.2f}°")
