"""
时间/探测器位置标定模块（独立子程序，默认启用，--skip-offset 跳过）。

读取 calibration/jul24_locked_xyz_additive_utc.csv（时间窗 + 每 DU 的
加性时间偏移 ns 与位置偏移 m）。对输入数据按每个事例的 GPS 时间码选择
最接近的时间窗，把标定值**正向**加到 DU 触发时间（整数 ns）和探测器坐标上。

两条使用路径：
  1) main.py（PWM/SWM 生产线）:
       apply_calibration(matching_file, det_pos_file, offset_csv)
       -> 写出 <matching>.toffset.yaml 与 <det_pos>.toffset.txt，返回新路径。
       由于各阶段缓存签名含 matching_file / det_pos_file，标定开关切换时
       缓存会自动失效，无需改动任何现有函数。
  2) reconstruction_candidate.efield_recons_from_efield_PWF:
       load_offset_windows + select_window 取单事件最近窗口，直接对
       du_nanoseconds 与 DU 坐标加偏移（见该文件内的标定块）。

标定参数一律取"最接近的时段"：事例时刻落在某窗口内则取该窗口，
否则取与窗口区间（含中点）距离最近的窗口。
"""

import os
import csv
import numpy as np
import yaml
from datetime import datetime
from logger_config import logger

GPS_UTC_OFFSET = 18  # 与 find_event.utils 一致：GPS -> Unix(UTC) 秒
EPOCH = datetime(1970, 1, 1)

DEFAULT_OFFSET_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibration", "jul24_locked_xyz_additive_utc.csv",
)


def _parse_utc(s):
    """'YYYY-MM-DD HH:MM:SS[.fff]'（UTC）→ Unix 秒。"""
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return (datetime.strptime(s, fmt) - EPOCH).total_seconds()
        except ValueError:
            continue
    raise ValueError(f"无法解析标定 CSV 时间: {s!r}")


def load_offset_windows(csv_path=DEFAULT_OFFSET_CSV):
    """
    读取标定 CSV，返回按窗口开始时间排序的列表：
      [(start_unix, end_unix, {du_id: (dt_ns, dx, dy, dz)}), ...]
    """
    windows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (_parse_utc(row["window_start_utc"]), _parse_utc(row["window_end_utc"]))
            windows.setdefault(key, {})[int(row["DU_ID"])] = (
                float(row["time_offset_add_ns"]),
                float(row["x_offset_m"]),
                float(row["y_offset_m"]),
                float(row["z_offset_m"]),
            )
    return sorted(((s, e, d) for (s, e), d in windows.items()), key=lambda w: w[0])


def select_window(event_utc_unix, windows):
    """
    为事例选择最接近的时间窗：落在某窗口内则取该窗口（距离 0）；
    否则取与窗口区间（含中点）距离最近的窗口。返回 (index, distance_s)。
    """
    best_i, best_d = -1, np.inf
    for i, (s, e, _) in enumerate(windows):
        if s <= event_utc_unix <= e:
            return i, 0.0
        mid = 0.5 * (s + e)
        d = min(abs(event_utc_unix - s), abs(event_utc_unix - e),
                abs(event_utc_unix - mid))
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def mean_position_offsets(windows):
    """每 DU 跨所有窗口的位置偏移平均值 {du_id: (dx, dy, dz)}（全局坐标用）。"""
    s_sum, s_cnt = {}, {}
    for (_, _, dus) in windows:
        for du, (_, dx, dy, dz) in dus.items():
            s_sum.setdefault(du, np.zeros(3))
            s_cnt[du] = s_cnt.get(du, 0) + 1
            s_sum[du] += np.array([dx, dy, dz])
    return {du: s_sum[du] / s_cnt[du] for du in s_sum}


def _write_fallback(path, content):
    """优先写在原文件同目录；不可写（只读挂载）时退回当前工作目录。"""
    try:
        with open(path, "w") as f:
            f.write(content)
        return path
    except OSError:
        path = os.path.basename(path)
        with open(path, "w") as f:
            f.write(content)
        return path


def apply_calibration(matching_file, det_pos_file, offset_csv=DEFAULT_OFFSET_CSV):
    """
    对 readroot 产出的 Trigger YAML 与 DU 坐标文件施加标定。

    - 时间：逐事例按 gps_time 选最接近窗口，该 DU 的 time 列表每个值
      正向加 dt_ns 并取整（下游解析器要求整数 ns）；
    - 坐标：每 DU 取所有窗口位置偏移的均值，正向加在坐标上，写出新的
      det-pos 文件（'ID x y z' 文本格式）。

    返回 (calibrated_yaml, calibrated_det_pos, stats)。
    """
    from find_event.io import load_data_from_file

    windows = load_offset_windows(offset_csv)
    mean_off = mean_position_offsets(windows)

    # ---------- 坐标 ----------
    positions = load_data_from_file(det_pos_file)
    out_pos_lines = []
    n_pos = 0
    for du_id in sorted(positions.keys(), key=lambda k: int(k)):
        coord = np.asarray(positions[du_id], dtype=float)
        off = mean_off.get(int(du_id))
        if off is not None:
            coord = coord + off
            n_pos += 1
        out_pos_lines.append(f"{du_id} {coord[0]:.4f} {coord[1]:.4f} {coord[2]:.4f}")
    calibrated_det_pos = _write_fallback(
        det_pos_file + ".toffset.txt", "\n".join(out_pos_lines) + "\n")

    # ---------- 时间 ----------
    with open(matching_file) as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"无法解析输入 YAML: {matching_file}")

    n_events = n_time = 0
    max_dist = 0.0
    window_use = {}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        gps = item.get("gps_time")
        time_map = item.get("time")
        if gps is None or not isinstance(time_map, dict):
            continue
        n_events += 1
        wi, dist = select_window(float(gps) - GPS_UTC_OFFSET, windows)
        max_dist = max(max_dist, dist)
        window_use[wi] = window_use.get(wi, 0) + 1
        if wi < 0:
            continue
        dus = windows[wi][2]
        new_map = {}
        for du, tval in time_map.items():
            off = dus.get(int(du))
            if off is None:
                new_map[du] = tval
                continue
            vals = tval if isinstance(tval, (list, tuple)) else [tval]
            new_map[du] = [int(round(float(t) + off[0])) for t in vals]
            n_time += len(vals)
        item["time"] = new_map

    out_yaml = os.path.splitext(matching_file)[0] + ".toffset.yaml"
    calibrated_yaml = _write_fallback(out_yaml, yaml.dump(payload))

    stats = dict(n_events=n_events, n_time_applied=n_time, n_pos_applied=n_pos,
                 n_windows=len(windows), max_window_distance_s=max_dist,
                 windows_used=len(window_use))
    logger.info(
        f"[Offset calibration] events={n_events}, time offsets applied={n_time}, "
        f"position offsets on {n_pos} DUs, max window distance={max_dist:.1f}s, "
        f"windows used={len(window_use)}/{len(windows)} -> {calibrated_yaml}, {calibrated_det_pos}"
    )
    return calibrated_yaml, calibrated_det_pos, stats
