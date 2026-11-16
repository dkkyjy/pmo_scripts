"""ROOT 读取、坐标表解析与因果筛选。

只依赖 reconstruction_common 与 readroot/uproot、图优化模块；不做拟合，也不碰天线响应。
文件末尾沿用原文件的兜底块：`readroot` 不可用时用内联等价实现，保证本文件可单独分发。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import uproot

from logger_config import logger
from _optimized_read_matching_times_graph import optimized_read_matching_times_graph

from reconstruction_common import gps_to_lst, DU_POS_TABLE

# readroot: 统一的 tree 探测 (teventadc / tadc) 与 CoREAS 基准时间解析。
# 若 readroot 不在 sys.path (例如仅分发本文件), 使用等价的内联实现以保证可移植性。
try:
    from readroot.common import find_event_tree_key, parse_sim_base_time
except ImportError:
    import re as _re
    from datetime import datetime as _datetime
    import calendar as _calendar

    _SIM_TIME_PATTERN = _re.compile(r"(\d{8})_(\d{6})")

    def find_event_tree_key(keys, file_name):
        for key in keys:
            if "teventadc" in str(key):
                return key, "teventadc"
        for key in keys:
            if "tadc" in str(key):
                return key, "tadc"
        for key in keys:                       # 仿真真值电场 / 真电压 (CoREAS/Dunhuang 的 L0 文件)
            if "tefield" in str(key):
                return key, "tefield"
        for key in keys:
            if "tvoltage" in str(key):
                return key, "tvoltage"
        raise KeyError(f"none of teventadc / tadc / tefield / tvoltage TTree found in file: {file_name}")

    def parse_sim_base_time(file_path):
        for part in reversed(str(file_path).replace("\\", "/").split("/")):
            m = _SIM_TIME_PATTERN.search(part)
            if not m:
                continue
            ds, ts = m.group(1), m.group(2)
            try:
                dt = _datetime(int(ds[0:4]), int(ds[4:6]), int(ds[6:8]),
                               int(ts[0:2]), int(ts[2:4]), int(ts[4:6]))
            except ValueError:
                continue
            return int(ds), int(ts), int(_calendar.timegm(dt.timetuple()))
        return None


# CoREAS 真值树：有 trace 但没有 du_nanoseconds，且 DU id 与 GP65 生产表不重合，
# 因此必须走 readroot 的探测分支 + CoREAS 坐标表（_resolve_coreas_det_pos）。
# 这是"哪些 kind 属于这一类"的唯一真相来源：新增同类树时只改这里。
_COREAS_TRUTH_TREES = ('tefield', 'tvoltage')


def read_root_file_from_path(root_path, event_number_int):
    """读取指定事件的 ADC 波形, 兼容真实 trigger 的 ``teventadc`` 与
    CoREAS/Dunhuang 仿真的 ``tadc`` 两种 tree schema。

    返回 ``(event_num, du_ids_all, gps_time, du_nanoseconds, LST_time,
    trace_1, trace_2, trace_3)``。``teventadc`` 直接提供 ``trace_1/2/3`` 与
    ``gps_time``; ``tadc`` 提供 ``trace_ch`` (形状 ``(n_du, 3, n_time)``) 且
    ``gps_time`` 为空, 由 run 目录名合成。
    """

    root_file = uproot.open(root_path)
    try:
        key, tree_kind = find_event_tree_key(list(root_file.keys()), root_path) # 探测 teventadc / tadc
        tree = root_file[key]

        if tree_kind == "teventadc":
            data = tree.arrays(["event_number", "du_id", "gps_time", "du_seconds", "du_nanoseconds", "trace_1", "trace_2", "trace_3"], library = "ak")
            data = data[int(event_number_int)]
            event_num, du_ids_all, gps_time, du_nanoseconds = data["event_number"], np.array(data["du_id"]), data["gps_time"], np.array(data["du_nanoseconds"])
            trace_1, trace_2, trace_3 = np.array(data["trace_1"]), np.array(data["trace_2"]), np.array(data["trace_3"])
        elif tree_kind == "tadc": # tadc (CoREAS / Dunhuang simulation)
            data = tree.arrays(["event_number", "du_id", "du_nanoseconds", "trace_ch"], library = "ak")
            data = data[int(event_number_int)]
            event_num, du_ids_all, du_nanoseconds = data["event_number"], np.array(data["du_id"]), np.array(data["du_nanoseconds"])
            trace_ch = np.array(data["trace_ch"].to_list()) # (n_du, 3, n_time)
            trace_1, trace_2, trace_3 = trace_ch[:, 0], trace_ch[:, 1], trace_ch[:, 2]
            gps_time = _synthesize_gps_times(root_path, int(event_num), len(du_ids_all))
        elif tree_kind in _COREAS_TRUTH_TREES:
            # 仿真真值 (L0): trace 形状 (n_du, 3, n_time), 无 gps_time / du_nanoseconds
            # 时间轴由 run 目录名合成, du_nanoseconds 置 0 (方向重建只用 fluence + 相对时间窗)
            # 只取这一条: 本函数是逐事例调用的, 整树读会把 N 个事例的 trace 反序列化 N 次。
            # entry_start/entry_stop 与旧的 data[int(event_number_int)] 等价(已实测核对)。
            _idx = int(event_number_int)
            data = tree.arrays(["event_number", "du_id", "trace"],
                               entry_start = _idx, entry_stop = _idx + 1, library = "ak")[0]
            event_num, du_ids_all = data["event_number"], np.array(data["du_id"])
            tr = np.array(data["trace"].to_list())        # (n_du, 3, n_time)
            if tr.ndim != 3:
                raise ValueError(f"{tree_kind}: 期望 trace 形状 (n_du,3,n_time), 实际 {tr.shape}")
            trace_1, trace_2, trace_3 = tr[:, 0], tr[:, 1], tr[:, 2]
            du_nanoseconds = np.zeros(len(du_ids_all), dtype = np.int64)
            logger.warning(
                f"{tree_kind}: 无到达时间，du_nanoseconds 置 0（{len(du_ids_all)} 根天线）；"
                "不要用此路径做 PWF/SWF —— 它们依赖真实时间差，置 0 会给出静默错误的方向"
            )
            gps_time = _synthesize_gps_times(root_path, int(event_num), len(du_ids_all))
    finally:
        root_file.close()

    LST_time = int(np.round(gps_to_lst(gps_time[2])[0]))

    logger.info(f"Trigger Time @ LST Time (hours) {LST_time}")

    return event_num, du_ids_all, gps_time, du_nanoseconds, LST_time, trace_1, trace_2, trace_3


def _synthesize_gps_times(root_path, event_number, n_du):
    """为无 ``gps_time`` 向量的 CoREAS ``tadc`` 树合成扁平 GPS 时间轴。

    复用 :func:`parse_sim_base_time` 从 run 目录名解析基准时间, 并按每个 DU 重复
    ``[date, hhmmss, epoch + event_number]`` 三元组, 使其与真实 ``teventadc`` 树的
    ``gps_time`` 布局 (长度 ``3 * n_du``) 一致。
    """
    base = parse_sim_base_time(root_path)
    if base is None:
        logger.warning(f"[WARN] 无法从路径解析 CoREAS 基准时间, gps_time 置零: {root_path}")
        return np.zeros(3 * n_du, dtype = int)
    date_int, hhmmss_int, epoch = base
    row = [int(date_int), int(hhmmss_int), int(epoch) + int(event_number)]
    return np.array(row * n_du, dtype = int)


def _read_du_position_table(path):
    """读取 DU 位置表, 兼容空格分隔 (GP65) 与逗号分隔 (CoREAS ``du_coords.txt``)。"""
    try:
        return np.loadtxt(path)  # GP65 使用空格/制表符分隔
    except ValueError:
        return np.loadtxt(path, delimiter = ',')  # CoREAS 使用逗号分隔


def _resolve_coreas_det_pos(root_path, det_pos_path = None):
    """为 CoREAS 输入选定 DU 坐标表 (tadc)。

    优先级: 显式 ``det_pos_path`` -> 环境变量 ``RECON_COREAS_DET_POS`` ->
    在 root 文件同级/上级目录查找 ``du_coords.txt``。CoREAS 的 DU id 与 GP65 生产
    表不重合, 必须使用独立的坐标表。
    """
    if det_pos_path:
        return det_pos_path
    env_path = os.environ.get('RECON_COREAS_DET_POS')
    if env_path:
        return env_path
    start = os.path.dirname(os.path.abspath(root_path))
    for base in (start, os.path.dirname(start)):
        candidate = os.path.join(base, 'du_coords.txt')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "CoREAS 输入需要 DU 坐标表 (du_coords.txt), 请通过 det_pos_path 参数或 "
        f"RECON_COREAS_DET_POS 环境变量指定: {root_path}"
    )


def _load_du_position_for_input(root_path, det_pos_path = None):
    """根据输入 tree 类型选定 DU 坐标表路径。

    ``teventadc`` (真实 trigger) 沿用生产表 :data:`DU_POS_TABLE`;
    ``tadc`` / ``tefield`` / ``tvoltage`` (CoREAS, 含 L0 真值) 使用
    :func:`_resolve_coreas_det_pos` 选定的独立坐标表 —— 它们的 DU id 与 GP65
    生产表不重合, 用错表会得到静默错误的位置。
    返回 ``(tree_kind, position_path)``; 具体坐标由 :func:`read_xyz_coordinates`
    读取, 以保证与生产路径完全一致。
    """
    probe_file = uproot.open(root_path)
    try:
        tree_kind = find_event_tree_key(list(probe_file.keys()), root_path)[1]
    finally:
        probe_file.close()
    if tree_kind == 'tadc' or tree_kind in _COREAS_TRUTH_TREES:
        path = _resolve_coreas_det_pos(root_path, det_pos_path)
    else:
        path = det_pos_path if det_pos_path is not None else DU_POS_TABLE
    return tree_kind, path


def read_xyz_coordinates(file_path, du_ids_all, rotate=True):
    # rotate=True : 对坐标施加 GRAND 站点布局旋转 (真实 trigger / teventadc 生产表需要)。
    # rotate=False: 直接使用坐标表原文 (CoREAS / starshape 的 du_coords.txt 已在重建所用
    #               坐标系中, 与 pmo 生成 PWM/SWM yaml 及 shower tshower 真值保持一致)。
    DU_position = _read_du_position_table(file_path) # 加载 DU 位置数据, 包含 DU ID 和坐标信息 (兼容空格/逗号分隔)

    DU_rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    DU_all_x_positions, DU_all_y_positions, DU_all_z_positions = np.array(DU_position[:, 1]), np.array(DU_position[:, 2]), np.array(DU_position[:, 3]) # 所有 DU 的位置列表 (根据提供的 DU_position 数据)

    DU_all_positions = np.stack((DU_all_x_positions, DU_all_y_positions, DU_all_z_positions), axis = 1) # 所有 DU 的位置矩阵 (M x 3)

    if rotate:
        DU_all_positions_rotated = np.dot(DU_rotation_matrix, DU_all_positions.T).T # 所有 DU 的位置矩阵经过旋转后的版本 (M x 3)
    else:
        DU_all_positions_rotated = DU_all_positions # CoREAS/starshape: 坐标表原文即为重建坐标系, 不旋转

    #--------------------------------------------------------------------------

    x_position, y_position, z_position = [], [], []
    
    for du in du_ids_all:

        x_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 1])
        y_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 2])
        z_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 3])

    x_position, y_position, z_position = np.array(x_position), np.array(y_position), np.array(z_position)

    DU_positions = np.stack((x_position, y_position, z_position), axis = 1) # 触发的 DU 的位置矩阵 (N x 3)

    if rotate:
        DU_positions_rotated = np.dot(DU_rotation_matrix, DU_positions.T).T # 旋转坐标系以适应 GRAND 的站点布局
    else:
        DU_positions_rotated = DU_positions # CoREAS/starshape: 坐标表原文即为重建坐标系, 不旋转

    #--------------------------------------------------------------------------

    return x_position, y_position, z_position, DU_positions_rotated, DU_all_positions_rotated


def causality_filter(efield_rec_dir, event_num, du_ids_all, gps_time, du_nanoseconds, trace_1, trace_2, trace_3, x_position, y_position, z_position, DU_positions_rotated):

    detector_positions_for_causal = {} # 构造 detector_positions 字典以满足函数的输入要求 (使用你当前已经提取过的坐标)
    
    for j, du_ in enumerate(du_ids_all): detector_positions_for_causal[str(du_)] = np.array([x_position[j], y_position[j], z_position[j]]) # 原函数内部还会自己调用一次 rotate_and_shift, 为了不发生两次旋转，我们直接传原始的未经旋转偏置的硬坐标
    

    temp_causal_file = os.path.join(efield_rec_dir, f'temp_causality_event_{event_num}_{os.getpid()}.txt') # 进程标识: 并发分片可处理同一个 event_num # 生成一个满足 parser 语法的纯文本行并写入临时文件
    
    
    match_str_list = [f"('{du_ids_all[j]}', {int(du_nanoseconds[j])})" for j in range(len(du_ids_all))] # 将 DU_id 和 nanosecond 时间戳组装成 '[('56', 15632890), ('103', 15639100), ...]' 
    
    match_str = ", ".join(match_str_list)
    
    line_to_write = f"{gps_time[2]}: [{match_str}]"  # gps_time[2] 用作标识
    
    with open(temp_causal_file, 'w') as tf: tf.write(line_to_write + '\n')

    
    logger.info("Executing strict causality cleaning via graph optimization...") # 调用引用的图论清理函数（设定最少容忍 4 个探测器）
    
    try: # 临时文件名带进程标识后不会互相覆盖, 所以异常路径必须自己清掉, 否则每个崩溃的进程都会留下一份
        matching_times = optimized_read_matching_times_graph(
            file_path = temp_causal_file, 
            detector_positions = detector_positions_for_causal, 
            min_detectors = 4)
    finally:
        if os.path.exists(temp_causal_file): os.remove(temp_causal_file) # 删除临时缓存文件

    
    if len(matching_times) == 0: # 分析返回列表，应用掩码
        
        logger.warning(f"Event {event_num} failed strict causality check completely. Skipping.")
        
        return None

    valid_du_list = matching_times[0][1]
    
    indices = []
    for du_id_str, ns_target in valid_du_list:
        du_target = int(du_id_str)
        match = np.where((du_ids_all == du_target) & (du_nanoseconds == ns_target))[0] # 在原始数组中寻找同时匹配 du_id 和 nanosecond 的行
        if len(match) > 0: indices.append(match[0])  # 取第一个匹配（理论上应该只有一个）
        else: logger.warning(f"[WARN] 未找到匹配 (DU={du_target}, ns={ns_target})，跳过")

    indices = np.array(indices)
    
    triggered_du_ids, du_nanoseconds_filtered = du_ids_all[indices], du_nanoseconds[indices] # 找出原阵列里保留下来的天线的索引
    
    trace_1_filtered, trace_2_filtered, trace_3_filtered = trace_1[indices], trace_2[indices], trace_3[indices] # 同样使用索引过滤波形数据
    
    DU_positions_rotated = DU_positions_rotated[indices] # 同样使用索引过滤位置数据
    
    n_triggered = len(triggered_du_ids)
    
    logger.info(f"Passed strict causality check: {n_triggered} DUs retained: {triggered_du_ids}")

    #---------------------------------------------------------------------------

    return triggered_du_ids, du_nanoseconds_filtered, trace_1_filtered, trace_2_filtered, trace_3_filtered, DU_positions_rotated, n_triggered
