import numpy as np
import sys
from collections import defaultdict
from datetime import datetime, timedelta

C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259  # m/ns
GPS_UTC_OFFSET = 18  # GPS与UTC的闰秒偏移

def gps_time_to_utc(gps_time):
    """
    将GPS时间转换为真正的UTC时间（不加8小时偏移）。
    GPS时间以1970-01-01为起点。
    """
    gps_epoch = datetime(1970, 1, 1)
    return gps_epoch + timedelta(seconds=(gps_time - GPS_UTC_OFFSET))

def load_data_from_file(filename):
    """Load detector position data from a file"""
    data = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        duid = parts[0]
                        try:
                            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                            data[duid] = coords
                        except ValueError:
                            print(f"Warning: Could not parse coordinates for DU {duid}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    return data 

def rotate_and_shift_coordinates(coord):
    """
    完全复现 _plot_time_ratio.py 中的坐标变换逻辑
    """
    rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 
    rotated = np.dot(rotation_matrix, coord)
    
    # 应用特定的平移偏移量
    # rotated[0] -= -3455.6213993713604
    # rotated[1] -= -1277.747
    # rotated[2] -= 24.02
    
    return rotated

def precompute_distances(detector_positions):
    """预计算所有探测器对之间的距离"""
    du_ids = list(detector_positions.keys())
    n = len(du_ids)
    dist_map = {}
    
    for du_id in du_ids:
        dist_map[(du_id, du_id)] = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = du_ids[i], du_ids[j]
            dist = np.linalg.norm(detector_positions[id1] - detector_positions[id2])
            dist_map[(id1, id2)] = dist
            dist_map[(id2, id1)] = dist
            
    return dist_map

def check_causality_strict(t1, t2, dist, c_ns=C_LIGHT_NS):
    """
    严格因果律检查
    规则：t_obs >= t_theo (Ratio >= 1.0) 才是合法的。
    Ratio < 1.0 意味着超光速，违背因果律，必须剔除。
    """
    if dist == 0:
        return True, float('inf')
    
    t_theo = dist / c_ns
    t_obs = abs(t1 - t2)
    
    ratio = t_obs / t_theo
    is_safe = ratio <= 1.02 
    return is_safe, ratio

def optimized_read_matching_times_graph(file_path, detector_positions, min_detectors=4, utc_start=None, utc_end=None):
    """
    步骤1: 因果律清洗
    返回格式: List[Tuple[gps_time, List[Tuple[du_id, time_ns]]]]
    
    可选参数:
        utc_start: str, UTC开始时间, 格式 "YYYY-MM-DD HH:MM:SS"
        utc_end:   str, UTC结束时间, 格式 "YYYY-MM-DD HH:MM:SS"
    """
    # 解析UTC时间范围
    utc_start_dt = None
    utc_end_dt = None
    if utc_start:
        utc_start_dt = datetime.strptime(utc_start, '%Y-%m-%d %H:%M:%S')
    if utc_end:
        utc_end_dt = datetime.strptime(utc_end, '%Y-%m-%d %H:%M:%S')
    if utc_start_dt or utc_end_dt:
        print(f"[Step 1] UTC time filter enabled: {utc_start} ~ {utc_end}")
    
    matching_times = []
    rotated_positions = {k: rotate_and_shift_coordinates(v) for k, v in detector_positions.items()}
    dist_map = precompute_distances(rotated_positions)
    
    total_lines = 0
    valid_events = 0
    skipped_by_time = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                total_lines += 1
                line = line.strip()
                if not line or ": [" not in line:
                    continue
                try:
                    gps_time_str, matches_str = line.split(": [", 1)
                    gps_time = float(gps_time_str.strip())
                    
                    # --- UTC 时间范围过滤 ---
                    if utc_start_dt or utc_end_dt:
                        event_utc = gps_time_to_utc(gps_time)
                        if utc_start_dt and event_utc < utc_start_dt:
                            skipped_by_time += 1
                            continue
                        if utc_end_dt and event_utc > utc_end_dt:
                            skipped_by_time += 1
                            continue
                    
                    matches_str = matches_str.rstrip("]")
                    
                    nodes = []
                    node_info = {}
                    du_triggers = defaultdict(list)
                    node_id = 0
                    matches_list = matches_str.split("), (")
                    
                    for match in matches_list:
                        match = match.strip("() ")
                        if not match: continue
                        parts = match.split(",")
                        if len(parts) < 2: continue
                        det_id = parts[0].strip().strip("'")
                        try:
                            time_ns = int(parts[1].strip())
                        except ValueError: continue
                        
                        if det_id not in rotated_positions: continue
                        
                        nodes.append((det_id, time_ns, node_id))
                        node_info[node_id] = (det_id, time_ns)
                        du_triggers[det_id].append((time_ns, node_id))
                        node_id += 1
                    
                    if len(du_triggers) < min_detectors:
                        continue
                    
                    # 构建因果律图
                    n_nodes = len(nodes)
                    neighbors = {i: set() for i in range(n_nodes)}
                    
                    for i in range(n_nodes):
                        du_i, time_i, _ = nodes[i]
                        for j in range(i + 1, n_nodes):
                            du_j, time_j, _ = nodes[j]
                            if du_i == du_j: continue
                            
                            dist = dist_map.get((du_i, du_j))
                            if dist is None: continue
                            
                            is_safe, _ = check_causality_strict(time_i, time_j, dist)
                            if is_safe:
                                neighbors[i].add(j)
                                neighbors[j].add(i)
                    
                    # 寻找最大团
                    node_degrees = {i: len(neighbors[i]) for i in range(n_nodes)}
                    sorted_candidates = sorted(range(n_nodes), key=lambda x: node_degrees[x], reverse=True)
                    
                    clique = []
                    for candidate in sorted_candidates:
                        if all(member in neighbors[candidate] for member in clique):
                            clique.append(candidate)
                    
                    # 后处理净化
                    final_clique = list(clique)
                    changed = True
                    while changed and len(final_clique) > min_detectors:
                        changed = False
                        worst_node = -1
                        worst_score = -1.0
                        
                        for node in final_clique:
                            du_n, time_n, _ = nodes[node]
                            score_sum = 0.0
                            has_violation = False
                            
                            for other in final_clique:
                                if other == node: continue
                                du_o, time_o, _ = nodes[other]
                                if du_n == du_o: continue
                                dist = dist_map.get((du_n, du_o), 0.0)
                                if dist == 0: continue
                                safe, ratio = check_causality_strict(time_n, time_o, dist)
                                if not safe:
                                    has_violation = True
                                    score_sum += (1.0 - ratio)
                            
                            if has_violation and score_sum > worst_score:
                                worst_score = score_sum
                                worst_node = node
                        
                        if worst_node != -1:
                            final_clique.remove(worst_node)
                            changed = True
                    
                    # 最终验证
                    is_final_safe = True
                    for i in range(len(final_clique)):
                        if not is_final_safe: break
                        for j in range(i+1, len(final_clique)):
                            n1, n2 = final_clique[i], final_clique[j]
                            d1, t1, _ = nodes[n1]
                            d2, t2, _ = nodes[n2]
                            if d1 == d2: continue
                            dist = dist_map.get((d1, d2), 0)
                            safe, _ = check_causality_strict(t1, t2, dist)
                            if not safe:
                                is_final_safe = False
                                break
                    
                    if is_final_safe and len(final_clique) >= min_detectors:
                        valid_matches = [(node_info[nid][0], node_info[nid][1]) for nid in final_clique]
                        matching_times.append((gps_time, valid_matches))
                        valid_events += 1
                        
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"Error: Input file {file_path} not found.")
        return []
    
    if skipped_by_time > 0:
        print(f"[Step 1] UTC filter: {skipped_by_time} events skipped by time range.")
    print(f"[Step 1] Causality Cleaning: {valid_events}/{total_lines} events retained.")
    return matching_times
