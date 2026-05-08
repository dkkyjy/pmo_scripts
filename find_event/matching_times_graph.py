from du_pair_theoretical import load_or_build_theoretical_rows
from pathlib import Path
from collections import defaultdict, deque
from logger_config import logger
import yaml


C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259  # m/ns

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
    
    #if t_obs == 0:
    #    if dist > 1e-6:
    #        return False, 0.0
    #    return True, 0.0
        
    ratio = t_obs / t_theo
    is_safe = ratio <= 1.02 
    return is_safe, ratio


def optimized_read_matching_times_graph(times_dict, signals_dict, det_pos_path, min_detectors=4, speed_of_light_tolerance=1.05, force_recompute=False):
    """
    步骤1: 因果律清洗
    返回格式: (times, signals, du_ids), 与 matching_times.py 一致
    """
    matching_times = {}

    logger.debug("=== 初始化阶段 ===")
    rows = load_or_build_theoretical_rows(Path(det_pos_path), force_recompute=force_recompute)
    dist_map = {f'{row.du_a} - {row.du_b}': row.distance_m for row in rows}
    logger.debug(f"✅ 完成距离表加载，共缓存 {len(dist_map)} 组探测器对距离")

    def get_dist(id1, id2):
        """Symmetric distance lookup."""
        k = f'{id1} - {id2}'
        if k in dist_map:
            return dist_map[k]
        else:
            k = f'{id2} - {id1}'
            return dist_map[k]

    total_lines = 0
    valid_events = 0

    logger.debug("=== 开始处理输入数据 ===")
    for key, matches in times_dict.items():
        total_lines += 1
        signals = signals_dict.get(key) if signals_dict else None

        nodes = []
        node_info = {}
        du_triggers = defaultdict(list)
        node_id = 0
        logger.debug(f"———— 处理事件 {key} ————")
        for det_id, value in matches.items():
            logger.debug(f"det_id: {det_id}, value: {value}")
            for time_ns in value:
                logger.debug(f"time_ns: {time_ns}")
                nodes.append((str(det_id), time_ns, node_id))
                node_info[node_id] = (str(det_id), time_ns)
                du_triggers[str(det_id)].append((time_ns, node_id))
                node_id += 1

        logger.debug(f"📊 节点解析统计: 总构建节点数: {len(nodes)}, 触发的探测器数: {len(du_triggers)}, 各探测器触发数: { {k:len(v) for k,v in du_triggers.items()} }")
        if len(du_triggers) < min_detectors:
            logger.warning(f"⚠️  触发探测器数 ({len(du_triggers)}) < 最小阈值 ({min_detectors})，跳过该事件")
            continue

        # 构建因果律图
        n_nodes = len(nodes)
        neighbors = {i: set() for i in range(n_nodes)}
        edge_count = 0
        logger.debug(f"🔗 开始构建因果律图 (共 {n_nodes} 个节点)")
        for i in range(n_nodes):
            du_i, time_i, _ = nodes[i]
            for j in range(i + 1, n_nodes):
                du_j, time_j, _ = nodes[j]
                if du_i == du_j:
                    continue
                dist = get_dist(str(du_i), str(du_j))
                if dist is None:
                    logger.warning(f"   ⚠️  无探测器对 {du_i}-{du_j} 距离数据，跳过")
                    continue
                is_safe, ratio = check_causality_strict(time_i, time_j, dist)
                if is_safe:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    edge_count += 2
                    logger.debug(f"   ✅ 连边: 节点{i}({du_i}) ↔ 节点{j}({du_j}) | 距离={dist:.2f}m | 观测时间差={abs(time_i-time_j)}ns | 比值={ratio:.4f}")
                else:
                    logger.debug(f"   ❌ 不连边: 节点{i}({du_i}) ↔ 节点{j}({du_j}) | 比值={ratio:.4f} (超阈值)")

        logger.debug(f"✅ 因果律图构建完成: 共 {edge_count//2} 条无向边, 节点度数统计: { {i:len(v) for i,v in neighbors.items()} }")

        # 寻找最大团
        node_degrees = {i: len(neighbors[i]) for i in range(n_nodes)}
        sorted_candidates = sorted(range(n_nodes), key=lambda x: node_degrees[x], reverse=True)
        logger.debug(f"🌀 开始贪心寻找最大团, 节点按度数降序: {sorted_candidates} (度数: { [node_degrees[x] for x in sorted_candidates] })")
        clique = []
        for candidate in sorted_candidates:
            is_compatible = all(member in neighbors[candidate] for member in clique)
            if is_compatible:
                clique.append(candidate)
                logger.debug(f"   ✅ 加入节点 {candidate} → 当前团: {clique}")
            else:
                logger.debug(f"   ❌ 跳过节点 {candidate} (与团内节点不兼容)")

        logger.debug(f"✅ 初始最大团筛选完成: 团内节点数={len(clique)}, 节点列表={clique}")

        # 后处理净化
        final_clique = list(clique)
        changed = True
        iteration = 0
        logger.debug(f"🧹 开始后处理净化 (Hard Filter)，初始团大小={len(final_clique)}")
        while changed and len(final_clique) > min_detectors:
            iteration += 1
            changed = False
            worst_node = -1
            worst_score = -1.0
            logger.debug(f"   迭代 {iteration}: 检查团内 {len(final_clique)} 个节点")
            for node in final_clique:
                du_n, time_n, _ = nodes[node]
                score_sum = 0.0
                has_violation = False
                violation_details = []
                for other in final_clique:
                    if other == node:
                        continue
                    du_o, time_o, _ = nodes[other]
                    if du_n == du_o:
                        continue
                    dist = get_dist(str(du_n), str(du_o))
                    if dist is None or dist == 0:
                        continue
                    safe, ratio = check_causality_strict(time_n, time_o, dist)
                    if not safe:
                        has_violation = True
                        score = 1.0 - ratio
                        score_sum += score
                        violation_details.append(f"节点{other} (ratio={ratio:.4f}, score={score:.4f})")
                if has_violation:
                    logger.debug(f"      节点 {node}({du_n}) 存在违规 → 违规项: {violation_details} | 总违规分={score_sum:.4f}")
                    if score_sum > worst_score:
                        worst_score = score_sum
                        worst_node = node
                else:
                    logger.debug(f"      节点 {node}({du_n}) 无违规 → 总违规分=0.0")
            if worst_node != -1:
                final_clique.remove(worst_node)
                changed = True
                logger.debug(f"   ❌ 剔除违规最严重节点 {worst_node} → 剩余团大小={len(final_clique)}")
            else:
                logger.debug(f"   ✅ 迭代 {iteration} 无违规节点，净化完成")

        logger.debug(f"✅ 后处理净化完成: 最终团大小={len(final_clique)}, 节点列表={final_clique}")

        # 最终验证
        is_final_safe = True
        logger.debug(f"✅ 开始最终因果律验证")
        for i in range(len(final_clique)):
            if not is_final_safe:
                break
            for j in range(i+1, len(final_clique)):
                n1, n2 = final_clique[i], final_clique[j]
                d1, t1, _ = nodes[n1]
                d2, t2, _ = nodes[n2]
                if d1 == d2:
                    logger.warning(f"   ⚠️  最终团内出现同探测器 {d1}，标记为不安全")
                    is_final_safe = False
                    break
                dist = get_dist(str(d1), str(d2))
                if dist is None:
                    dist = 0
                safe, ratio = check_causality_strict(t1, t2, dist)
                if not safe:
                    logger.debug(f"   ❌ 最终团内违规: 节点{n1}({d1}) ↔ 节点{n2}({d2}) | ratio={ratio:.4f}")
                    is_final_safe = False
                    break

        if is_final_safe and len(final_clique) >= min_detectors:
            valid_matches = [(node_info[nid][0], node_info[nid][1]) for nid in final_clique]
            matching_times[key] = valid_matches
            valid_events += 1
            logger.debug(f"🎉 该事件有效！保留探测器数: {len(final_clique)}, 最终保留触发: {valid_matches}")
        else:
            logger.debug(f"❌ 该事件无效（团大小不足/因果律违规）")

    logger.debug(f"[Step 1] Causality Cleaning: {valid_events}/{total_lines} events retained.")

    times = {}
    signals = {}
    du_ids = {}
    for key, matches in matching_times.items():
        times[key] = {det_id: ns for det_id, ns in matches}
        signals[key] = None
        du_ids[key] = [det_id for det_id, _ in matches]

    return times, signals, du_ids
