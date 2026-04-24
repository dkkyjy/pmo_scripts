#!/usr/bin/env python3
"""
基于球面波的宇宙线射电信号模拟
- 各向同性采样：天顶角在指定范围内按 sin(theta) 分布
- 入射轴与地面交点（击中点）在探测器XY边界1.5倍内随机
- 只采样入射方向轴线CONE_ANGLE_DEG°锥形范围内的探测器

注意：大天顶角(>70°)时2°锥形可能无法覆盖足够探测器，建议调整参数
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import sys

# ==================== 可调参数 ====================
N_EVENTS = 100000  # 事例数
MIN_DET_PER_EVENT = 5
MAX_DET_PER_EVENT = 10
CONE_ANGLE_DEG = 1.5  # 锥形角度（度）- 如失败请增大此值
THETA_MIN_DEG = 50.0  # 最小天顶角
THETA_MAX_DEG = 88.0  # 最大天顶角 - 2°锥形只能在小天顶角(<30°)工作
MIN_SOURCE_HEIGHT = 3000.0  # 固定信号源高度(m)
MAX_SOURCE_HEIGHT = 8000.0  # 固定信号源高度(m)
XY_EXTEND_FACTOR = 1.5
MAX_ATTEMPTS = 100  # 增加尝试次数
GPS_START = 1773219543.0

# 物理常数
C_LIGHT_NS = 299792458.0 / 1e9
COORD_FILE = "../_gp65_rtksort.txt"

# 存储统计量
theta_list_deg = []
phi_list_deg = []
impact_x_list = []
impact_y_list = []
source_x_list = []
source_y_list = []
source_z_list = []
n_det_list = []
cone_angle_list = []


def load_du_coords(file_path):
    """加载DU编号和三维坐标"""
    du_coords = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            du_id = parts[0]
            if not du_id.isdigit():
                continue
            du_id = int(du_id)
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                du_coords[du_id] = np.array([x, y, z], dtype=np.float64)
            except ValueError:
                continue
    return du_coords


def compute_xy_bounds(du_coords):
    """计算探测器阵列的XY边界，并扩展"""
    x_coords = [pos[0] for pos in du_coords.values()]
    y_coords = [pos[1] for pos in du_coords.values()]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    x_min_ext = x_center - (x_center - x_min) * XY_EXTEND_FACTOR
    x_max_ext = x_center + (x_max - x_center) * XY_EXTEND_FACTOR
    y_min_ext = y_center - (y_center - y_min) * XY_EXTEND_FACTOR
    y_max_ext = y_center + (y_max - y_center) * XY_EXTEND_FACTOR
    
    return x_min_ext, x_max_ext, y_min_ext, y_max_ext


def sample_isotropic_theta():
    """各向同性采样天顶角 P(theta) ∝ sin(theta)"""
    theta_min_rad = np.radians(THETA_MIN_DEG)
    theta_max_rad = np.radians(THETA_MAX_DEG)
    
    cos_min = np.cos(theta_max_rad)
    cos_max = np.cos(theta_min_rad)
    
    cos_theta = np.random.uniform(cos_max, cos_min)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def generate_spherical_wave_arrival(du_coords, xy_bounds):
    """
    生成球面波到达时间
    固定信号源高度，计算所需距离
    """
    x_min, x_max, y_min, y_max = xy_bounds
    cone_angle_rad = np.radians(CONE_ANGLE_DEG)
    
    for attempt in range(MAX_ATTEMPTS):
        # 随机击中点
        impact_x = np.random.uniform(x_min, x_max)
        impact_y = np.random.uniform(y_min, y_max)
        impact_point = np.array([impact_x, impact_y, 0.0])
        
        # 各向同性采样天顶角和方位角
        theta_deg = sample_isotropic_theta()
        theta_rad = np.radians(theta_deg)
        phi_deg = np.random.uniform(0, 360)
        phi_rad = np.radians(phi_deg)
        
        # 根据固定高度计算距离：h = D * cos(theta) => D = h / cos(theta)
        cos_theta = np.cos(theta_rad)
        if cos_theta < 0.01:  # 避免除零
            continue
        distance = np.random.uniform(MIN_SOURCE_HEIGHT, MAX_SOURCE_HEIGHT) / cos_theta
        
        # 方向向量（从源指向地面）
        dir_vec = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            -cos_theta
        ])
        
        # 信号源位置
        source_pos = impact_point - distance * dir_vec
        axis_vec = dir_vec  # 从源指向地面/击中点
        
        # 筛选锥形内的探测器
        cone_du_list = []
        for du_id, det_pos in du_coords.items():
            r_vec = det_pos - source_pos
            r_norm = np.linalg.norm(r_vec)
            if r_norm < 1.0:
                continue
            r_vec_norm = r_vec / r_norm
            cos_angle = np.dot(r_vec_norm, axis_vec)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle <= cone_angle_rad:
                cone_du_list.append((du_id, r_norm, np.degrees(angle)))
        
        if len(cone_du_list) >= MIN_DET_PER_EVENT:
            # 按距离排序，选择最近的n_det个
            cone_du_list.sort(key=lambda x: x[1])
            n_det = random.randint(MIN_DET_PER_EVENT, 
                                  min(MAX_DET_PER_EVENT, len(cone_du_list)))
            selected_du = cone_du_list[:n_det]
            
            # 记录数据
            impact_x_list.append(impact_x)
            impact_y_list.append(impact_y)
            source_x_list.append(source_pos[0])
            source_y_list.append(source_pos[1])
            source_z_list.append(source_pos[2])
            n_det_list.append(n_det)
            theta_list_deg.append(theta_deg)
            phi_list_deg.append(phi_deg)
            cone_angle_list.append(max([x[2] for x in selected_du]))
            
            # 计算到达时间
            det_time = {}
            for du_id, dist, _ in selected_du:
                det_time[du_id] = round(dist / C_LIGHT_NS)
            
            return det_time, source_pos, impact_point, n_det, theta_deg, phi_deg, distance
    
    return None, None, None, 0, 0, 0, 0


def format_gps_and_output(du_coords, xy_bounds):
    """生成GPS时间和触发列表"""
    gps_current = GPS_START
    successful_events = 0
    failed_events = 0
    
    for event_idx in range(N_EVENTS):
        result = generate_spherical_wave_arrival(du_coords, xy_bounds)
        trigger_dict, source_pos, impact_point, n_det, theta_deg, phi_deg, distance = result
        
        if trigger_dict is None:
            failed_events += 1
            if failed_events == 1 or failed_events % 10 == 0:
                print(f"警告：已失败 {failed_events} 个事件", file=sys.stderr)
                print(f"  建议：增大CONE_ANGLE_DEG（当前{CONE_ANGLE_DEG}°）或减小THETA_MAX_DEG（当前{THETA_MAX_DEG}°）", file=sys.stderr)
            continue
        
        successful_events += 1
        
        all_trigger_times = list(trigger_dict.values())
        base_ns = min(all_trigger_times)
        gps_second = int(gps_current)
        gps_frac = base_ns / 1e9
        gps_time = gps_second + gps_frac
        
        trigger_list = [(du_id, t) for du_id, t in trigger_dict.items()]
        trigger_str = ", ".join([f"({du}, {t})" for du, t in trigger_list])
        print(f"{gps_time:.9f}: [{trigger_str}]")
        
        gps_step = np.random.uniform(0.1, 0.5)
        gps_current += gps_step
    
    return successful_events, failed_events


def plot_distributions(du_coords):
    """绘制分布图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 左上：击中点XY分布
    ax1 = axes[0, 0]
    du_x = [pos[0] for pos in du_coords.values()]
    du_y = [pos[1] for pos in du_coords.values()]
    ax1.scatter(du_x, du_y, c='red', s=30, marker='s', label='Detectors', zorder=5, edgecolors='black')
    if impact_x_list:
        ax1.scatter(impact_x_list, impact_y_list, c='blue', s=5, alpha=0.5, label='Impact points', zorder=1)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Impact Points (XY)\n{len(impact_x_list)} events', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 中上：Theta分布
    ax2 = axes[0, 1]
    if theta_list_deg:
        ax2.hist(theta_list_deg, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax2.axvline(np.mean(theta_list_deg), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean={np.mean(theta_list_deg):.1f}°')
    ax2.set_xlabel('Theta (Zenith Angle) / °', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Zenith ({THETA_MIN_DEG}°-{THETA_MAX_DEG}°, sinθ)', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 右上：Phi分布
    ax3 = axes[0, 2]
    if phi_list_deg:
        ax3.hist(phi_list_deg, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(phi_list_deg), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(phi_list_deg):.1f}°')
    ax3.set_xlabel('Phi (Azimuth) / °', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Azimuth (Uniform)', fontsize=13)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 左下：信号源高度
    ax4 = axes[1, 0]
    if source_z_list:
        ax4.hist(np.array(source_z_list)/1000, bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax4.axvline(MIN_SOURCE_HEIGHT/1000, color='red', linestyle='--', linewidth=2,
                   label=f'Fixed={MIN_SOURCE_HEIGHT/1000:.1f} km')
    ax4.set_xlabel('Source Height (km)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Source Height', fontsize=13)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 中下：探测器数量
    ax5 = axes[1, 1]
    if n_det_list:
        unique_counts = sorted(set(n_det_list))
        counts = [n_det_list.count(x) for x in unique_counts]
        ax5.bar(unique_counts, counts, color='coral', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Detectors', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.set_title(f'Detectors per Event ({CONE_ANGLE_DEG}° cone)', fontsize=13)
        ax5.set_xticks(unique_counts)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 右下：实际锥形角度
    ax6 = axes[1, 2]
    if cone_angle_list:
        ax6.hist(cone_angle_list, bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax6.axvline(np.mean(cone_angle_list), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(cone_angle_list):.2f}°')
        ax6.axvline(CONE_ANGLE_DEG, color='blue', linestyle='--', linewidth=2,
                   label=f'Limit={CONE_ANGLE_DEG}°')
    ax6.set_xlabel('Max Angle / °', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Actual Cone Angle', fontsize=13)
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_cone_isotropic.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: distribution_cone_isotropic.png", file=sys.stderr)


if __name__ == "__main__":
    print("="*60, file=sys.stderr)
    print("宇宙线球面波模拟 (各向同性 + 2°锥形)", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    du_coords = load_du_coords(COORD_FILE)
    if not du_coords:
        print("错误：未加载到探测器坐标！", file=sys.stderr)
        exit(1)
    
    print(f"\n加载了 {len(du_coords)} 个探测器", file=sys.stderr)
    
    xy_bounds = compute_xy_bounds(du_coords)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print("参数配置:", file=sys.stderr)
    print(f"  天顶角范围: {THETA_MIN_DEG}° - {THETA_MAX_DEG}° (建议: 50-70°)", file=sys.stderr)
    print(f"  锥形角度: {CONE_ANGLE_DEG}° (如失败请增大到5°或10°)", file=sys.stderr)
    print(f"  变化信号源高度: {MIN_SOURCE_HEIGHT/1000:.1f} - {MAX_SOURCE_HEIGHT/1000:.1f} km", file=sys.stderr)
    print(f"  每事件探测器数: {MIN_DET_PER_EVENT} - {MAX_DET_PER_EVENT}", file=sys.stderr)
    print(f"  目标事件数: {N_EVENTS}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    
    successful, failed = format_gps_and_output(du_coords, xy_bounds)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"结果: 成功 {successful} / 目标 {N_EVENTS}, 失败 {failed}", file=sys.stderr)
    
    if successful == 0:
        print("\n错误：所有事件都失败了！", file=sys.stderr)
        print("解决方案（请修改代码顶部参数）:", file=sys.stderr)
        print(f"  1. 增大 CONE_ANGLE_DEG: {CONE_ANGLE_DEG}° → 5° 或 10°", file=sys.stderr)
        print(f"  2. 减小 THETA_MAX_DEG: {THETA_MAX_DEG}° → 60° 或 70°", file=sys.stderr)
        print(f"  3. 减小 MIN_DET_PER_EVENT: {MIN_DET_PER_EVENT} → 3 或 4", file=sys.stderr)
        exit(1)
    
    if theta_list_deg:
        plot_distributions(du_coords)
        
        print(f"\n统计:", file=sys.stderr)
        print(f"  天顶角: {np.mean(theta_list_deg):.1f}° ± {np.std(theta_list_deg):.1f}°", file=sys.stderr)
        print(f"  方位角: {np.mean(phi_list_deg):.1f}°", file=sys.stderr)
        print(f"  探测器数: {np.mean(n_det_list):.1f}", file=sys.stderr)
