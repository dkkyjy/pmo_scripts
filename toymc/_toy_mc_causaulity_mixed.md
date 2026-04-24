# Toy MC Causality Simulation Walkthrough

*2026-03-27T06:49:14Z by Showboat 0.6.1*
<!-- showboat-id: 2a00398b-0ce2-46f1-ac21-8a23c908ca4e -->

首先，代码导入了必要的库并定义了一些物理常数和模拟参数。光速定义为 /usr/bin/bash.299792458$ m/ns，模拟生成 00,000$ 个事件，每个事件涉及 $ 到 $ 个探测器。

```bash
sed -n '1,26p' _toy_mc_causaulity_gp65_histogram.py
```

```output
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
from datetime import datetime, timedelta

# 光速 (m/ns)
C_LIGHT_NS = 299792458.0 / 1e9
# 读取探测器坐标文件
COORD_FILE = "_gp65_rtksort.txt"
# 生成的信号数量（可自定义）
N_EVENTS = 100000  # 增加数量，让分布更明显
# 每个信号点亮的探测器数范围 [5,8]
MIN_DET_PER_EVENT = 5
MAX_DET_PER_EVENT = 8
# GPS起始时间（示例起始为1773019543，可自定义）
GPS_START = 1773019543.0
# 信号源到原点的距离范围 (m)：5km ~ 50km
MIN_SOURCE_DIST = 5000.0
MAX_SOURCE_DIST = 50000.0

# 存储所有信号源的theta和phi（角度制）
theta_list_deg = []
phi_list_deg = []


```

`load_du_coords` 函数负责从指定的文件中读取探测器的坐标（ID, X, Y, Z）。它过滤掉 ID 小于 1000 的记录。

```bash
sed -n '27,53p' _toy_mc_causaulity_gp65_histogram.py
```

```output
def load_du_coords(file_path):
    """加载DU编号和三维坐标，过滤注释和无效行"""
    du_coords = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 过滤注释行、空行
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            du_id = parts[0]
            # 过滤非数字DU编号（如CS）
            if not du_id.isdigit():
                continue
            du_id = int(du_id)
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                du_coords[du_id] = np.array([x, y, z], dtype=np.float64)
            except ValueError:
                continue
    return du_coords


```

`generate_spherical_wave_arrival` 是模拟的核心。它随机生成一个信号源位置，然后计算球面波到达各个随机选定的探测器的时间。

```bash
sed -n '54,100p' _toy_mc_causaulity_gp65_histogram.py
```

```output
def generate_spherical_wave_arrival(du_coords, n_det):
    """
    生成球面波到达探测器的时间（仅上半天球信号源，禁止单个DU多次触发）
    :param du_coords: 所有探测器坐标字典 {du_id: [x,y,z]}
    :param n_det: 本次点亮的探测器数
    :return: 触发探测器的时间字典 {du_id: t, ...}（每个DU仅1个时间）
    """
    # 随机选择n_det个探测器（保证无重复）
    selected_du = random.sample(list(du_coords.keys()), n_det)
    
    # 1. 随机生成信号源单位方向向量（仅上半天球：theta ∈ [0, π/2]）
    phi = np.random.uniform(0, 2 * np.pi)  # 方位角：0~360°（全方位）
    theta = np.arccos(np.random.uniform(0.05, 0.6))  # 极角：0~90°（上半天球）
    
    # 转换为角度制并存储
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    theta_list_deg.append(theta_deg)
    phi_list_deg.append(phi_deg)
    
    dir_vec = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # 2. 随机生成信号源到原点的距离（5~50km）
    r0 = np.random.uniform(MIN_SOURCE_DIST, MAX_SOURCE_DIST)
    # 信号源位置：从原点沿-dir_vec方向（入射方向）移动r0距离
    source_pos = -r0 * dir_vec
    
    # 3. 计算每个探测器到信号源的直线距离
    det_dist_to_source = {}
    for du in selected_du:
        det_pos = du_coords[du]
        # 探测器与信号源的欧氏距离
        dist = np.linalg.norm(det_pos - source_pos)
        det_dist_to_source[du] = dist
    
    # 4. 计算到达时间（ns）= 距离 / 光速（每个DU仅1个时间）
    det_time = {du: round(dist / C_LIGHT_NS) for du, dist in det_dist_to_source.items()}
    
    # 打印当前信号源参数（含极角验证）
    #print(f"  信号源参数：距离原点 {r0/1000:.2f} km，极角 {theta_deg:.2f}°（上半天球），方位角 {phi_deg:.2f}°")
    return det_time


```

`format_gps_and_output` 协调整个生成流程。它遍历事件索引，调用物理模型生成触发时间，并构建符合 `read_header.py` 规范的 YAML 数据结构。最后，它将所有数据写入 `toy_mc_events.yaml`。

```bash
sed -n '101,168p' _toy_mc_causaulity_gp65_histogram.py
```

```output
def format_gps_and_output(du_coords):
    """生成最终输出格式的GPS时间和触发列表（无重复DU），并输出为YAML文件"""
    gps_current = GPS_START
    yaml_data = {}
    
    # GPS epoch is 1980-01-06 00:00:00 UTC
    gps_epoch = datetime(1980, 1, 6)
    
    for event_idx in range(N_EVENTS):
        #print(f"=== 生成第 {event_idx+1} 个信号 ===")
        # 随机选择本次点亮的探测器数
        n_det = random.randint(MIN_DET_PER_EVENT, MAX_DET_PER_EVENT)
        # 生成球面波到达时间（仅上半天球，无重复DU触发）
        trigger_dict = generate_spherical_wave_arrival(du_coords, n_det)
        
        # 提取所有触发时间，找到基准纳秒（用于GPS的小数部分）
        all_trigger_times = list(trigger_dict.values())
        base_ns = min(all_trigger_times)  # 基准纳秒，GPS小数部分由其组成
        
        # 计算GPS时间：整数秒 + 基准纳秒/1e9
        gps_second = int(gps_current)
        gps_frac = base_ns / 1e9
        gps_time = gps_second + gps_frac
        
        # 计算 datetime
        # GPS time does not account for leap seconds in this simple conversion, 
        # but it matches the typical format requirement.
        event_datetime = gps_epoch + timedelta(seconds=gps_second)
        datetime_str = event_datetime.strftime("%Y-%m-%dT%H:%M:%S")
        
        # 触发时间转换为：绝对纳秒（匹配示例格式，每个DU仅1个元组）
        # trigger_list = [(du_id, t) for du_id, t in trigger_dict.items()]
        
        # 格式化输出字符串
        # trigger_str = ", ".join([f"({du}, {t})" for du, t in trigger_list])
        #print(f"{gps_time:.9f}: [{trigger_str}]\n")
        
        # 构建符合 read_header.py 格式的 payload
        event_number = event_idx + 1
        
        # time 字典: {du_id: [time_ns]}
        time_map = {int(du_id): [int(t)] for du_id, t in trigger_dict.items()}
        list_du_id = list(time_map.keys())
        
        payload = {
            "run_number": 1,  # 模拟数据，固定为1
            "event_number": event_number,
            "datetime": datetime_str,
            "gps_time": gps_second,
            "time": time_map,
            "du_id": list_du_id,
            "file": "toy_mc_simulation.root",
            "index": event_idx,
        }
        
        yaml_data[str(event_number)] = payload
        
        # GPS时间步进（0.1~0.5秒随机，模拟信号间隔）
        gps_step = np.random.uniform(0.1, 0.5)
        gps_current += gps_step
        
    # 写入 YAML 文件
    output_filename = "toy_mc_events.yaml"
    with open(output_filename, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"成功生成 {N_EVENTS} 个事件，并保存至 {output_filename}")


```

`plot_theta_phi_distribution` 函数使用 Matplotlib 绘制生成的信号源天顶角（Theta）和方位角（Phi）的分布直方图。

```bash
sed -n '169,216p' _toy_mc_causaulity_gp65_histogram.py
```

```output
def plot_theta_phi_distribution():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    """绘制theta（极角）和phi（方位角）的分布直方图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== 左图：theta（极角）分布 ==========
    ax1.hist(theta_list_deg, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    # 计算关键统计量
    theta_mean = np.mean(theta_list_deg)
    theta_median = np.median(theta_list_deg)
    theta_90 = np.percentile(theta_list_deg, 90)
    # 标注统计量
    ax1.axvline(theta_mean, color='red', linestyle='--', linewidth=2, label=f'均值: {theta_mean:.2f}°')
    ax1.axvline(theta_median, color='green', linestyle='--', linewidth=2, label=f'中位数: {theta_median:.2f}°')
    ax1.axvline(theta_90, color='orange', linestyle='--', linewidth=2, label=f'90%分位数: {theta_90:.2f}°')
    # 样式设置
    ax1.set_xlabel('Theta (极角) / °', fontsize=12)
    ax1.set_ylabel('信号源数量', fontsize=12)
    ax1.set_title('上半天球信号源极角 (Theta) 分布', fontsize=14)
    ax1.set_xlim(0, 90)  # 上半天球极角范围0~90°
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ========== 右图：phi（方位角）分布 ==========
    ax2.hist(phi_list_deg, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    # 计算关键统计量
    phi_mean = np.mean(phi_list_deg)
    phi_median = np.median(phi_list_deg)
    phi_90 = np.percentile(phi_list_deg, 90)
    # 标注统计量
    ax2.axvline(phi_mean, color='red', linestyle='--', linewidth=2, label=f'均值: {phi_mean:.2f}°')
    ax2.axvline(phi_median, color='green', linestyle='--', linewidth=2, label=f'中位数: {phi_median:.2f}°')
    ax2.axvline(phi_90, color='orange', linestyle='--', linewidth=2, label=f'90%分位数: {phi_90:.2f}°')
    # 样式设置
    ax2.set_xlabel('Phi (方位角) / °', fontsize=12)
    ax2.set_ylabel('信号源数量', fontsize=12)
    ax2.set_title('信号源方位角 (Phi) 分布', fontsize=14)
    ax2.set_xlim(0, 360)  # 方位角范围0~360°
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 整体布局
    plt.tight_layout()
    plt.savefig('theta_phi_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


```

最后，在 `if __name__ == '__main__':` 块中，脚本加载探测器坐标，启动生成流程。

```bash
sed -n '217,230p' _toy_mc_causaulity_gp65_histogram.py
```

```output
if __name__ == "__main__":
    # 加载探测器坐标
    du_coords = load_du_coords(COORD_FILE)
    if not du_coords:
        print("未加载到有效探测器坐标！")
    else:
        print(f"成功加载 {len(du_coords)} 个探测器坐标，开始生成模拟信号（仅上半天球）...\n")
        # 生成并输出模拟数据
        format_gps_and_output(du_coords)
        # 绘制theta和phi的分布直方图
        print("\n=== 绘制theta/phi分布直方图 ===")
        plot_theta_phi_distribution()
        
        # 打印关键统计信息
```
