import numpy as np
from scipy.optimize import minimize
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque
from itertools import combinations
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy import linalg

C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259  # m/ns
def estimate_initial_direction_robust(positions, t_s, c= C_LIGHT_NS ):
    """
    使用 SVD/最小二乘法稳健地估计平面波初值 (Zenith, Azimuth, t0)。
    
    参数:
        positions: (N, 3) numpy array, 探测器坐标 [x, y, z] (单位: 米)
        t_s: (N,) numpy array, 触发时间 (单位: 秒)
        c: 光速 (m/s)
        
    返回:
        zenith_deg: 天顶角 (度), 0=天顶, 90=水平
        azimuth_deg: 方位角 (度), 0=北, 90=东 (根据坐标系定义调整)
        t0_est: 估计的时间零点 (秒)
        success: 布尔值，表示估算是否成功
    """
    n_det = len(positions)
    
    # 1. 基本检查
    if n_det < 3:
        # 探测器太少，无法确定平面，返回默认天顶方向
        return 0.0, 0.0, np.median(t_s), False
    
    positions = np.array(positions)
    t_s = np.array(t_s)
    
    # 2. 去中心化 (提高数值稳定性)
    pos_mean = np.mean(positions, axis=0)
    t_mean = np.mean(t_s)
    
    A = positions - pos_mean          # (N, 3)
    b = (t_s - t_mean) * c            # (N,) 将时间差转换为距离差
    
    # 3. 使用最小二乘法求解平面法向量 v
    # 模型: A * v ≈ b
    # 其中 v = [nx, ny, nz] 是波传播方向的余弦分量 (方向矢量)
    # 物理意义: t_i - t_mean = (r_i - r_mean) . v / c
    try:
        # lstsq 返回解 v, 残差, rank, singular_values
        v, residuals, rank, s = linalg.lstsq(A, b)
    except linalg.LinAlgError:
        return 0.0, 0.0, np.median(t_s), False
        
    # 4. 物理约束检查与修正
    # 理论上 |v| 应该 <= 1 (因为 v 是单位方向矢量在坐标轴上的投影)
    # 如果 |v| > 1，说明由于噪声导致拟合出的“视速度”小于光速，这是不可能的。
    # 我们将其投影到单位球面上，保留方向，丢弃错误的速度大小信息。
    v_norm = np.linalg.norm(v)
    
    if v_norm < 1e-6:
        # 几乎零向量，可能是所有探测器同时触发或几何中心对称且噪声大
        return 0.0, 0.0, t_mean, False
    
    if v_norm > 1.0:
        # 发生“超光速”拟合假象（实际是亚光速假象，即 dt 太大），强制归一化
        # 这比原来代码中强行计算 arccos(>1) 或分支处理要物理得多
        v = v / v_norm
        # 可选：这里可以记录一个警告，说明该事件噪声较大
        # print(f"Warning: Fitted speed < c, normalizing direction. Norm was {v_norm:.2f}")
    else:
        # 如果 v_norm <= 1，说明拟合非常完美或噪声很小，直接使用
        # 但为了作为方向初值，通常也建议归一化，除非你想保留曲率信息（平面波不需要）
        v = v / v_norm

    # 5. 计算天顶角和方位角
    # 假设 v 指向波的传播方向 (从源到探测器)
    # Zenith (theta): 与 Z 轴的夹角. cos(theta) = vz
    # 注意：如果 v 是波矢量 k 的方向，那么源的方向是 -v。
    # 通常重建中，Zenith=0 表示来自天顶（向下传播），此时 v_z 应该是正的 (如果 z 轴向上) 
    # 或者 v_z 是负的 (如果 z 轴向下)。
    # 假设标准坐标系：Z 轴向上。波从天顶来，传播方向向下，v_z < 0。
    # 但通常我们定义天顶角为入射方向与天顶的夹角。
    # 让我们假设 v 是波前的法向量，指向波传播的方向。
    
    vz = v[2]
    vx = v[0]
    vy = v[1]
    
    # 确保 acos 的参数在 [-1, 1] 之间 (防止浮点数误差)
    cos_theta = np.clip(vz, -1.0, 1.0)
    
    # 如果 v 指向下方 (vz < 0)，则天顶角 < 90。
    # 如果 v 指向上方 (vz > 0)，则意味着波从地下上来？或者坐标系定义问题。
    # 通常宇宙线/大气簇射是从上往下，vz 应该是负值 (如果 Z 向上)。
    # 天顶角定义为入射方向与垂直向上的夹角。
    # 入射方向 = -v (如果 v 是传播方向)。
    # 所以 cos(zenith) = (-v) . (0,0,1) = -vz
    # 如果 vz = -1 (向下传), cos(zenith) = 1 -> zenith = 0. 正确。
    # 如果 vz = 0 (水平传), cos(zenith) = 0 -> zenith = 90. 正确。
    
    zenith_rad = np.arccos(-vz) 
    zenith_deg = np.degrees(zenith_rad)
    
    # 方位角 Azimuth: 在 XY 平面的投影角度
    # 入射方向的水平分量是 (-vx, -vy)
    azimuth_rad = np.arctan2(-vy, -vx)
    azimuth_deg = np.degrees(azimuth_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0
        
    # 6. 估算 t0
    # t0 定义为波前通过坐标原点 (0,0,0) 的时间，或者通过阵列中心的时间
    # 这里我们估算通过阵列中心 (pos_mean) 的时间，这对优化器更友好
    # t_center = t_mean - (pos_mean . v) / c ? 
    # 其实去中心化后，t_mean 就是波前通过 pos_mean 的近似时间
    # 更精确的 t0 (通过原点) = t_mean - (pos_mean . v) / c
    # 但作为初值，直接给 t_mean 通常足够，或者给通过原点的时间
    t0_est = t_mean - np.dot(pos_mean, v) / c
    
    return zenith_deg, azimuth_deg, t0_est, True

def calculate_azimuth(direction_vector):
    """
    计算方位角（azimuth），它是方向矢量在x-y平面的投影与x轴的夹角。
    """
    x, y, z = direction_vector
    azimuth = np.degrees(np.arctan2(y, x))
    if azimuth < 0:
        azimuth += 360
    return azimuth

def estimate_initial_direction(detector_positions, times, c):
    # 估计初始方向的函数实现（省略详细代码）
    """
    估计平面波的初始入射方向和时间。
    :param detector_positions: 探测器的坐标，字典形式 {det_id: position_array}
    :param times: 探测器接收到信号的时间，字典形式 {det_id: time}
    :param c: 波速，通常为光速
    :return: 平面波的初始入射方向和时间 (direction, t0)
    """

    # 提取探测器的位置和时间
    positions = np.array([detector_positions[det_id] for det_id in times.keys()])
    ant_id = np.array([det_id for det_id in times.keys()])
    t_ns = np.array([times[det_id] for det_id in times.keys()])  # 纳秒时间
    #print("The event consists of DUs",times.keys(),times)

    min_index = np.argmin(t_ns)
    max_index = np.argmax(t_ns)
    delta_t = t_ns[max_index] - t_ns[min_index]

    min_position = positions[min_index]
    max_position = positions[max_index]
    #print("earliest ",ant_id[min_index],min_position)
    #print("latest ",ant_id[max_index],max_position)

    distance = np.linalg.norm(max_position - min_position)

    dx = max_position[0] - min_position[0]
    dy = max_position[1] - min_position[1]

    azimuth = np.arctan2(dy, dx) * 180 / np.pi 
    azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
    #zenith = 20
    #print("parameters check:", round(azimuth,3),"deg",delta_t,"ns",round(distance,3),"m" )
    #print("Try to assign direction",delta_t,distance/c)
    if delta_t > distance/c :
        #print("case 1 : ",delta_t,distance/c)
        zenith = np.arccos( (distance/c)/delta_t )* 180/np.pi
    if delta_t < distance/c :
        #print("ratio : ",delta_t,distance/c)
        zenith = 90 - np.arccos( delta_t/(distance/c) )* 180/np.pi
    if ant_id[min_index] == ant_id[max_index] or abs(delta_t ) < 10 :
        print("Warning~~~~~~~~~~~All DUs triggered at the same time, please confirm.T is ",t_ns)
        zenith = 0

    t_s = t_ns  # 转换为秒
    t_s_mean = np.mean(t_s)
    t_s_min = np.min(t_s)
    t_s_min = np.median(t_s)
    #t_s_min = times['1075']
    #print("Roughly to estimate Distance,zenith,azimuth : ", np.around(distance,2),"m",np.around(zenith,2),"deg",np.around(azimuth,2),"deg",t_s_mean,t_s_min)

    def residuals(params):
        theta, phi, t0 = params
        direction = np.array([
            np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
            np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
            np.cos(np.deg2rad(theta))
        ])
        predicted_times = t0 - np.dot(positions, direction) / c
        residuals = []
        for i in range(len(predicted_times)):
            for j in range(i + 1, len(predicted_times)):
                residuals.append(predicted_times[i] - predicted_times[j] - (t_s[i] - t_s[j]))
        return residuals

    # 初始猜测值
    initial_guess = [zenith, azimuth, t_s_min]  # 初始方向角度和平均时间

    # 最小化残差
    result = least_squares(residuals, initial_guess)

    # 提取结果
    theta, phi, t0 = result.x
    direction = np.array([
        np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
        np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
        np.cos(np.deg2rad(theta))
    ])

    return direction, t0


def calculate_PWM_chi_square(matches, detector_positions, direction, t0, c):
    """
    计算卡方值。
    :param matches: 匹配的探测器和时间，字典形式 {det_id: ns}
    :param detector_positions: 探测器的坐标，字典形式 {det_id: position_array}
    :param direction: 平面波的入射方向
    :param t0: 平面波的初始时间
    :param c: 波速，通常为光速
    :return: 卡方值
    """
    total_error = 0.0
    # times = {detector_id: ns for detector_id, ns in matches.items()}
    #print("Times ", times)

    # 提取探测器的位置和时间
    positions = np.array([detector_positions[det_id] for det_id, ns in matches.items()])
    det_ids = list(matches.keys())
    t_ns = list(matches.values())

    def theoretical_time_difference(pos1, pos2):
        return np.dot(pos2 - pos1, direction) / c

    # 计算两两探测器之间的时间差
    for i in range(len(det_ids)):
        for j in range(i + 1, len(det_ids)):
            det_id_i = det_ids[i]
            det_id_j = det_ids[j]

            pos_i = detector_positions[det_id_i]
            pos_j = detector_positions[det_id_j]
            #t_j = t_j + id_time_dict[int(det_id_j)]
            #t_i = t_i + id_time_dict[int(det_id_i)]
            t_j = t_ns[j]
            t_i = t_ns[i]
            '''
            if time_data.get(int(det_id_j)):
                t_j = t_j + time_data.get(int(det_id_j))['mean']
            if time_data.get(int(det_id_i)):
                t_i = t_i + time_data.get(int(det_id_i))['mean']
            '''
            measured_time_difference = (t_j - t_i)
            theoretical_time_diff = theoretical_time_difference(pos_i, pos_j)
            chi_square_contribution = ( ((measured_time_difference - theoretical_time_diff)/6.0) ** 2)

            #print(f"chi2_calculation: detectors {det_id_i} and {det_id_j}, measured_diff: {measured_time_difference}, theoretical_diff: {theoretical_time_diff}, contribution: {chi_square_contribution}")
            #print(get_time_offset(det_id_j),get_time_offset(det_id_i))

            total_error += chi_square_contribution
    reduced_total_error = total_error/(len(matches) - 3) 
    return reduced_total_error

def plane_wave_model(matching_times, matching_signals, detector_positions):
    """
    Fit each event using the plane wave model and return directions and related information that meet the criteria.

    Returns:
      directions, zeniths, azimuths, chi_squares
    """
    index = 0
    directions = {}
    zeniths = {}
    azimuths = {}
    chi_squares = {}
    c = C_LIGHT_NS

    for key, times in matching_times.items():
        index += 1
        if index % 1000 == 1:
            print(f'PWM EventNo{index}/{len(matching_times)}')
        
        # In estimation.py, times is already a dict {det_id: ns}
        # In the original plane_wave_model.py, matches was a list of (det_id, ns)
        # We now assume times is a dict {det_id: ns} to match estimation.py
        matches = {key: value for key, value in times.items()}
        # matches = {key: value[0] for key, value in times.items()}

        
        t_ns = np.array([matches[det_id] for det_id in matches.keys()])
        current_pos = np.array([detector_positions[det_id] for det_id in matches.keys()])

        def objective_function(params):
            theta, phi, t0 = params
            direction = np.array([
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta))
            ])
            total_error = 0.0
            det_ids = list(matches.keys())
            num_detectors = len(det_ids)
            for i in range(num_detectors):
                for j in range(i + 1, num_detectors):
                    det_id_i = det_ids[i]
                    det_id_j = det_ids[j]
                    pos_i = detector_positions[det_id_i]
                    pos_j = detector_positions[det_id_j]
                    time_i = matches[det_id_i]
                    time_j = matches[det_id_j]
                    predicted_time_i = t0 + np.dot(pos_i, direction) / c
                    predicted_time_j = t0 + np.dot(pos_j, direction) / c
                    relative_time_theoretical = predicted_time_i - predicted_time_j
                    relative_time_measured = (time_i - time_j) / 1e0
                    total_error += ((relative_time_theoretical - relative_time_measured) / 6) ** 2
            return total_error

        zenith_init, azimuth_init, t0_init, success = estimate_initial_direction_robust(current_pos, t_ns)
        bounds = [(0, 95), (-180, 180), (-np.inf, np.inf)]
        if not success:
            zenith_init = 45.0
            azimuth_init = 34.0
            t0_init = np.median(t_ns)
            print("Warning: Robust initial estimation failed, using defaults.")

        best_result = None
        min_chi2 = np.inf
        candidates = [
            (zenith_init, azimuth_init, t0_init),
            (35, azimuth_init, t0_init),
            (55.0, azimuth_init, t0_init),
            (85.0, azimuth_init, t0_init)
        ]

        for guess in candidates:
            try:
                res = minimize(objective_function, guess, method='L-BFGS-B', bounds=bounds)
                if res.success and res.fun < min_chi2:
                    min_chi2 = res.fun
                    best_result = res
            except:
                continue

        if best_result is None:
            print(f"Warning: Event {index} optimization failed for all candidates, skipping.")
            continue

        theta = best_result.x[0]
        if theta > 90: theta = 180 - theta
        phi = best_result.x[1]
        t0 = best_result.x[2]
        direction = np.array([
            np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
            np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
            np.cos(np.deg2rad(theta))
        ])
        azimuth = calculate_azimuth(direction)
        # zenith = calculate_zenith(direction) # If you have calculate_zenith
        zenith = theta # Since theta is zenith in degrees
        chi_square = calculate_PWM_chi_square(matches, detector_positions, direction, t0, c)

        # In estimation.py style, we return all events that were successfully fitted
        directions[key] = direction
        zeniths[key] = float(zenith)
        azimuths[key] = float(azimuth)
        chi_squares[key] = float(chi_square)

    return directions, zeniths, azimuths, chi_squares
