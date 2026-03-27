import numpy as np
from scipy.optimize import minimize
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import List, Dict, Set, Tuple, Optional
from scipy import linalg
import re
GPS_UTC_OFFSET = 18
c = 299792458.0/1e9/1.000
np.set_printoptions(precision=3)
def gps_to_utc(gps_time):
    """
    将GPS时间转换为UTC时间。
    """
    gps_epoch = datetime(1970, 1, 1)
    utc_time = gps_epoch + timedelta(seconds=(gps_time - GPS_UTC_OFFSET + 8*3600))
    return utc_time

def azimuth_boundary_penalty(phi, width=10.0, strength=1e3):
    """
    对方位角在0°或360°边界附近施加软惩罚，缓解0度堆积效应。
    
    参数:
        phi: 方位角度数（可能超出[0,360]范围）
        width: 边界影响宽度（度），默认30°
        strength: 惩罚强度系数
    
    返回:
        penalty: 添加到目标函数的惩罚值
    """
    # 规范化到[0,360]计算到边界的距离
    phi_norm = phi % 360
    dist_0 = phi_norm
    dist_360 = 360 - phi_norm
    min_dist = min(dist_0, dist_360)
    
    # 软惩罚：边界附近二次增长
    if min_dist < width:
        factor = (width - min_dist) / width
        return strength * (factor ** 2)
    return 0.0

def estimate_initial_direction_robust(positions, t_s, c=299792458.0):
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

def fit_3param(matches, detector_positions, c, initial_guesses=None):
    """
    3参数球面波拟合（rho, theta, phi），t0通过解析方式求解。
    
    参数:
        matches: 探测器列表 [(det_id, ns), ...]
        detector_positions: 探测器位置字典
        c: 光速
        initial_guesses: 初始猜测列表，每个元素为 [rho, theta, phi]
    
    返回:
        rho, theta, phi, t0, chi2, min_chi2, success
    """
    if len(matches) != 4:
        return None, None, None, None, np.inf, np.inf, False
    
    times = {d: ns for d, ns in matches}
    t_mean_ns = np.mean(list(times.values()))
    
    def obj_3param(p):
        rho_3, theta_3, phi_3 = p
        src_3 = rho_3 * np.array([
            np.sin(np.deg2rad(theta_3))*np.cos(np.deg2rad(phi_3)),
            np.sin(np.deg2rad(theta_3))*np.sin(np.deg2rad(phi_3)),
            np.cos(np.deg2rad(theta_3))
        ])
        travel_times_3 = [np.linalg.norm(detector_positions[did] - src_3) / c 
                          for did, _ in matches]
        measured_times_3 = [ns for _, ns in matches]
        t0_3 = np.mean(measured_times_3) - np.mean(travel_times_3)
        err_3 = sum(((tt + t0_3 - mt) / 6.0) ** 2 
                    for tt, mt in zip(travel_times_3, measured_times_3))
        penalty_3 = azimuth_boundary_penalty(phi_3, width=10.0, strength=5e2)
        return err_3 + penalty_3
    
    bounds_3param = [(2e3, 6e4), (30, 95), (-30, 390)]
    
    # 默认候选初值
    if initial_guesses is None:
        initial_guesses = [
            [3e4, 60.0, 0.0],
            [2e4, 45.0, 0.0],
            [4e4, 75.0, 0.0],
        ]
    
    best_res = None
    best_chi2 = np.inf
    
    for guess in initial_guesses:
        try:
            res = minimize(obj_3param, guess, method='L-BFGS-B', bounds=bounds_3param)
            if res.success and res.fun < best_chi2:
                best_chi2 = res.fun
                best_res = res
        except:
            continue
    
    if best_res is None:
        return None, None, None, None, np.inf, np.inf, False
    
    rho, theta, phi = best_res.x
    phi = phi % 360
    
    # 镜像修正
    if theta > 90:
        theta = 180 - theta
    
    # 解析计算t0
    src = rho * np.array([
        np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
        np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
        np.cos(np.deg2rad(theta))
    ])
    travel_times = [np.linalg.norm(detector_positions[did] - src) / c 
                    for did, _ in matches]
    measured_times = [ns for _, ns in matches]
    t0 = np.mean(measured_times) - np.mean(travel_times)
    
    # 计算chi2（自由度=1）
    chi2 = best_chi2 / max(1, len(matches) - 3)
    
    return rho, theta, phi, t0, chi2, best_chi2, True


def calculate_spherical_chi_square(matches, detector_positions, source_position, t0, c):
    """
    计算卡方值。
    :param matches: 探测器接收到信号的时间，列表形式 [(det_id, time)]
    :param detector_positions: 探测器的坐标，字典形式 {det_id: position_array}
    :param direction: 平面波的入射方向
    :param t0: 平面波的初始时间
    :param c: 波速，通常为光速
    :return: 卡方值
    """
    total_error = 0.0
    times = {detector_id: ns for detector_id, ns in matches}
    #print("Times ", times)

    # 提取探测器的位置和时间
    positions = np.array([detector_positions[det_id] for det_id in times.keys()])
    t_ns = np.array([times[det_id] for det_id in times.keys()])
    average_t_value = np.mean(t_ns)

    for detector_id, ns in matches:
        detector_position = np.array(detector_positions[detector_id])
        distance = np.linalg.norm(detector_position - source_position)
        theoretical_time = t0 + distance / c  # 理论接收时间
        #observed_time = ns - average_t_value + id_time_dict[int(detector_id)] #/ 1e9  # 将纳秒转为秒
        observed_time = ns - average_t_value  #/ 1e9  # 将纳秒转为秒
        '''
        if int(detector_id) == 2001 : 
            ns -= 270.0
        if time_data.get(int(detector_id)):
            observed_time = ns - average_t_value + time_data.get(int(detector_id))['mean'] #/ 1e9  # 将纳秒转为秒
        else :
           observed_time = ns - average_t_value  #/ 1e9  # 将纳秒转为秒
        '''
        residual = ( (observed_time - theoretical_time )/6e0)**2
        print(f"chi2_calculation_sperical: detectors {detector_id}, measured_diff: {observed_time:.2f}, theoretical_diff: {theoretical_time:.2f}, contribution: {residual:.2f}")
        total_error += residual 
    reduced_total_err = total_error/(len(matches) - 4)
    return reduced_total_err

def spherical_wave_model(matching_times, matching_signals, detector_positions, initial_directions):
    """球面波重建模型"""
    source_positions = {}
    chi_squares = {}
    c = 299792458.0/1e9/1.000

    for i, (key, times) in enumerate(matching_times.items()):
        if key not in initial_directions:
            continue
        if len(times) < 5:
            continue
        
        matches = list(times.items())
        t_ns = np.array(list(times.values()))
        current_pos = np.array([detector_positions[det_id] for det_id in times.keys()])
        t_mean_ns = np.mean(t_ns)
        min_index = np.argmin(t_ns)
        initial_direction = initial_directions[key]
        initial_t0 = t_ns[min_index] - t_mean_ns - 7.3e3/c
        index=i+1
        if(index%50 == 1 ) :
            print(f"SWM EventNo.{index}/{len(initial_directions)}", initial_direction, initial_t0)
        initial_rho = 7.3e3  # Initial estimated source distance
        vector = np.array([initial_direction[0], initial_direction[1], initial_direction[2]])
        norm = np.linalg.norm(vector)
        initial_phi = np.arctan2(initial_direction[1], initial_direction[0]) * (180 / np.pi)
        initial_theta = np.arccos(initial_direction[2] / norm) * (180 / np.pi)

        zenith_init, azimuth_init, t0_init, success = estimate_initial_direction_robust(current_pos, t_ns, c)
        if not success:
            zenith_init = 45.0
            azimuth_init = 0.0
            t0_init = np.median(list(times.values()))
            print("Warning: Robust initial estimation failed, using defaults.")
        def obj(p):
            rho, theta, phi, t0 = p
            src = rho * np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                                  np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                                  np.cos(np.deg2rad(theta))])
            err = 0
            for did, ns in matches:
                tp = np.linalg.norm(detector_positions[did] - src) / c + t0
                err += ((tp - (ns - t_mean_ns)) / 6.0) ** 2
            return err
        
        bounds = [(2e3, 6e4), (30, 95), (-30, 390), (-np.inf, np.inf)]
        best_result = None
        min_chi2 = np.inf

        candidates = [
            (initial_rho, zenith_init, azimuth_init, t0_init),
            (initial_rho, initial_theta, initial_phi, initial_t0),
            (3e4, 55.0, initial_phi, initial_t0),
            (3e4, 60.0, initial_phi, initial_t0),
            (3e4, 65.0, initial_phi, initial_t0),
            (3e4, 70.0, initial_phi, initial_t0),
            (3e4, 75.0, initial_phi, initial_t0),
            (4e4, 80.0, initial_phi, initial_t0),
            (2e4, 85.0, initial_phi, initial_t0)
        ]
        
        for guess in candidates:
            try:
                res = minimize(obj, guess, method='L-BFGS-B', bounds=bounds)
                if res.success and res.fun < min_chi2:
                    min_chi2 = res.fun
                    best_result = res
            except:
                continue
        
        if best_result:
            rho, theta, phi, t0 = best_result.x
            if theta - 90 > 0 : theta = 180 - theta
            src = rho * np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                                  np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                                  np.cos(np.deg2rad(theta))])
            chi2 = min_chi2 / max(1, len(matches)-4)
            if chi2 < 1e-2: chi2=max(chi2,1e-2)
            
            original_chi2 = chi2
            needs_3param_fit = False
             
            while chi2 > 10 and len(matches) >= 5:
                if len(matches) == 5 and chi2 > 200:
                    best_4du_chi2 = np.inf
                    best_4du_matches = None
                    best_4du_params = None
                    
                    for combo in combinations(matches, 4):
                        combo_matches = list(combo)
                        
                        candidates_4du = [
                            [rho, theta, phi],
                            [rho*1.2, theta, phi+90],
                            [rho*0.8, theta, phi-120],
                            [rho, min(95, theta+10), phi-30],
                            [rho, max(30, theta-10), phi],
                        ]
                        
                        rho_4, theta_4, phi_4, t0_4, chi2_4du, min_chi2_4, success_4 = fit_3param(
                            combo_matches, detector_positions, c, candidates_4du
                        )
                        
                        if success_4 and chi2_4du < best_4du_chi2:
                            best_4du_chi2 = chi2_4du
                            best_4du_matches = combo_matches
                            best_4du_params = [rho_4, theta_4, phi_4]
                            best_4du_t0 = t0_4
                            best_4du_min_chi2 = min_chi2_4
                    
                    if best_4du_matches and best_4du_chi2 < chi2 * 0.5:
                        matches = best_4du_matches
                        chi2 = best_4du_chi2
                        rho, theta, phi = best_4du_params
                        t0 = best_4du_t0
                        min_chi2 = best_4du_min_chi2
                        src = rho * np.array([
                            np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                            np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                            np.cos(np.deg2rad(theta))
                        ])
                        needs_3param_fit = True
                        break
                
                du_contributions = []
                for did, ns in matches:
                    tp = np.linalg.norm(detector_positions[did] - src) / c + t0
                    contribution = ((tp - (ns - np.mean(np.array([m[1] for m in matches])))) / 6.0) ** 2
                    du_contributions.append((contribution, did, ns))
                
                du_contributions.sort(reverse=True)
                removed_du = None
                for contrib, did_to_remove, _ in du_contributions:
                    if len(matches) - 1 >= 3:
                        new_matches = [(d, ns) for d, ns in matches if d != did_to_remove]
                        n_new = len(new_matches)
                        
                        if n_new == 4:
                            candidates_4du = [
                                [rho, theta, phi],
                                [rho*1.2, theta, phi+20],
                                [rho*0.8, theta, phi-30],
                                [rho, min(95, theta+10), phi-90],
                                [rho, max(30, theta-10), phi],
                                [rho, theta, (phi+180)%360],
                            ]
                            
                            rho_4, theta_4, phi_4, t0_4, chi2_4, min_chi2_4, success_4 = fit_3param(
                                new_matches, detector_positions, c, candidates_4du
                            )
                            
                            if success_4 and chi2_4 < chi2 * 0.5:
                                matches = new_matches
                                removed_du = did_to_remove
                                rho, theta, phi, t0 = rho_4, theta_4, phi_4, t0_4
                                chi2 = chi2_4
                                min_chi2 = min_chi2_4
                                src = rho * np.array([
                                    np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                                    np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                                    np.cos(np.deg2rad(theta))
                                ])
                                needs_3param_fit = True
                            break
                        else:
                            def obj_reduced(p):
                                rho_r, theta_r, phi_r, t0_r = p
                                phi_norm = phi_r % 360
                                src_r = rho_r * np.array([
                                    np.sin(np.deg2rad(theta_r))*np.cos(np.deg2rad(phi_norm)),
                                    np.sin(np.deg2rad(theta_r))*np.sin(np.deg2rad(phi_norm)),
                                    np.cos(np.deg2rad(theta_r))
                                ])
                                err_r = 0
                                for d_r, ns_r in new_matches:
                                    tp_r = np.linalg.norm(detector_positions[d_r] - src_r) / c + t0_r
                                    err_r += ((tp_r - (ns_r - np.mean(np.array([m[1] for m in new_matches])))) / 6.0) ** 2
                                penalty = azimuth_boundary_penalty(phi_r, width=10.0, strength=5e2)
                                return err_r + penalty
                            
                            candidates_n = [
                                [rho, theta, phi, t0],
                                [rho*1.2, theta, phi, t0],
                                [rho*0.8, theta, phi, t0],
                                [rho, min(95, theta+10), phi, t0],
                                [rho, max(30, theta-10), phi, t0],
                                [rho, theta, (phi+180)%360, t0],
                            ]
                            
                            best_res_reduced = None
                            best_chi2_reduced = np.inf
                            for guess_n in candidates_n:
                                try:
                                    res_n = minimize(obj_reduced, guess_n, 
                                                    method='L-BFGS-B', bounds=bounds)
                                    if res_n.success and res_n.fun < best_chi2_reduced:
                                        best_chi2_reduced = res_n.fun
                                        best_res_reduced = res_n
                                except:
                                    continue
                            
                            if best_res_reduced:
                                new_chi2_val = best_res_reduced.fun / max(1, n_new - 4)
                                if new_chi2_val < chi2 * 0.5:
                                    rho, theta, phi, t0 = best_res_reduced.x
                                    phi = phi % 360
                                    if theta - 90 > 0:
                                        theta = 180 - theta
                                    src = rho * np.array([
                                        np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                                        np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                                        np.cos(np.deg2rad(theta))
                                    ])
                                    chi2 = new_chi2_val
                                    min_chi2 = best_res_reduced.fun
                                    matches = new_matches
                                    removed_du = did_to_remove
                                    break
                
                if removed_du:
                    pass
                elif needs_3param_fit:
                    break
                else :
                    break
            
            chi2 = max(chi2, 1e-3)
            
            source_positions[key] = src
            chi_squares[key] = chi2

    return source_positions, chi_squares
