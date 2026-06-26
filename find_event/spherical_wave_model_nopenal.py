import numpy as np
from scipy.optimize import minimize
import os
from datetime import datetime, timedelta
from scipy import linalg
from logger_config import logger

# Numba JIT 编译加速
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Warning: numba not installed. Falling back to standard Python.")
    logger.debug("Install with: pip install numba")

GPS_UTC_OFFSET = 18
c = 299792458.0/1e9/1.000259
np.set_printoptions(precision=3)
DEFAULT_TIME_OFFSET_FILE = "_offset_byairplane_wyb_complete.txt"


def phi_to_uv(phi):
    """将方位角 phi（度）转换为 (u, v) 参数化表示。"""
    return (np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)))


class SWMRecoverableFitError(Exception):
    """Raised when the no-penalty SWM fit fails in a recoverable way."""


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False)
    def obj_uv_with_jac_numba(p, positions, times_array, t_mean_ns, c_val, sigma):  # pragma: no cover
        """
        Numba JIT 编译的UV参数化目标函数。
        
        参数:
            p: [rho, theta, u, v, t0] 参数数组
            positions: (N, 3) 探测器位置数组
            times_array: (N,) 观测时间数组
            t_mean_ns: 时间均值
            c_val: 光速
            sigma: 时间分辨率
        
        返回:
            err: 目标函数值
            jac: (5,) 梯度数组 [drho, dtheta, du, dv, dt0]
        """
        rho = p[0]
        theta = p[1]
        u = p[2]
        v = p[3]
        t0 = p[4]
        
        # 归一化 u,v
        norm = np.sqrt(u*u + v*v)
        if norm > 1e-10:
            u_norm = u / norm
            v_norm = v / norm
        else:
            u_norm = u
            v_norm = v
        
        # 角度转换
        deg2rad = np.pi / 180.0
        theta_rad = theta * deg2rad
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        
        # 方向向量 n = [sinθ·u, sinθ·v, cosθ]
        n_vec_x = sin_theta * u_norm
        n_vec_y = sin_theta * v_norm
        n_vec_z = cos_theta
        
        # 源位置
        src_x = rho * n_vec_x
        src_y = rho * n_vec_y
        src_z = rho * n_vec_z
        
        # ∂n/∂theta = [cosθ·u, cosθ·v, -sinθ]
        dn_dtheta_x = cos_theta * u_norm
        dn_dtheta_y = cos_theta * v_norm
        dn_dtheta_z = -sin_theta
        
        # 归一化导数
        if norm > 1e-10:
            du_norm_du = (v*v) / (norm*norm*norm + 1e-20)
            du_norm_dv = (-u*v) / (norm*norm*norm + 1e-20)
            dv_norm_du = (-u*v) / (norm*norm*norm + 1e-20)
            dv_norm_dv = (u*u) / (norm*norm*norm + 1e-20)
        else:
            du_norm_du = 1.0
            du_norm_dv = 0.0
            dv_norm_du = 0.0
            dv_norm_dv = 1.0
        
        err = 0.0
        jac_rho = 0.0
        jac_theta = 0.0
        jac_u = 0.0
        jac_v = 0.0
        jac_t0 = 0.0
        
        n_du = positions.shape[0]
        
        for i in range(n_du):
            # 探测器到源的向量
            diff_x = positions[i, 0] - src_x
            diff_y = positions[i, 1] - src_y
            diff_z = positions[i, 2] - src_z
            
            # 距离
            dist = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            dist_safe = dist + 1e-10
            
            # 理论时间和残差
            tp = dist / c_val + t0
            t_obs = times_array[i] - t_mean_ns
            residual = (tp - t_obs) / sigma
            err += residual * residual
            
            # 距离对各参数的导数
            ddist_drho = -(diff_x * n_vec_x + diff_y * n_vec_y + diff_z * n_vec_z) / dist_safe
            ddist_dtheta = -rho * (diff_x * dn_dtheta_x + diff_y * dn_dtheta_y + diff_z * dn_dtheta_z) / dist_safe
            
            # ∂dist/∂u_norm 和 ∂dist/∂v_norm
            ddist_du_norm = -rho * sin_theta * diff_x / dist_safe
            ddist_dv_norm = -rho * sin_theta * diff_y / dist_safe
            
            # 链式法则得到 ∂dist/∂u 和 ∂dist/∂v
            ddist_du = ddist_du_norm * du_norm_du + ddist_dv_norm * dv_norm_du
            ddist_dv = ddist_du_norm * du_norm_dv + ddist_dv_norm * dv_norm_dv
            
            # 残差对各参数的导数
            dr_drho = ddist_drho / (sigma * c_val)
            dr_dtheta = ddist_dtheta * deg2rad / (sigma * c_val)
            dr_du = ddist_du / (sigma * c_val)
            dr_dv = ddist_dv / (sigma * c_val)
            dr_dt0 = 1.0 / sigma
            
            jac_rho += 2 * residual * dr_drho
            jac_theta += 2 * residual * dr_dtheta
            jac_u += 2 * residual * dr_du
            jac_v += 2 * residual * dr_dv
            jac_t0 += 2 * residual * dr_dt0
        
        return err, np.array([jac_rho, jac_theta, jac_u, jac_v, jac_t0])

def _obj_uv_with_jac_python(p, matches, detector_positions, t_mean_ns):
    """UV参数化目标函数（纯 Python，带解析梯度），参数为 [rho, theta, u, v, t0]。
    
    供 spherical_wave_model 内部及 DU 剔除循环复用。
    """
    rho, theta, u, v, t0 = p
    sigma = 6.0
    
    # 归一化 u,v
    norm = np.sqrt(u*u + v*v)
    if norm > 1e-10:
        u_norm, v_norm = u/norm, v/norm
    else:
        u_norm, v_norm = u, v
    
    # 方向向量
    theta_rad = np.deg2rad(theta)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    n_vec = np.array([sin_theta * u_norm, sin_theta * v_norm, cos_theta])
    src = rho * n_vec
    
    # 预计算方向导数
    dn_dtheta = np.array([cos_theta * u_norm, cos_theta * v_norm, -sin_theta])
    
    if norm > 1e-10:
        du_norm_du = (v*v) / (norm*norm*norm + 1e-20)
        du_norm_dv = (-u*v) / (norm*norm*norm + 1e-20)
        dv_norm_du = (-u*v) / (norm*norm*norm + 1e-20)
        dv_norm_dv = (u*u) / (norm*norm*norm + 1e-20)
    else:
        du_norm_du, du_norm_dv = 1.0, 0.0
        dv_norm_du, dv_norm_dv = 0.0, 1.0
    
    err = 0.0
    jac = np.zeros(5)  # [drho, dtheta, du, dv, dt0]
    
    for did, ns in matches:
        det_pos = detector_positions[did]
        diff = det_pos - src
        dist = np.linalg.norm(diff)
        dist_safe = dist + 1e-10
        
        tp = dist / c + t0
        t_obs = ns - t_mean_ns
        residual = (tp - t_obs) / sigma
        err += residual ** 2
        
        # 距离对各参数的导数
        ddist_drho = -np.dot(diff, n_vec) / dist_safe
        ddist_dtheta = -rho * np.dot(diff, dn_dtheta) / dist_safe
        
        ddist_du_norm = -rho * sin_theta * diff[0] / dist_safe
        ddist_dv_norm = -rho * sin_theta * diff[1] / dist_safe
        ddist_du = ddist_du_norm * du_norm_du + ddist_dv_norm * dv_norm_du
        ddist_dv = ddist_du_norm * du_norm_dv + ddist_dv_norm * dv_norm_dv
        
        # 残差对各参数的导数
        dr_drho = ddist_drho / (sigma * c)
        dr_dtheta = ddist_dtheta * (np.pi/180.0) / (sigma * c)
        dr_du = ddist_du / (sigma * c)
        dr_dv = ddist_dv / (sigma * c)
        dr_dt0 = 1.0 / sigma
        
        jac[0] += 2 * residual * dr_drho
        jac[1] += 2 * residual * dr_dtheta
        jac[2] += 2 * residual * dr_du
        jac[3] += 2 * residual * dr_dv
        jac[4] += 2 * residual * dr_dt0
    
    return err, jac

def gps_to_utc(gps_time):
    """
    将GPS时间转换为UTC时间。
    """
    gps_epoch = datetime(1970, 1, 1)
    utc_time = gps_epoch + timedelta(seconds=(gps_time - GPS_UTC_OFFSET + 8*3600))
    return utc_time

def load_time_data(file_path=DEFAULT_TIME_OFFSET_FILE):
    """从文件加载时间偏移数据并返回字典。"""
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                id_val = int(parts[0])
                data_dict[id_val] = {
                    'mean': float(parts[1])
                    #'mean': float(parts[1]),
                    #'dispersion': float(parts[2])
                }
    return data_dict


def get_time_data(file_path=DEFAULT_TIME_OFFSET_FILE):
    """按需读取时间偏移数据，缺文件时安全回退为空字典。"""
    if not os.path.exists(file_path):
        logger.warning(f"Optional SWM offset file not found: {file_path}; continuing without offsets.")
        return {}
    return load_time_data(file_path)

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
    4参数球面波拟合（rho, theta, u, v），使用(u,v)参数化消除周期性边界问题。
    t0通过解析方式求解。
    
    参数:
        matches: 探测器列表 [(det_id, ns), ...]
        detector_positions: 探测器位置字典
        c: 光速
        initial_guesses: 初始猜测列表，每个元素为 [rho, theta, phi] 或 [rho, theta, u, v]
    
    返回:
        rho, theta, phi, t0, chi2, min_chi2, success
    """
    if len(matches) != 4:
        return None, None, None, None, np.inf, np.inf, False
    
    times = {d: ns for d, ns in matches}
    t_mean_ns = np.mean(list(times.values()))
    
    def obj_4param_uv(p):
        """UV参数化的目标函数，无周期性边界问题"""
        rho, theta, u, v = p
        
        # 归一化 u,v 确保单位圆
        norm = np.sqrt(u*u + v*v)
        if norm > 1e-10:
            u, v = u/norm, v/norm
        
        # 方向向量（直接用 u,v 代替 cosφ, sinφ）
        sin_theta = np.sin(np.deg2rad(theta))
        cos_theta = np.cos(np.deg2rad(theta))
        n_vec = np.array([sin_theta * u, sin_theta * v, cos_theta])
        
        src = rho * n_vec
        
        # 计算理论时间和残差
        travel_times = [np.linalg.norm(detector_positions[int(did)] - src) / c 
                       for did, _ in matches]
        measured_times = [ns for _, ns in matches]
        t0 = np.mean(measured_times) - np.mean(travel_times)
        err = sum(((tt + t0 - mt) / 6.0) ** 2 for tt, mt in zip(travel_times, measured_times))
        
        return err  # 无惩罚！
    
    # UV参数边界：(-1.5, 1.5) 给优化器足够缓冲空间
    bounds_uv = [(2e3, 2e5), (0, 105), (-1.5, 1.5), (-1.5, 1.5)]
    
    # 默认候选初值 - 8方向 × 2距离，均匀覆盖
    if initial_guesses is None:
        candidates_uv = []
        # 8个主要方向
        for phi_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            u, v = phi_to_uv(phi_deg)
            candidates_uv.append([3e4, 60.0, u, v])  # 中等距离
            candidates_uv.append([2e4, 45.0, u, v])  # 近距离
    else:
        # 转换传入的 [rho, theta, phi] 为 [rho, theta, u, v]
        candidates_uv = []
        for guess in initial_guesses:
            if len(guess) == 3:
                rho, theta, phi = guess
                u, v = phi_to_uv(phi)
                candidates_uv.append([rho, theta, u, v])
            elif len(guess) == 4:
                # 已经是 [rho, theta, u, v] 格式
                candidates_uv.append(guess)
    
    best_res = None
    best_chi2 = np.inf
    
    for guess in candidates_uv:
        res = minimize(obj_4param_uv, guess, method='L-BFGS-B', bounds=bounds_uv)
        if res.fun < best_chi2:
            best_chi2 = res.fun
            best_res = res
    
    if best_res is None:
        return None, None, None, None, np.inf, np.inf, False
    
    rho, theta, u_fit, v_fit = best_res.x
    
    # 归一化确保正确
    norm = np.sqrt(u_fit*u_fit + v_fit*v_fit)
    if norm > 1e-10:
        u_fit, v_fit = u_fit/norm, v_fit/norm
    
    # 从 u,v 恢复 phi
    phi = np.rad2deg(np.arctan2(v_fit, u_fit))
    if phi < 0:
        phi += 360
    
    # 镜像修正
    if theta > 90:
        theta = 180 - theta
    
    # 解析计算t0
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    n_vec = np.array([sin_theta * u_fit, sin_theta * v_fit, cos_theta])
    src = rho * n_vec
    
    travel_times = [np.linalg.norm(detector_positions[int(did)] - src) / c 
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

    # 提取探测器的位置和时间
    positions = np.array([detector_positions[det_id] for det_id in times.keys()])
    t_ns = list(times.values())
    average_t_value = np.mean(t_ns)

    for detector_id, ns in matches:
        detector_position = np.array(detector_positions[detector_id])
        distance = np.linalg.norm(detector_position - source_position)
        theoretical_time = t0 + distance / c
        observed_time = ns - average_t_value
        residual = ((observed_time - theoretical_time) / 6e0) ** 2
        logger.debug(
            f"chi2_calculation_sperical: detectors {detector_id}, "
            f"measured_diff: {observed_time:.2f}, "
            f"theoretical_diff: {theoretical_time:.2f}, contribution: {residual:.2f}"
        )
        total_error += residual 
    reduced_total_err = total_error/(len(times) - 4)
    return reduced_total_err


def _build_src_vector(rho, theta, phi):
    """根据球坐标构建源位置向量。"""
    return rho * np.array([
        np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
        np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
        np.cos(np.deg2rad(theta))
    ])


def _verify_removed_du(removed_du_id, src, t0, t_mean_ns, detector_positions, matches, sigma_t=6.0):
    """验证被剔除的 DU 是否确实异常。

    用新拟合参数计算被剔除 DU 的理论到达时间，并与实际时间对比。
    只有当残差 > 3σ 时才确认剔除。

    返回:
        (confirmed, n_sigma, residual): 是否确认剔除、sigma 倍数、残差值
    """
    removed_du_data = [(d, ns) for d, ns in matches if d == removed_du_id]
    if not removed_du_data:
        return False, 0.0, 0.0
    removed_du_time = removed_du_data[0][1]
    removed_du_pos = detector_positions[int(removed_du_id)]

    tp_removed = np.linalg.norm(removed_du_pos - src) / c + t0
    residual = tp_removed - (removed_du_time - t_mean_ns)
    n_sigma = abs(residual) / sigma_t

    return n_sigma > 3.0, n_sigma, residual


def _verify_multiple_removed_dus(removed_du_ids, src, t0, t_mean_ns, detector_positions, matches, sigma_t=6.0):
    """验证多个被剔除的 DU 是否确实异常。

    对每个被剔除的 DU 独立调用 _verify_removed_du，返回合并后的验证结果。
    只有当所有 DU 都被确认异常时才视为整体确认。

    参数:
        removed_du_ids: 被剔除的 DU ID 列表。
        src, t0, t_mean_ns, detector_positions, matches, sigma_t:
            与 _verify_removed_du 相同。

    返回:
        (all_confirmed, max_n_sigma, max_residual):
            all_confirmed — 所有 DU 均确认异常
            max_n_sigma — 各 DU 中最大的 sigma 倍数
            max_residual — 各 DU 中最大的残差值
    """
    results = [
        _verify_removed_du(du_id, src, t0, t_mean_ns, detector_positions, matches, sigma_t)
        for du_id in removed_du_ids
    ]
    all_confirmed = all(r[0] for r in results)
    max_sigma = max(r[1] for r in results)
    max_residual = max(r[2] for r in results)
    return all_confirmed, max_sigma, max_residual


def spherical_wave_model(matching_times, detector_positions, initial_directions, save_name='SWM', prune_du1=True, prune_du2=True):  # pragma: no cover
    """球面波重建模型
    
    Parameters
    ----------
    prune_du : bool, optional
        是否启用高卡方 DU 剔除优化循环，默认 True。
        设为 False 可跳过剔除步骤直接使用初始拟合结果。
    """
    results = {}
    chi_squares = {}
    all_matches = {}  # 存储每个事件最终的DU列表
    
    for i, (event_key, times) in enumerate(matching_times.items()): 
        if i % 1000 == 0:
            logger.info(f"SWM EventNo{i}/{len(matching_times)}")
        if len(times) < 6: continue
        
        t_ns = np.array(list(times.values()))
        pos_list = np.array([detector_positions[int(det_id)] for det_id in times.keys()])
        t_mean_ns = np.mean(t_ns)
        min_index = np.argmin(t_ns)
        max_index = np.argmax(t_ns)
        initial_direction = initial_directions[event_key]
        initial_t0 = t_ns[min_index] - t_mean_ns - 7.3e3/c
        current_du_ids = list(times.keys())
        index=i+1
        if(index%500 == 1 ) :
            logger.debug(
                f"SWM EventNo.{index}/{len(initial_directions)} "
                f"{initial_direction} {initial_t0}"
            )
        initial_rho = 7.3e3  # Initial estimated source distance
        vector = np.array([initial_direction[0], initial_direction[1], initial_direction[2]])
        norm = np.linalg.norm(vector)
        initial_phi = np.arctan2(initial_direction[1], initial_direction[0]) * (180 / np.pi)
        initial_theta = np.arccos(initial_direction[2] / norm) * (180 / np.pi)

        
        
        zenith_init, azimuth_init, t0_init, success = estimate_initial_direction_robust(pos_list, t_ns, c)
        if not success:
            # 如果估算失败（例如探测器太少），使用默认保守初值
            zenith_init = 45.0
            azimuth_init = 0.0
            t0_init = np.median(times)
            logger.debug("Warning: Robust initial estimation failed, using defaults.")
        
        # 为 Numba 准备数据（转换为 numpy 数组）
        if NUMBA_AVAILABLE:
            # 提取探测器位置和时间到数组
            du_ids_list = list(times.keys())
            n_du = len(du_ids_list)
            positions_array = np.zeros((n_du, 3))
            times_array = np.zeros(n_du)
            for idx, did in enumerate(du_ids_list):
                positions_array[idx] = detector_positions[int(did)]
                times_array[idx] = times[did]
            sigma_numba = 6.0  # 时间分辨率
        
        def obj_uv_with_jac(p):
            """UV参数化目标函数（带解析梯度），参数为 [rho, theta, u, v, t0]。
            委托给模块级 _obj_uv_with_jac_python，捕获 matches, detector_positions, t_mean_ns。"""
            return _obj_uv_with_jac_python(p, matches, detector_positions, t_mean_ns)
        
        ##----------new loop start-----------------------
        best_result = None
        min_chi2 = np.inf

        # UV参数化候选初值列表 - 16个方向 × 2距离，完全避免phi边界问题
        # 参数格式：[rho, theta, u, v, t0]
        candidates_uv = [
            # 8主要方向
            (3e4, 60.0, 1.0, 0.0, t0_init),      # phi=0
            (3e4, 60.0, 0.707, 0.707, t0_init),  # phi=45
            (3e4, 60.0, 0.0, 1.0, t0_init),      # phi=90
            (3e4, 60.0, -0.707, 0.707, t0_init), # phi=135
            (3e4, 60.0, -1.0, 0.0, t0_init),     # phi=180
            (3e4, 60.0, -0.707, -0.707, t0_init),# phi=225
            (3e4, 60.0, 0.0, -1.0, t0_init),     # phi=270
            (3e4, 60.0, 0.707, -0.707, t0_init), # phi=315
            # 8中间方向 + 近距离
            (2e4, 45.0, 0.924, 0.383, t0_init),  # phi=22.5
            (2e4, 45.0, 0.383, 0.924, t0_init),  # phi=67.5
            (2e4, 45.0, -0.383, 0.924, t0_init), # phi=112.5
            (2e4, 45.0, -0.924, 0.383, t0_init), # phi=157.5
            (2e4, 45.0, -0.924, -0.383, t0_init),# phi=202.5
            (2e4, 45.0, -0.383, -0.924, t0_init),# phi=247.5
            (2e4, 45.0, 0.383, -0.924, t0_init), # phi=292.5
            (2e4, 45.0, 0.924, -0.383, t0_init), # phi=337.5
        ]
        options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 30000}
        
        # UV参数边界 [rho, theta, u, v, t0]
        bounds_uv = [(2e3, 2e5), (5, 95), (-2.0, 2.0), (-2.0, 2.0), (-np.inf, np.inf)]
        
        # 选择优化函数：Numba版本（更快）或 Python版本
        if NUMBA_AVAILABLE:
            # Numba UV版本
            def obj_uv_numba_wrapper(p):
                return obj_uv_with_jac_numba(p, positions_array, times_array, t_mean_ns, c, sigma_numba)
            obj_to_use = obj_uv_numba_wrapper
        else:
            # Python版本
            obj_to_use = obj_uv_with_jac
        
        for guess in candidates_uv:
            res = minimize(obj_to_use, guess, method='L-BFGS-B', bounds=bounds_uv, 
                          options=options, jac=True)
            if res.fun < min_chi2:
                min_chi2 = res.fun
                best_result = res
        
        # 使用 best_result 作为最终重建结果
        ##----------new loop end-----------------------
        logger.debug(
            f"  Event {index}: 32-candidate optimization done, "
            f"best_chi2={min_chi2:.1f}, "
            f"result={'valid' if best_result and best_result.success else 'failed'}"
        )
        
        # 新增：如果所有候选都失败或卡方过大，使用平面波近似作为备用
        if best_result is None or min_chi2 > 1e6:
            logger.debug(
                f"  Event {index}: Spherical fit failed or chi2 too large "
                f"(min_chi2={min_chi2:.1e}), "
                f"using plane wave approximation "
                f"(zenith={zenith_init:.1f}°, azimuth={azimuth_init:.1f}°, t0={t0_init:.1f}ns)"
            )
            # 使用平面波方向作为球面波近似（假设源在无穷远）
            rho_fallback = 1e5  # 远距离近似
            theta_fallback = zenith_init
            phi_fallback = azimuth_init
            t0_fallback = t0_init
            src_fallback = _build_src_vector(rho_fallback, theta_fallback, phi_fallback)
            # 计算近似卡方
            err_fallback = 0
            for did, ns in times.items():
                tp = np.linalg.norm(detector_positions[int(did)] - src_fallback) / c + t0_fallback
                err_fallback += ((tp - (ns - t_mean_ns)) / 6.0) ** 2
            chi2_fallback = err_fallback / max(1, len(times)-4)
            
            results[event_key] = src_fallback
            chi_squares[event_key] = chi2_fallback
            continue  # 跳到下一个事件
        
        if best_result:
            rho, theta, u_fit, v_fit, t0 = best_result.x
            # 从u,v恢复phi
            norm = np.sqrt(u_fit*u_fit + v_fit*v_fit)
            if norm > 1e-10:
                u_fit, v_fit = u_fit/norm, v_fit/norm
            phi = np.rad2deg(np.arctan2(v_fit, u_fit))
            if phi < 0:
                phi += 360
            if theta - 90 > 0 : theta = 180 - theta #mannually correct the mirror effect 
            src = _build_src_vector(rho, theta, phi)
            chi2 = min_chi2 / max(1, len(times)-4)
            #if chi2 < 1e-2: chi2=max(chi2,1e-2)
            
            logger.debug(
                f"  Event {index}: initial fit -> "
                f"rho={rho:.0f}m, theta={theta:.1f}°, phi={phi:.1f}°, "
                f"t0={t0:.1f}ns, chi2={chi2:.1f}, nDU={len(times)}, "
                f"prune_du1={prune_du1}, prune_du2={prune_du2}"
            )
            
            # ===== 高卡方事件逐个DU剔除优化 =====
            # 对 chi2 > 100 且 DU数 >= 6 的事件，尝试剔除异常DU重新拟合
            if prune_du1:
                original_chi2 = chi2  # 保存原始卡方用于显示
                needs_3param_fit = False  # 标记是否需要后续3参数拟合（当DU=4时）
                matches = [(d, ns) for d, ns in times.items()]  # 初始化 matches（列表形式，供闭包和剔除逻辑使用）
                 
            while prune_du1 and chi2 > 10 and len(times) >= 5:
                logger.debug(
                    f"  Event {index}: [prune-loop-1] iterating, "
                    f"chi2={chi2:.1f}, nDU={len(times)}"
                )
                # ===== 特殊处理：当DU=5且chi2>200时，尝试所有4-DU组合（方案1）=====
                if len(times) == 5 and chi2 > 200:
                    logger.debug(
                        f"  Event {index}: [prune-loop-1] 5-DU special handling, "
                        f"chi2={chi2:.1f} > 200, trying C(5,4)=5 combos"
                    )
                    best_4du_chi2 = np.inf
                    best_4du_matches = None
                    best_4du_times = None
                    best_4du_t_mean = None
                    removed_du_id = None
                    
                    # 尝试所有C(5,4)=5种4-DU组合
                    best_4du_params = None  # 保存最佳拟合参数
                    from itertools import combinations
                    for combo in combinations(times.items(), 4):
                        combo_matches = list(combo)
                        combo_times = {d: ns for d, ns in combo_matches}
                        combo_t_mean = np.mean(list(combo_times.values()))

                        # 使用统一的3参数拟合函数
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
                            best_4du_times = combo_times
                            best_4du_t_mean = combo_t_mean
                            best_4du_params = [rho_4, theta_4, phi_4]
                            best_4du_t0 = t0_4
                            best_4du_min_chi2 = min_chi2_4
                            # 记录被剔除的DU
                            combo_dus = set(d for d, _ in combo_matches)
                            all_dus = set(d for d, _ in times.items())
                            removed_du_id = list(all_dus - combo_dus)[0]
                    
                    # 如果找到更好的4-DU组合，验证被剔除的DU是否真的有问题
                    if best_4du_matches and best_4du_chi2 < chi2 * 0.5:
                        rho_new, theta_new, phi_new = best_4du_params
                        t0_new = best_4du_t0
                        src_new = _build_src_vector(rho_new, theta_new, phi_new)

                        confirmed, n_sigma, residual_removed = _verify_removed_du(
                            removed_du_id, src_new, t0_new, best_4du_t_mean,
                            detector_positions, matches
                        )

                        logger.debug(
                            f"  Event {index}: [prune-loop-1] 5-DU special: "
                            f"verify removed DUs={removed_du_id}, "
                            f"confirmed={confirmed}, n_sigma={n_sigma:.1f}, "
                            f"residual={residual_removed:.1f}ns, "
                            f"best_4du_chi2={best_4du_chi2:.1f}"
                        )

                        if confirmed:
                            matches = best_4du_matches
                            times = best_4du_times
                            t_mean_ns = best_4du_t_mean
                            chi2 = best_4du_chi2
                            rho, theta, phi = best_4du_params
                            t0 = best_4du_t0
                            min_chi2 = best_4du_min_chi2
                            src = src_new
                            needs_3param_fit = True
                            logger.debug(
                                f"  Event {index}: [prune-loop-1] 5-DU optimal removal "
                                f"(removed DU {removed_du_id}, "
                                f"residual={residual_removed:.1f}ns/{n_sigma:.1f}σ), "
                                f"chi2: {original_chi2:.1f} -> {chi2:.1f}"
                            )
                            break
                        else:
                            logger.debug(
                                f"  Event {index}: [prune-loop-1] Rejected removal of DU "
                                f"{removed_du_id} "
                                f"(residual={residual_removed:.1f}ns/"
                                f"{n_sigma:.1f}σ < 3σ), keeping all 5 DUs"
                            )
                # ===== 方案1特殊处理结束 =====
                
                # 计算每个DU的卡方贡献
                du_contributions = []
                for did, ns in times.items():
                    tp = np.linalg.norm(detector_positions[int(did)] - src) / c + t0
                    contribution = ((tp - (ns - t_mean_ns)) / 6.0) ** 2
                    du_contributions.append((contribution, did, ns))
                
                # # 按贡献排序（从大到小）
                # du_contributions.sort(reverse=True)
                # top3 = [(f"DU{did}({contrib:.1f})") for contrib, did, _ in du_contributions[:3]]
                # logger.debug(
                #     f"  Event {index}: [prune-loop-1] top3 DU contributions: "
                #     f"{', '.join(top3)}, total chi2={chi2:.1f}"
                # )

                # 遍历所有 DU，收集 chi2 改善的候选，最后选取 chi2 最小的
                best_removal = None  # (chi2, state_dict)
                for contrib, did_to_remove, _ in du_contributions:
                    if len(times) - 1 >= 3:  # 确保剩余至少4个DU（但4个DU时用3参数）
                        # 构建剔除后的matches
                        new_matches = [(d, ns) for d, ns in times.items() if d != did_to_remove]
                        n_new = len(new_matches)

                        # 重新计算均值
                        new_times = {d: ns for d, ns in new_matches}
                        new_t_mean_ns = np.mean(list(new_times.values()))
                        
                        # 根据剩余DU数量选择策略
                        if n_new == 4:
                            # ===== 恰好4个DU：3参数拟合 =====
                            candidates_4du = [
                                [rho, theta, phi],
                                [rho*1.2, theta, phi+20],
                                [rho*0.8, theta, phi-30],
                                [rho, min(95, theta+10), phi-90],
                                [rho, max(30, theta-10), phi],
                                [rho, theta, (phi+180)%360],
                            ]
                            if phi < 20:
                                candidates_4du.extend([[rho, theta, phi-30], [rho, theta, 360+phi-30]])
                            elif phi > 340:
                                candidates_4du.extend([[rho, theta, phi+30], [rho, theta, phi-360-30]])
                            
                            rho_4, theta_4, phi_4, t0_4, chi2_4, min_chi2_4, success_4 = fit_3param(
                                new_matches, detector_positions, c, candidates_4du
                            )

                            if success_4 and chi2_4 < chi2 * 0.8:
                                src_4 = _build_src_vector(rho_4, theta_4, phi_4)
                                if best_removal is None or chi2_4 < best_removal[0]:
                                    best_removal = (chi2_4, {
                                        'removed_du': did_to_remove,
                                        'matches': new_matches,
                                        'times': new_times,
                                        't_mean_ns': new_t_mean_ns,
                                        'rho': rho_4, 'theta': theta_4, 'phi': phi_4,
                                        't0': t0_4, 'chi2': chi2_4,
                                        'min_chi2': min_chi2_4,
                                        'src': src_4,
                                        'needs_3param_fit': True,
                                    })
                        else:
                            # ===== 多于4个DU：使用4参数拟合（带解析梯度）=====
                            # 准备 Numba 数据（如果可用）
                            if NUMBA_AVAILABLE:
                                new_positions_array = np.zeros((n_new, 3))
                                new_times_array = np.zeros(n_new)
                                for idx_new, (d_new, ns_new) in enumerate(new_matches):
                                    new_positions_array[idx_new] = detector_positions[int(d_new)]
                                    new_times_array[idx_new] = ns_new
                                # Numba UV包装函数
                                def obj_reduced_uv_with_jac(p):
                                    return obj_uv_with_jac_numba(p, new_positions_array, new_times_array, 
                                                                 new_t_mean_ns, c, sigma_numba)
                            else:
                                # Python UV版本，委托给模块级 _obj_uv_with_jac_python
                                def obj_reduced_uv_with_jac(p):
                                    return _obj_uv_with_jac_python(p, new_matches, detector_positions, new_t_mean_ns)
                            
                            # UV参数化候选初值（基于当前拟合参数）
                            u_cur, v_cur = phi_to_uv(phi)
                            candidates_n_uv = [
                                [rho, theta, u_cur, v_cur, t0],
                                [rho*1.2, theta, u_cur, v_cur, t0],
                                [rho*0.8, theta, u_cur, v_cur, t0],
                                [rho, min(95, theta+10), u_cur, v_cur, t0],
                                [rho, max(30, theta-10), u_cur, v_cur, t0],
                            ]
                            # 添加8个主要方向作为备选
                            for phi_g in [0, 45, 90, 135, 180, 225, 270, 315]:
                                u_g, v_g = phi_to_uv(phi_g)
                                candidates_n_uv.append([rho, theta, u_g, v_g, t0])
                            
                            # 多初值优化（使用解析梯度）
                            best_res_reduced = None
                            best_chi2_reduced = np.inf
                            for guess_n in candidates_n_uv:
                                res_n = minimize(obj_reduced_uv_with_jac, guess_n, 
                                                method='L-BFGS-B', bounds=bounds_uv, options=options, jac=True)
                                if res_n.fun < best_chi2_reduced:
                                    best_chi2_reduced = res_n.fun
                                    best_res_reduced = res_n
                            
                            if best_res_reduced:
                                new_chi2_val = best_chi2_reduced / max(1, n_new - 4)
                                # 如果新卡方显著改善（至少降低30%），进一步验证被剔除的DU
                                if new_chi2_val < chi2 * 0.7:
                                    rho_new, theta_new, u_new, v_new, t0_new = best_res_reduced.x
                                    norm_new = np.sqrt(u_new*u_new + v_new*v_new)
                                    if norm_new > 1e-10:
                                        u_new, v_new = u_new/norm_new, v_new/norm_new
                                    phi_new = np.rad2deg(np.arctan2(v_new, u_new))
                                    if phi_new < 0:
                                        phi_new += 360
                                    if theta_new - 90 > 0:
                                        theta_new = 180 - theta_new
                                    src_new = _build_src_vector(rho_new, theta_new, phi_new)

                                    confirmed, n_sigma, residual_removed = _verify_removed_du(
                                        did_to_remove, src_new, t0_new, new_t_mean_ns,
                                        detector_positions, matches
                                    )
                                    
                                    logger.debug(
                                            f"  Event {index}: [prune-loop-1] removed DUs {did_to_remove}, "
                                            f"confirmed={confirmed}, n_sigma={n_sigma:.1f}, "
                                            f"residual_removed={residual_removed:.1f}ns, "
                                            f"new_chi2_val={new_chi2_val:.1f}"
                                    )

                                    if confirmed:
                                        if best_removal is None or new_chi2_val < best_removal[0]:
                                            best_removal = (new_chi2_val, {
                                                'removed_du': did_to_remove,
                                                'matches': new_matches,
                                                'times': new_times,
                                                't_mean_ns': new_t_mean_ns,
                                                'rho': rho_new, 'theta': theta_new, 'phi': phi_new,
                                                't0': t0_new, 'chi2': new_chi2_val,
                                                'min_chi2': best_res_reduced.fun,
                                                'src': src_new,
                                                'needs_3param_fit': False,
                                            })
                                    # else:
                                    #     logger.debug(
                                    #         f"  Event {index}: [prune-loop-1] Rejected removal "
                                    #         f"of DU {did_to_remove} "
                                    #         f"(residual={residual_removed:.1f}ns/"
                                    #         f"{n_sigma:.1f}σ < 3σ), trying next DU"
                                    #     )

                # 选取 chi2 最小的候选应用
                if best_removal is not None:
                    b = best_removal[1]
                    removed_du = b['removed_du']
                    matches = b['matches']
                    times = b['times']
                    t_mean_ns = b['t_mean_ns']
                    rho, theta, phi, t0 = b['rho'], b['theta'], b['phi'], b['t0']
                    chi2 = b['chi2']
                    min_chi2 = b['min_chi2']
                    src = b['src']
                    needs_3param_fit = b['needs_3param_fit']
                    fit_label = "3-param" if needs_3param_fit else "4-param"
                    logger.debug(
                        f"  Event {index}: [prune-loop-1] Best removal: DU {removed_du} "
                        f"({fit_label}), "
                        f"chi2: {original_chi2:.1f} -> {chi2:.1f}"
                    )
                else:
                    removed_du = None

                if removed_du :
                    # 成功剔除DU并优化（可能是4参数或3参数拟合）
                    fit_type = "3-param" if needs_3param_fit else "4-param"
                    logger.debug(
                        f"  Event {index}: [prune-loop-1] Removed DU {removed_du} "
                        f"({fit_type}), "
                        f"chi2 improved: {original_chi2:.1f} -> {chi2:.1f}, "
                        f"remaining DUs: {len(times)}"
                    )
                    # 如果已经用3参数拟合过（即当前DU=4），退出while循环
                    if needs_3param_fit:
                        break
                    # 否则继续while循环尝试进一步改善（DU>4且chi2可能仍>10）
                else:
                    # 没有成功剔除任何DU（所有尝试都失败或chi2未改善）
                    logger.debug(
                        f"  Event {index}: [prune-loop-1] no DU removed, "
                        f"chi2={chi2:.1f}, exiting prune loop 1"
                    )
                    break
            
            # ===== 高卡方优化结束 =====
            # (prune_du=False 时跳过上述整个剔除循环)
            
            # ===== 高卡方事例一次性去除两个DU优化 ====
            if prune_du2:
                original_chi2 = chi2  # 保存原始卡方用于显示
                needs_3param_fit = False  # 标记是否需要后续3参数拟合（当DU=4时）
                matches = [(d, ns) for d, ns in times.items()]  # 初始化 matches（列表形式，供闭包和剔除逻辑使用）
                 
            while prune_du2 and chi2 > 10 and len(times) >= 6:
                logger.debug(
                    f"  Event {index}: [prune-loop-2] entering with "
                    f"chi2={chi2:.1f}, nDU={len(times)}"
                )
                # ===== 特殊处理：当DU=6且chi2>200时，尝试所有4-DU组合（方案1）=====
                if len(times) == 6 and chi2 > 200:
                    logger.debug(
                        f"  Event {index}: [prune-loop-2] 6-DU special handling, "
                        f"chi2={chi2:.1f} > 200, trying C(6,4)=15 combos"
                    )
                    
                    best_4du_chi2 = np.inf
                    best_4du_matches = None
                    best_4du_times = None
                    best_4du_t_mean = None
                    removed_du_id = None
                    
                    # 尝试所有C(6,4)=15种4-DU组合
                    best_4du_params = None  # 保存最佳拟合参数
                    from itertools import combinations
                    for combo in combinations(times.items(), 4):
                        combo_matches = list(combo)
                        combo_times = {d: ns for d, ns in combo_matches}
                        combo_t_mean = np.mean(list(combo_times.values()))
                        
                        # 使用统一的3参数拟合函数
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
                            best_4du_times = combo_times
                            best_4du_t_mean = combo_t_mean
                            best_4du_params = [rho_4, theta_4, phi_4]
                            best_4du_t0 = t0_4
                            best_4du_min_chi2 = min_chi2_4
                            # 记录被剔除的DU
                            combo_dus = set(d for d, _ in combo_matches)
                            all_dus = set(d for d, _ in times.items())
                            removed_du_id = list(all_dus - combo_dus)
                    # 如果找到更好的4-DU组合，验证被剔除的DU是否真的有问题
                    if best_4du_matches and best_4du_chi2 < chi2 * 0.5:
                        logger.debug(
                            f"  Event {index}: [prune-loop-2] 6-DU special: "
                            f"best 4-DU combo chi2={best_4du_chi2:.1f}, "
                            f"removed DUs: {removed_du_id}"
                        )
                        rho_new, theta_new, phi_new = best_4du_params
                        t0_new = best_4du_t0
                        src_new = _build_src_vector(rho_new, theta_new, phi_new)

                        confirmed, n_sigma, residual_removed = _verify_multiple_removed_dus(
                            removed_du_id, src_new, t0_new, best_4du_t_mean,
                            detector_positions, matches
                        )

                        logger.debug(
                            f"  Event {index}: [prune-loop-2] 6-DU special: "
                            f"verify removed DUs={removed_du_id}, "
                            f"confirmed={confirmed}, n_sigma={n_sigma:.1f}, "
                            f"residual={residual_removed:.1f}ns, "
                            f"best_4du_chi2={best_4du_chi2:.1f}"
                        )
                        
                        if confirmed:
                            matches = best_4du_matches
                            times = best_4du_times
                            t_mean_ns = best_4du_t_mean
                            chi2 = best_4du_chi2
                            rho, theta, phi = best_4du_params
                            t0 = best_4du_t0
                            min_chi2 = best_4du_min_chi2
                            src = src_new
                            needs_3param_fit = True
                            logger.debug(
                                f"  Event {index}: [prune-loop-2] 6-DU optimal removal "
                                f"(removed DU {removed_du_id}, "
                                f"residual={residual_removed:.1f}ns/{n_sigma:.1f}σ), "
                                f"chi2: {original_chi2:.1f} -> {chi2:.1f}"
                            )
                            break
                        else:
                            logger.debug(
                                f"  Event {index}: [prune-loop-2] Rejected removal of DU "
                                f"{removed_du_id} "
                                f"(residual={residual_removed:.1f}ns/"
                                f"{n_sigma:.1f}σ < 3σ), keeping all 5 DUs"
                            )
                # ===== 方案1特殊处理结束 =====
                
                # 计算每个DU的卡方贡献
                du_contributions = []
                for did, ns in times.items():
                    tp = np.linalg.norm(detector_positions[int(did)] - src) / c + t0
                    contribution = ((tp - (ns - t_mean_ns)) / 6.0) ** 2
                    du_contributions.append((contribution, did, ns))
                
                # # 按贡献排序（从大到小）
                # du_contributions.sort(reverse=True)
                # top3 = [(f"DU{did}({contrib:.1f})") for contrib, did, _ in du_contributions[:3]]
                # logger.debug(
                #     f"  Event {index}: [prune-loop-2] top3 DU contributions: "
                #     f"{', '.join(top3)}, total chi2={chi2:.1f}"
                # )
                
                # 遍历所有 DU 对组合，收集 chi2 改善的候选，最后选取 chi2 最小的
                best_removal = None  # (chi2, state_dict)
                from itertools import combinations
                for did_to_remove in combinations(times.items(), 2):
                    all_dus = set(d for d, _ in times.items())
                    removed_dus = set(d for d, _ in did_to_remove)
                    contrib = list(all_dus - removed_dus)

                    if len(times) - 1 >= 3:  # 确保剩余至少4个DU（但4个DU时用3参数）
                        # 构建剔除后的matches
                        new_matches = [(d, ns) for d, ns in times.items() if d in contrib]
                        n_new = len(new_matches)

                        # 重新计算均值
                        new_times = {d: ns for d, ns in new_matches}
                        new_t_mean_ns = np.mean(list(new_times.values()))
                        
                        # 根据剩余DU数量选择策略
                        if n_new == 4:
                            # ===== 恰好4个DU：3参数拟合 =====
                            candidates_4du = [
                                [rho, theta, phi],
                                [rho*1.2, theta, phi+20],
                                [rho*0.8, theta, phi-30],
                                [rho, min(95, theta+10), phi-90],
                                [rho, max(30, theta-10), phi],
                                [rho, theta, (phi+180)%360],
                            ]
                            if phi < 20:
                                candidates_4du.extend([[rho, theta, phi-30], [rho, theta, 360+phi-30]])
                            elif phi > 340:
                                candidates_4du.extend([[rho, theta, phi+30], [rho, theta, phi-360-30]])
                            
                            rho_4, theta_4, phi_4, t0_4, chi2_4, min_chi2_4, success_4 = fit_3param(
                                new_matches, detector_positions, c, candidates_4du
                            )

                            if success_4 and chi2_4 < chi2 * 0.8:
                                src_4 = _build_src_vector(rho_4, theta_4, phi_4)
                                if best_removal is None or chi2_4 < best_removal[0]:
                                    best_removal = (chi2_4, {
                                        'removed_du': did_to_remove,
                                        'matches': new_matches,
                                        'times': new_times,
                                        't_mean_ns': new_t_mean_ns,
                                        'rho': rho_4, 'theta': theta_4, 'phi': phi_4,
                                        't0': t0_4, 'chi2': chi2_4,
                                        'min_chi2': min_chi2_4,
                                        'src': src_4,
                                        'needs_3param_fit': True,
                                    })
                        else:
                            # ===== 多于4个DU：使用4参数拟合（带解析梯度）=====
                            # 准备 Numba 数据（如果可用）
                            if NUMBA_AVAILABLE:
                                new_positions_array = np.zeros((n_new, 3))
                                new_times_array = np.zeros(n_new)
                                for idx_new, (d_new, ns_new) in enumerate(new_matches):
                                    new_positions_array[idx_new] = detector_positions[int(d_new)]
                                    new_times_array[idx_new] = ns_new
                                # Numba UV包装函数
                                def obj_reduced_uv_with_jac(p):
                                    return obj_uv_with_jac_numba(p, new_positions_array, new_times_array, 
                                                                 new_t_mean_ns, c, sigma_numba)
                            else:
                                # Python UV版本，委托给模块级 _obj_uv_with_jac_python
                                def obj_reduced_uv_with_jac(p):
                                    return _obj_uv_with_jac_python(p, new_matches, detector_positions, new_t_mean_ns)
                            
                            # UV参数化候选初值（基于当前拟合参数）
                            u_cur, v_cur = phi_to_uv(phi)
                            candidates_n_uv = [
                                [rho, theta, u_cur, v_cur, t0],
                                [rho*1.2, theta, u_cur, v_cur, t0],
                                [rho*0.8, theta, u_cur, v_cur, t0],
                                [rho, min(95, theta+10), u_cur, v_cur, t0],
                                [rho, max(30, theta-10), u_cur, v_cur, t0],
                            ]
                            # 添加8个主要方向作为备选
                            for phi_g in [0, 45, 90, 135, 180, 225, 270, 315]:
                                u_g, v_g = phi_to_uv(phi_g)
                                candidates_n_uv.append([rho, theta, u_g, v_g, t0])
                            
                            # 多初值优化（使用解析梯度）
                            best_res_reduced = None
                            best_chi2_reduced = np.inf
                            for guess_n in candidates_n_uv:
                                res_n = minimize(obj_reduced_uv_with_jac, guess_n, 
                                                method='L-BFGS-B', bounds=bounds_uv, options=options, jac=True)
                                if res_n.fun < best_chi2_reduced:
                                    best_chi2_reduced = res_n.fun
                                    best_res_reduced = res_n
                            
                            if best_res_reduced:
                                new_chi2_val = best_chi2_reduced / max(1, n_new - 4)
                                # 如果新卡方显著改善（至少降低30%），进一步验证被剔除的DU
                                if new_chi2_val < chi2 * 0.7:
                                    rho_new, theta_new, u_new, v_new, t0_new = best_res_reduced.x
                                    norm_new = np.sqrt(u_new*u_new + v_new*v_new)
                                    if norm_new > 1e-10:
                                        u_new, v_new = u_new/norm_new, v_new/norm_new
                                    phi_new = np.rad2deg(np.arctan2(v_new, u_new))
                                    if phi_new < 0:
                                        phi_new += 360
                                    if theta_new - 90 > 0:
                                        theta_new = 180 - theta_new
                                    src_new = _build_src_vector(rho_new, theta_new, phi_new)

                                    confirmed, n_sigma, residual_removed = _verify_multiple_removed_dus(
                                        [d[0] for d in did_to_remove], src_new, t0_new, new_t_mean_ns,
                                        detector_positions, matches
                                    )
                                    logger.debug(
                                            f"  Event {index}: [prune-loop-2] both removed DUs "
                                            f"{did_to_remove}, "
                                            f"confirmed={confirmed}, n_sigma={n_sigma:.1f}, "
                                            f"residual={residual_removed:.1f}ns, "
                                            f"new_chi2_val={new_chi2_val:1f}"
                                    )
                                    
                                    if confirmed:
                                        if best_removal is None or new_chi2_val < best_removal[0]:
                                            best_removal = (new_chi2_val, {
                                                'removed_du': did_to_remove,
                                                'matches': new_matches,
                                                'times': new_times,
                                                't_mean_ns': new_t_mean_ns,
                                                'rho': rho_new, 'theta': theta_new, 'phi': phi_new,
                                                't0': t0_new, 'chi2': new_chi2_val,
                                                'min_chi2': best_res_reduced.fun,
                                                'src': src_new,
                                                'needs_3param_fit': False,
                                            })
                                    # else:
                                    #     logger.debug(
                                    #         f"  Event {index}: [prune-loop-2] Rejected removal "
                                    #         f"of DU {did_to_remove} "
                                    #         f"(residual={residual_removed:.1f}ns/"
                                    #         f"{n_sigma:.1f}σ < 3σ), trying next DU"
                                    #     )

                # 选取 chi2 最小的候选应用
                if best_removal is not None:
                    b = best_removal[1]
                    removed_du = b['removed_du']
                    matches = b['matches']
                    times = b['times']
                    t_mean_ns = b['t_mean_ns']
                    rho, theta, phi, t0 = b['rho'], b['theta'], b['phi'], b['t0']
                    chi2 = b['chi2']
                    min_chi2 = b['min_chi2']
                    src = b['src']
                    needs_3param_fit = b['needs_3param_fit']
                    fit_label = "3-param" if needs_3param_fit else "4-param"
                    logger.debug(
                        f"  Event {index}: [prune-loop-2] Best removal: DU {removed_du} "
                        f"({fit_label}), "
                        f"chi2: {original_chi2:.1f} -> {chi2:.1f}"
                    )
                else:
                    removed_du = None
                
                if removed_du :
                    # 成功剔除DU并优化（可能是4参数或3参数拟合）
                    fit_type = "3-param" if needs_3param_fit else "4-param"
                    logger.debug(
                        f"  Event {index}: [prune-loop-2] Removed DU {removed_du} "
                        f"({fit_type}), "
                        f"chi2 improved: {original_chi2:.1f} -> {chi2:.1f}, "
                        f"remaining DUs: {len(times)}"
                    )
                    # 如果已经用3参数拟合过（即当前DU=4），退出while循环
                    if needs_3param_fit:
                        break
                    # 否则继续while循环尝试进一步改善（DU>4且chi2可能仍>10）
                else:
                    # 没有成功剔除任何DU（所有尝试都失败或chi2未改善）
                    logger.debug(
                        f"  Event {index}: [prune-loop-2] no DU removed, "
                        f"chi2={chi2:.1f}, exiting prune loop 2"
                    )
                    break
            

            # ===== 高卡方进一步优化结束 =====
            # (prune_du=False 时跳过上述整个剔除循环)
            


            # ===== 4-DU 事件已在前面处理完成 =====
            # 所有4-DU事件（通过方案1或正常剔除）都已调用fit_3param完成拟合
            
            results[event_key] = src
            chi_squares[event_key] = chi2
            all_matches[event_key] = times

            epsilon = 1e-12
            zenith = np.arccos(src[2] / (np.linalg.norm(src) + epsilon ) ) * 180 / np.pi
            azimuth = np.arctan2(src[1], src[0]) * 180 / np.pi 
            azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)
            # 根据 DU 数量选择正确的自由度：4 DU 时用 3 参数（自由度=1），>4 DU 时用 4 参数
            dof = max(1, len(times) - 3) if len(times) == 4 else max(1, len(times) - 4)
            chi_square = min_chi2 / dof
            if chi_square > 1e3 and index%2000 ==1 :
                logger.debug(
                    f"Event No.{index}, LargeChi {chi_square:.1e}, "
                    f"Number of DUs:{len(times)}"
                )
            if index % 500 == 1:
                logger.debug(
                    f"  Event {index}: final -> "
                    f"zenith={zenith:.1f}°, azimuth={azimuth:.1f}°, "
                    f"rho={np.linalg.norm(src):.0f}m, chi2={chi_square:.1f}, "
                    f"nDU={len(times)}, t0={t0:.1f}ns"
                )
    return results, chi_squares, all_matches