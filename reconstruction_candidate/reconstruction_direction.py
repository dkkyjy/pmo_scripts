"""PWF / SWF / ADF 拟合器、方向重建入口，以及它们共用的模型函数。

内容：
  - 三个拟合器 PWF / SWF / ADF 与它们的模型函数；
  - ADF 最小化的唯一实现 ``_adf_minuit`` 及其四个薄壳
    （``ADF_fit`` / ``ADF_fit_fixdw`` / ``ADF_fit_analytic`` / ``ADF_fit_analytic_fixdw``）；
  - 方向求解入口（``recons_direction`` / ``recons_direction_swf_adf`` /
    ``reconstruct_direction_only`` / ``recons_direction_struct``）与
    :class:`DirectionResult` 声明的 20 字段方向契约（长度在 import 时断言）；
  - 被编排模块按名 import 的公开符号 ``find_core`` 与 ``magnetic_field_vector``；
  - 能量反演复用的模型函数与修正量 ``che_fit``（及其内部的 ``che_fit_chi2``）、``f_adf``、
    ``density_and_alpha_correction``。

边界：它不读 ROOT、不做因果筛选、不碰天线响应，也不需要 ``parameter_files/``——
那些留在 ``reconstruction_candidate_fluence_error_fixed``；依赖方向因此是单向的
（该模块 import 本模块，本模块不 import 它）。

数值行为与原文件完全一致：本模块的内容是从原文件原样搬来的，任何计算逻辑的改动
都应作为独立的一次变更，而不是顺手做在搬迁或清理里。
"""

import os
import sys

import numpy as np
import iminuit
from typing import NamedTuple
from scipy.stats import chi2, norm

# 本模块要能被单独 import：它按扁平顶层名引入 logger_config / AiresInfoFunctions 等，
# 因此自己把模块目录与仓库根放进 sys.path，不依赖调用方先 import 旧模块。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger_config import logger
import AiresInfoFunctions as aires
import coordinatesystems, helper

def sigma(dchi2,freedom):
    s=norm.isf(chi2.sf(dchi2,freedom)/2)
    return s

c = 299792458 # 光速 (m/s)

xList, yList, zList = np.linspace(-3e5, 3e5, 4), np.linspace(-3e5, 3e5, 4), np.linspace(1e3, 2e4, 4) # 定义网格搜索的 Xmax 位置列表, 单位为m

magnetic_field_vector = helper.spherical_to_cartesian(np.pi / 2 - np.deg2rad(60.79), 180 * np.pi / 180) # GRAND 站点的地磁场矢量, 单位矢量, 笛卡尔坐标系下

class PWF:
    """
    平面波前 (Plane Wave Front) 拟合类。
    
    用于定义平面波前拟合的目标函数 (chi2)。
    """
    def __init__(self, x, y, z, t):
        """
        初始化 PWF 类。
        
        参数:
            x, y, z (numpy.ndarray): 天线位置坐标。
            t (numpy.ndarray): 信号到达时间。
        """
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        
    def chi2(self, theta, phi, rc):
        """
        计算卡方值 (Chi-squared)。
        
        目标是最小化观测到的时间差与平面波模型预测的时间差之间的差异。
        
        参数:
            theta (float): 天顶角 (度)。
            phi (float): 方位角 (度)。
            rc (float): 相对光速因子 (通常接近 1)。
            
        返回:
            float: Chi2 值。
        """
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        # print('pars:', theta, phi, rc)
        sint, cost = np.sin(theta), np.cos(theta)
        sinp, cosp = np.sin(phi), np.cos(phi)
        
        # 计算位置差和时间差矩阵
        dxx = self.x.reshape(1, -1) - self.x.reshape(-1, 1)
        dyy = self.y.reshape(1, -1) - self.y.reshape(-1, 1)
        dzz = self.z.reshape(1, -1) - self.z.reshape(-1, 1)
        dtt = self.t.reshape(1, -1) - self.t.reshape(-1, 1)
        
        # 平面波方程残差: r * n - c * t
        # 这里使用的是差分形式
        fmin = dxx * sint * cosp + dyy * sint * sinp + dzz * cost - rc * 299792458 * dtt
        # 修正卡方公式：如果是基于时间分辨率 (6 ns) 计算误差 (sigma)，分母应该是方差 (sigma**2)
        chi2 = (fmin ** 2 / (6e-9 * 299792458) ** 2).sum()
        
        return chi2
        
    def __call__(self, theta, phi, rc):
        """
        使类实例可调用，以便 iminuit 使用。
        """
        chi2 = self.chi2(theta, phi, rc)
        return chi2

def PWF_fit(x, y, z, ant_t, theta0, phi0):
    """
    执行平面波前拟合。
    
    使用 iminuit 最小化 PWF 类的 chi2 函数，以求解最佳的到达方向 (theta, phi)。
    
    参数:
        x, y, z (numpy.ndarray): 天线位置。
        ant_t (numpy.ndarray): 天线信号到达时间。
        theta0 (float): 天顶角初始猜测值。
        phi0 (float): 方位角初始猜测值。
    
    返回:
        tuple: (theta, phi, fval)
            theta (float): 拟合得到的天顶角。
            phi (float): 拟合得到的方位角。
            fval (float): 最小化的函数值 (chi2)。
    """
    m = iminuit.Minuit(PWF(x, y, z, ant_t), theta = theta0, phi = phi0, rc = 1)
    m.limits = ([90, 180], [0, 360], [0, 1]) # 参数限制范围
    m.fixed = (False, False, True) # 固定 rc = 1
    m.errors = (1, 1, 0.1) # 初始误差步长，rc虽固定但error需为正数以避免警告
    m.errordef = 1
    m.tol = 1e-5
    m.migrad() # 执行最小化
    m.hesse()  # 计算海森矩阵 (误差估计)

    theta = m.values["theta"]
    phi = m.values["phi"]
    fval = m.fval
    
    theta_err = m.errors["theta"]
    phi_err = m.errors["phi"]
    return theta, phi, fval / (len(x) - 2), theta_err, phi_err

def find_core(x, y, z, theta, phi, z_det):
    """
    计算簇射轴与给定海拔平面 z = z_det 的交点(芯位)，并返回 Xmax 到芯位的距离。

    参数:
        x, y, z (float): Xmax 坐标。
        theta (float): 天顶角(度)。
        phi (float): 方位角(度)。
        z_det (float): 探测器平面海拔(米)。

    返回:
        tuple: (core_x, core_y, core_z, dmax)
            core_x, core_y, core_z (float): 芯位坐标。
            dmax (float): Xmax 到芯位的几何距离。

    说明:
        若轴向 z 分量过小(近水平)，直线与平面交点不稳定，函数返回 (nan, nan, nan, inf)。
    """
    
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)

    # 与 helper.spherical_to_cartesian 的角度约定保持一致
    ux = np.sin(theta_rad) * np.cos(phi_rad)
    uy = np.sin(theta_rad) * np.sin(phi_rad)
    uz = np.cos(theta_rad)

    if np.abs(uz) < 1e-12:
        return np.nan, np.nan, np.nan, np.inf

    s = (z_det - z) / uz
    core_x = x + s * ux
    core_y = y + s * uy
    core_z = z_det
    dmax = np.sqrt((core_x - x) ** 2 + (core_y - y) ** 2 + (core_z - z) ** 2)

    return core_x, core_y, core_z, dmax

class SWF:
    """
    球面波前 (Spherical Wave Front) 拟合类。
    
    用于定义球面波前拟合的目标函数 (chi2)，以重建簇射源点位置。
    """
    def __init__(self, x, y, z, t):
        """
        初始化 SWF 类。
        
        参数:
            x, y, z (numpy.ndarray): 天线位置坐标。
            t (numpy.ndarray): 信号到达时间。
        """
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        
    def chi2(self, xs, ys, zs, rc):
        """
        计算卡方值 (Chi-squared)。
        
        目标是最小化观测到的时间与球面波模型预测的时间之间的差异。
        模型假设信号从源点 (xs, ys, zs) 以球面波形式传播。
        
        参数:
            xs, ys, zs (float): 假设的源点坐标。
            rc (float): 相对光速因子 (未使用，保留接口)。
            
        返回:
            float: Chi2 值。
        """
        dx = self.x - xs
        dy = self.y - ys
        dz = self.z - zs
        
        n_eff_mean = np.mean(aires.GetZHSEffectiveRefractionIndex(xs, ys, zs, np.mean(self.x), np.mean(self.y), np.mean(self.z))) # 获取有效折射率
        # 所有天线共用一个有效折射率, 合适吗????
        
        dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        ts = (self.t * c / n_eff_mean - dr).mean() # unit of t is s, 估算发射时间 ts
        fmin = (self.t * c / n_eff_mean - ts) - dr # 计算残差: (t_obs - t_emit) * c/n - distance
        chi2 = (fmin ** 2 / (6e-9 * 299792458) ** 2).sum()
        return chi2
        
    def __call__(self, xs, ys, zs, rc):
        """
        使类实例可调用，以便 iminuit 使用。
        """
        chi2 = self.chi2(xs, ys, zs, rc)
        return chi2

def SWF_fit(x, y, z, ant_t, x0, y0, z0, simplex=True):
    """
    执行球面波前拟合。
    
    使用 iminuit 寻找最佳的源点位置 (xs, ys, zs)。
    
    参数:
        x, y, z (numpy.ndarray): 天线位置。
        ant_t (numpy.ndarray): 信号到达时间。
        x0, y0, z0 (float): 源点坐标的初始猜测值。
        
    返回:
        tuple: (xs, ys, zs, fval)
        xs, ys, zs (float): 拟合得到的源点坐标。
        fval (float): 最小化的函数值 (chi2)。
    """
    m = iminuit.Minuit(SWF(x, y, z, ant_t), xs = x0, ys = y0, zs = z0, rc = 1)
    
    m.limits = ([-3e5, 3e5], [-3e5, 3e5], [1e3, 5e4], [0, 1]) # 坐标限制范围
    # 源坐标(x, y, z)的范围限制目前合理吗?????

    m.fixed = (False, False, False, True) # 固定 rc
    m.errordef = 1
    m.tol = 0.0001
    if simplex:
        m.simplex()  # 先用单纯形法粗略搜索 (单起点已有好初值时无需)
    m.migrad()  # 再用 MIGRAD 算法精确最小化
    m.hesse()   # 计算海森矩阵 (误差估计)
    xs = m.values["xs"]
    ys = m.values["ys"]
    zs = m.values["zs"]
    fval = m.fval
    xs_err = m.errors["xs"]
    ys_err = m.errors["ys"]
    zs_err = m.errors["zs"]
    return xs, ys, zs, fval, xs_err, ys_err, zs_err

class ADF:
    """
    角分布函数 (Angular Distribution Function) 拟合类。
    
    用于定义切伦科夫辐射角分布的拟合目标函数 (chi2)。
    考虑了地磁效应修正。
    """
    def __init__(self, x, y, z, p, xs, ys, zs, obs_altitude, p_sigma=None):
        """
        初始化 ADF 类。
        
        参数:
            x, y, z (numpy.ndarray): 天线位置。
            p (numpy.ndarray): 观测到的信号幅度或能量通量。
            xs, ys, zs (float): 簇射源点 (Xmax) 坐标。
            obs_altitude (float): 观测站海拔高度 (米)，用于计算芯位和距离。
            p_sigma (numpy.ndarray): 【新增】每根天线 Fluence 的物理传播误差 (来自 e_recons 协方差),
                用作 chi2 归一化分母。若为 None, 回退到旧的经验公式 (向后兼容)。
        """
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.obs_altitude = obs_altitude
        self.p_sigma = p_sigma
        
        n = np.mean(aires.GetZHSEffectiveRefractionIndex(xs, ys, zs, np.mean(self.x), np.mean(self.y), np.mean(self.z))) # 计算源点到平均天线位置的有效折射率
        self.wc = np.arccos(1 / n) # 计算切伦科夫角 (Cherenkov angle)
    
        
    def chi2(self, theta, phi, dw, A, B, rc):
        """
        计算卡方值 (Chi-squared)。
        
        参数:
            theta (float): 簇射轴天顶角 (度)。
            phi (float): 簇射轴方位角 (度)。
            dw (float): 切伦科夫环宽度参数。
            A (float): 幅度参数。
            B (float): 地磁不对称参数 (通常固定)。
            rc (float): 未使用。
            
        返回:
            float: Chi2 值。
        """
        
        cs = coordinatesystems.cstrafo(np.deg2rad(theta), np.deg2rad(phi), magnetic_field_vector) # 坐标变换到 vxB 系统
        
        corex, corey, corez, dXmax = find_core(self.xs, self.ys, self.zs, theta, phi, self.obs_altitude) # 直接计算芯位坐标和 Xmax 到芯位的距离
        shower_core = np.array([corex, corey, corez])
        
        pos = np.array([self.x, self.y, self.z]).T
        pos_vxB = cs.transform_to_vxB_vxvxB(pos, shower_core)

        xmax_vxB = cs.transform_to_vxB_vxvxB(np.array([[self.xs, self.ys, self.zs]]), shower_core) #坐标系一致的写法, 先把 Xmax 也变到 vxB, 再在同一坐标系下算几何量
        #print("Xmax in vxB coordinates:", xmax_vxB) ; exit()
        #r_vxB = pos_vxB - xmax_vxB # Xmax -> 天线 向量(vxB)
        #l_ = np.linalg.norm(r_vxB, axis = 1) # 传播距离
        #sin_omega = np.sqrt(r_vxB[:, 0] ** 2 + r_vxB[:, 1] ** 2) / l_
        #omega = np.arcsin(np.clip(sin_omega, 0.0, 1.0)) # 每个天线与簇射轴线夹角(0 ~ pi/2)
        
        #l_ = np.sqrt((pos_vxB[:, 0] - self.xs) ** 2 + (pos_vxB[:, 1] - self.ys) ** 2 + (pos_vxB[:, 2] - self.zs) ** 2)
        l_ = np.sqrt((pos_vxB[:, 0] - xmax_vxB[0]) ** 2 + (pos_vxB[:, 1] - xmax_vxB[1]) ** 2 + (pos_vxB[:, 2] - xmax_vxB[2]) ** 2)
        #omega = np.arcsin(np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2) / l_) # 计算每个天线的视角 (viewing angle) omega
        omega = np.arcsin(np.clip(np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2) / np.maximum(l_, 1e-10), 0.0, 1.0)) # 数值稳定性修正, 避免输入 arcsin 的值超过 [0, 1]
        
        # 地磁效应修正
        shower_axis = helper.spherical_to_cartesian(np.deg2rad(theta), np.deg2rad(phi))
        eta = helper.get_normalized_angle(np.arctan2(pos_vxB[:, 1], pos_vxB[:, 0]))
        alpha = np.arccos(np.dot(shower_axis, magnetic_field_vector))
        f_GeoM = 1 + B * np.sin(alpha) ** 2 * np.cos(eta)
        
        # 距离衰减修正 (Early-Late)
        # ant2Xmax_dis = np.dot(pos, -shower_axis)
        # early_late = np.square((dXmax + ant2Xmax_dis) / dXmax)
        
        ant2Xmax_dis = ((self.x - self.xs) ** 2 + (self.y - self.ys) ** 2 + (self.z - self.zs) ** 2) ** 0.5 # 天线到 Xmax 的距离
        early_late = np.square(ant2Xmax_dis / dXmax)

        fluence_decay = self.p * early_late

        # if len(self.x) > 50: self.wc = omega[fluence_decay == max(fluence_decay)] # 如果天线数量足够多，尝试从数据中更新切伦科夫角 wc
        # 代码风险: 是否可能出现数组wc ????
        if len(self.x) > 50: self.wc = omega[np.argmax(fluence_decay)]

        f_Cerenkov = 1 / (1 + 4 * (((np.tan(omega) / np.tan(self.wc)) ** 2 - 1) / dw) ** 2) # 切伦科夫函数模型

        # 预测值
        adf = A * f_Cerenkov * f_GeoM / early_late
        
        
        # 计算残差
        # 【修改】归一化分母改用 e_recons 协方差传播得到的物理 Fluence 误差 (self.p_sigma),
        # 不再使用"10%*p + 1%*max(p)"的经验公式; 若未提供 p_sigma 则回退到旧公式 (向后兼容)。
        denom = self.p_sigma if self.p_sigma is not None else (0.1 * self.p + 0.01 * np.max(self.p))
        fmin = (self.p - adf) / denom
        chi2 = (fmin ** 2).sum()
        
        return chi2
        
    def __call__(self, theta, phi, dw, A, B, rc):
        chi2 = self.chi2(theta, phi, dw, A, B, rc)
        return chi2

def ADF_fit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, ant_p_sigma=None):
    """见 :func:`_adf_minuit`; 本函数只固定 amp / fix_dw 两个开关。"""
    r = _adf_minuit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, obs_altitude,
                    dw0 = dw0, A0 = A0, ant_p_sigma = ant_p_sigma, amp = "free", fix_dw = False)
    return r

def ADF_fit_fixdw(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, ant_p_sigma=None):
    """见 :func:`_adf_minuit`; 本函数只固定 amp / fix_dw 两个开关。"""
    r = _adf_minuit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, obs_altitude,
                    dw0 = dw0, A0 = A0, ant_p_sigma = ant_p_sigma, amp = "free", fix_dw = True)
    return r

# 20 元组的位置契约（:func:`recons_direction_swf_adf` 的返回值）。
# 这是**唯一**允许出现下标的地方：其余代码一律走 :class:`DirectionResult` 的具名字段。
_TUPLE_FIELDS = (
    "zenith_init", "azimuth_init", "chi2_plane_init",
    "x_max", "y_max", "z_max",
    "chi2_sph", "zenith", "azimuth", "dw", "A", "fmin", "wc",
    "zenith_err", "azimuth_err", "dw_err", "A_err",
    "x_max_err", "y_max_err", "z_max_err",
)
_TUPLE_LEN = len(_TUPLE_FIELDS)
assert _TUPLE_LEN == 20, _TUPLE_LEN

class DirectionTuple(tuple):
    """`recons_direction_swf_adf` 的 20 元组返回，额外带一个 `dw_fixed` 属性。

    为什么不用 `rec_dw_err == 0` 当哨兵：dw 被固定时误差不可得，写成 0.0 会被读成"误差为零"。
    dw_err 现在固定时为 NaN，而 NaN 无法区分"固定"与"拟合失败"，所以固定与否单独带出来。
    保持 tuple 子类：长度、下标、解包全部与旧契约一致。
    """

    dw_fixed = False

    def __new__(cls, values, dw_fixed = False):
        obj = super().__new__(cls, values)
        obj.dw_fixed = dw_fixed
        return obj

class DirectionResult(NamedTuple):
    """方向重建的具名结果。

    ``raw`` 是 20 元组原件，迁移期兜底；新代码请只读具名字段。
    """

    zenith_init: float
    azimuth_init: float
    zenith: float
    azimuth: float
    dw: float
    A: float
    wc: float
    x_max: float
    y_max: float
    z_max: float
    chi2_plane: float
    chi2_sph: float
    fmin: float
    zenith_err: float
    azimuth_err: float
    dw_err: float
    A_err: float
    dw_fixed: bool
    x_max_err: float
    y_max_err: float
    z_max_err: float
    raw: tuple

def direction_result_from_tuple(t) -> DirectionResult:
    """把 :func:`recons_direction_swf_adf` 的返回值包成 :class:`DirectionResult`。

    下标只在这里出现，并且由 ``_TUPLE_FIELDS`` 驱动；长度不符直接抛错而不是静默错位。
    """
    if len(t) != _TUPLE_LEN:
        raise ValueError(f"方向返回元组长度应为 {_TUPLE_LEN}，实际 {len(t)}")
    by_name = dict(zip(_TUPLE_FIELDS, t))
    return DirectionResult(
        zenith_init = by_name["zenith_init"], azimuth_init = by_name["azimuth_init"],
        zenith = by_name["zenith"], azimuth = by_name["azimuth"],
        dw = by_name["dw"], A = by_name["A"], wc = by_name["wc"],
        x_max = by_name["x_max"], y_max = by_name["y_max"], z_max = by_name["z_max"],
        chi2_plane = by_name["chi2_plane_init"], chi2_sph = by_name["chi2_sph"],
        fmin = by_name["fmin"],
        zenith_err = by_name["zenith_err"], azimuth_err = by_name["azimuth_err"],
        dw_err = by_name["dw_err"], A_err = by_name["A_err"],
        dw_fixed = bool(getattr(t, "dw_fixed", False)),
        x_max_err = by_name["x_max_err"], y_max_err = by_name["y_max_err"],
        z_max_err = by_name["z_max_err"],
        raw = tuple(t),
    )

def recons_direction_struct(x, y, z, t, p, init_input, obs_altitude, **kwargs) -> DirectionResult:
    """与 :func:`recons_direction_swf_adf` 同参，但返回 :class:`DirectionResult`。"""
    return direction_result_from_tuple(
        recons_direction_swf_adf(x, y, z, t, p, init_input, obs_altitude, **kwargs)
    )

def reconstruct_direction_only(x, y, z, t, p, init_input, obs_altitude,
                               p_sigma = None, x_xmax = None, y_xmax = None,
                               z_xmax = None, chi_square_external = None) -> DirectionResult:
    """只做方向重建：不建目录、不写盘、不做能量重建（R2 的分离入口）。

    与 ``efield_recons_from_efield_PWF`` 内的方向那一段完全同参，
    调用方拿到 :class:`DirectionResult` 即可，不需要触发任何 IO。
    """
    raw = recons_direction_swf_adf(x, y, z, t, p, init_input, obs_altitude,
                                   p_sigma, x_xmax, y_xmax, z_xmax, chi_square_external)
    return direction_result_from_tuple(raw)

def recons_direction_swf_adf(x, y, z, t, p, init_input, obs_altitude, p_sigma=None, x_xmax_external=None, y_xmax_external=None, z_xmax_external=None, chi_square_external=None):  # time given in ns
    """
    联合重建簇射方向和 Xmax 位置。
    
    首先使用 SWF_fit 网格搜索最佳的源点位置 (Xmax)。
    然后使用 ADF_fit 网格搜索最佳的方向 (theta, phi) 和切伦科夫参数。
    
    参数:
        x, y, z (numpy.ndarray): 天线位置。
        t (numpy.ndarray): 到达时间。
        p (numpy.ndarray): 信号幅度/能量。
        init_input (tuple): 初始猜测 (zenith_ini, azimuth_ini, chi2Plane)。
        obs_altitude (float): 观测高度，用于计算芯位。
        
    返回:
        tuple: 重建结果 (初始角度, 平面波chi2, 重建Xmax坐标, 球面波chi2, 重建角度, dw, A, 最小chi2)。
    """
    if x_xmax_external is not None and y_xmax_external is not None and z_xmax_external is not None:
        # 外部给定 Xmax 源点 (来自 candidates YAML 的 SWM 结果): 用作 SWF_fit 的单一初值,
        # 仅执行一次拟合 (不跑 4x4x4 网格)。SWM 源点与内部天线坐标同为重建坐标系(未旋转),
        # 故无需坐标变换。chi_square_external 仅作日志参考, 真值改用精炼后的 fval。
        # 初值已落在全局极小附近, simplex 无用: 跳过它结果与网格相同而耗时减半
        # (见 tools/sim_dunhuang/swf_seed_validity.md)。网格分支仍保留 simplex 作保底。
        x0, y0, z0 = float(x_xmax_external), float(y_xmax_external), float(z_xmax_external)
        chi2_external_ref = float(chi_square_external) if chi_square_external is not None else np.nan
        rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph, rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err = SWF_fit(x, y, z, t, x0, y0, z0, simplex=False)
        logger.info(f"[SWM] SWF 单点初值拟合 (x0={x0:.1f}, y0={y0:.1f}, z0={z0:.1f}, chi2_ext={chi2_external_ref}): "
                    f"({rec_x_xmax:.1f}, {rec_y_xmax:.1f}, {rec_z_xmax:.1f}), chi2 = {chi2Sph:.2e}")
    else:
        rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph = np.inf, np.inf, np.inf, np.inf
        rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err = np.inf, np.inf, np.inf

        for x0 in xList: # 网格搜索最佳 Xmax 位置 (SWF 拟合)
            for y0 in yList:
                for z0 in zList:
                    x_xmax, y_xmax, z_xmax, fval, x_err, y_err, z_err = SWF_fit(x, y, z, t, x0, y0, z0)
                    if fval < chi2Sph:
                        chi2Sph = fval
                        rec_x_xmax = x_xmax
                        rec_y_xmax = y_xmax
                        rec_z_xmax = z_xmax
                        rec_x_xmax_err = x_err
                        rec_y_xmax_err = y_err
                        rec_z_xmax_err = z_err
                    
    zenith_ini, azimuth_ini, chi2Plane  = init_input
    rec_dw, rec_A, rec_theta, rec_phi, fmin = np.inf, np.inf, np.inf, np.inf, np.inf
    rec_dw_err, rec_A_err, rec_theta_err, rec_phi_err = 0, 0, 0, 0
    # dw 是否被固定（而不是“误差为零”）：下游 dof / dE_dw 判据一律看这个布尔，
    # 不再用 rec_dw_err == 0 当哨兵（固定时 rec_dw_err 现在是 NaN）。
    rec_dw_fixed = False
    
    
    thetalist = np.linspace(zenith_ini - 0.5, zenith_ini + 0.5, 3)
    philist = np.linspace(azimuth_ini - 0.5, azimuth_ini + 0.5, 3) # 这两个参数范围是否给的太窄, 真实情况应当是 0-90, 0-360 ????
    dwList = np.linspace(0.1, 4., 4)
    # A 由解析式给出 (ADF_fit_analytic), 因此不再需要 Alist 那层初值循环:
    # 3x3 theta/phi x 4 dw = 36 个起点, 实测与 A0 网格版方向一致到 1e-6 度、chi2 逐位相同 (<adf_speedup.md>)。
    p_sigma_masked = p_sigma[~np.isnan(p)] if p_sigma is not None else None
    _xm, _ym, _zm = x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)]
    _pm = p[~np.isnan(p)]

    for theta0 in thetalist: # 网格搜索最佳方向和 ADF 参数 (A 解析消去)
        for phi0 in philist:
            for dw0 in dwList:
                theta, phi, dw, A, fval = ADF_fit_analytic(_xm, _ym, _zm, _pm, rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, obs_altitude, p_sigma_masked)
                if fval < fmin:
                    fmin = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A

    # 网格只给点估计; 在最佳点跑一次 ADF_fit 取误差(顺便再精化一步), 保持返回签名不变。
    if np.isfinite(rec_theta):
        _th, _ph, _dw, _A, _fv, _te, _pe, _de, _ae = ADF_fit(
            _xm, _ym, _zm, _pm, rec_x_xmax, rec_y_xmax, rec_z_xmax,
            rec_theta, rec_phi, rec_dw, rec_A, obs_altitude, p_sigma_masked)
        if _fv <= fmin:
            fmin, rec_theta, rec_phi, rec_dw, rec_A = _fv, _th, _ph, _dw, _A
            rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err = _te, _pe, _de, _ae
        else:
            rec_theta_err = rec_phi_err = rec_dw_err = rec_A_err = np.nan

    if rec_dw > 3.499 or rec_dw < 0.101: # 如果拟合得到的 dw 超出合理范围，使用参数化公式固定 dw 重新拟合

        dXmax = find_core(rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta, rec_phi, obs_altitude)[3] # 计算 Xmax 到芯位的距离
        XmaxDis_km = dXmax / 1e3

        dw0 = 227.958137 / (176.414293 * XmaxDis_km ** 2 + 112.917133 * XmaxDis_km) + 1.351487 # dw 参数化公式

        fmin_fixdw = np.inf

        for theta0 in thetalist: # 网格搜索最佳方向, 固定 dw, A 解析消去
            for phi0 in philist:
                theta, phi, dw, A, fval = ADF_fit_analytic_fixdw(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, obs_altitude, p_sigma_masked)
                if fval < fmin_fixdw:
                    fmin_fixdw = fval
                    rec_theta = theta
                    rec_phi = phi
                    rec_dw = dw
                    rec_A = A

        # 固定 dw 的网格同样只给点估计: 在最佳点跑一次 ADF_fit_fixdw 取误差
        _t, _p, _d, _a, _fv, rec_theta_err, rec_phi_err, _de, rec_A_err = ADF_fit_fixdw(
            x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)],
            rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta, rec_phi, dw0, rec_A,
            obs_altitude, p_sigma_masked)
        if _fv <= fmin_fixdw:
            fmin_fixdw, rec_theta, rec_phi, rec_A = _fv, _t, _p, _a
        rec_dw_err = np.nan          # 固定 dw -> 误差不可得；固定与否看 rec_dw_fixed
        rec_dw_fixed = True

        fmin = fmin_fixdw

        # rec_theta, rec_phi, rec_dw, rec_A, fmin = ADF_fit_fixdw(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, A0, obs_altitude)
    
        logger.info(f"dw out of range, refit with fixed dw: {rec_dw}")
    
    try: # 重新调用 ADF 类计算切伦科夫角 wc，注意这里使用了拟合得到的参数
        temp_adf = ADF(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, obs_altitude, (p_sigma[~np.isnan(p)] if p_sigma is not None else None))
        temp_adf.chi2(rec_theta, rec_phi, rec_dw, rec_A, 0.01, 1)
        wc_adf = temp_adf.wc
        logger.info(f"Fitted wc: {np.asarray(wc_adf).reshape(-1)[0]} rec_dw: {rec_dw}")
    except Exception as e:
        logger.warning(f"Could not calculate wc for display: {e}")
    
    return DirectionTuple((zenith_ini, azimuth_ini, chi2Plane, rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph / (len(x) - 3), rec_theta, rec_phi, rec_dw, rec_A, fmin / (len(p[~np.isnan(p)]) - (3 if rec_dw_fixed else 4)), np.asarray(wc_adf).reshape(-1)[0], rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err, rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err,), dw_fixed = rec_dw_fixed) # wc_adf 兼容写法(标量/单元素数组都可)

def f_Che(par, omega_):
    """
    切伦科夫函数模型。
    
    参数:
        par (list): [幅度, 切伦科夫角, 宽度]。
        omega_ (float): 视角。
    """

    return par[0] / (1 + 4 * (((np.tan(omega_) / np.tan(par[1])) ** 2 - 1) / par[2]) ** 2)


def f_adf(par, omega_, alpha_, phi_):
    """
    带有地磁不对称修正的 ADF 函数。
    
    参数:
        par (list): [幅度, 切伦科夫角, 宽度]。
        omega_ (float): 视角。
        alpha_ (float): 地磁角。
        phi_ (float): 方位角。
    """
    asym_corr = (1 + 0.01 * np.sin(alpha_) ** 2 * np.cos(phi_))

    return asym_corr * par[0] / (1 + 4 * (((np.tan(omega_) / np.tan(par[1])) ** 2 - 1) / par[2]) ** 2)

def che_fit_chi2(omega_, data, amp0, w_c0, domega0, fix_dw):
    """
    切伦科夫函数拟合。
    
    使用 iminuit 进行拟合。
    
    参数:
        omega_ (numpy.ndarray): 视角。
        data (numpy.ndarray): 能量通量。
        amp0 (float): 幅度初始值。
        w_c0 (float): 切伦科夫角初始值。
        domega0 (float): 宽度初始值。
    """
    
    def che_fit_predict(amp, w_c, domega):
        predicted = f_Che([amp, w_c, domega], omega_)
        # return ((data - predicted) ** 2).sum()
        return ((data - predicted) ** 2 / (0.1 * data + 0.01 * np.max(data)) ** 2).sum()

    initial_params = {'amp': amp0, 'w_c': w_c0, 'domega': domega0}

    m = iminuit.Minuit(che_fit_predict, **initial_params)
    m.limits = [(amp0 / 10, amp0 * 1000), (1e-4, 1.0), (0.01, 5.0)]
    if fix_dw == 1: m.fixed = (False, False, True) # 固定 dw 参数
    else: m.fixed = (False, False, False) # 全部参数自由拟合
    m.errordef = 1
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    amp_fit, amp_err = m.values["amp"], m.errors["amp"]
    w_c_fit, w_c_err = m.values["w_c"], m.errors["w_c"]
    domega_fit, domega_err = m.values["domega"], m.errors["domega"]
    chi2 = m.fval
    
    return amp_fit, w_c_fit, domega_fit, chi2, amp_err, w_c_err, domega_err

def che_fit(omega_, data, amp0, w_c0, domega0):

    amp, w_c, domega, chi2_min = np.inf, np.inf, np.inf, np.inf
    amp_err, w_c_err, domega_err = 0, 0, 0

    amp_list, w_c_list, domega_list = np.linspace(amp0 / 2, amp0 * 3, 5), np.linspace(1e-4, w_c0 + 0.05, 5), np.linspace(0.01, domega0 + 2.0, 5)

    for amp_ in amp_list:
        for w_c_ in w_c_list:
            for domega_ in domega_list:
                fit_results = che_fit_chi2(omega_, data, amp_, w_c_, domega_, 0) # fix_dw = 0, 全参数自由拟合
                if fit_results[3] < chi2_min:
                    amp, w_c, domega = fit_results[0], fit_results[1], fit_results[2]
                    chi2_min = fit_results[3]
                    amp_err, w_c_err, domega_err = fit_results[4], fit_results[5], fit_results[6]

    # if domega > 3.499 or domega < 0.101: # 如果拟合得到的 dw 超出合理范围，使用参数化公式固定 dw 重新拟合

    #     chi2_min = np.inf

    #     for amp_ in amp_list:
    #         for w_c_ in w_c_list:
    #             fit_results = che_fit_chi2(omega_, data, amp_, w_c_, domega0, 1) # fix_dw = 1, 固定 dw 参数
    #             if fit_results[3] < chi2_min:
    #                 amp, w_c = fit_results[0], fit_results[1]
    #                 domega = fit_results[2]
    #                 chi2_min = fit_results[3]

    return amp, w_c, domega, chi2_min / (len(omega_) - 3), amp_err, w_c_err, domega_err

def ADF_constant_fit(fluence, early_late, fluence_sigma):
    """
    常数模型拟合: Fluence = A / early_late
    使用 iminuit 最小化 χ²，与 ADF 模型嵌套（Δdof = 3）。
    
    参数:
        fluence        : 原始能流 [eV/m^2]
        early_late     : 距离衰减因子 (d/dXmax)^2
        fluence_sigma  : 能流物理误差 [eV/m^2]
    
    返回:
        A, A_err, chi2, ndof
    """
    
    def ADF_constant(A): return np.sum((fluence - A / early_late) ** 2 / fluence_sigma ** 2)
    
    m = iminuit.Minuit(ADF_constant, A = np.median(fluence * early_late)) # 初始猜测 A 为中位数
    m.errordef = 1
    m.migrad()
    m.hesse()
    
    return m.values["A"], m.errors["A"], m.fval / (len(fluence) - 1)

def density_and_alpha_correction(E_em_GeV, rho, alpha):
    """
    计算地磁修正后的能量。
    
    参数:
        E_em_GeV (float): 重建的地磁辐射能量 (GeV)。
        rho (float): 大气密度 (kg/m^3)。
        alpha (float): 地磁角。

    返回:
        float: 地磁修正后的能量 (GeV)。

    公式说明:
        参考自: https://arxiv.org/pdf/2507.06698v1 (公式7和图3)
    """

    density_correction = 1 - 47.90 + 47.90 * np.exp(-0.01 * (rho - 0.23)) - 0.03 * rho ** (-1) + 0.19
    
    return E_em_GeV / (density_correction ** 2 * np.sin(alpha) ** 1.06)

def _adf_chi2_profiled(x, y, z, p, xs, ys, zs, obs_altitude, p_sigma,
                       theta, phi, dw, wc0, B = 0.01):
    """给定 (theta, phi, dw), 解析求出最优幅度 A 及对应 chi2。

    模型对 A 是线性的: adf = A * g, g = f_Cerenkov*f_GeoM/early_late,
    chi2 = sum(((p - A*g)/denom)^2) -> A* = sum(p*g/denom^2)/sum(g^2/denom^2),
    chi2_min = sum(p^2/denom^2) - (sum(p*g/denom^2))^2 / sum(g^2/denom^2)。
    几何与 ADF.chi2 逐行一致 (含 len(x)>50 时用 omega[argmax(p*early_late)] 更新 wc)。
    """
    cs = coordinatesystems.cstrafo(np.deg2rad(theta), np.deg2rad(phi), magnetic_field_vector)
    corex, corey, corez, dXmax = find_core(xs, ys, zs, theta, phi, obs_altitude)
    shower_core = np.array([corex, corey, corez])

    pos = np.array([x, y, z]).T
    pos_vxB = cs.transform_to_vxB_vxvxB(pos, shower_core)
    xmax_vxB = cs.transform_to_vxB_vxvxB(np.array([[xs, ys, zs]]), shower_core)

    l_ = np.sqrt((pos_vxB[:, 0] - xmax_vxB[0]) ** 2
                 + (pos_vxB[:, 1] - xmax_vxB[1]) ** 2
                 + (pos_vxB[:, 2] - xmax_vxB[2]) ** 2)
    omega = np.arcsin(np.clip(np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2)
                              / np.maximum(l_, 1e-10), 0.0, 1.0))

    shower_axis = helper.spherical_to_cartesian(np.deg2rad(theta), np.deg2rad(phi))
    eta = helper.get_normalized_angle(np.arctan2(pos_vxB[:, 1], pos_vxB[:, 0]))
    alpha = np.arccos(np.dot(shower_axis, magnetic_field_vector))
    f_GeoM = 1 + B * np.sin(alpha) ** 2 * np.cos(eta)

    ant2Xmax_dis = np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)
    early_late = np.square(ant2Xmax_dis / dXmax)

    wc = wc0
    if len(x) > 50:
        wc = omega[np.argmax(p * early_late)]

    f_Cerenkov = 1.0 / (1.0 + 4.0 * (((np.tan(omega) / np.tan(wc)) ** 2 - 1.0) / dw) ** 2)
    g = f_Cerenkov * f_GeoM / early_late

    denom = p_sigma if p_sigma is not None else (0.1 * p + 0.01 * np.max(p))
    w = 1.0 / np.square(denom)
    s_pg, s_gg, s_pp = np.sum(p * g * w), np.sum(g * g * w), np.sum(p * p * w)
    if not np.isfinite(s_gg) or s_gg <= 0:
        return np.inf, 0.0
    a_best = s_pg / s_gg
    return float(s_pp - s_pg * s_pg / s_gg), float(a_best)

def ADF_fit_analytic(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, obs_altitude,
                     ant_p_sigma = None):
    """见 :func:`_adf_minuit`; 本函数只固定 amp / fix_dw 两个开关。"""
    r = _adf_minuit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, obs_altitude,
                    dw0 = dw0, A0 = None, ant_p_sigma = ant_p_sigma, amp = "profiled", fix_dw = False)
    return r[:5]

def ADF_fit_analytic_fixdw(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, obs_altitude,
                     ant_p_sigma = None):
    """见 :func:`_adf_minuit`; 本函数只固定 amp / fix_dw 两个开关。"""
    r = _adf_minuit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, obs_altitude,
                    dw0 = dw0, A0 = None, ant_p_sigma = ant_p_sigma, amp = "profiled", fix_dw = True)
    return r[:5]

def recons_direction(x, y, z, t, p, init_angles, obs_altitude, p_sigma = None,
                     x_xmax = None, y_xmax = None, z_xmax = None,
                     chi_square_external = None, method = "ADF"):
    """只做**方向**重建, 不做能量重建, 也不落盘。

    与 :func:`efield_recons_from_efield_PWF` 的区别: 后者是"电压->反演电场->SWF->ADF"
    的完整链路, 会把结果写到 ``./Reconstruction/<file>_event_N/`` 并附带能量信息;
    本函数只回答"这个事例的来向是多少", 因此

      - 不创建/读写任何目录 (并行安全);
      - 不做电场反演 (能量重建与方向无关);
      - PWM / SWM 的先验直接由调用者通过参数传入。

    参数
    ----
    x, y, z : array_like
        天线坐标 [m], 形状 (n_du,)。
    t : array_like
        到达时间, **单位秒** (不是 ns)。ADF/SWF 需要绝对时间对齐, PWF 只用差值。
    p : array_like
        fluence, 形状 (n_du,)。可用真电场 Σ|E|²、真电压 ΣV²、或带噪 ADC 的
        峰值对齐 ±N 样点窗内 ΣV² (见 :func:`adf_angles_from_voc`)。
    init_angles : (zenith, azimuth) 或 (zenith, azimuth, chi2)
        PWM (或任一平面波) 结果作初值 [deg]。
    obs_altitude : float
        观测面海拔 [m]。
    x_xmax, y_xmax, z_xmax, chi_square_external : optional
        SWM 结果作先验 (Xmax 源点 [m] 与外部 chi2)。给定则跳过内部 SWF 网格,
        直接用该点做 ADF 几何中心。
    method : {"ADF", "PWF", "SWF"}
        "ADF" (默认) —— 切伦科夫环幅度分布拟合, 内部 36 起点 + 解析幅度 A;
        "PWF" —— 平面波前拟合, 初值取 ``init_angles``;
        "SWF" —— 球面波前拟合, 初值取 ``init_angles`` + (若给出) SWM 源点。

    返回
    ----
    dict
        ``method``, ``zenith``, ``azimuth`` [deg], ``chi2``, 以及对应方法的附加量
        (ADF: ``dw``/``A``/``wc``/``xmax``; SWF: ``x_max``/``y_max``/``z_max``;
        PWF: ``chi2_per_dof``)。失败时 ``zenith``/``azimuth`` 为 NaN。
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    t = np.asarray(t, float); p = np.asarray(p, float)
    zenith_ini, azimuth_ini = float(init_angles[0]), float(init_angles[1])
    chi2_plane = float(init_angles[2]) if len(init_angles) > 2 else 0.0
    out = dict(method = method, zenith = np.nan, azimuth = np.nan, chi2 = np.nan,
               dw = np.nan, A = np.nan, wc = np.nan, x_max = np.nan,
               y_max = np.nan, z_max = np.nan, chi2_per_dof = np.nan)

    ok = np.isfinite(p) & (p > 0)
    if ok.sum() < 5:
        logger.warning(f"recons_direction: 有效天线数 {int(ok.sum())} < 5, 跳过")
        return out

    if method.upper() == "PWF":
        theta, phi, fval_dof, theta_err, phi_err = PWF_fit(
            x[ok], y[ok], z[ok], t[ok], zenith_ini, azimuth_ini)
        out.update(zenith = float(theta), azimuth = float(phi),
                   chi2_per_dof = float(fval_dof),
                   chi2 = float(fval_dof) * max(len(t[ok]) - 2, 1))
        return out

    if method.upper() == "SWF":
        # SWF 拟合的是源点 (xs, ys, zs), 返回 (xs, ys, zs, fval, xs_err, ys_err, zs_err)。
        # 方向由源点相对阵列给出, 需由调用者换算; 这里只汇报源点与 chi2。
        if x_xmax is None:
            # 无 SWM 先验: 从 init_angles 沿视线推一个 10 km 处的初值
            d = 10000.0
            x_xmax = -d * np.sin(np.radians(zenith_ini)) * np.cos(np.radians(azimuth_ini))
            y_xmax = -d * np.sin(np.radians(zenith_ini)) * np.sin(np.radians(azimuth_ini))
            z_xmax = obs_altitude + d * np.cos(np.radians(zenith_ini))
            simplex = True
        else:
            simplex = False
        xs, ys, zs, fval, xs_err, ys_err, zs_err = SWF_fit(
            x[ok], y[ok], z[ok], t[ok], x_xmax, y_xmax, z_xmax, simplex = simplex)
        out.update(x_max = float(xs), y_max = float(ys), z_max = float(zs),
                   chi2 = float(fval),
                   zenith = float(np.degrees(np.arccos(np.clip(
                       (zs - obs_altitude) / max(np.hypot(xs, ys, zs - obs_altitude), 1e-9), -1, 1)))),
                   azimuth = float(np.degrees(np.arctan2(-ys, -xs)) % 360.0))
        return out

    # ---- 默认: ADF ----
    try:
        r = recons_direction_swf_adf(x, y, z, t, p, [zenith_ini, azimuth_ini, chi2_plane],
                         obs_altitude, p_sigma, x_xmax, y_xmax, z_xmax, chi_square_external)
    except Exception as exc:                      # noqa: BLE001
        logger.warning(f"recons_direction: recons_angle 失败: {exc}")
        return out
    res = direction_result_from_tuple(r)
    out.update(zenith = float(res.zenith), azimuth = float(res.azimuth),
               dw = float(res.dw), A = float(res.A), wc = float(res.wc))
    return out

# 向后兼容别名: 旧名 recons_angle 保留 (SWF->ADF 方向重建驱动, 返回 20 元组)。
# 仓库内没有代码调用它；还在用这个名字的文档只有 README.md:316
# （reconstruction_candidate/reconstruction_candidate.md 讲的是旧模块自己的同名函数）。
recons_angle = recons_direction_swf_adf

def _adf_minuit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, obs_altitude,
                dw0 = 1.0, A0 = None, ant_p_sigma = None,
                amp = "free", fix_dw = False):
    """ADF 拟合的唯一实现。四个公开变体只是本函数的薄壳。

    两个正交开关:

    ``amp``      —— 幅度的处理方式
        ``"free"``     : A 交给 Minuit（与旧 ``ADF_fit`` 完全一致: B=0.01/rc=1 固定,
                         A 边界 [A0/2, A0*1e3], migrad+hesse 取误差）
        ``"profiled"`` : A 解析消去（profile likelihood, 见 :func:`_adf_chi2_profiled`）,
                         只优化 theta/phi[/dw]，误差不可得 -> NaN

    ``fix_dw``   —— ``dw0`` 是固定值(True)还是初值(False)

    返回 ``(theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err)``;
    ``amp="profiled"`` 时四个误差均为 NaN (``fix_dw`` 时 ``dw_err`` 也是 NaN ——
    "参数被固定"不是"误差为零"，调用方要判断固定与否请看自己传的 ``fix_dw``)。
    """
    if amp == "profiled":
        wc0 = float(ADF(x, y, z, ant_p, xs, ys, zs, obs_altitude, ant_p_sigma).wc)
        if fix_dw:
            def _f(theta, phi):
                return _adf_chi2_profiled(x, y, z, ant_p, xs, ys, zs, obs_altitude,
                                          ant_p_sigma, theta, phi, dw0, wc0)[0]
            m = iminuit.Minuit(_f, theta = theta0, phi = phi0)
            m.limits = ([theta0 - 0.8, theta0 + 0.8], [phi0 - 0.8, phi0 + 0.8])
        else:
            def _f(theta, phi, dw):
                return _adf_chi2_profiled(x, y, z, ant_p, xs, ys, zs, obs_altitude,
                                          ant_p_sigma, theta, phi, dw, wc0)[0]
            m = iminuit.Minuit(_f, theta = theta0, phi = phi0, dw = dw0)
            m.limits = ([theta0 - 0.8, theta0 + 0.8], [phi0 - 0.8, phi0 + 0.8], [0.01, 5])
        m.errordef = 1
        m.tol = 1e-4
        m.migrad()
        theta, phi = float(m.values["theta"]), float(m.values["phi"])
        dw = dw0 if fix_dw else float(m.values["dw"])
        fval, a_best = _adf_chi2_profiled(x, y, z, ant_p, xs, ys, zs, obs_altitude,
                                          ant_p_sigma, theta, phi, dw, wc0)
        return (theta, phi, dw, a_best, fval,
                np.nan, np.nan, np.nan, np.nan)

    # ---- amp == "free" (旧 ADF_fit / ADF_fit_fixdw 的行为, 逐行等价) ----
    if A0 is None:
        raise ValueError("amp='free' 需要 A0 (幅度初值)")
    m = iminuit.Minuit(
        ADF(x, y, z, ant_p, xs, ys, zs, obs_altitude, ant_p_sigma),
        theta = theta0, phi = phi0, dw = dw0, A = A0, B = 0.01, rc = 1,
    )
    m.limits = (
        [theta0 - 0.8, theta0 + 0.8],
        [phi0 - 0.8, phi0 + 0.8],
        [0.01, 5],
        [A0 / 2, A0 * 1e3], [0, 0.01], [0, 1],
    )
    m.fixed = (False, False, bool(fix_dw), False, True, True)   # B / rc 恒固定
    m.errordef = 1
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    theta, theta_err = m.values["theta"], m.errors["theta"]
    phi, phi_err = m.values["phi"], m.errors["phi"]
    dw, dw_err = m.values["dw"], (np.nan if fix_dw else m.errors["dw"])
    A, A_err = m.values["A"], m.errors["A"]
    return theta, phi, dw, A, m.fval, theta_err, phi_err, dw_err, A_err
