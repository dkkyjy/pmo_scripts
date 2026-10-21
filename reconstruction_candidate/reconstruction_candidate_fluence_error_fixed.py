import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.fftpack import ifft, fft # 引入快速傅里叶变换和逆变换函数
from scipy.fftpack import fftfreq
from scipy.interpolate import interp1d
from scipy.signal import hilbert, windows # 引入希尔伯特变换和窗函数
from scipy import integrate # 引入数值积分函数
import h5py # 引入h5py库用于读取.mat文件
import iminuit
from multiprocessing.pool import Pool
import io
import glob
import contextlib
import scipy.io
import uproot
import scienceplots

plt.rcParams.update({"text.usetex": False})
plt.style.use(['science', 'grid', 'no-latex'])
# ── 快速模式: RECON_SKIP_PLOTS=1 时把所有 savefig 置空 (Agg 后端延迟渲染,
#    PNG 光栅化 + 编码 + ceph 写盘是单事件主要耗时来源; 统计只需 summary/
#    shower_output.txt, 不需要诊断图)。未设该环境变量时行为不变。──
if os.environ.get('RECON_SKIP_PLOTS'):
    plt.savefig = lambda *a, **k: None



from scipy.stats import chi2, norm

def sigma(dchi2,freedom):
    s=norm.isf(chi2.sf(dchi2,freedom)/2)
    return s

# 这部分函数为官方可下载函数

#--------------------------------------------------------------------------

# 这部分为自定义函数, 见文件夹python代码文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#--------------------------------------------------------------------------
# 路径解析层: 所有外部资源都相对本文件定位, 不再依赖任何机器相关的绝对路径,
# 以增强可移植性 (可在任意机器/集群直接运行)。
#   - parameter_files/ : EFL / PG_ALL / Z_ant / 30_250galactic 等参数与数据文件
#   - 每个资源常量都可用同名环境变量覆盖, 便于跨站点部署:
#       RECON_EFL_MAT            天线有效长度 (EFL) 数据 (.mat)
#       RECON_DU_POS_TABLE       生产 DU 位置表 (GP65, teventadc 输入用)
#       RECON_DU_ROTATION_FILE   DU 安装转角表
#       RECON_COREAS_DET_POS     CoREAS du_coords.txt (tadc 输入用)
#--------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAM_DIR = os.path.join(_HERE, 'parameter_files')
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

def _resolve_path(env_var, default):
    """返回环境变量 ``env_var`` 指定的路径, 未设置时回退到 ``default``。"""
    return os.environ.get(env_var, default)

def _resolve_data_file(env_var, filename):
    """在 parameter_files/ 下解析数据文件, 可用环境变量 ``env_var`` 覆盖。"""
    return _resolve_path(env_var, os.path.join(_PARAM_DIR, filename))

# 生产 (GP65) DU 位置表: 历史上位于仓库根目录的 _gp65_rtksort.txt。
DU_POS_TABLE = _resolve_path('RECON_DU_POS_TABLE', os.path.join(_REPO_ROOT, '_gp65_rtksort.txt'))

# DU 安装转角表 (RTK 实测)。
DU_ROTATION_FILE = _resolve_data_file('RECON_DU_ROTATION_FILE', '_rtk_fred_rotation_order.txt')

#--------------------------------------------------------------------------

from logger_config import logger  # 统一日志输出 (loguru), 替代 print
from equivalent import CEL # 引入计算天线有效长度的函数
from galacticnoise_get import gala, gala_new # 引入获取银河噪声的函数
from Time_domain_Shower_Edata_get import time_data_get # 引入获取时域簇射电场数据的函数
# from vad2voc import vad2voc # 引入从Vad计算Voc的核心函数
from vad2voc_fixed import vad2voc # 引入从Vad计算Voc的核心函数 (新版, 支持直接传真实频点)
from interpolation_ import inter
from complex_expansion import expan
import AiresInfoFunctions as aires
import coordinatesystems, helper
import models as atm
from _optimized_read_matching_times_graph import optimized_read_matching_times_graph

#--------------------------------------------------------------------------
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
        raise KeyError(f"neither teventadc nor tadc TTree found in file: {file_name}")

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

#--------------------------------------------------------------------------

# 定义一些全局参数和常量

c = 299792458 # 光速 (m/s)

tukey_wind = windows.tukey(200) # 创建一个长度为200的Tukey窗函数
time_wind = 100 # 定义时间窗口大小

flower, fupper = 30, 201 # 定义频率范围的下界和上界, MHz

thetaList, phiList = np.linspace(90, 180, 6), np.linspace(0, 360, 6) # 定义初始角度列表, 用于波前拟合的初始猜测

xList, yList, zList = np.linspace(-3e5, 3e5, 4), np.linspace(-3e5, 3e5, 4), np.linspace(1e3, 2e4, 4) # 定义网格搜索的 Xmax 位置列表, 单位为m

N, f0, f1 = 2000, 1.0, np.linspace(0, 1000, 1001) # 定义频率点数和频率范围, MHz

# magnetic_field_vector = helper.spherical_to_cartesian(np.deg2rad(60.79) + np.pi / 2, 0) # GRAND 站点的地磁场矢量, 单位矢量, 笛卡尔坐标系下
magnetic_field_vector = helper.spherical_to_cartesian(np.pi / 2 - np.deg2rad(60.79), 180 * np.pi / 180) # GRAND 站点的地磁场矢量, 单位矢量, 笛卡尔坐标系下
logger.info(f"Magnetic field vector (cartesian): {magnetic_field_vector}")
# ? ? ? ? ? 地磁场矢量方位角 0.36 deg ?

#--------------------------------------------------------------------------

efield_rec_dir = './Reconstruction/' # 结果保存目录
# efield_rec_dir = './Reconstruction_test/' # 结果保存目录, 用于测试筛选事件
# efield_rec_dir = './Reconstruction_colleague/'
if not os.path.exists(efield_rec_dir): os.makedirs(efield_rec_dir)

#--------------------------------------------------------------------------

def gps_to_lst(gps_times):
    GPS_UTC_OFFSET = 18
    LON_DEG = 93.97111533
    gps_times = np.array(gps_times)
    valid_mask = np.isfinite(gps_times)
    if not np.any(valid_mask): return None
    valid_gps = gps_times[valid_mask]
    unix_times = valid_gps - GPS_UTC_OFFSET
    jd = 2440587.5 + unix_times / 86400.0
    d = jd - 2451545.0
    T = d / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T**2 - T**3 / 38710000.0
    gmst_hr = (gmst_deg / 15.0) % 24
    lon_hr = LON_DEG / 15.0
    lst_hr = (gmst_hr + lon_hr) % 24
    return lst_hr

#--------------------------------------------------------------------------

def rotation_m(theta, phi):
    # 定义旋转矩阵，用于坐标系转换
    # 输入: theta (天顶角), phi (方位角)
    # 输出: 3x3 旋转矩阵
    return np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
                     [np.cos(theta) * np.sin(phi),  np.cos(phi), np.sin(theta) * np.sin(phi)],
                     [             -np.sin(theta),            0,               np.cos(theta)]])

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def e_recons(Voc, Lce, Vnoise):
    # 电场重建函数, 最小二乘法 (3分量), 频域输入: Voc (开路电压), Lce (天线有效长度), Vnoise (噪声电压)
    Et_rec, Ep_rec = np.array([]), np.array([]) # 初始化重建的theta, phi分量电场数组
    invM_Ett, invM_Epp, invM_Etp = np.array([]), np.array([]), np.array([])
    
    for ff in range(fupper - flower): # 遍历频率范围
        freq = ff + flower # 当前频率
        V1, V2, V3 = Voc[freq, :] # 获取当前频率下三个端口的电压
        
        A = np.array([[Lce[freq, 0, 0], Lce[freq, 1, 0]],
                      [Lce[freq, 0, 1], Lce[freq, 1, 1]],
                      [Lce[freq, 0, 2], Lce[freq, 1, 2]]]) # 构建天线响应矩阵 A (3x2矩阵，假设电场只有theta和phi分量)
        
        sig_V = np.array([[abs(Vnoise[freq, 0]) ** 2, 0, 0], 
                          [0, abs(Vnoise[freq, 1]) ** 2, 0],
                          [0, 0, abs(Vnoise[freq, 2]) ** 2]]) # 构建噪声协方差矩阵 sig_V (对角阵), 使用噪声幅度平方作为方差
        
        #AT = A.T # A的转置
        AT = A.conj().T # A的共轭转置, 因为A是复数矩阵 ; 这样做会更好 ? ? ?
        
        #sig_V_inv = np.linalg.inv(sig_V) # 噪声矩阵求逆
        sig_V_inv = np.linalg.solve(sig_V, np.eye(sig_V.shape[0], dtype = sig_V.dtype)) # 用 solve 等价计算 sig_V_inv，避免显式求逆放大数值误差

        ATsig = np.dot(AT, sig_V_inv) # AT * sig_V_inv
        ATVA = np.dot(ATsig, A) # AT * sig_V_inv * A (信息矩阵)
        
        #ATVA_inv = np.linalg.inv(ATVA) # 信息矩阵求逆 (协方差矩阵)
        ATVA_inv = np.linalg.solve(ATVA, np.eye(ATVA.shape[0], dtype = ATVA.dtype)) # 用 solve 等价计算 ATVA_inv，避免显式求逆带来的病态放大
    
        invAT = np.dot(ATVA_inv, AT) # (AT * sig_V_inv * A)^-1 * AT
        A_ = np.dot(invAT, sig_V_inv) # 最终的重建矩阵 (最小二乘解)
        V = np.array([V1, V2, V3]) # 电压向量
        et, ep = np.dot(A_, V) # 重建电场 (theta, phi)
        
        Et_rec, Ep_rec = np.append(Et_rec, et), np.append(Ep_rec, ep) # 保存结果
        invM_Ett, invM_Epp = np.append(invM_Ett, ATVA_inv[0, 0]), np.append(invM_Epp, ATVA_inv[1, 1]) # 保存不确定度信息
        invM_Etp = np.append(invM_Etp, ATVA_inv[0, 1]) # 保存协方差信息 (可选)

    E_inv = np.array([invM_Ett, invM_Epp, invM_Etp]) # 组合不确定度
    
    return (Et_rec), (Ep_rec), (E_inv) # 返回重建电场和不确定度


def e_recons_real_MHz(Voc, Lce, Vnoise):
    """真实频点版: Voc / Lce / Vnoise 第一维为同一频点数组 (30-250MHz 内的真实 rfft 频点)."""
    nfreq = Voc.shape[0]
    Et_rec, Ep_rec = np.zeros(nfreq, dtype = complex), np.zeros(nfreq, dtype = complex)
    invM_Ett = np.zeros(nfreq); invM_Epp = np.zeros(nfreq); invM_Etp = np.zeros(nfreq)

    for k in range(nfreq):
        V1, V2, V3 = Voc[k, :]
        A = np.array([[Lce[k, 0, 0], Lce[k, 1, 0]],
                      [Lce[k, 0, 1], Lce[k, 1, 1]],
                      [Lce[k, 0, 2], Lce[k, 1, 2]]])          # (3,2)
        sig_V = np.diag(np.abs(Vnoise[k, :]) ** 2)            # (3,3) 对角
        # sig_V_inv = np.linalg.solve(sig_V, np.eye(3))
        sig_V_inv = np.linalg.solve(sig_V, np.eye(sig_V.shape[0], dtype = sig_V.dtype)) # 用 solve 等价计算 sig_V_inv，避免显式求逆放大数值误差
        AT = A.conj().T
        ATVA = AT @ sig_V_inv @ A
        # ATVA_inv = np.linalg.solve(ATVA, np.eye(2))
        ATVA_inv = np.linalg.solve(ATVA, np.eye(ATVA.shape[0], dtype = ATVA.dtype)) # 用 solve 等价计算 ATVA_inv，避免显式求逆带来的病态放大
        A_ = ATVA_inv @ AT @ sig_V_inv
        et, ep = A_ @ np.array([V1, V2, V3])
        Et_rec[k], Ep_rec[k] = et, ep
        invM_Ett[k], invM_Epp[k], invM_Etp[k] = ATVA_inv[0, 0], ATVA_inv[1, 1], ATVA_inv[0, 1]

    return Et_rec, Ep_rec, np.array([invM_Ett, invM_Epp, invM_Etp])

#--------------------------------------------------------------------------

# GRAND 天线有效长度 (EFL) 数据文件, 默认取 parameter_files/ 下的副本, 可用
# 环境变量 RECON_EFL_MAT 覆盖。
REfile_grand = _resolve_data_file('RECON_EFL_MAT', 'EFL_data_3.2m_XYZ_20250313.mat')
if not os.path.exists(REfile_grand):
    raise FileNotFoundError(
        f"未找到 GRAND 天线有效长度数据文件: {REfile_grand}\n"
        f"请将其放入 {_PARAM_DIR}/ , 或用环境变量 RECON_EFL_MAT 指向实际路径。"
    )

mat_data = scipy.io.loadmat(REfile_grand)
EFL_all_grand = mat_data['EFL_all']

def get_lec_grand(e_theta, e_phi, N, f0):
    """
    获取 GRAND 天线的有效长度 (Effective Length)。
    
    该函数从加载的 GRAND 天线响应数据中插值获取特定方向的有效长度，并将其转换到球面坐标系。
    
    参数:
        e_theta (float): 天顶角 (度)。
        e_phi (float): 方位角 (度)。
        N (int): 频率点数。
        f0 (float): 频率步长。
    
    返回:
        numpy.ndarray: 球面坐标系下的有效长度矩阵 (N x 3 x 3)。
    """

    # 插值获取辐射方向图数据
    e_radiation = inter(EFL_all_grand, e_theta, e_phi)
    Lce = np.zeros((N, 3, 3), dtype = complex)
    
    # 展开复数系数
    for i in range(3):
        for p in range(3):
            f, Lce[:, i, p] = expan(N, f0, 30, 250, e_radiation[:, p, i])
            
    # 坐标转换：笛卡尔 -> 球面
    R_sph2cart = rotation_m(e_theta * np.pi / 180, e_phi * np.pi / 180)
    R_cart2sph = np.linalg.inv(np.matrix(R_sph2cart))
    Lce_sphere_ = np.zeros((N, 3, 3), dtype = complex)
    for i in range(2000):
        for j in range(3):
            Lce_sphere_[i,:,j] = np.dot(R_cart2sph,Lce[i,:,j])
    
    return Lce_sphere_

#--------------------------------------------------------------------------

def _interp_complex_1d(f_src, y_src, f_tar, kind = "cubic"):
    """
    复数一维插值 (实部/虚部分别插值, 与 vad2voc 风格一致).
    范围外的频点置 0 (与 get_lec_grand 中 expan 在 30-250 MHz 之外为 0 的行为一致).
    """
    f_src = np.asarray(f_src, dtype = float)
    f_tar = np.asarray(f_tar, dtype = float)
    y_src = np.asarray(y_src).reshape(len(f_src), -1)      # (Nsrc, 1)

    out = np.empty((f_tar.size, y_src.shape[1]), dtype = np.complex128)
    for k in range(y_src.shape[1]):
        re_f = interp1d(f_src, y_src[:, k].real, kind = kind, bounds_error = False, fill_value = 0.0)
        im_f = interp1d(f_src, y_src[:, k].imag, kind = kind, bounds_error = False, fill_value = 0.0)
        out[:, k] = re_f(f_tar) + 1j * im_f(f_tar)

    return out[:, 0]   # 返回 1D


# 真实频点版有效长度: 可指定天顶角/方位角/任意频点数组
def get_lec_grand_real_MHz(e_theta, e_phi, freq_target_MHz):
    """
    获取 GRAND 天线在指定真实频点上的有效长度 (球面坐标系).

    参数
    ----
    e_theta : float, 天顶角 (度), 与 get_lec_grand 完全一致.
    e_phi : float, 方位角 (度), 与 get_lec_grand 完全一致.
    freq_target_MHz : array_like
        目标频点数组 (MHz), 可为真实 FFT 频点 (如 0.488 MHz 间隔).
        仅 30-250 MHz 内的频点有效, 范围外置 0.

    返回
    ----
    Lce_sphere_real : (Nfreq_target, 3, 3) complex
        索引 [频点, 球面分量, 端口]:
          分量 0 = theta, 1 = phi, 2 = r;
          端口 0/1/2 = X/Y/Z.
        与 get_lec_grand 返回语义完全一致 (可直接给 e_recons 用 [:, 0, p] 与 [:, 1, p]),
        只是频率轴由 1 MHz 网格换成你指定的 freq_target_MHz.
    """
    freq_target_MHz = np.asarray(freq_target_MHz, dtype = float)

    # 1) 取该方向的辐射响应 (与原函数相同)
    e_radiation = inter(EFL_all_grand, e_theta, e_phi)     # (221, 3 ports, 3 comps)

    nfreq = freq_target_MHz.size
    Lce_cart_real = np.zeros((nfreq, 3, 3), dtype = complex)   # [freq, comp, port]

    # 2) 把 1 MHz 网格频响插值到目标真实频点
    for i in range(3):        # 笛卡尔分量
        for p in range(3):    # 端口
            Lce_cart_real[:, i, p] = _interp_complex_1d(
                np.arange(30, 251, 1), e_radiation[:, p, i], freq_target_MHz, kind = "cubic"
            )

    # 3) 笛卡尔 -> 球面 (与原函数相同, 修掉了硬编码 2000 的隐患)
    R_sph2cart = rotation_m(e_theta * np.pi / 180.0, e_phi * np.pi / 180.0)
    R_cart2sph = np.asarray(np.linalg.inv(np.matrix(R_sph2cart)))

    Lce_sphere_real = np.zeros((nfreq, 3, 3), dtype = complex)
    for i in range(nfreq):
        for j in range(3):    # 端口
            Lce_sphere_real[i, :, j] = R_cart2sph @ Lce_cart_real[i, :, j]

    return Lce_sphere_real

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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
        else: # tadc (CoREAS / Dunhuang simulation)
            data = tree.arrays(["event_number", "du_id", "du_nanoseconds", "trace_ch"], library = "ak")
            data = data[int(event_number_int)]
            event_num, du_ids_all, du_nanoseconds = data["event_number"], np.array(data["du_id"]), np.array(data["du_nanoseconds"])
            trace_ch = np.array(data["trace_ch"].to_list()) # (n_du, 3, n_time)
            trace_1, trace_2, trace_3 = trace_ch[:, 0], trace_ch[:, 1], trace_ch[:, 2]
            gps_time = _synthesize_gps_times(root_path, int(event_num), len(du_ids_all))
    finally:
        root_file.close()

    LST_time = int(np.round(gps_to_lst(gps_time[2])[0]))

    logger.info(f"Trigger Time @ LST Time (hours) {LST_time}")

    return event_num, du_ids_all, gps_time, du_nanoseconds, LST_time, trace_1, trace_2, trace_3

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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
    ``tadc`` (CoREAS) 使用 :func:`_resolve_coreas_det_pos` 选定的独立坐标表。
    返回 ``(tree_kind, position_path)``; 具体坐标由 :func:`read_xyz_coordinates`
    读取, 以保证与生产路径完全一致。
    """
    probe_file = uproot.open(root_path)
    try:
        tree_kind = find_event_tree_key(list(probe_file.keys()), root_path)[1]
    finally:
        probe_file.close()
    if tree_kind == 'tadc':
        path = _resolve_coreas_det_pos(root_path, det_pos_path)
    else:
        path = det_pos_path if det_pos_path is not None else DU_POS_TABLE
    return tree_kind, path

#--------------------------------------------------------------------------

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

# 注: "全部 DU 的旋转坐标"(仅用于绘图背景点) 不再作为模块级全局状态, 而是由
# efield_recons_from_efield_PWF / energy_restruction 在选定正确坐标表后按需局部加载,
# 以避免模块导入副作用并支持不同探测器布局。

#--------------------------------------------------------------------------

def causality_filter(efield_rec_dir, event_num, du_ids_all, gps_time, du_nanoseconds, trace_1, trace_2, trace_3, x_position, y_position, z_position, DU_positions_rotated):

    detector_positions_for_causal = {} # 构造 detector_positions 字典以满足函数的输入要求 (使用你当前已经提取过的坐标)
    
    for j, du_ in enumerate(du_ids_all): detector_positions_for_causal[str(du_)] = np.array([x_position[j], y_position[j], z_position[j]]) # 原函数内部还会自己调用一次 rotate_and_shift, 为了不发生两次旋转，我们直接传原始的未经旋转偏置的硬坐标
    

    temp_causal_file = os.path.join(efield_rec_dir, f'temp_causality_event_{event_num}.txt') # 生成一个满足 parser 语法的纯文本行并写入临时文件
    
    
    match_str_list = [f"('{du_ids_all[j]}', {int(du_nanoseconds[j])})" for j in range(len(du_ids_all))] # 将 DU_id 和 nanosecond 时间戳组装成 '[('56', 15632890), ('103', 15639100), ...]' 
    
    match_str = ", ".join(match_str_list)
    
    line_to_write = f"{gps_time[2]}: [{match_str}]"  # gps_time[2] 用作标识
    
    with open(temp_causal_file, 'w') as tf: tf.write(line_to_write + '\n')

    
    logger.info("Executing strict causality cleaning via graph optimization...") # 调用引用的图论清理函数（设定最少容忍 4 个探测器）
    
    matching_times = optimized_read_matching_times_graph(
        file_path = temp_causal_file, 
        detector_positions = detector_positions_for_causal, 
        min_detectors = 4)

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

#--------------------------------------------------------------------------

def ADC_trace_to_Voc_time(trace_1, trace_2, trace_3):

    '''
    # 给定 3 个通道的 Vad 时域数据, 计算对应的 Voc 时域数据
    # 输入: trace_1, trace_2, trace_3: 3 个通道的时域数据, 形状为 (N_time, 3)
    # 输出: Voc_time: 对应的 Voc 时域数据, 形状为 (N_time, 3)
    '''

    N_time = len(trace_1) # 动态获取长度, 支持 1024 或 512.
        
    ADC_single_du_time = np.array([trace_1, trace_2, trace_3]).T # 形状 (N_time, 3)，每列对应一个通道的时域数据

    #-----------------------------------------------------------------------
    
    ADC_single_du_rfft = np.fft.rfft(ADC_single_du_time, axis = 0) # 对每个通道进行实数快速傅里叶变换，得到频域数据，形状为 (N_time/2 + 1, 3)
    
    freqs_real_MHz = np.fft.rfftfreq(N_time, d = 2e-9) / 1e6 # 计算对应的频率点, 单位 MHz, 长度为 N_time/2 + 1

    ADC_single_du_rfft[freqs_real_MHz > 120, :] = 0 # 频域滤波: 将 120 MHz 以上的频点置零, 根据实际情况调整截止频率

    ADC_single_du_time_filtered = np.fft.irfft(ADC_single_du_rfft, n = N_time, axis = 0) # 对滤波后的频域数据进行逆变换，得到滤波后的时域数据，形状为 (N_time, 3)

    #-----------------------------------------------------------------------
        
    band_mask = (freqs_real_MHz >= 30) & (freqs_real_MHz <= 250) # 频带掩码, 选取有效频率范围内的频点
        
    freqs_band = freqs_real_MHz[band_mask] # 选取有效频率范围内的频点, 长度根据实际数据而定, 满足 vad2voc 的输入要求

    freqs_band_hz = freqs_band * 1e6 # 26-08-26 新加代码

    ADC_band = ADC_single_du_rfft[band_mask, :] # 获取高精度的原始信号有效频带 (完全不经过任何降采样插值)

    Voc_rfft_band = vad2voc(ADC_band, target_freq_hz = freqs_band_hz) # 直接在真实频点上一步计算 Voc

    #-----------------------------------------------------------------------
    
    window_real = np.pad(windows.tukey(len(freqs_band) - 8, alpha = 0.5), (4, 4), mode = 'constant')
    
    Voc_rfft_band *= window_real[:, None]

    #-----------------------------------------------------------------------

    Voc_full_rfft = np.zeros_like(ADC_single_du_rfft, dtype = complex) 
    # 必须创建一个完整的频点数组, 包括直流分量, 再把对应频点数据放上去, 才可以逆傅立叶变换
    
    Voc_full_rfft[band_mask, :] = Voc_rfft_band # 拼装回完整的 RFFT 数组中进行 IRFFT
    
    Voc_time = np.fft.irfft(Voc_full_rfft, n = N_time, axis = 0) * (109.86) # units of uV.

    #-----------------------------------------------------------------------

    return Voc_time, ADC_single_du_time_filtered, N_time, freqs_real_MHz, freqs_band

#--------------------------------------------------------------------------

def Voc_fft_real_MHz(Voc_time, freqs_real_MHz, ADC_single_du_time_filtered, time_wind):

    """
    # 给定开路电压 Voc_time, 在峰值位置加窗函数, 傅立叶变换, 只取位于 30--250 MHz频段内的数据, 输出用于电场重建.
    """

    vx_std, vy_std, vz_std = np.std(Voc_time[int(len(Voc_time) * 0.75) : , :], axis = 0) # 计算噪声标准差
    stds_V = np.array([vx_std, vy_std, vz_std])

    # 2026-09-07 新加代码
    peak_pos = np.argmax(abs(Voc_time[0 : 300, :]), axis = 0) # 获取每个通道的峰值位置索引, 以便后续加窗使用

    peaks_V = np.max(abs(Voc_time[0 : 300, :]), axis = 0) # 获取每个通道的峰值电压, 用于信号筛选和加窗判断

    #----------------------------------------------------------------------

    Voc_time_windows = np.zeros(Voc_time.shape) # 初始化加窗后的时域 Voc 数组, 形状与原始 Voc_time 相同

    for cmp in range(3): 
            
        best_ch = np.argmax(peaks_V) # 确定峰值的中心基准点.
        
        if peaks_V[best_ch] / stds_V[best_ch] > 5:
            
            # if peaks_V[cmp] / stds_V[cmp] > 5: center_idx = peak_pos[cmp]
            
            # else: 
            
            center_idx = peak_pos[best_ch] # 2026-09-07 修正代码
        
        else: # center_idx = np.argmax(np.abs(ADC_single_du_time_filtered))

            # 2026-09-07 修正代码
            center_idx = np.argmax(np.max(np.abs(ADC_single_du_time_filtered[0 : 300, :]), axis = 1)) - 25
        
        
        left_ideal, right_ideal = center_idx - time_wind, center_idx + time_wind # 计算理想窗口游标
        
        left_actual, right_actual = max(0, left_ideal), min(len(Voc_time), right_ideal) # 计算实际能切出的合法边界 (限制在 0 到 数组最大长度 之间)
        
        
        crop_start, crop_end = left_actual - left_ideal, 2 * time_wind - (right_ideal - right_actual) # 判断合法边界到理想边界失去了多少点，从而裁剪对应的 Tukey 窗并赋值

        if right_actual > left_actual: # 确保存在有效窗口切片
            
            Voc_time_windows[left_actual : right_actual, cmp] = Voc_time[left_actual : right_actual, cmp] * tukey_wind[crop_start : crop_end]

    Voc_fft_windows = np.fft.rfft(Voc_time_windows, axis = 0) # 将加窗后的时域信号重新变回频域, 得到无瑕的 rfft 形式，提供极干净的边界供物理方程匹配

    # Voc_fft_windows = np.fft.rfft(Voc_time, axis = 0)

    #----------------------------------------------------------------------------------

    band_mask = (freqs_real_MHz >= 30) & (freqs_real_MHz <= 250)

    Voc_fft_band = Voc_fft_windows[band_mask, :]

    return Voc_fft_band

#--------------------------------------------------------------------------

# DU 转角: RTK 实测角 -> 重建代码方位角约定 -> 把 X/Y 端口旋到标称系

def read_du_rotation_angles(file_path, usecols = (1, 3, 4), angle_overrides = None):
    """
    读取每个 DU 的安装转角, 返回 {du_id: phi}。

    phi 的约定 = **重建代码的方位角约定**: 正北 = 0 度, 逆时针为正。

    文件 (_rtk_fred_rotation_order.txt) 列:
        第 1 列 = DU ID, 第 3 列 = X 通道相对正北的夹角 α (度), 第 4 列 = Y 通道相对正北的夹角 β (度);
    文件里的角度是 **正北 = 0 度, 顺时针为正**。

    推导:
        X 标称指向正北 -> "X 与正北的夹角" = α
        Y 标称指向正东 -> "Y 与正东的夹角" = β - 90
        两者平均 = 该 DU 的整体安装转角 (顺时针为正):  phi_rtk = (α + (β - 90)) / 2
        代码方位角是逆时针为正, 与顺时针差一个符号:      phi = -phi_rtk
    """
    data = np.loadtxt(file_path, usecols = usecols)

    rot_dict = {}
    for du_id, ang_x, ang_y in data:
        phi_rtk = 0.5 * (ang_x + (ang_y - 90.0))          # 顺时针为正的安装转角
        phi     = -phi_rtk                                # 换成代码约定 (逆时针为正)
        if angle_overrides is not None:
            phi += angle_overrides.get(int(du_id), 0.0)   # 个别 DU 的安装特例 (同样按逆时针为正)
        rot_dict[int(du_id)] = phi
    return rot_dict

def rotate_xy_ports(arr, angle_deg, axis = -1):
    """
    把 X(0)/Y(1) 两个端口的极化基旋到标称参考系 (X->正北, Y->正东); Z 端口不动。

    angle_deg : 该 DU 的安装转角, **必须是重建代码的方位角约定 (正北=0, 逆时针为正)**,
                即 read_du_rotation_angles 返回的值。

    变换 (实际端口电压 -> 标称端口电压):
        V_X_nominal =  cos(phi) * V_X_physical + sin(phi) * V_Y_physical
        V_Y_nominal = -sin(phi) * V_X_physical + cos(phi) * V_Y_physical

    适用于 (Nfreq, 3) 的 Voc, 也可用于 (Nfreq, 3, 3) 的有效长度 (端口都在 axis 轴, 默认最后一维)。
    """
    phi = np.deg2rad(angle_deg)
    c, s = np.cos(phi), np.sin(phi)

    out  = np.array(arr, copy = True)
    sl_x = [slice(None)] * out.ndim ; sl_x[axis] = 0   # X 端口
    sl_y = [slice(None)] * out.ndim ; sl_y[axis] = 1   # Y 端口

    X = out[tuple(sl_x)].copy()
    Y = out[tuple(sl_y)].copy()
    out[tuple(sl_x)] =  c * X + s * Y     # 标称 X (正北)
    out[tuple(sl_y)] = -s * X + c * Y     # 标称 Y (正东)
    return out

# 每个 DU 的转角表 (已换算成代码的方位角约定: 逆时针为正)。
# 文件缺失时不修正转角 (等价于所有 DU 转角为 0), 保证链路仍可运行。
if os.path.exists(DU_ROTATION_FILE):
    DU_ROTATION_ANGLES = read_du_rotation_angles(DU_ROTATION_FILE)
else:
    logger.warning(f"[WARN] 未找到 DU 转角表 {DU_ROTATION_FILE}, 跳过端口转角修正。"
          f"如需启用, 请将其放入 {_PARAM_DIR}/ 或用环境变量 RECON_DU_ROTATION_FILE 指定。")
    DU_ROTATION_ANGLES = {}

#--------------------------------------------------------------------------

def calculate_energy_fluence_and_error(Et_rec, Ep_rec, E_inv, freqs_real_MHz, N_time):

    E_rec_complex = np.zeros((len(freqs_real_MHz), 2), dtype = complex)

    band_mask = (freqs_real_MHz >= 30) & (freqs_real_MHz <= 250)

    E_rec_complex[band_mask, 0], E_rec_complex[band_mask, 1] = Et_rec, Ep_rec

    E_rec_time = np.fft.irfft(E_rec_complex, n = N_time, axis = 0)

    #----------------------------------------------------------------------

    E_rec_amp = np.sqrt(E_rec_time[:, 0] ** 2 + E_rec_time[:, 1] ** 2) # 计算重建电场的振幅，供后续物理方程正演使用

    # Fluence = np.sum(E_rec_amp[np.argmax(E_rec_amp) -50 : np.argmax(E_rec_amp) + 50] ** 2) # 计算重建电场的 fluence, 供后续物理方程正演使用
    Fluence, Fluence_theta, Fluence_phi = np.sum(E_rec_amp ** 2), np.sum(E_rec_time[:, 0] ** 2), np.sum(E_rec_time[:, 1] ** 2) 
    # 计算重建电场的 fluence，供后续物理方程正演使用, 同时分离 theta 和 phi 分量的 fluence 以供分析
    Fluence_theta_phi = np.sum(E_rec_time[:, 0] * E_rec_time[:, 1]) # 计算 theta 和 phi 分量的 fluence 交叉项, 用于分析两者之间的相关性

    transform_factor = 6.242e18 * 1e-12 * 2e-9 / 377 # 根据实际情况调整能量因子, 以使得 fluence 的单位为 eV/m^2

    Fluence *= transform_factor ; Fluence_theta *= transform_factor ; Fluence_phi *= transform_factor ; Fluence_theta_phi *= transform_factor

    #----------------------------------------------------------------------

    invM_Ett, invM_Epp, invM_Etp = np.real(E_inv[0]), np.real(E_inv[1]), E_inv[2]

    var_freq_sum = np.sum((invM_Ett ** 2 + 2 * invM_Ett * np.abs(Et_rec) ** 2) + (invM_Epp ** 2 + 2 * invM_Epp * np.abs(Ep_rec) ** 2))
    var_freq_sum += np.sum(2 * np.abs(invM_Etp) ** 2 + 4 * np.real(invM_Etp * np.conj(Et_rec) * Ep_rec)) # 加上 Et 和 Ep 的协方差项, 以考虑两者之间的相关性

    parseval_factor = 2.0 / N_time  # N = N_time 时域点数, 频段内均非DC/Nyquist频点
    Fluence_sigma = np.sqrt(var_freq_sum) * parseval_factor * transform_factor
    
    Fluence_frequency = parseval_factor * np.sum(np.abs(Et_rec) ** 2 + np.abs(Ep_rec) ** 2) * transform_factor

    logger.info(f"Reconstructed E-field Fluence frequency = {Fluence_frequency:.3e} eV/m^2, time {Fluence:.3e} eV/m^2, relative error = {(Fluence_frequency / Fluence - 1.0):.3f}")
    logger.info(f"Reconstructed E-field Fluence = {Fluence:.3e} eV/m^2 ± {Fluence_sigma:.3e} eV/m^2")
    
    #----------------------------------------------------------------------

    return E_rec_time, Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi, Fluence_sigma

#-------------------------------------------------------------------------

def re_calculate_Voc_from_reconstruction(LST_time, N_time, freqs_band, freqs_real_MHz, Lce_sphere, Et_rec, Ep_rec):

    _, galactic_v_complex_double = gala_new((LST_time, 0), N_time)

    noise_mask = freqs_band > 120.0

    galactic_v_complex_double[noise_mask, :] = 0.0 # 将高于 120 MHz 的银河噪声频点置零, 以便后续重建 Voc 时不引入高频噪声


    Voc_recalc_complex_band = np.zeros((len(freqs_band), 3), dtype = complex) # 初始化输出电压的频域容器
    
    for k in range(len(freqs_band)):
        for p in range(3):
            Voc_recalc_complex_band[k, p] = (Lce_sphere[k, 0, p] * Et_rec[k] + Lce_sphere[k, 1, p] * Ep_rec[k] + galactic_v_complex_double[k, p]) # 直接在频域下根据物理方程重建 Voc，注意这里的 galactic noise 也是频域表示, 直接乘以对应的 Lce 分量后叠加到 Voc 中, 完美保护时间相位


    Voc_recalc_complex_real_MHz = np.zeros((len(freqs_real_MHz), 3), dtype = complex)

    band_mask = (freqs_real_MHz >= 30) & (freqs_real_MHz <= 250)

    Voc_recalc_complex_real_MHz[band_mask, :] = Voc_recalc_complex_band

    Voc_recalc_time = np.fft.irfft(Voc_recalc_complex_real_MHz, n = N_time, axis = 0) # 将重新计算的频域电压进行 IRFFT，得到时域波形，供与原始 Voc 的时域波形进行对比

    return Voc_recalc_time

#--------------------------------------------------------------------------

def efield_recons_from_efield_PWF(root_path, event_number_int, det_pos_path = None, valid_du_ids = None, theta = None, phi = None, prior_times = None): # 基于平面波前拟合 (PWF) 的电场重建
    # 可选外部先验 (来自 matched_PWM.yaml 的 PWM 结果):
    #   valid_du_ids: 给定则用 np.isin 直接筛天线, 替换内部 causality_filter 图优化筛选;
    #   prior_times : 给定则用 yaml 的到达时间 (du_id -> ns) 替换 ROOT 的 du_nanoseconds;
    #   theta / phi : yaml 的 zenith / azimuth (deg)。换算为 PWF 内部坐标作单一初值,
    #                 仍执行一次 PWF_fit (不跑网格搜索)。

    logger.info(f"Starting E-field reconstruction using PWF from {root_path} , event:  {int(event_number_int)}")

    filename = os.path.splitext(os.path.basename(root_path))[0] + f"_event_{int(event_number_int)}" # 得到文件名和事件编号

    if not os.path.exists(efield_rec_dir + filename + '/'): os.makedirs(efield_rec_dir + filename + '/')

    #---------------------------------------------------------------------------
    
    event_num, du_ids_all, gps_time, du_nanoseconds, LST_time, trace_1, trace_2, trace_3 \
    = read_root_file_from_path(root_path, event_number_int)

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    
    # 依输入 tree 类型选择 DU 坐标表: teventadc -> 生产表; tadc (CoREAS) -> du_coords.txt。
    # 旋转策略: 仅真实 trigger (teventadc) 需施加 GRAND 布局旋转; CoREAS/starshape (tadc) 不旋转。
    tree_kind, du_pos_path = _load_du_position_for_input(root_path, det_pos_path)
    x_position, y_position, z_position, DU_positions_rotated, DU_all_positions_rotated = read_xyz_coordinates(du_pos_path, du_ids_all, rotate = (tree_kind == 'teventadc'))

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    
    if valid_du_ids is not None:
        # 外部给定 DU 子集 (PWM): 直接 np.isin 筛选, 替换内部 causality_filter 图优化。
        # prior_times 非空时, 用 yaml 的到达时间 (du_id -> ns) 替换 ROOT 的 du_nanoseconds。
        du_ns_lookup = None
        if prior_times is not None:
            du_ns_lookup = {}
            for k, v in prior_times.items():
                if isinstance(v, (list, tuple)):
                    v = v[0] if len(v) else None
                if v is not None:
                    du_ns_lookup[int(k)] = float(v)
        causal_mask = np.isin(du_ids_all, np.asarray(valid_du_ids))
        triggered_du_ids        = du_ids_all[causal_mask]
        if du_ns_lookup is not None:
            missing = [int(d) for d in triggered_du_ids if int(d) not in du_ns_lookup]
            if missing:
                logger.warning(f"[PWM] yaml prior_times 缺少 {len(missing)} 个 DU 的时间, 回退到 ROOT: {missing}")
            du_nanoseconds_filtered = np.array([
                du_ns_lookup[int(d)] if int(d) in du_ns_lookup else du_nanoseconds[i]
                for i, d in enumerate(du_ids_all) if causal_mask[i]
            ])
            logger.info(f"[PWM] 使用 yaml 先验到达时间替换 ROOT du_nanoseconds ({len(triggered_du_ids)} DUs)")
        else:
            du_nanoseconds_filtered = du_nanoseconds[causal_mask]
        trace_1_filtered = trace_1[causal_mask]
        trace_2_filtered = trace_2[causal_mask]
        trace_3_filtered = trace_3[causal_mask]
        DU_positions_rotated = DU_positions_rotated[causal_mask]
        n_triggered = len(triggered_du_ids)
        logger.info(f"[PWM] 使用外部 valid_du_ids 筛选天线: {n_triggered} DUs retained: {triggered_du_ids}")
        if n_triggered < 4:
            logger.warning(f"Event {event_num} has only {n_triggered} DUs after valid_du_ids filter (< 4 minimum). Skipping.")
            return -1
    else:
        causal_out = causality_filter(efield_rec_dir, event_num, du_ids_all, gps_time, du_nanoseconds, trace_1, trace_2, trace_3, \
                                      x_position, y_position, z_position, DU_positions_rotated)

        if causal_out is None: return -1

        triggered_du_ids, du_nanoseconds_filtered, trace_1_filtered, trace_2_filtered, trace_3_filtered, DU_positions_rotated, n_triggered = causal_out

    #---------------------------------------------------------------------------

    if theta is not None and phi is not None:
        # yaml 的 zenith / azimuth 换算为 PWF 内部坐标 (θ, φ) 作单一初值,
        # 仍执行一次 PWF_fit (不再跑 6x6 网格搜索)。
        # 换算 (对固定版输出侧 rec_theta=180-θ_fit, rec_phi=φ_fit+180 的逆):
        #   θ_fit0 = 180 - zenith ,  φ_fit0 = azimuth - 180
        # 注意 φ 必须减 180: 漏掉时 (曾为 `phi % 360`) PWF 会被引向 φ_fit≈0 的退化解,
        # 实测 60 例中 33 例χ² 比网格搜索劣化 1e4~1e5 倍; 补上后 58/60 与网格最优一致。
        theta0 = 180.0 - float(theta)
        phi0 = (float(phi) - 180.0) % 360.0
        theta_fit, phi_fit, fval, theta_fit_err, phi_fit_err = PWF_fit(DU_positions_rotated[:, 0], DU_positions_rotated[:, 1], DU_positions_rotated[:, 2], du_nanoseconds_filtered * 1e-9, theta0, phi0)
        chi2Plane = fval
        rec_theta_plane, rec_phi_plane = 180 - theta_fit, phi_fit + 180
        rec_theta_plane_err, rec_phi_plane_err = theta_fit_err, phi_fit_err
        if rec_phi_plane > 360: rec_phi_plane = rec_phi_plane - 360
        logger.info(f"[PWM] PWF 单点初值拟合 (theta0={theta0:.2f}, phi0={phi0:.2f}): theta = {rec_theta_plane:.2f} deg, phi = {rec_phi_plane:.2f} deg, chi2 = {chi2Plane:.2e}")
    else:
        rec_theta_plane, rec_phi_plane, chi2Plane = np.inf, np.inf, np.inf # 平面波前拟合 (PWF) 寻找最佳角度
        rec_theta_plane_err, rec_phi_plane_err = np.inf, np.inf

        for theta0 in thetaList:
            for phi0 in phiList:
                theta, phi, fval, theta_err, phi_err = PWF_fit(DU_positions_rotated[:, 0], DU_positions_rotated[:, 1], DU_positions_rotated[:, 2], du_nanoseconds_filtered * 1e-9, theta0, phi0)
                if fval < chi2Plane:
                    chi2Plane, rec_theta_plane, rec_phi_plane = fval, theta, phi
                    rec_theta_plane_err, rec_phi_plane_err = theta_err, phi_err

        rec_theta_plane, rec_phi_plane = 180 - rec_theta_plane, rec_phi_plane + 180
        if rec_phi_plane > 360: rec_phi_plane = rec_phi_plane - 360

        logger.info(f"Best PWF fit: theta = {rec_theta_plane:.2f} deg, phi = {rec_phi_plane:.2f} deg, chi2/dof = {chi2Plane:.2e}")

    #---------------------------------------------------------------------------

    plt.figure(figsize=(8, 6)) # 绘制触发天线位置图
    plt.scatter(-1 * DU_all_positions_rotated[:, 1], DU_all_positions_rotated[:, 0], c = 'lightgray', label = 'All DUs')
    plt.scatter(-1 * DU_positions_rotated[:, 1], DU_positions_rotated[:, 0], c = du_nanoseconds_filtered - np.median(du_nanoseconds_filtered), label = 'Triggered DUs', s = 100, cmap = 'viridis', edgecolors = 'black')
    cbar = plt.colorbar()
    cbar.set_label('Relative Time [ns]', fontsize = 16)
    cbar.ax.tick_params(labelsize = 12)
    for j_, du_ in enumerate(triggered_du_ids): plt.text(-1 * DU_positions_rotated[j_, 1], DU_positions_rotated[j_, 0], str(du_), fontsize = 10, fontweight = 'bold', ha = 'left', va = 'bottom', color = 'black')
    plt.arrow(0, 0, -1 * 500 * np.sin(np.deg2rad(rec_theta_plane)) * np.sin(np.deg2rad(rec_phi_plane)), 500 * np.sin(np.deg2rad(rec_theta_plane)) * np.cos(np.deg2rad(rec_phi_plane)), color = 'red', width = 20, head_width = 80, head_length = 120, label = 'PWM Fit')
    plt.xlabel("West-East [m]", fontsize = 16)
    plt.ylabel("South-North [m]", fontsize = 16)
    plt.title("Triggered DUs Position", fontsize = 18)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
    for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
    plt.tick_params(axis='x', pad=6)
    plt.tick_params(axis='y', pad=6)
    plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_Triggered_DUs_Position.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    saved_data = np.zeros([n_triggered, 18]) # 16 列 + 末尾 2 列 PWF 误差 (theta_err, phi_err) # 新增第16列: Fluence_sigma (物理传播误差, 见下方计算)
    E_rec_time_all_DUs = [] # 初始化一个列表来保存每个 DU 的重建电场时域数据

    for i in range(n_triggered):

        #-----------------------------------------------------------------------

        Voc_time, ADC_single_du_time_filtered, N_time, freqs_real_MHz, freqs_band \
        = ADC_trace_to_Voc_time(trace_1_filtered[i], trace_2_filtered[i], trace_3_filtered[i])
        
        #-----------------------------------------------------------------------

        plt.figure(figsize=(10, 4)) # 绘制时域ADC波形图
        plt.plot(ADC_single_du_time_filtered[:, 0], color = 'red', linewidth = 1.0, label = 'Port X')
        plt.plot(ADC_single_du_time_filtered[:, 1], color = 'green', linewidth = 1.0, label = 'Port Y')
        plt.plot(ADC_single_du_time_filtered[:, 2], color = 'blue', linewidth = 1.0, label = 'Port Z')
        plt.title(f"DU {triggered_du_ids[i]}: ADC Time Domain", fontsize = 18)
        plt.xlabel("Time index (2 ns interval)", fontsize = 16)
        plt.ylabel("ADC", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_DU_{triggered_du_ids[i]}_ADC_Traces.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #-----------------------------------------------------------------------

        plt.figure(figsize=(10, 4)) # 绘制时域 Voc 波形图
        plt.plot(Voc_time[:, 0], color = 'red', linewidth = 1.0, label = 'Port X')
        plt.plot(Voc_time[:, 1], color = 'green', linewidth = 1.0, label = 'Port Y')
        plt.plot(Voc_time[:, 2], color = 'blue', linewidth = 1.0, label = 'Port Z')
        plt.title(f"DU {triggered_du_ids[i]}: Voc Time Domain (After IRFFT)", fontsize = 18)
        plt.xlabel("Time index (2 ns interval)", fontsize = 16)
        plt.ylabel("Voltage [uV]", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Time_Domain.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #----------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------

        Voc_fft_band = Voc_fft_real_MHz(Voc_time, freqs_real_MHz, ADC_single_du_time_filtered, time_wind)

        # 转角修正: 把该 DU 的 X/Y 端口旋到标称参考系 (X->正北, Y->正东)
        du_angle = DU_ROTATION_ANGLES.get(int(triggered_du_ids[i]), 0.0)   # 无记录的 DU 不修正
        Voc_fft_band = rotate_xy_ports(Voc_fft_band, du_angle)

        #--------------------------------------------------------------------------------------------------------

        Lce_sphere = get_lec_grand_real_MHz(rec_theta_plane, rec_phi_plane, freqs_band)
        # 获取对应方向的天线有效长度, 指定频率点 freqs_band.

        _, galactic_v_complex_double = gala_new((LST_time, 0), N_time)

        Et_rec, Ep_rec, E_inv = e_recons_real_MHz(Voc_fft_band, Lce_sphere, galactic_v_complex_double)

        #--------------------------------------------------------------------------------------------------------

        plt.figure(figsize=(10, 7)) # 绘制重建电场的频域图
        plt.plot(freqs_band, np.abs(Et_rec) * 2 / N_time, 'r-', label = 'Theta component')
        plt.plot(freqs_band, np.abs(Ep_rec) * 2 / N_time, 'b-', label = 'Phi component')
        plt.title(f"DU {triggered_du_ids[i]}: Reconstructed E-field in Frequency Domain", fontsize = 18)
        plt.xlabel("Frequency [MHz]", fontsize = 16)
        plt.ylabel("Electric Field [uV/m]", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_DU_{triggered_du_ids[i]}_Efield_Reconstruction_Frequency.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #------------------------------------------------------------------------------------------------------------------

        E_rec_time, Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi, Fluence_sigma \
        = calculate_energy_fluence_and_error(Et_rec, Ep_rec, E_inv, freqs_real_MHz, N_time)

        #-------------------------------------------------------------------------------------------

        plt.figure(figsize=(10, 4)) # 绘制重建电场的时域波形图
        plt.plot(E_rec_time[:, 0], color = 'red', linewidth = 1.0, label = 'Theta component')
        plt.plot(E_rec_time[:, 1], color = 'blue', linewidth = 1.0, label = 'Phi component')
        # plt.plot(E_rec_amp, color = 'black', linewidth = 1.5, label = 'Total')
        plt.title(f"DU {triggered_du_ids[i]}: Reconstructed E-field in Time Domain", fontsize = 18)
        plt.xlabel("Time index (2 ns interval)", fontsize = 16)
        plt.ylabel("Electric Field [uV/m]", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_DU_{triggered_du_ids[i]}_Efield_Reconstruction_Time.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #------------------------------------------------------------------------------------------------------------------

        Voc_recalc_time = re_calculate_Voc_from_reconstruction(LST_time, N_time, freqs_band, freqs_real_MHz, Lce_sphere, Et_rec, Ep_rec)

        fig, axs = plt.subplots(3, 1, figsize = (12, 10), sharex = True, gridspec_kw = {'hspace': 0.0})
        ports = ['X', 'Y', 'Z']

        for ch in range(3):
            ax = axs[ch]
            ax.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
            ax.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
            for spine in ax.spines.values(): spine.set_linewidth(1.5)
            for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight('bold')
            ax.tick_params(axis='x', pad=6)
            ax.plot(Voc_time[:, ch], 'k-', alpha = 0.8, linewidth = 1.5, label = 'Original Voc') # 原始 Voc 时域波形 (经过 IRFFT 后的结果)
            ax.plot(Voc_recalc_time[:, ch], 'r--', linewidth = 1.5, label='Recalculated Voc') # 根据重建电场重新计算的 Voc 时域波形
            ax.set_ylabel(f'Voltage {ports[ch]} [uV]', fontsize = 18)
            ax.set_ylim(1.1 * np.min(Voc_time[50 : 250, :]), 1.1 * np.max(Voc_time[50 : 250, :])) # 根据原始 Voc 的范围动态调整 y 轴范围
            if ch == 0: 
                ax.legend(loc='upper left', fontsize = 15)
                ax.set_xlim(50, 250) # 只显示峰值附近的部分时间窗口以便对比
                ax.set_title('Voc Consistency Check', fontsize = 18)

        axs[-1].set_xlabel('Time index (2 ns interval)', fontsize = 18)
        plt.savefig(os.path.join(efield_rec_dir + filename + '/', f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Consistency_Check.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #-------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------

        saved_data[i] = [event_num, gps_time[3 * i + 2], LST_time, triggered_du_ids[i], rec_theta_plane, rec_phi_plane, chi2Plane, du_nanoseconds_filtered[i], DU_positions_rotated[i, 0], DU_positions_rotated[i, 1], DU_positions_rotated[i, 2], Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi, Fluence_sigma, rec_theta_plane_err, rec_phi_plane_err] # 保存 DU ID, 三通道峰值电压, 重建电场的 fluence 及其物理误差到数组中

        E_rec_time_all_DUs.append(E_rec_time.copy()) # 将每个触发 DU 的重建电场时域波形保存到列表中, 供后续分析使用

    #---------------------------------------------------------------------------

    np.savetxt(efield_rec_dir + filename + '/' + f'/event_{event_num}_reconstruction_results.txt', saved_data) # 将结果保存为 文本文件, 包含每个触发 DU 的 ID, 到达时间, 位置坐标, 和重建电场的 fluence
    
    #---------------------------------------------------------------------------

    E_rec_time_all_DUs = np.array(E_rec_time_all_DUs) # 将列表转换为 numpy 数组, 形状为 (触发 DU 数量, 时间点数量, 2) 对应 (DU, 时间, E-field 分量)

    E_rec_time_all_DUs_2d = E_rec_time_all_DUs.reshape(E_rec_time_all_DUs.shape[0], -1) # 将 3D 数组重塑为 2D 数组, 形状为 (触发 DU 数量, 时间点数量 * E-field 分量数量)，以便保存为 CSV 文件

    np.savetxt(efield_rec_dir + filename + '/' + f'/event_{event_num}_E_rec_time_all_DUs.txt', E_rec_time_all_DUs_2d, delimiter = ' ', header = 'E_theta_time_0, E_phi_time_0, E_theta_time_1, E_phi_time_1, ..., E_theta_time_N, E_phi_time_N') # 将重建电场的时域波形保存为 txt 文件, 每行对应一个触发 DU, 列对应时间点和 E-field 分量

    #---------------------------------------------------------------------------

    return 0

    #---------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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
    """
    执行 ADF 拟合。
    
    拟合参数: theta, phi, dw (宽度), A (幅度)。
    固定参数: B (地磁参数), rc。
    ant_p_sigma: 【新增】每根天线 Fluence 的物理误差, 传给 ADF 类用作 chi2 归一化分母。
    """
    m = iminuit.Minuit(
        ADF(x, y, z, ant_p, xs, ys, zs, obs_altitude, ant_p_sigma),
        theta = theta0,
        phi = phi0,
        dw = dw0,
        A = A0,
        B = 0.01,
        rc = 1,
    )
    # 设置参数限制
    m.limits = (
        [theta0 - 0.8, theta0 + 0.8],
        [phi0 - 0.8, phi0 + 0.8],
        [0.01, 5],
        [A0 / 2, A0 * 1e3], [0, 0.01],
        [0, 1],
    )
    # 固定 B 和 rc
    m.fixed = (False, False, False, False, True, True)
    m.errordef = 1
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    theta, theta_err = m.values["theta"], m.errors["theta"]
    phi, phi_err = m.values["phi"], m.errors["phi"]
    dw, dw_err = m.values["dw"], m.errors["dw"]
    A, A_err = m.values["A"], m.errors["A"]
    fval = m.fval
    return theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err


def ADF_fit_fixdw(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, ant_p_sigma=None):
    """
    执行固定宽度的 ADF 拟合。
    
    当自由拟合的 dw 不合理时使用。固定 dw 为传入的 dw0。
    ant_p_sigma: 【新增】每根天线 Fluence 的物理误差, 传给 ADF 类用作 chi2 归一化分母。
    """
    m = iminuit.Minuit(
        ADF(x, y, z, ant_p, xs, ys, zs, obs_altitude, ant_p_sigma),
        theta = theta0,
        phi = phi0,
        dw = dw0,
        A = A0,
        B = 0.01,
        rc = 1,
    )
    m.limits = (
        [theta0 - 0.8, theta0 + 0.8],
        [phi0 - 0.8, phi0 + 0.8],
        [0.01, 5],
        [A0 / 2, A0 * 1e3], [0, 0.01],
        [0, 1],
    )
    m.fixed = (False, False, True, False, True, True) # 固定 dw, B, rc
    m.errordef = 1
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    theta, theta_err = m.values["theta"], m.errors["theta"]
    phi, phi_err = m.values["phi"], m.errors["phi"]
    dw, dw_err = m.values["dw"], 0 # dw 固定, 误差设为 0
    A, A_err = m.values["A"], m.errors["A"]
    fval = m.fval
    return theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err

#--------------------------------------------------------------------------

def recons_angle(x, y, z, t, p, init_input, obs_altitude, p_sigma=None, x_xmax_external=None, y_xmax_external=None, z_xmax_external=None, chi_square_external=None):  # time given in ns
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
    
    
    thetalist = np.linspace(zenith_ini - 0.5, zenith_ini + 0.5, 3)
    philist = np.linspace(azimuth_ini - 0.5, azimuth_ini + 0.5, 3) # 这两个参数范围是否给的太窄, 真实情况应当是 0-90, 0-360 ????
    dwList = np.linspace(0.1, 4., 4)
    Alist = np.linspace(max(p), 3 * max(p), 3)
    
    for theta0 in thetalist: # 网格搜索最佳方向和 ADF 参数
        for phi0 in philist:
            for dw0 in dwList:
                for A0 in Alist:
                    p_sigma_masked = p_sigma[~np.isnan(p)] if p_sigma is not None else None
                    theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err = ADF_fit(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, A0, obs_altitude, p_sigma_masked)
                    if fval < fmin:
                        fmin = fval
                        rec_theta = theta
                        rec_phi = phi
                        rec_dw = dw
                        rec_A = A
                        rec_theta_err = theta_err
                        rec_phi_err = phi_err
                        rec_dw_err = dw_err
                        rec_A_err = A_err

    if rec_dw > 3.499 or rec_dw < 0.101: # 如果拟合得到的 dw 超出合理范围，使用参数化公式固定 dw 重新拟合

        dXmax = find_core(rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta, rec_phi, obs_altitude)[3] # 计算 Xmax 到芯位的距离
        XmaxDis_km = dXmax / 1e3

        dw0 = 227.958137 / (176.414293 * XmaxDis_km ** 2 + 112.917133 * XmaxDis_km) + 1.351487 # dw 参数化公式

        fmin_fixdw = np.inf

        for theta0 in thetalist: # 网格搜索最佳方向和 ADF 参数, 固定 dw
            for phi0 in philist:
                for A0 in Alist:
                    p_sigma_masked = p_sigma[~np.isnan(p)] if p_sigma is not None else None
                    theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err = ADF_fit_fixdw(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, A0, obs_altitude, p_sigma_masked)
                    if fval < fmin_fixdw:
                        fmin_fixdw = fval
                        rec_theta = theta
                        rec_phi = phi
                        rec_dw = dw
                        rec_A = A
                        rec_theta_err = theta_err
                        rec_phi_err = phi_err
                        rec_A_err = A_err
                        rec_dw_err = 0 # 固定 dw 后误差设为 0

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
    
    return zenith_ini, azimuth_ini, chi2Plane, rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph / (len(x) - 3), rec_theta, rec_phi, rec_dw, rec_A, fmin / (len(p[~np.isnan(p)]) - (3 if rec_dw_err == 0 else 4)), np.asarray(wc_adf).reshape(-1)[0], rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err, rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err # wc_adf 兼容写法(标量/单元素数组都可)

#--------------------------------------------------------------------------

def f_Che(par, omega_):
    """
    切伦科夫函数模型。
    
    参数:
        par (list): [幅度, 切伦科夫角, 宽度]。
        omega_ (float): 视角。
    """

    return par[0] / (1 + 4 * (((np.tan(omega_) / np.tan(par[1])) ** 2 - 1) / par[2]) ** 2)


def f_che_residual(pars, w_, data):
    """
    计算切伦科夫拟合残差。
    
    参数:
        pars (lmfit.Parameters): 拟合参数。
        w_ (numpy.ndarray): 视角。
        data (numpy.ndarray): 观测数据。
    """
    vals = pars.valuesdict()
    amp = vals['amp']
    w_c = vals['w_c']
    domega = vals['domega']
    
    fche = amp * (1 / (1 + 4 * (((np.tan(w_) / np.tan(w_c)) ** 2 - 1) / domega) ** 2))
    
    fmin = fche - data
    chi2 = (fmin ** 2).sum()
    return chi2


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

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------

def energy_restruction(root_path, event_number_int, det_pos_path = None, x_xmax = None, y_xmax = None, z_xmax = None, chi_square_external = None):
    # 可选外部先验 (来自 candidates YAML 的 SWM 结果): x_xmax/y_xmax/z_xmax/chi_square_external
    # 给定则透传给 recons_angle, 跳过其内部 SWF 网格搜索。

    logger.info(f"Starting energy reconstruction using ADF from {root_path} , event:  {int(event_number_int)}")

    filename = os.path.splitext(os.path.basename(root_path))[0] + f"_event_{int(event_number_int)}" # 得到文件名和事件编号

    filename_escaped = filename.replace("_", "-")

    #-------------------------------------------------------------------------------------------------

    # 全部 DU 的坐标 (绘图背景点): 按输入 tree 类型选定坐标表后读取; 旋转策略同 efield 阶段。
    tree_kind, du_pos_path = _load_du_position_for_input(root_path, det_pos_path)
    DU_all_positions_rotated = read_xyz_coordinates(du_pos_path, [], rotate = (tree_kind == 'teventadc'))[4]

    #-------------------------------------------------------------------------------------------------

    fluence_data = np.loadtxt(glob.glob(os.path.join(efield_rec_dir + filename + '/', '*reconstruction_results.txt'))[0])

    du_ids = fluence_data[:, 3].astype(int) # 从保存的重建结果中提取天线 ID 列

    rec_theta_plane, rec_phi_plane, chi2Plane = fluence_data[:, 4], fluence_data[:, 5], fluence_data[:, 6] # 从保存的重建结果中提取平面波重建的角度和 chi2 列

    nanosecond_time = fluence_data[:, 7] * 1e-9 # 从保存的重建结果中提取纳秒时间列

    positions_x, positions_y, positions_z = fluence_data[:, 8], fluence_data[:, 9], fluence_data[:, 10] # 从保存的重建结果中提取位置坐标列

    positions_all = np.array([positions_x, positions_y, positions_z]).T

    Fluence, Fluence_theta, Fluence_phi = fluence_data[:, 11], fluence_data[:, 12], fluence_data[:, 13] # 从保存的重建结果中提取 fluence 列
    Fluence_theta_phi = fluence_data[:, 14]
    Fluence_sigma = fluence_data[:, 15] # 【新增】物理传播得到的 Fluence 误差, 用作 ADF 拟合的 chi2 归一化分母

    obs_altitude = np.mean(positions_z) # 使用天线位置的平均高度作为观测高度

    #-------------------------------------------------------------------------------------------------

    E_rec_time_all_DUs_data = np.loadtxt(glob.glob(os.path.join(efield_rec_dir + filename + '/', '*E_rec_time_all_DUs.txt'))[0])

    n_time_rec = E_rec_time_all_DUs_data.shape[1] // 2 # 从数据推断 trace 长度 (1024 / 512 均可), 避免硬编码
    E_rec_time_sph = E_rec_time_all_DUs_data.reshape(E_rec_time_all_DUs_data.shape[0], n_time_rec, 2) # 

    #-------------------------------------------------------------------------------------------------

    init_input = [rec_theta_plane[0], rec_phi_plane[0], chi2Plane[0]] # 使用平面波重建结果作为 ADF 重建的初始输入

    rec_theta_plane, rec_phi_plane, chi2Plane, rec_x_xmax, rec_y_xmax, rec_z_xmax, \
    chi2Sph, rec_theta_sph, rec_phi_sph, rec_dw, rec_A, fmin, rec_wc, \
    rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err, \
    rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err = \
    recons_angle(positions_x, positions_y, positions_z, nanosecond_time, Fluence, init_input, obs_altitude, Fluence_sigma, x_xmax, y_xmax, z_xmax, chi_square_external)


    logger.info(f"Reconstructed Xmax coordinates (m) and chi2: {rec_x_xmax} {rec_y_xmax} {rec_z_xmax} {chi2Sph}")

    logger.info(f"Reconstructed shower direction (theta, phi in degrees and errors): {rec_theta_sph} {rec_theta_err} {rec_phi_sph} {rec_phi_err}")

    logger.info(f"Reconstructed ADF parameters (A, wc, dw, fmin/DOF): {rec_A} {rec_wc} {rec_dw} {fmin}")

    logger.info(f"Reconstructed ADF parameter errors (A_err, wc_err, dw_err): {rec_A_err} 0 {rec_dw_err}")

    #-------------------------------------------------------------------------------------------------

    height = helper.get_local_altitude([rec_x_xmax, rec_y_xmax, rec_z_xmax + 1280]) # 计算重建的 Xmax 坐标对应的海拔高度
    logger.info(f"Local altitude at reconstructed Xmax (m): {height}")

    #-------------------------------------------------------------------------------------------------

    rho = atm.get_density(height, model = 1) * 1e-3 # 使用重建的 Xmax 高度计算大气密度
    logger.info(f"Reconstructed Xmax altitude: {height} rho at reconstructed Xmax (g/cm^3): {rho}")

    #-------------------------------------------------------------------------------------------------

    shower_axis = helper.spherical_to_cartesian(np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph))
    logger.info(f"Shower axis unit vector: {shower_axis}")

            
    alpha = np.arccos(np.dot(shower_axis, magnetic_field_vector) / (np.linalg.norm(shower_axis) * np.linalg.norm(magnetic_field_vector)))
    logger.info(f"alpha: {alpha * 180 / np.pi}")

    #-------------------------------------------------------------------------------------------------

    corex, corey, corez, dXmax = find_core(rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta_sph, rec_phi_sph, obs_altitude)
    shower_core = np.array([corex, corey, corez])
    logger.info(f"Shower core: {shower_core} dXmax: {dXmax}")


    #-------------------------------------------------------------------------------------------------

    nanosecond_time_relative = nanosecond_time - np.median(nanosecond_time) # 相对时间, 以中位数为基准
    nanosecond_time_order = np.argsort(np.argsort(nanosecond_time_relative))  # 排名: 最早->最小, 最晚->最大


    X_grid, Y_grid = np.linspace(-10000, 10000, 400), np.linspace(-10000, 10000, 400)
    XX_plot, YY_plot = np.meshgrid(X_grid, Y_grid)

    physics_Y, physics_X = -XX_plot, YY_plot # 转换为物理坐标系 (西-东, 南-北)
    physics_Z = np.zeros_like(physics_X) + obs_altitude # 平面图，Z 坐标为 0

    obs_X, obs_Y, obs_Z = rec_x_xmax - physics_X, rec_y_xmax - physics_Y, rec_z_xmax - physics_Z # 观测点坐标 (Xmax)

    L_grid = np.sqrt(obs_X ** 2 + obs_Y ** 2 + obs_Z ** 2) # 传播距离
    omega_grid = np.arccos(np.clip(shower_axis[0] * obs_X / L_grid + shower_axis[1] * obs_Y / L_grid + shower_axis[2] * obs_Z / L_grid, -1.0, 1.0)) # 视角

    plt.figure(figsize = (8, 6))
    plt.scatter(-1 * DU_all_positions_rotated[:, 1], DU_all_positions_rotated[:, 0], c = 'lightgray')
    plt.scatter(float(-1 * corey), float(corex), c = 'red', s = 100, marker = '.', label = 'Shower Core')
    # cbar = plt.scatter(-1 * positions_y, positions_x, c = Fluence, s = 100 + ((Fluence - np.min(Fluence)) / (np.max(Fluence) - np.min(Fluence))) * 300, edgecolors = 'k', cmap = 'viridis')
    cbar = plt.scatter(-1 * positions_y, positions_x, c = Fluence, s = 50 + nanosecond_time_order / max(len(nanosecond_time_order) - 1, 1) * 350, edgecolors = 'k', cmap = 'viridis')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(r'Fluence$\rm [eV/m^2]$', fontsize = 16)
    for i in range(len(positions_x)): plt.text(-1 * positions_y[i], positions_x[i], str(int(du_ids[i])), fontsize = 10, ha = 'left', va = 'bottom', color = 'black')
    contours = plt.contour(XX_plot, YY_plot, omega_grid, levels = [rec_wc], colors = 'teal', alpha = 0.7, linewidths = 2, linestyles = '--')
    plt.arrow(-1 * corey, float(corex), -500 * np.sin(np.deg2rad(rec_theta_sph)) * np.sin(np.deg2rad(rec_phi_sph)), 500 * np.sin(np.deg2rad(rec_theta_sph)) * np.cos(np.deg2rad(rec_phi_sph)), head_width = 50, head_length = 100, fc = 'red', ec = 'red')
    plt.xlabel('West-East [m]', fontsize = 16)
    plt.ylabel('South-North [m]', fontsize = 16)
    plt.title(f'{filename_escaped} \n z = {rec_theta_sph:.2f} deg, a = {rec_phi_sph:.2f} deg, chi2/dof = {fmin:.2f}', fontsize = 12)
    plt.xlim(-5000, 3000)
    plt.ylim(-6000, 2000)
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
    for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
    plt.tick_params(axis='x', pad=6)
    plt.tick_params(axis='y', pad=6)
    plt.legend(fontsize = 14, loc = 'lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.axis('equal')
    plt.savefig(os.path.join(efield_rec_dir + filename + '/', 'shower_core_position.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()

    #-------------------------------------------------------------------------------------------------

    cs = coordinatesystems.cstrafo(np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph), magnetic_field_vector)
    
    fluences_sph = np.array([Fluence_theta, Fluence_phi, np.zeros(len(Fluence_phi))]).T
    
    R_sph2cart = rotation_m(rec_theta_sph * np.pi / 180, rec_phi_sph * np.pi / 180)
    
    fluences_cart = np.dot(fluences_sph, R_sph2cart.T)
    
    fluence_vxB = np.array([cs.transform_to_vxB_vxvxB(fluence) for fluence in fluences_cart])
    fluence_vxB = abs(fluence_vxB)
    
    #-------------------------------------------------------------------------------------------------

    pos_vxB = cs.transform_to_vxB_vxvxB(positions_all, shower_core)

    #-------------------------------------------------------------------------------------------------
    
    ant2Xmax_dis = ((positions_x - rec_x_xmax) ** 2 + (positions_y - rec_y_xmax) ** 2 + (positions_z - rec_z_xmax) ** 2) ** 0.5
    early_late = np.square(ant2Xmax_dis / dXmax)

    # print("Early-Late correction factors:", early_late, flush = True)
    

    # f_vxB, f_vxvxB = f_vxB_pure * early_late, f_vxvxB_pure * early_late
    f_vxB, f_vxvxB = fluence_vxB[:, 0] * early_late, fluence_vxB[:, 1] * early_late
    phi = helper.get_normalized_angle(np.arctan2(pos_vxB[:, 1], pos_vxB[:, 0]))
    vxB_axis = np.abs(np.sin(phi)) < np.abs(np.sin(10 * np.pi / 180)) # 定义一个小范围来判断是否接近 vxB 轴线, 避免数值问题
    f_tot = np.sum(fluence_vxB, axis = -1) * early_late
    # f_tot = f_vxB + f_vxvxB

    # print("phi (degrees):", phi * 180 / np.pi, flush = True)
    # print("f_vxB:", f_vxB, flush = True)
    # print("f_vxvxB:", f_vxvxB, flush = True)
    # print("Total fluence (f_tot):", f_tot, flush = True)


    #-------------------------------------------------------------------------------------------------
    '''
    ADF_A_constant, ADF_A_err_constant, ADF_chi2_constant = ADF_constant_fit(Fluence[~np.isnan(Fluence)], early_late[~np.isnan(Fluence)], Fluence_sigma[~np.isnan(Fluence)])

    print("ADF constant fit results: A =", ADF_A_constant, "A_err =", ADF_A_err_constant, "chi2/dof =", ADF_chi2_constant, flush = True)

    if rec_dw_err == 0: 
        delta_ADF_chi2 = ADF_chi2_constant * (len(Fluence[~np.isnan(Fluence)]) - 1) - fmin * (len(Fluence[~np.isnan(Fluence)]) - 3) # 计算 Δχ² = χ²_constant - χ²_ADF, 并乘以自由度 (dof = N - 3)
        delta_ADF_chi2_sigma = sigma(delta_ADF_chi2, 2) # 将 Δχ² 转换为 σ 级别

    else: 
        delta_ADF_chi2 = ADF_chi2_constant * (len(Fluence[~np.isnan(Fluence)]) - 1) - fmin * (len(Fluence[~np.isnan(Fluence)]) - 4) # 计算 Δχ² = χ²_constant - χ²_ADF, 并乘以自由度 (dof = N - 4)
        delta_ADF_chi2_sigma = sigma(delta_ADF_chi2, 3) # 将 Δχ² 转换为 σ 级别

    print("Delta ADF chi2:", delta_ADF_chi2, "Delta ADF chi2 sigma:", delta_ADF_chi2_sigma, flush = True)
    '''
    #-------------------------------------------------------------------------------------------------

    f_geo_pos = np.zeros(len(positions_x))

    peak_vB_voltage, peak_vvB_voltage = np.zeros(len(positions_x)), np.zeros(len(positions_x))

    for ant in range(len(positions_x)):

        E_sph_t = np.zeros((E_rec_time_sph.shape[1], 3))
        E_sph_t[:, 0], E_sph_t[:, 1] = E_rec_time_sph[ant, :, 0], E_rec_time_sph[ant, :, 1]

        E_cart_t = np.dot(E_sph_t, R_sph2cart.T)

        E_shower_t = cs.transform_to_vxB_vxvxB(E_cart_t.T).T

        E_vxB_t, E_vxvxB_t = E_shower_t[:, 0], E_shower_t[:, 1]

        #--------------------------------------------------------------------------------------------------------------------

        env_vxB, env_vxvxB = np.abs(hilbert(E_vxB_t)), np.abs(hilbert(E_vxvxB_t)) # Hilbert envelope of each component

        env_tot = np.sqrt(env_vxB ** 2 + env_vxvxB ** 2) # Total envelope

        t_peak, half = np.argmax(env_tot), np.max(env_tot) / 2.0

        left = np.where(env_tot[:t_peak] < half)[0]
        t_start = left[-1] + 1 if len(left) > 0 else 0

        right = np.where(env_tot[t_peak:] < half)[0]
        t_end = t_peak + right[0] if len(right) > 0 else len(env_tot)

        peak_vB_voltage[ant], peak_vvB_voltage[ant] = np.mean(E_vxB_t[t_start : t_end]), np.mean(E_vxvxB_t[t_start : t_end])

        # peak_vB_voltage[ant], peak_vvB_voltage[ant] = E_vxB_t[np.argmax(env_tot)], E_vxvxB_t[np.argmax(env_tot)]

        #--------------------------------------------------------------------------------------------------------------------
        
        phi_iter = phi[ant]
        denom = np.abs(np.sin(phi_iter)) if np.abs(np.sin(phi_iter)) >= 1e-4 else 1e-4 * np.sign(np.sin(phi_iter)) + 1e-12 # 避免除以零
        cot_phi = np.cos(phi_iter) / denom

        E_geo_t = E_vxB_t - E_vxvxB_t * cot_phi

        f_geo_pos[ant] = np.sum(E_geo_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377) # 这里的地磁修正公式在 vxB_axis 区域会出现除以 sin(phi) 的问题，使用 cot_phi 来避免，同时整个表达式在 vxB_axis 区域被设置为 0

        # print(np.sum(E_vxB_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377), np.sum(E_vxvxB_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377), flush = True)

    #------------------------- 计算视角 omega -----------------------------------------------
    
    obs = np.array([rec_x_xmax - positions_x, rec_y_xmax - positions_y, rec_z_xmax - positions_z])
    l = np.sqrt((positions_x - rec_x_xmax) ** 2 + (positions_y - rec_y_xmax) ** 2 + (positions_z - rec_z_xmax) ** 2)

    u_ant = obs / l
    # w = np.arccos(np.clip(np.dot(shower_axis, u_ant), -1.0 + 1e-10, 1.0 - 1e-10)) # 计算每个天线与簇射轴的夹角 omega，并进行数值稳定性处理以避免 arccos 输入超出 [-1, 1]
    w = np.arccos(np.dot(shower_axis, u_ant))
    # omega_c = w[np.argmax(f_geo_pos)]
    omega_c = float(np.median((lambda m: w[m][np.argsort(f_geo_pos[m])[-max(3, int(0.1 * np.sum(m))):]])(~vxB_axis))) # 取 f_geo_pos 最大的几个点对应的 omega 的中位数作为切伦科夫角, 避免单点异常


    # print("Viewing angles (omega in degrees):", w * 180 / np.pi, flush = True)
    # print("omega_c (rad):", omega_c, flush = True)

    #-------------------------------------------------------------------------------------------------

    angle_meas = np.arctan2(peak_vvB_voltage, peak_vB_voltage) * 180 / np.pi
    # print("Measured polarization angles (degrees):", angle_meas, flush = True)

    angle_expected = np.arctan2(0.10 * pos_vxB[:, 1] / np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2), 1.0 + 0.10 * pos_vxB[:, 0] / np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2)) * 180 / np.pi
    # print("Expected polarization angles (degrees):", angle_expected, flush = True)

    angle_meas, angle_expected = angle_meas % 180, angle_expected % 180 # 将角度折叠到 [0, 180) 范围内，因为偏振方向具有 180 度周期性

    delta_angle = np.minimum(np.abs(angle_meas - angle_expected), 180 - np.abs(angle_meas - angle_expected)) # 计算测量和期望偏振角之间的最小差值，考虑到周期性
    logger.info(f"Difference (degrees): {np.abs(delta_angle)}")

    mean_delta_angle, median_delta_angle = np.mean(np.abs(delta_angle)), np.median(np.abs(delta_angle))
    logger.info(f"Mean difference (degrees): {mean_delta_angle} Median difference (degrees): {median_delta_angle}")

    #-------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------
    
    '''
    #------排除指定 DU(测试用，注释掉则显示全部)--------------------------------------
    _exclude_du = {1046, 1054, 1065, 1074}
    _show_du = np.array([d not in _exclude_du for d in du_ids])

    du_ids, pos_vxB, angle_meas, delta_angle = du_ids[_show_du], pos_vxB[_show_du], angle_meas[_show_du], delta_angle[_show_du]
    #-------<<<>>>---------------------------------------------------------------
    '''
    
    plt.figure(figsize=(9, 9))
    plt.scatter(pos_vxB[:, 0], pos_vxB[:, 1], c = 'blue', s = 50, alpha = 0.8)
    for i in range(len(du_ids)): 
        
        if i == 0: 
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200, 0, head_width = 5, head_length = 20, fc = 'red', ec = 'red', label = 'Geomagnetic')
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, -1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, head_width = 5, head_length = 20, fc = 'blue', ec = 'blue', label = 'Charge-excess')
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200 -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, -1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, head_width = 5, head_length = 20, fc = 'black', ec = 'black', label = 'Expected')
            # plt.plot([pos_vxB[i, 0], 0], [pos_vxB[i, 1], 0], 'k--', linewidth = 0.5, alpha = 0.5)

            if (-200 -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20) * (-200 * np.cos(np.deg2rad(angle_meas[i]))) \
            + (-1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20) * (-200 * np.sin(np.deg2rad(angle_meas[i]))) < 0:
                plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], 200 * np.cos(np.deg2rad(angle_meas[i])), 200 * np.sin(np.deg2rad(angle_meas[i])), head_width=5, head_length=20, fc='green', ec='green', label = 'Measured')
            else:
                plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200 * np.cos(np.deg2rad(angle_meas[i])), -200 * np.sin(np.deg2rad(angle_meas[i])), head_width=5, head_length=20, fc='green', ec='green', label = 'Measured')
    
        else:
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200, 0, head_width = 5, head_length = 20, fc = 'red', ec = 'red')
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, -1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, head_width = 5, head_length = 20, fc = 'blue', ec = 'blue')
            plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200 -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, -1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20, head_width = 5, head_length = 20, fc = 'black', ec = 'black')
            # plt.plot([pos_vxB[i, 0], 0], [pos_vxB[i, 1], 0], 'k--', linewidth = 0.5, alpha = 0.5)

            if (-200 -1 * pos_vxB[i, 0] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20) * (-200 * np.cos(np.deg2rad(angle_meas[i]))) \
            + (-1 * pos_vxB[i, 1] / np.sqrt(pos_vxB[i, 0] ** 2 + pos_vxB[i, 1] ** 2) * 20) * (-200 * np.sin(np.deg2rad(angle_meas[i]))) < 0:
                plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], 200 * np.cos(np.deg2rad(angle_meas[i])), 200 * np.sin(np.deg2rad(angle_meas[i])), head_width=5, head_length=20, fc='green', ec='green')
            else:
                plt.arrow(pos_vxB[i, 0], pos_vxB[i, 1], -200 * np.cos(np.deg2rad(angle_meas[i])), -200 * np.sin(np.deg2rad(angle_meas[i])), head_width=5, head_length=20, fc='green', ec='green')

        plt.text(pos_vxB[i, 0] - 150, pos_vxB[i, 1] + 40, rf"{du_ids[i]},$\Delta\beta$={delta_angle[i]:.1f}deg", fontsize = 12, color = 'black')
    
    # plt.scatter(0, 0, c = 'black', s = 150, marker = '+')
    plt.title(f'{filename_escaped} \n z = {rec_theta_sph:.2f} deg, a = {rec_phi_sph:.2f} deg, chi2/dof = {fmin:.2f}', fontsize = 12)
    plt.xlabel(r'$\hat{v} \times \hat{B}$ [m]', fontsize = 20)
    plt.ylabel(r'$\hat{v} \times (\hat{v} \times \hat{B})$ [m]', fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
    for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
    plt.tick_params(axis='x', pad=6)
    plt.legend(fontsize = 12, loc = 'best')
    x_axis_center = (pos_vxB[:, 0].max() + pos_vxB[:, 0].min()) / 2 # 计算数据在 vxB 平面上的中心位置
    y_axis_center = (pos_vxB[:, 1].max() + pos_vxB[:, 1].min()) / 2
    half_range = max(pos_vxB[:, 0].max() - pos_vxB[:, 0].min(), pos_vxB[:, 1].max() - pos_vxB[:, 1].min()) / 2 # 取 x 和 y 跨度的最大值，确保两轴范围一致
    margin = half_range * 0.15 # 加上 10~20% 边距，让箭头不贴边
    plt.xlim(x_axis_center - half_range - margin, x_axis_center + half_range + margin)
    plt.ylim(y_axis_center - half_range - margin, y_axis_center + half_range + margin)
    # plt.axis("square")
    plt.axis("equal")
    plt.savefig(os.path.join(efield_rec_dir, filename, 'polarization_map_shower_plane.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    #--------------------------------------------------------------------------------------------------

    XmaxDis_km = dXmax / 1e3
    dw = 227.958137 / (176.414293 * XmaxDis_km ** 2 + 112.917133 * XmaxDis_km) + 1.351487 # dw 经验参数化公式

    #---------------------- 拟合切伦科夫分布 ---------------------------------------------------
    
    omega_c = rec_wc # 使用 ADF 重建得到的切伦科夫角作为初始值，避免拟合陷入局部最小值
    chi2_che = che_fit(w[~vxB_axis], f_geo_pos[~vxB_axis], max(f_geo_pos), omega_c, dw)

    f_che_pars = np.array([chi2_che[0], chi2_che[1], chi2_che[2]])

    chi2_che_err = np.array([chi2_che[4], chi2_che[5], chi2_che[6]])

    logger.info(f"Initial Che parameters: {f_che_pars} with errors: {chi2_che_err} Chi2/dof {chi2_che[3]}")

    logger.info(f'Asymmetric ADF parameters: {f"{rec_A:.5e}"} {rec_wc} {rec_dw}')

    #---------------------- 计算辐射能量 ------------------------------------------------------
    #----------------------------------------------------------------------------------------
    
    e_rad_geo_fit = integrate.quad(lambda omega: omega * f_Che(f_che_pars, omega), 0, 0.1)[0] # 积分计算地磁辐射能量

    factor = 1 # 0.5e-9 / 377 * (1e-6) ** 2 * 6.241509e18 # Unit conversion factor

    E_rad_GeV = e_rad_geo_fit * factor * 2 * np.pi * dXmax ** 2
    logger.info(f"Reconstructed E_rad (GeV): {E_rad_GeV / 1e9}")

    correction_E_geo = density_and_alpha_correction(E_rad_GeV, rho, alpha)
    logger.info(f"Density and alpha corrected E_geo (GeV): {correction_E_geo / 1e9}")
    
    #----------------------------------------------------------------------------------------

    def compute_energy_che(a, w, d):

        e_rad = integrate.quad(lambda omega: omega * f_Che([a, w, d], omega), 0, 0.1)[0]

        e_rad_GeV = e_rad * factor * 2 * np.pi * dXmax ** 2

        e_rad_GeV_corrected = density_and_alpha_correction(e_rad_GeV, rho, alpha)

        return np.array([e_rad_GeV / 1e9, e_rad_GeV_corrected / 1e9, (e_rad_GeV_corrected / 1e9 / 0.07) ** (1 / 2.03)]) # 

    E_che_EeV_center = compute_energy_che(*f_che_pars)

    dEamp = (compute_energy_che(f_che_pars[0] + chi2_che_err[0], f_che_pars[1], f_che_pars[2]) - compute_energy_che(f_che_pars[0] - chi2_che_err[0], f_che_pars[1], f_che_pars[2])) / 2

    dEwc = (compute_energy_che(f_che_pars[0], f_che_pars[1] + chi2_che_err[1], f_che_pars[2]) - compute_energy_che(f_che_pars[0], f_che_pars[1] - chi2_che_err[1], f_che_pars[2])) / 2

    dEdw = (compute_energy_che(f_che_pars[0], f_che_pars[1], f_che_pars[2] + chi2_che_err[2]) - compute_energy_che(f_che_pars[0], f_che_pars[1], f_che_pars[2] - chi2_che_err[2])) / 2

    for energy_list in range(3):
        
        E_che_EeV_err = np.sqrt(dEamp[energy_list] ** 2 + dEwc[energy_list] ** 2 + dEdw[energy_list] ** 2)

        logger.info(f"Reconstructed E_em (EeV) from Cherenkov fit: {float(E_che_EeV_center[energy_list]):.2f} ± {float(E_che_EeV_err):.2f} EeV")

    #----------------------------------------------------------------------------------------
    
    adf_pars = np.array([rec_A, rec_wc, rec_dw])

    def integrand(omega, phi): return omega * f_adf(adf_pars, omega, alpha, phi)

    E_em_all_two, error = integrate.dblquad(integrand, 
                                            0, 2 * np.pi,  # Outer limits (phi): 0 to 2 * pi (360 degrees)
                                            lambda phi: 0, lambda phi: 0.1  # Inner limits (omega): 0 to 0.1 rad
                                            )

    E_em_GeV_two = E_em_all_two * factor * dXmax ** 2
    
    logger.info(f"Reconstructed E_em (GeV) from asymmetric ADF: {E_em_GeV_two / 1e9}")

    E_em_corrected_two = density_and_alpha_correction(E_em_GeV_two, rho, alpha)

    logger.info(f"Density and alpha corrected E_em (GeV) from asymmetric ADF: {E_em_corrected_two / 1e9}")
    
    #----------------------------------------------------------------------------------------
    
    def compute_energy_adf(A, wc, dw):

        adf_pars = np.array([A, wc, dw])

        def integrand(omega, phi): return omega * f_adf(adf_pars, omega, alpha, phi)

        e_rad_all, _ = integrate.dblquad(integrand, 
                                            0, 2 * np.pi,  # Outer limits (phi): 0 to 2 * pi (360 degrees)
                                            lambda phi: 0, lambda phi: 0.1  # Inner limits (omega): 0 to 0.1 rad
                                            )

        e_rad_GeV = e_rad_all * factor * dXmax ** 2

        e_rad_GeV_corrected = density_and_alpha_correction(e_rad_GeV, rho, alpha)

        return np.array([e_rad_GeV / 1e9, e_rad_GeV_corrected / 1e9, (e_rad_GeV_corrected / 1e9 / 0.08) ** (1 / 1.91)]) #
    
    E_cent_adf = compute_energy_adf(*adf_pars)
    
    dE_A_adf = (compute_energy_adf(adf_pars[0] + rec_A_err, adf_pars[1], adf_pars[2]) - compute_energy_adf(adf_pars[0] - rec_A_err, adf_pars[1], adf_pars[2])) / 2
    
    dE_dw_adf = (compute_energy_adf(adf_pars[0], adf_pars[1], adf_pars[2] + rec_dw_err) - compute_energy_adf(adf_pars[0], adf_pars[1], adf_pars[2] - rec_dw_err)) / 2

    for energy_list in range(3):

        E_adf_EeV_err = np.sqrt(dE_A_adf[energy_list] ** 2 + dE_dw_adf[energy_list] ** 2)

        logger.info(f"Reconstructed E_em (EeV) from asymmetric ADF: {float(E_cent_adf[energy_list]):.2f} ± {float(E_adf_EeV_err):.2f} EeV")

    #------------------------------------------------------------------------------------------------
    #------------------------------ 绘制切伦科夫拟合结果 ------------------------------------------------

    logger.info(f"energy flux: {Fluence * early_late}")
    logger.info(f"energy flux error: {Fluence_sigma * early_late}")

    plt.figure(figsize = (8, 6))

    w_line = np.linspace(0, 0.1, 200)
    f_che_line, f_che_line_as = f_Che(f_che_pars, w_line), f_Che(adf_pars, w_line) # 这里的 phi 参数对拟合曲线影响较小，暂时固定为 0 来展示 ADF 曲线

    # plt.plot(w_line * 180 / np.pi, f_che_line, color = 'blue', label = 'Fitted distribution')
    # plt.scatter(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], color = 'blue', label = 'Data points')
    plt.plot(w_line * 180 / np.pi, f_che_line_as, color = 'red')#, label = 'Asymmetric ADF function')
    plt.errorbar(w * 180 / np.pi, Fluence * early_late, yerr = Fluence_sigma * early_late, fmt = 's', color = 'red')
    # plt.errorbar(w * 180 / np.pi, Fluence * early_late, yerr = 0.1 * Fluence * early_late + 0.01 * np.max(Fluence * early_late), fmt = 's',  color = 'red')
    for j_, du_ in enumerate(du_ids): plt.text(w[j_] * 180 / np.pi, Fluence[j_] * early_late[j_], f'DU{int(du_)}', fontsize=8, color='gray', alpha=0.8, ha='left', va='bottom')
    # plt.errorbar(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], yerr = f_geo_pos[~vxB_axis] * 0.1 + 0.01 * np.max(f_geo_pos[~vxB_axis]), fmt = 'o', color = 'blue', label = 'Data points')
    # plt.errorbar(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], yerr = np.sqrt(f_geo_pos[~vxB_axis]), fmt = 'o', color = 'blue', label = 'Data points')
    # plt.axvline(x = f_che_pars[1] * 180 / np.pi, color = 'blue', linestyle = '--', label = f'Cherenkov angle: {f_che_pars[1] * 180 / np.pi:.2f} deg')
    plt.axvline(x = rec_wc * 180 / np.pi, color = 'red', linestyle = '--', label = f'ADF Cherenkov angle: {rec_wc * 180 / np.pi:.2f} deg')
    plt.xlabel(r'$\omega$ [deg]', fontsize = 16)
    plt.ylabel(r'Fluence [$\rm eV/m^2$]', fontsize = 16)
    # plt.title('Cherenkov Angle Distribution', fontsize = 18)
    plt.title(f'{filename_escaped} \n z = {rec_theta_sph:.2f} deg, a = {rec_phi_sph:.2f} deg, chi2/dof = {fmin:.2f}', fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
    for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
    plt.tick_params(axis='x', pad=6)
    plt.tick_params(axis='y', pad=6)
    plt.legend(fontsize = 12, loc = 'upper right')
    plt.xlim(0, 3.0)
    plt.grid(False)
    # plt.ylim(0, max(f_geo_pos) * 1.1)
    plt.savefig(os.path.join(efield_rec_dir + filename + '/', 'che_fit_figure.png'), dpi = 300)
    plt.close()

    #----------------------------------------------------------------------------------------

    shower_output = np.array([rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph, \
                              rec_theta_sph, rec_theta_err, rec_phi_sph, rec_phi_err, \
                              rec_A, rec_A_err, rec_wc, 0, rec_dw, rec_dw_err, fmin, \
                              height, rho, alpha, \
                              corex, corey, corez, dXmax, \
                              chi2_che[0], chi2_che[4], chi2_che[1], chi2_che[5], chi2_che[2], chi2_che[6], chi2_che[3] / (len(w[~vxB_axis]) - 1), \
                              E_cent_adf[0], np.sqrt(dE_A_adf[0] ** 2 + dE_dw_adf[0] ** 2), \
                              E_cent_adf[1], np.sqrt(dE_A_adf[1] ** 2 + dE_dw_adf[1] ** 2), \
                              E_cent_adf[2], np.sqrt(dE_A_adf[2] ** 2 + dE_dw_adf[2] ** 2), \
                              E_che_EeV_center[0], np.sqrt(dEamp[0] ** 2 + dEwc[0] ** 2 + dEdw[0] ** 2), \
                              E_che_EeV_center[1], np.sqrt(dEamp[1] ** 2 + dEwc[1] ** 2 + dEdw[1] ** 2), \
                              E_che_EeV_center[2], np.sqrt(dEamp[2] ** 2 + dEwc[2] ** 2 + dEdw[2] ** 2), \
                              rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err
                              ])

    np.savetxt(os.path.join(efield_rec_dir + filename + '/', 'shower_output.txt'), shower_output, fmt = '%.6e')

    #----------------------------------------------------------------------------------------

    logger.info("------------------------------------------------------------------------------------------------")
    logger.info("\n")

    #----------------------------------------------------------------------------------------

    return w, Fluence * early_late, delta_angle

#--------------------------------------------------------------------------

# 示例 / 命令行入口: 仅在直接运行本文件时执行, 避免被 import 时触发重型计算。
# 通过环境变量 RECON_ROOT 与 RECON_EVENT 指定输入 root 文件与事件号:
#   RECON_ROOT=/path/to/adc_*.root RECON_EVENT=0 python reconstruction_candidate_fluence_error_fixed.py
# 若设置了 RECON_ROOT 但未设置 RECON_EVENT, 默认事件号为 0。
if __name__ == '__main__':
    _demo_root = os.environ.get('RECON_ROOT')
    if _demo_root:
        _demo_event = int(os.environ.get('RECON_EVENT', '0'))
        efield_recons_from_efield_PWF(_demo_root, _demo_event)
        energy_restruction(_demo_root, _demo_event)
    else:
        logger.info("未设置 RECON_ROOT, 跳过内置示例。"
              "调用 efield_recons_from_efield_PWF(root_path, event_number, det_pos_path=None) "
              "与 energy_restruction(root_path, event_number, det_pos_path=None) 即可。")

#--------------------------------------------------------------------------
