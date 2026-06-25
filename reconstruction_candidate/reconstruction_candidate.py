import os
import sys
import glob
import numpy as np
import scipy.io
import uproot
import iminuit
import scienceplots
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import windows
from scipy import integrate
from logger_config import logger

plt.rcParams.update({"text.usetex": False})
plt.style.use(['science', 'grid', 'no-latex'])

# 这部分函数为官方可下载函数

#--------------------------------------------------------------------------

# 这部分为自定义函数, 见文件夹python代码文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from equivalent import CEL # 引入计算天线有效长度的函数
from galacticnoise_get import gala # 引入获取银河噪声的函数
from Time_domain_Shower_Edata_get import time_data_get # 引入获取时域簇射电场数据的函数
from vad2voc import vad2voc # 引入从Vad计算Voc的核心函数
from interpolation_ import inter
from complex_expansion import expan
import AiresInfoFunctions as aires
import coordinatesystems, helper
import models as atm
from _optimized_read_matching_times_graph import optimized_read_matching_times_graph

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
# 惰性加载模块级全局变量，避免 import 时触发文件 IO 和目录创建

_DU_POSITION = None
_DU_ALL_POSITIONS_ROTATED = None
_DU_ROTATION_MATRIX = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

efield_rec_dir = './Reconstruction/' # 结果保存目录

def _ensure_output_dir():
    """惰性创建结果输出目录。"""
    if not os.path.exists(efield_rec_dir):
        os.makedirs(efield_rec_dir)

def _load_du_position():
    """惰性加载 DU 位置数据并计算旋转后坐标。"""
    global _DU_POSITION, _DU_ALL_POSITIONS_ROTATED
    if _DU_POSITION is None:
        _DU_POSITION = np.loadtxt("_gp65_rtksort_2002_DU7.txt")
        x = np.array(_DU_POSITION[:, 1])
        y = np.array(_DU_POSITION[:, 2])
        z = np.array(_DU_POSITION[:, 3])
        all_positions = np.stack((x, y, z), axis=1)
        _DU_ALL_POSITIONS_ROTATED = np.dot(_DU_ROTATION_MATRIX, all_positions.T).T
    return _DU_POSITION, _DU_ALL_POSITIONS_ROTATED

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
    n_freqs = fupper - flower
    Et_rec = np.zeros(n_freqs, dtype=complex)
    Ep_rec = np.zeros(n_freqs, dtype=complex)
    invM_Ett = np.zeros(n_freqs)
    invM_Epp = np.zeros(n_freqs)
    
    for idx, ff in enumerate(range(flower, fupper)): # 遍历频率范围
        freq = ff # 当前频率
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
        
        Et_rec[idx], Ep_rec[idx] = et, ep # 保存结果
        invM_Ett[idx], invM_Epp[idx] = ATVA_inv[0, 0], ATVA_inv[1, 1] # 保存不确定度信息

    E_inv = np.array([invM_Ett, invM_Epp]) # 组合不确定度
    
    return (Et_rec), (Ep_rec), (E_inv) # 返回重建电场和不确定度

#--------------------------------------------------------------------------
_PARAM_DIR = os.path.join(os.path.dirname(__file__), 'parameter_files')
REfile_grand = os.path.join(_PARAM_DIR, 'EFL_data_3.2m_XYZ_20250313.mat') # 注意: 这里也是硬编码的路径, 指向 GRAND 天线的有效长度数据文件

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

    theta = m.values["theta"]
    phi = m.values["phi"]
    fval = m.fval
    
    return theta, phi, fval

#--------------------------------------------------------------------------

def efield_recons_from_efield_PWF(root_path, event_number_int): # 基于平面波前拟合 (PWF) 的电场重建

    logger.info(f"Starting E-field reconstruction using PWF from {root_path}, event: {int(event_number_int)}")

    _ensure_output_dir()
    DU_position, DU_all_positions_rotated = _load_du_position()

    filename = os.path.splitext(os.path.basename(root_path))[0] + f"_event_{int(event_number_int)}" # 得到文件名和事件编号
    event_dir = os.path.join(efield_rec_dir, filename)
    if not os.path.exists(event_dir): os.makedirs(event_dir)

    #---------------------------------------------------------------------------

    root_file = uproot.open(root_path)
    try:
        tree = root_file['teventadc'] # 获取 ADC 数据的 Tree
        
        data = tree.arrays(["event_number", "du_id", "gps_time", "du_seconds", "du_nanoseconds", "trace_1", "trace_2", "trace_3"], library = "ak")

        data = data[int(event_number_int)] # 直接索引到指定 Event 的数据
    finally:
        root_file.close()

    event_num, du_ids_all, gps_time, du_nanoseconds = data["event_number"], np.array(data["du_id"]), data["gps_time"], np.array(data["du_nanoseconds"])

    LST_time = int(np.round(gps_to_lst(gps_time[2])[0]))

    logger.info(f"Trigger Time @ LST Time (hours): {LST_time}")

    trace_1, trace_2, trace_3 = np.array(data["trace_1"]), np.array(data["trace_2"]), np.array(data["trace_3"]) # 获取三通道波形数据
    
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    x_position, y_position, z_position = [], [], []
    
    for du in du_ids_all:

        x_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 1])
        y_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 2])
        z_position.append(DU_position[np.where(DU_position[:, 0] == du)[0][0], 3])

    x_position, y_position, z_position = np.array(x_position), np.array(y_position), np.array(z_position)

    DU_positions = np.stack((x_position, y_position, z_position), axis = 1) # 触发的 DU 的位置矩阵 (N x 3)
    DU_positions_rotated = np.dot(_DU_ROTATION_MATRIX, DU_positions.T).T # 旋转坐标系以适应 GRAND 的站点布局

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    detector_positions_for_causal = {} # 构造 detector_positions 字典以满足函数的输入要求 (使用你当前已经提取过的坐标)
    
    for j, du_ in enumerate(du_ids_all): detector_positions_for_causal[str(du_)] = np.array([x_position[j], y_position[j], z_position[j]]) # 原函数内部还会自己调用一次 rotate_and_shift, 为了不发生两次旋转，我们直接传原始的未经旋转偏置的硬坐标
    

    temp_causal_file = os.path.join(efield_rec_dir, f'temp_causality_event_{event_num}.txt') # 生成一个满足 parser 语法的纯文本行并写入临时文件
    
    
    match_str_list = [f"('{du_ids_all[j]}', {int(du_nanoseconds[j])})" for j in range(len(du_ids_all))] # 将 DU_id 和 nanosecond 时间戳组装成 '[('56', 15632890), ('103', 15639100), ...]' 
    
    match_str = ", ".join(match_str_list)
    
    line_to_write = f"{gps_time[2]}: [{match_str}]"  # gps_time[2] 用作标识
    
    with open(temp_causal_file, 'w') as tf: tf.write(line_to_write + '\n')

    
    logger.info("Executing strict causality cleaning via graph optimization...")  # 调用引用的图论清理函数（设定最少容忍 4 个探测器）
    
    matching_times = optimized_read_matching_times_graph(
        file_path = temp_causal_file, 
        detector_positions = detector_positions_for_causal, 
        min_detectors = 4)

    if os.path.exists(temp_causal_file): os.remove(temp_causal_file) # 删除临时缓存文件

    
    if len(matching_times) == 0: # 分析返回列表，应用掩码
        
        logger.warning(f"Event {event_num} failed strict causality check completely. Skipping.")
        
        return -1
        
    
    valid_du_list = matching_times[0][1]
    
    valid_du_ids = np.array([int(item[0]) for item in valid_du_list]) # matching_times 结构为: [(gps_time, [('du_id', ns), ('du_id', ns)])]
    

    causal_mask = np.isin(du_ids_all, valid_du_ids) # 使用保留合规 DU 的掩码更新所有矩阵
    
    triggered_du_ids, du_nanoseconds_filtered = du_ids_all[causal_mask], du_nanoseconds[causal_mask] # 找出原阵列里保留下来的天线的索引
    
    trace_1_filtered, trace_2_filtered, trace_3_filtered = trace_1[causal_mask], trace_2[causal_mask], trace_3[causal_mask]
    
    DU_positions_rotated = DU_positions_rotated[causal_mask]
    
    n_triggered = len(triggered_du_ids)
    
    if n_triggered < 4:
        logger.warning(f"Event {event_num} has only {n_triggered} DUs after causality filter (< 4 minimum). Skipping.")
        return -1
    
    logger.info(f"Passed strict causality check: {n_triggered} DUs retained: {triggered_du_ids}")

    #---------------------------------------------------------------------------

    rec_theta_plane, rec_phi_plane, chi2Plane = np.inf, np.inf, np.inf # 平面波前拟合 (PWF) 寻找最佳角度

    for theta0 in thetaList:
        for phi0 in phiList:
            theta, phi, fval = PWF_fit(DU_positions_rotated[:, 0], DU_positions_rotated[:, 1], DU_positions_rotated[:, 2], du_nanoseconds_filtered * 1e-9, theta0, phi0)
            if fval < chi2Plane:
                chi2Plane, rec_theta_plane, rec_phi_plane = fval / (len(triggered_du_ids) - 1), theta, phi
            
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
    plt.savefig(os.path.join(event_dir, f'event_{event_num}_Triggered_DUs_Position.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    # 列: event_num, gps_time, LST, du_id, theta_plane, phi_plane, chi2Plane,
    #      du_ns, pos_x, pos_y, pos_z, Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi
    N_SAVED_COLS = 15
    saved_data = np.zeros([n_triggered, N_SAVED_COLS])
    E_rec_time_all_DUs = [] # 初始化一个列表来保存每个 DU 的重建电场时域数据

    for i in range(n_triggered):
        
        #-----------------------------------------------------------------------

        N_time = len(trace_1_filtered[i]) # 动态获取长度, 支持 1024 或 512.
        
        ADC_single_du_time = np.array([trace_1_filtered[i], trace_2_filtered[i], trace_3_filtered[i]]).T # 形状 (N_time, 3)，每列对应一个通道的时域数据

        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------

        ADC_single_du_rfft = np.fft.rfft(ADC_single_du_time, axis = 0) # 对每个通道进行实数快速傅里叶变换，得到频域数据，形状为 (N_time/2 + 1, 3)
        
        freqs_real_MHz = np.fft.rfftfreq(N_time, d = 2e-9) / 1e6 # 计算对应的频率点, 单位 MHz, 长度为 N_time/2 + 1

        ADC_single_du_rfft[freqs_real_MHz > 120, :] = 0 # 频域滤波: 将 120 MHz 以上的频点置零, 根据实际情况调整截止频率

        ADC_single_du_time_filtered = np.fft.irfft(ADC_single_du_rfft, n = N_time, axis = 0) # 对滤波后的频域数据进行逆变换，得到滤波后的时域数据，形状为 (N_time, 3)


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
        plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_ADC_Traces.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #-----------------------------------------------------------------------
        
        freqs_1MHz = np.arange(30, 251, 1) # 长度 221, 对应于 vad2voc.py 的频点要求.
        
        band_mask = (freqs_real_MHz >= 30) & (freqs_real_MHz <= 250) # 频带掩码, 选取有效频率范围内的频点
        
        freqs_band = freqs_real_MHz[band_mask] # 选取有效频率范围内的频点, 长度根据实际数据而定, 满足 vad2voc 的输入要求

        
        dummy_ADC_1MHz = np.ones((len(freqs_1MHz), 3), dtype = complex) # 制造一个振幅为1的“白板探针”信号

        H_vad2voc_1MHz = vad2voc(dummy_ADC_1MHz) # 获取 vad2voc 在 1 MHz 频点上的系统的固有频域复数传递函数 H(f)

        H_real_freq = np.zeros((len(freqs_band), 3), dtype = complex)
        
        ADC_1MHz_interp = np.zeros((len(freqs_1MHz), 3), dtype = complex) # 为旧绘图代码提供占位

        for ch in range(3):
            # 对 vad2voc 的自身传递函数进行极坐标插值（S参数非常平滑，插值无损）
            amp_H, phase_H = np.abs(H_vad2voc_1MHz[:, ch]), np.unwrap(np.angle(H_vad2voc_1MHz[:, ch])) # 获取振幅和相位，注意相位需要 unwrap 以避免跳变
            interp_H_amp = interp1d(freqs_1MHz, amp_H, kind = 'cubic', fill_value = 'extrapolate')(freqs_band)
            interp_H_phase = interp1d(freqs_1MHz, phase_H, kind = 'cubic', fill_value = 'extrapolate')(freqs_band)
            H_real_freq[:, ch] = interp_H_amp * np.exp(1j * interp_H_phase)
            
            
            amp, phase = np.abs(ADC_single_du_rfft[:, ch]), np.unwrap(np.angle(ADC_single_du_rfft[:, ch])) # 直接对原始频域数据进行极坐标插值，得到在 1 MHz 网格上的振幅和相位插值结果
            interp_amp = interp1d(freqs_real_MHz, amp, kind = 'cubic', fill_value = 'extrapolate')
            interp_phase = interp1d(freqs_real_MHz, phase, kind = 'cubic', fill_value = 'extrapolate')
            ADC_1MHz_interp[:, ch] = interp_amp(freqs_1MHz) * np.exp(1j * interp_phase(freqs_1MHz))


        plt.figure(figsize=(10, 7)) # 绘制频域插值检查图，比较原始频域数据和插值结果
        plt.plot(freqs_real_MHz[band_mask], np.abs(ADC_single_du_rfft[band_mask, 0]), 'r-', alpha=0.5, label='X Original')
        plt.plot(freqs_real_MHz[band_mask], np.abs(ADC_single_du_rfft[band_mask, 1]), 'g-', alpha=0.5, label='Y Original')
        plt.plot(freqs_real_MHz[band_mask], np.abs(ADC_single_du_rfft[band_mask, 2]), 'b-', alpha=0.5, label='Z Original')
        plt.plot(freqs_1MHz, np.abs(ADC_1MHz_interp[:, 0]), 'k--', label='X Interpolated')
        plt.plot(freqs_1MHz, np.abs(ADC_1MHz_interp[:, 1]), 'm--', label='Y Interpolated')
        plt.plot(freqs_1MHz, np.abs(ADC_1MHz_interp[:, 2]), 'c--', label='Z Interpolated')
        plt.title(f"DU {triggered_du_ids[i]}: ADC FFT Interpolation Check", fontsize = 18)
        plt.xlabel("Frequency [MHz]", fontsize = 16)
        plt.ylabel("ADC", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_ADC_Interpolation_Comparison.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #---------------------------------------------------------------------------------------

        Voc_1MHz = vad2voc(ADC_1MHz_interp) # 

        # window_1mhz = windows.tukey(len(freqs_1MHz)) # 处理频谱泄露的高精度窗, 长度与 freqs_1MHz 相同, 用于保护插值后的频域数据在时域转换中的相位信息
        window_1mhz = np.pad(windows.tukey(len(freqs_1MHz) - 4, alpha = 0.5), (2, 2), mode = 'constant')

        for ch in range(3): Voc_1MHz[:, ch] = Voc_1MHz[:, ch] * window_1mhz # 直接对插值后的频域数据乘以窗函数, 保护时域转换中的相位信息, 避免时域波形的泄露和失真

        ADC_band = ADC_single_du_rfft[band_mask, :] # 获取高精度的原始信号有效频带 (完全不经过任何降采样插值)

        Voc_rfft_band = np.zeros_like(ADC_band, dtype = complex)
        
        # window_real = windows.tukey(len(freqs_band)) # 处理频谱泄露的高精度窗
        window_real = np.pad(windows.tukey(len(freqs_band) - 8, alpha = 0.5), (4, 4), mode = 'constant')
        
        for ch in range(3): Voc_rfft_band[:, ch] = ADC_band[:, ch] * H_real_freq[:, ch] * window_real # 直接执行频域相乘: 原生信号 * 插值后的高精度硬件响应 * 窗函数 ！ (完美保护时间相位)


        Voc_full_rfft = np.zeros_like(ADC_single_du_rfft, dtype = complex) 
        
        Voc_full_rfft[band_mask, :] = Voc_rfft_band # 拼装回完整的 RFFT 数组中进行 IRFFT
        
        Voc_time = np.fft.irfft(Voc_full_rfft, n = N_time, axis = 0) * (109.86) # units of uV.
        # if i == 0: print(len(Voc_time), N_time, np.argmax(np.abs(ADC_single_du_time[:, 1])), np.argmax(np.abs(Voc_time[:, 1])), flush = True)

        
        plt.figure(figsize=(10, 7))
        plt.plot(freqs_1MHz, np.abs(Voc_1MHz[:, 0]), color = 'red', linewidth = 1.0, label = 'Port X')
        plt.plot(freqs_1MHz, np.abs(Voc_1MHz[:, 1]), color = 'green', linewidth = 1.0, label = 'Port Y')
        plt.plot(freqs_1MHz, np.abs(Voc_1MHz[:, 2]), color = 'blue', linewidth = 1.0, label = 'Port Z')
        plt.plot(freqs_band, np.abs(Voc_rfft_band[:, 0]), 'r--')
        plt.plot(freqs_band, np.abs(Voc_rfft_band[:, 1]), 'g--')
        plt.plot(freqs_band, np.abs(Voc_rfft_band[:, 2]), 'b--')
        plt.title(f"DU {triggered_du_ids[i]}: Voc Frequency Domain (After Interpolation)", fontsize = 18)
        plt.xlabel("Frequency [MHz]", fontsize = 16)
        plt.ylabel("ADC", fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
        plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
        for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
        plt.tick_params(axis='x', pad=6)
        plt.tick_params(axis='y', pad=6)
        plt.legend(fontsize = 14, loc = 'upper right')
        plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Frequency_Comparison.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()


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
        plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Time_Domain.png'), dpi = 300, bbox_inches = 'tight')
        plt.close()

        #----------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------

        vx, vy, vz = Voc_time.T # 分离三个通道的 Voc 时域数据

        vx_std, vy_std, vz_std = np.std(Voc_time[int(len(Voc_time) * 0.75) : , :], axis = 0) # 计算噪声标准差
        stds_V = np.array([vx_std, vy_std, vz_std])
        
        peak_pos = np.argmax(abs(Voc_time), axis = 0) # 获取每个通道的峰值位置索引, 以便后续加窗使用

        peaks_V = np.max(abs(Voc_time), axis = 0) # 获取每个通道的峰值电压, 用于信号筛选和加窗判断
        
        # print(f"DU {triggered_du_ids[i]}: Peak Voltages of Voc = {peaks_V}, Peak positions of Voc = {peak_pos}, Noise Stds = {stds_V}", flush = True)
        
        #----------------------------------------------------------------------------------

        trig_cmp, pos_sel = [], []
        
        for cmp in range(3): 
            if peaks_V[cmp] / stds_V[cmp] > 5: # 寻找触发通道, 大于5倍噪声标准差的峰值
                pos_sel.append(peak_pos[cmp]), trig_cmp.append(cmp) # 记录满足条件的通道索引和对应的峰值位置索引

        if len(pos_sel) > 0: pos_diff, pos_sel = max(pos_sel) - min(pos_sel), np.array(pos_sel) # 计算峰值位置的最大差异
            
        else: pos_diff, pos_sel = 0, np.array([np.argmax(np.abs(ADC_single_du_time[:, 1]))]) # 如果没有任何通道满足条件, 以 Y 通道的峰值位置作为默认值, 并且 pos_diff 设为0以避免后续条件判断出错
            
        # print("Trigged channels:", trig_cmp, "Peak position difference:", pos_diff, flush = True)

        #----------------------------------------------------------------------------------
        
        Voc_time_windows = np.zeros(Voc_time.shape) # 初始化加窗后的时域 Voc 数组, 形状与原始 Voc_time 相同
        
        '''
        for cmp in range(3): # 加窗, 以满足条件的通道为主, 以 Tukey 窗保护峰值附近的 200 个采样点 (±100)，其他部分置零, 提供极干净的边界供物理方程匹配
            
            # if peaks_V[1] / stds_V[1] > 5: # 以 Y 通道为基准进行加窗判断
            if peaks_V[np.argmax(peaks_V)] / stds_V[np.argmax(peaks_V)] > 5: # 以峰值电压与噪声标准差的比值最大的通道为基准进行加窗判断, 以适应不同通道可能存在的信号强度差异

                if peaks_V[cmp] / stds_V[cmp] > 5: 
                    
                    Voc_time_windows[peak_pos[cmp] - 100 : peak_pos[cmp] + 100, cmp] = Voc_time[peak_pos[cmp] - 100 : peak_pos[cmp] + 100, cmp] * tukey_wind
            
                    # Voc_time_windows[peak_pos[1] - 100 : peak_pos[1] + 100, cmp] = Voc_time[peak_pos[1] - 100 : peak_pos[1] + 100, cmp] * tukey_wind

                else: Voc_time_windows[peak_pos[np.argmax(peaks_V)] - 100 : peak_pos[np.argmax(peaks_V)] + 100, cmp] = Voc_time[peak_pos[np.argmax(peaks_V)] - 100 : peak_pos[np.argmax(peaks_V)] + 100, cmp] * tukey_wind
        
            else:

                Voc_time_windows[np.argmax(np.abs(ADC_single_du_time[:, np.argmax(peaks_V)])) - 100 : np.argmax(np.abs(ADC_single_du_time[:, np.argmax(peaks_V)])) + 100, cmp] = Voc_time[np.argmax(np.abs(ADC_single_du_time[:, np.argmax(peaks_V)])) - 100 : np.argmax(np.abs(ADC_single_du_time[:, np.argmax(peaks_V)])) + 100, cmp] * tukey_wind
        '''
            
        for cmp in range(3): 
            
            best_ch = np.argmax(peaks_V) # 确定峰值的中心基准点。若主通道达到置信度，则依据各通道自身或主通道，否则回归 ADC
            
            if peaks_V[best_ch] / stds_V[best_ch] > 5:
                
                if peaks_V[cmp] / stds_V[cmp] > 5: center_idx = peak_pos[cmp]
                
                else: center_idx = peak_pos[best_ch]
            
            else: # center_idx = np.argmax(np.abs(ADC_single_du_time))

                center_idx = np.argmax(np.max(np.abs(ADC_single_du_time[0 : 300, :]), axis = 1))
            
            
            left_ideal, right_ideal = center_idx - 100, center_idx + 100 # 计算理想窗口游标
            
            left_actual, right_actual = max(0, left_ideal), min(len(Voc_time), right_ideal) # 计算实际能切出的合法边界 (限制在 0 到 数组最大长度 之间)
            
            
            crop_start, crop_end = left_actual - left_ideal, 200 - (right_ideal - right_actual) # 判断合法边界到理想边界失去了多少点，从而裁剪对应的 Tukey 窗并赋值
            
            if right_actual > left_actual: # 确保存在有效窗口切片
                
                Voc_time_windows[left_actual : right_actual, cmp] = Voc_time[left_actual : right_actual, cmp] * tukey_wind[crop_start : crop_end]

        Voc_fft_windows = np.fft.rfft(Voc_time_windows, axis = 0) # 将加窗后的时域信号重新变回频域, 得到无瑕的 rfft 形式，提供极干净的边界供物理方程匹配

        # Voc_fft_windows = np.fft.rfft(Voc_time, axis = 0)

        #----------------------------------------------------------------------------------

        Voc_fft_band_interp = np.zeros((300, 3), dtype = np.complex128) # 插值回频段网络供原版重建方程读取

        for ch in range(3): # 此时信号是加过窗的, 直接把 fft 做插值给 30~250 MHz.
            amp_ch, phase_ch = np.abs(Voc_fft_windows[:, ch]), np.unwrap(np.angle(Voc_fft_windows[:, ch])) # 获取加窗后频域数据的振幅和相位
            interp_amp_ch = interp1d(freqs_real_MHz, amp_ch, kind = 'cubic', fill_value = 'extrapolate') # 振幅插值到 1 MHz 网格
            interp_phase_ch = interp1d(freqs_real_MHz, phase_ch, kind = 'cubic', fill_value = 'extrapolate') # 相位插值到 1 MHz 网格
            Voc_fft_band_interp[30 : 251, ch] = interp_amp_ch(freqs_1MHz) * np.exp(1j * interp_phase_ch(freqs_1MHz)) # 直接在 1 MHz 网格上构造插值后的频域数据，供原版重建方程读取


        Lce_sphere = get_lec_grand(rec_theta_plane, rec_phi_plane, N, f0) # 获取对应方向的天线有效长度, Lce 不需要插值了，我们直接传原本 2000 个频点的 Lce_sphere 即可，内部会读取 [30:251]
        # Lce_sphere = get_lec_grand(80.5, 185.9, N, f0)

        [galactic_v_complex_double, galactic_v_time] = gala(LST_time, N, f0, f1, 0) # 获取银河噪声, 这里的参数需要根据实际情况调整, LST = 18 是一个示例值, Vnoise 也不需要插值，直接传 galactic_v_complex_double ，内部会读取
        # print("Galactic noise complex spectrum:", galactic_v_complex_double[30 : 251], flush = True)
        

        Et_rec, Ep_rec, E_inv = e_recons(Voc_fft_band_interp, Lce_sphere, galactic_v_complex_double)

        # print(f"Reconstructed E-field (Et, Ep) at 30 MHz: {Et_rec[0]:.2e} uV/m, {Ep_rec[0]:.2e} uV/m", flush = True)


        if False:
            plt.figure(figsize=(10, 7)) # 绘制重建电场的频域图
            plt.plot(np.arange(flower, fupper), np.abs(Et_rec), 'r-', label = 'Theta component')
            plt.plot(np.arange(flower, fupper), np.abs(Ep_rec), 'b-', label = 'Phi component')
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
            plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Efield_Reconstruction_Frequency.png'), dpi = 300, bbox_inches = 'tight')
            plt.close()

        #------------------------------------------------------------------------------------------------------------------

        E_rec_complex_band = np.zeros((300, 2), dtype = complex)
    
        E_rec_complex_band[flower : fupper, 0], E_rec_complex_band[flower : fupper, 1] = Et_rec, Ep_rec

        E_rec_time = np.fft.irfft(E_rec_complex_band, n = 500, axis = 0) # 直接对重建的频域电场进行 IRFFT，得到时域电场波形，供后续物理方程正演使用
        # Et_rec, Ep_rec, E_rec_complex_band 频域下都为 1 MHz间隔, 逆傅立叶变换后 500 个时间点, 间隔为 2 ns.

        E_rec_amp = np.sqrt(E_rec_time[:, 0] ** 2 + E_rec_time[:, 1] ** 2) # 计算重建电场的振幅，供后续物理方程正演使用

        # Fluence = np.sum(E_rec_amp[np.argmax(E_rec_amp) -50 : np.argmax(E_rec_amp) + 50] ** 2) # 计算重建电场的 fluence，供后续物理方程正演使用
        Fluence, Fluence_theta, Fluence_phi = np.sum(E_rec_amp ** 2), np.sum(E_rec_time[:, 0] ** 2), np.sum(E_rec_time[:, 1] ** 2) # 计算重建电场的 fluence，供后续物理方程正演使用, 同时分离 theta 和 phi 分量的 fluence 以供分析
        Fluence_theta_phi = np.sum(E_rec_time[:, 0] * E_rec_time[:, 1]) # 计算 theta 和 phi 分量的 fluence 交叉项, 用于分析两者之间的相关性

        Fluence *= (6.242e18 * 1e-12 * 2e-9 / 377) # 根据实际情况调整能量因子, 以使得 fluence 的单位为 eV/m^2
        Fluence_theta *= (6.242e18 * 1e-12 * 2e-9 / 377)
        Fluence_phi *= (6.242e18 * 1e-12 * 2e-9 / 377)
        Fluence_theta_phi *= (6.242e18 * 1e-12 * 2e-9 / 377)

        if False:
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
            plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Efield_Reconstruction_Time.png'), dpi = 300, bbox_inches = 'tight')
            plt.close()

        #------------------------------------------------------------------------------------------------------------------
        
        Voc_recalc_complex_band = np.zeros((300, 3), dtype = complex) # 初始化输出电压的频域容器
        
        for ff in range(flower, fupper):
            for p in range(3):
                Voc_recalc_complex_band[ff, p] = (Lce_sphere[ff, 0, p] * E_rec_complex_band[ff, 0] + Lce_sphere[ff, 1, p] * E_rec_complex_band[ff, 1] + galactic_v_complex_double[ff, p]) # 直接在频域下根据物理方程重建 Voc，注意这里的 galactic noise 也是频域表示, 直接乘以对应的 Lce 分量后叠加到 Voc 中, 完美保护时间相位

        Voc_recalc_time = np.fft.irfft(Voc_recalc_complex_band, n = 500, axis = 0) # 将重新计算的频域电压进行 IRFFT，得到时域波形，供与原始 Voc 的时域波形进行对比

        if False:
            plt.figure(figsize=(10, 4)) # 绘制重新计算的 Voc 时域波形图
            plt.plot(Voc_recalc_time[:, 0], color = 'red', linewidth = 1.0, label = 'Port X')
            plt.plot(Voc_recalc_time[:, 1], color = 'green', linewidth = 1.0, label = 'Port Y')
            plt.plot(Voc_recalc_time[:, 2], color = 'blue', linewidth = 1.0, label = 'Port Z')
            plt.title(f"DU {triggered_du_ids[i]}: Recalculated Voc from Reconstructed E-field (Time Domain)", fontsize = 18)
            plt.xlabel("Time index (2 ns interval)", fontsize = 16)
            plt.ylabel("Voltage [uV]", fontsize = 16)
            plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
            plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
            for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
            for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
            plt.tick_params(axis='x', pad=6)
            plt.tick_params(axis='y', pad=6)
            plt.legend(fontsize = 14, loc = 'upper right')
            plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Recalculated_Time.png'), dpi = 300, bbox_inches = 'tight')
            plt.close()

        #---------------------------------------------------------------------------------

        if False:
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
                ax.set_ylim(1.1 * np.min(Voc_time[100 : 200, :]), 1.1 * np.max(Voc_time[100 : 200, :])) # 根据原始 Voc 的范围动态调整 y 轴范围
                if ch == 0: 
                    ax.legend(loc='upper left', fontsize = 15)
                    ax.set_xlim(100, 200) # 只显示峰值附近的部分时间窗口以便对比
                    ax.set_title('Voc Consistency Check', fontsize = 18)
    
            axs[-1].set_xlabel('Time index (2 ns interval)', fontsize = 18)
            plt.savefig(os.path.join(event_dir, f'event_{event_num}_DU_{triggered_du_ids[i]}_Voc_Consistency_Check.png'), dpi = 300, bbox_inches = 'tight')
            plt.close()

        #-------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------

        saved_data[i] = [event_num, gps_time[3 * i + 2], LST_time, triggered_du_ids[i], rec_theta_plane, rec_phi_plane, chi2Plane, du_nanoseconds_filtered[i], DU_positions_rotated[i, 0], DU_positions_rotated[i, 1], DU_positions_rotated[i, 2], Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi] # 保存 DU ID, 三通道峰值电压, 和重建电场的 fluence 到数组中

        E_rec_time_all_DUs.append(E_rec_time.copy()) # 将每个触发 DU 的重建电场时域波形保存到列表中, 供后续分析使用

    #---------------------------------------------------------------------------

    np.savetxt(os.path.join(event_dir, f'event_{event_num}_reconstruction_results.txt'), saved_data) # 将结果保存为 文本文件, 包含每个触发 DU 的 ID, 到达时间, 位置坐标, 和重建电场的 fluence
    
    #---------------------------------------------------------------------------

    E_rec_time_all_DUs = np.array(E_rec_time_all_DUs) # 将列表转换为 numpy 数组, 形状为 (触发 DU 数量, 时间点数量, 2) 对应 (DU, 时间, E-field 分量)

    E_rec_time_all_DUs_2d = E_rec_time_all_DUs.reshape(E_rec_time_all_DUs.shape[0], -1) # 将 3D 数组重塑为 2D 数组, 形状为 (触发 DU 数量, 时间点数量 * E-field 分量数量)，以便保存为 CSV 文件

    np.savetxt(os.path.join(event_dir, f'event_{event_num}_E_rec_time_all_DUs.txt'), E_rec_time_all_DUs_2d, delimiter = ' ', header = 'E_theta_time_0, E_phi_time_0, E_theta_time_1, E_phi_time_1, ..., E_theta_time_N, E_phi_time_N') # 将重建电场的时域波形保存为 txt 文件, 每行对应一个触发 DU, 列对应时间点和 E-field 分量

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


def SWF_fit(x, y, z, ant_t, x0, y0, z0):
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
    m.simplex() # 先用单纯形法粗略搜索
    m.migrad()  # 再用 MIGRAD 算法精确最小化
    m.hesse()   # 计算海森矩阵 (误差估计)
    xs = m.values["xs"]
    ys = m.values["ys"]
    zs = m.values["zs"]
    fval = m.fval
    return xs, ys, zs, fval

#--------------------------------------------------------------------------

class ADF:
    """
    角分布函数 (Angular Distribution Function) 拟合类。
    
    用于定义切伦科夫辐射角分布的拟合目标函数 (chi2)。
    考虑了地磁效应修正。
    """
    def __init__(self, x, y, z, p, xs, ys, zs, obs_altitude):
        """
        初始化 ADF 类。
        
        参数:
            x, y, z (numpy.ndarray): 天线位置。
            p (numpy.ndarray): 观测到的信号幅度或能量通量。
            xs, ys, zs (float): 簇射源点 (Xmax) 坐标。
            obs_altitude (float): 观测站海拔高度 (米)，用于计算芯位和距离。
        """
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.obs_altitude = obs_altitude
        
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
        fmin = (self.p - adf) / (0.1 * self.p + 0.01 * np.max(self.p)) # 归一化残差, 避免高振幅天线主导拟合
        # fmin = self.p - adf
        chi2 = (fmin ** 2).sum()
        
        return chi2
        
    def __call__(self, theta, phi, dw, A, B, rc):
        chi2 = self.chi2(theta, phi, dw, A, B, rc)
        return chi2


def _adf_fit_impl(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, fix_dw=False):
    """ADF 拟合的内部实现。fix_dw 控制是否固定 dw 参数。"""
    m = iminuit.Minuit(
        ADF(x, y, z, ant_p, xs, ys, zs, obs_altitude),
        theta=theta0, phi=phi0, dw=dw0, A=A0, B=0.01, rc=1,
    )
    m.limits = (
        [theta0 - 0.8, theta0 + 0.8],
        [phi0 - 0.8, phi0 + 0.8],
        [0.01, 5],
        [A0 / 2, A0 * 1e3], [0, 0.01],
        [0, 1],
    )
    m.fixed = (False, False, fix_dw, False, True, True)  # B 和 rc 始终固定
    m.errordef = 1
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    theta, theta_err = m.values["theta"], m.errors["theta"]
    phi, phi_err = m.values["phi"], m.errors["phi"]
    dw = m.values["dw"]
    dw_err = 0 if fix_dw else m.errors["dw"]
    A, A_err = m.values["A"], m.errors["A"]
    fval = m.fval
    return theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err


def ADF_fit(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude):
    """执行 ADF 拟合。拟合参数: theta, phi, dw, A。固定: B, rc。"""
    return _adf_fit_impl(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, fix_dw=False)


def ADF_fit_fixdw(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude):
    """执行固定宽度的 ADF 拟合。固定: dw, B, rc。"""
    return _adf_fit_impl(x, y, z, ant_p, xs, ys, zs, theta0, phi0, dw0, A0, obs_altitude, fix_dw=True)

#--------------------------------------------------------------------------

def recons_angle(x, y, z, t, p, init_input, obs_altitude):  # time given in ns
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
    rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph = np.inf, np.inf, np.inf, np.inf
    
    for x0 in xList: # 网格搜索最佳 Xmax 位置 (SWF 拟合)
        for y0 in yList:
            for z0 in zList:
                x_xmax, y_xmax, z_xmax, fval = SWF_fit(x, y, z, t, x0, y0, z0)
                if fval < chi2Sph:
                    chi2Sph = fval / (len(x) - 1) # 归一化 chi2
                    rec_x_xmax = x_xmax
                    rec_y_xmax = y_xmax
                    rec_z_xmax = z_xmax
                    
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
                    theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err = ADF_fit(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, A0, obs_altitude)
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
                    theta, phi, dw, A, fval, theta_err, phi_err, dw_err, A_err = ADF_fit_fixdw(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, theta0, phi0, dw0, A0, obs_altitude)
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
        temp_adf = ADF(x[~np.isnan(p)], y[~np.isnan(p)], z[~np.isnan(p)], p[~np.isnan(p)], rec_x_xmax, rec_y_xmax, rec_z_xmax, obs_altitude)
        temp_adf.chi2(rec_theta, rec_phi, rec_dw, rec_A, 0.01, 1)
        wc_adf = temp_adf.wc
        logger.info(f"Fitted wc: {np.asarray(wc_adf).reshape(-1)[0]}, rec_dw: {rec_dw}")
    except Exception as e:
        logger.warning(f"Could not calculate wc for display: {e}")
    
    return zenith_ini, azimuth_ini, chi2Plane, rec_x_xmax, rec_y_xmax, rec_z_xmax, chi2Sph, rec_theta, rec_phi, rec_dw, rec_A, fmin / (len(x) - 1), np.asarray(wc_adf).reshape(-1)[0], rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err # wc_adf 兼容写法(标量/单元素数组都可)

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

    return amp, w_c, domega, chi2_min, amp_err, w_c_err, domega_err

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

def energy_restruction(root_path, event_number_int):

    logger.info(f"Starting energy reconstruction using ADF from {root_path}, event: {int(event_number_int)}")

    _, DU_all_positions_rotated = _load_du_position()

    filename = os.path.splitext(os.path.basename(root_path))[0] + f"_event_{int(event_number_int)}" # 得到文件名和事件编号
    event_dir = os.path.join(efield_rec_dir, filename)

    #-------------------------------------------------------------------------------------------------

    fluence_data = np.loadtxt(glob.glob(os.path.join(event_dir, '*reconstruction_results.txt'))[0])

    du_ids = fluence_data[:, 3].astype(int) # 从保存的重建结果中提取天线 ID 列

    rec_theta_plane, rec_phi_plane, chi2Plane = fluence_data[:, 4], fluence_data[:, 5], fluence_data[:, 6] # 从保存的重建结果中提取平面波重建的角度和 chi2 列

    nanosecond_time = fluence_data[:, 7] * 1e-9 # 从保存的重建结果中提取纳秒时间列

    positions_x, positions_y, positions_z = fluence_data[:, 8], fluence_data[:, 9], fluence_data[:, 10] # 从保存的重建结果中提取位置坐标列

    positions_all = np.array([positions_x, positions_y, positions_z]).T

    Fluence, Fluence_theta, Fluence_phi = fluence_data[:, 11], fluence_data[:, 12], fluence_data[:, 13] # 从保存的重建结果中提取 fluence 列
    Fluence_theta_phi = fluence_data[:, 14]

    obs_altitude = np.mean(positions_z) # 使用天线位置的平均高度作为观测高度

    #-------------------------------------------------------------------------------------------------

    E_rec_time_all_DUs_data = np.loadtxt(glob.glob(os.path.join(event_dir, '*E_rec_time_all_DUs.txt'))[0])

    E_rec_time_sph = E_rec_time_all_DUs_data.reshape(E_rec_time_all_DUs_data.shape[0], 500, 2) # 

    #-------------------------------------------------------------------------------------------------

    init_input = [rec_theta_plane[0], rec_phi_plane[0], chi2Plane[0]] # 使用平面波重建结果作为 ADF 重建的初始输入

    rec_theta_plane, rec_phi_plane, chi2Plane, rec_x_xmax, rec_y_xmax, rec_z_xmax, \
    chi2Sph, rec_theta_sph, rec_phi_sph, rec_dw, rec_A, fmin, rec_wc, \
    rec_theta_err, rec_phi_err, rec_dw_err, rec_A_err = \
    recons_angle(positions_x, positions_y, positions_z, nanosecond_time, Fluence, init_input, obs_altitude)


    logger.info(f"Reconstructed Xmax coordinates (m) and chi2: {rec_x_xmax} {rec_y_xmax} {rec_z_xmax} {chi2Sph}")

    logger.info(f"Reconstructed shower direction (theta, phi in degrees and errors): {rec_theta_sph} {rec_theta_err} {rec_phi_sph} {rec_phi_err}")

    logger.info(f"Reconstructed ADF parameters (A, wc, dw, fmin/DOF): {rec_A} {rec_wc} {rec_dw} {fmin}")

    logger.info(f"Reconstructed ADF parameter errors (A_err, wc_err, dw_err): {rec_A_err} 0 {rec_dw_err}")

    # 返回结构化 dict 供上游函数直接使用（同时保持 print 兼容性）
    energy_result = dict(
        rec_x_xmax=rec_x_xmax, rec_y_xmax=rec_y_xmax, rec_z_xmax=rec_z_xmax, chi2Sph=chi2Sph,
        rec_theta_sph=rec_theta_sph, rec_theta_err=rec_theta_err,
        rec_phi_sph=rec_phi_sph, rec_phi_err=rec_phi_err,
        rec_A=rec_A, rec_wc=rec_wc, rec_dw=rec_dw, fmin_per_dof=fmin,
        rec_A_err=rec_A_err, rec_dw_err=rec_dw_err,
    )

    #-------------------------------------------------------------------------------------------------

    height = helper.get_local_altitude([rec_x_xmax, rec_y_xmax, rec_z_xmax + 1280]) # 计算重建的 Xmax 坐标对应的海拔高度
    logger.info(f"Local altitude at reconstructed Xmax (m): {height}")

    #-------------------------------------------------------------------------------------------------

    rho = atm.get_density(height, model = 1) * 1e-3 # 使用重建的 Xmax 高度计算大气密度
    logger.info(f"Reconstructed Xmax altitude: {height}, rho at reconstructed Xmax (g/cm^3): {rho}")

    #-------------------------------------------------------------------------------------------------

    shower_axis = helper.spherical_to_cartesian(np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph))
    logger.info(f"Shower axis unit vector: {shower_axis}")

            
    alpha = np.arccos(np.dot(shower_axis, magnetic_field_vector) / (np.linalg.norm(shower_axis) * np.linalg.norm(magnetic_field_vector)))
    logger.info(f"alpha: {alpha * 180 / np.pi}")

    #-------------------------------------------------------------------------------------------------

    corex, corey, corez, dXmax = find_core(rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta_sph, rec_phi_sph, obs_altitude)
    shower_core = np.array([corex, corey, corez])
    logger.info(f"Shower core: {shower_core}, dXmax: {dXmax}")


    #-------------------------------------------------------------------------------------------------

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
    cbar = plt.scatter(-1 * positions_y, positions_x, c = Fluence, s = 100 + ((Fluence - np.min(Fluence)) / (np.max(Fluence) - np.min(Fluence))) * 300, edgecolors = 'k', cmap = 'viridis')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(r'Fluence$\rm [eV/m^2]$', fontsize = 16)
    for i in range(len(positions_x)): plt.text(-1 * positions_y[i], positions_x[i], str(int(du_ids[i])), fontsize = 10, ha = 'left', va = 'bottom', color = 'black')
    contours = plt.contour(XX_plot, YY_plot, omega_grid, levels = [rec_wc], colors = 'teal', alpha = 0.7, linewidths = 2, linestyles = '--')
    plt.arrow(-1 * corey, float(corex), -500 * np.sin(np.deg2rad(rec_theta_sph)) * np.sin(np.deg2rad(rec_phi_sph)), 500 * np.sin(np.deg2rad(rec_theta_sph)) * np.cos(np.deg2rad(rec_phi_sph)), head_width = 50, head_length = 100, fc = 'red', ec = 'red')
    plt.xlabel('West-East [m]', fontsize = 16)
    plt.ylabel('South-North [m]', fontsize = 16)
    plt.title('Shower Core Position', fontsize = 16)
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
    plt.savefig(os.path.join(event_dir, 'shower_core_position.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()

    #-------------------------------------------------------------------------------------------------

    cs = coordinatesystems.cstrafo(np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph), magnetic_field_vector)
    
    fluences_sph = np.array([Fluence_theta, Fluence_phi, np.zeros(len(Fluence_phi))]).T
    
    R_sph2cart = rotation_m(rec_theta_sph * np.pi / 180, rec_phi_sph * np.pi / 180)
    

    #------------------------

    # e_theta_sph, e_phi_sph = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]) # 球坐标系下的单位矢量 (theta, phi)

    # e_theta_cart, e_phi_cart = np.dot(e_theta_sph, R_sph2cart.T), np.dot(e_phi_sph, R_sph2cart.T) # 转换到笛卡尔坐标系下的单位矢量

    # e_theta_shower, e_phi_shower = cs.transform_to_vxB_vxvxB(e_theta_cart), cs.transform_to_vxB_vxvxB(e_phi_cart) # 转换到 shower 坐标系下的单位矢量

    # cos_alpha, sin_alpha = e_theta_shower[0], e_phi_shower[0] # 

    '''
    vec_theta, vec_phi = R_sph2cart[:, 0], R_sph2cart[:, 1]

    p_theta = cs.transform_to_vxB_vxvxB(vec_theta, core = np.zeros(3))
    p_phi = cs.transform_to_vxB_vxvxB(vec_phi, core = np.zeros(3))
    '''
    
    # f_vxB_pure =  Fluence_theta * (cos_alpha ** 2) + Fluence_phi * (sin_alpha ** 2) + 2 * Fluence_theta_phi * cos_alpha * sin_alpha
    
    # f_vxvxB_pure = Fluence_theta * (sin_alpha ** 2) + Fluence_phi * (cos_alpha ** 2) - 2 * Fluence_theta_phi * cos_alpha * sin_alpha
    
    #------------------------

    
    fluences_cart = np.dot(fluences_sph, R_sph2cart.T)
    
    fluence_vxB = np.array([cs.transform_to_vxB_vxvxB(fluence) for fluence in fluences_cart])
    fluence_vxB = abs(fluence_vxB)
    
    #-------------------------------------------------------------------------------------------------

    pos_vxB = cs.transform_to_vxB_vxvxB(positions_all, shower_core)

    #----------------------------
    '''
    pos_xmax = cs.transform_to_vxB_vxvxB(np.array([[rec_x_xmax, rec_y_xmax, rec_z_xmax]]), shower_core)

    obs_vxB = np.array([pos_xmax[0] - pos_vxB[:, 0], pos_xmax[1] - pos_vxB[:, 1], pos_xmax[2] - pos_vxB[:, 2]]).T

    l_vxB = np.linalg.norm(obs_vxB, axis = 1)

    w_vxB = np.arcsin(np.sqrt(obs_vxB[:, 0] ** 2 + obs_vxB[:, 1] ** 2) / l_vxB) # 计算每个天线在 vxB 坐标系下的视角 omega_vxB

    print("Viewing angles in vxB coordinates (degrees):", w_vxB * 180 / np.pi, flush = True)
    '''
    #----------------------------

    #-------------------------------------------------------------------------------------------------
    
    ant2Xmax_dis = ((positions_x - rec_x_xmax) ** 2 + (positions_y - rec_y_xmax) ** 2 + (positions_z - rec_z_xmax) ** 2) ** 0.5
    early_late = np.square(ant2Xmax_dis / dXmax)

    # print("Early-Late correction factors:", early_late, flush = True)
    

    # f_vxB, f_vxvxB = f_vxB_pure * early_late, f_vxvxB_pure * early_late
    # f_vxB, f_vxvxB = fluence_vxB[:, 0] * early_late, fluence_vxB[:, 1] * early_late
    phi = helper.get_normalized_angle(np.arctan2(pos_vxB[:, 1], pos_vxB[:, 0]))
    vxB_axis = np.abs(np.sin(phi)) < np.abs(np.sin(10 * np.pi / 180)) # 定义一个小范围来判断是否接近 vxB 轴线, 避免数值问题
    # f_tot = np.sum(fluence_vxB, axis = -1) * early_late
    # f_tot = f_vxB + f_vxvxB

    # print("phi (degrees):", phi * 180 / np.pi, flush = True)
    # print("f_vxB:", f_vxB, flush = True)
    # print("f_vxvxB:", f_vxvxB, flush = True)
    # print("Total fluence (f_tot):", f_tot, flush = True)


    #-------------------------------------------------------------------------------------------------

    f_geo_pos = np.zeros(len(positions_x))

    for ant in range(len(positions_x)):

        E_sph_t = np.zeros((E_rec_time_sph.shape[1], 3))
        E_sph_t[:, 0], E_sph_t[:, 1] = E_rec_time_sph[ant, :, 0], E_rec_time_sph[ant, :, 1]

        E_cart_t = np.dot(E_sph_t, R_sph2cart.T)

        E_shower_t = cs.transform_to_vxB_vxvxB(E_cart_t.T).T

        E_vxB_t, E_vxvxB_t = E_shower_t[:, 0], E_shower_t[:, 1]

        phi_iter = phi[ant]
        denom = np.abs(np.sin(phi_iter)) if np.abs(np.sin(phi_iter)) >= 1e-4 else 1e-4 * np.sign(np.sin(phi_iter)) + 1e-12 # 避免除以零
        cot_phi = np.cos(phi_iter) / denom

        E_geo_t = E_vxB_t - E_vxvxB_t * cot_phi

        f_geo_pos[ant] = np.sum(E_geo_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377) # 这里的地磁修正公式在 vxB_axis 区域会出现除以 sin(phi) 的问题，使用 cot_phi 来避免，同时整个表达式在 vxB_axis 区域被设置为 0

        # print(np.sum(E_vxB_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377), np.sum(E_vxvxB_t ** 2) * early_late[ant] * (6.242e18 * 1e-12 * 2e-9 / 377), flush = True)

    #-------------------------------------------------------------------------------------------------

    # f_geo_pos = np.zeros_like(f_vxB)
    
    # f_geo_pos[~vxB_axis] = np.square(np.sqrt(f_vxB[~vxB_axis]) - (np.cos(phi[~vxB_axis]) / np.abs(np.sin(phi[~vxB_axis]))) * np.sqrt(f_vxvxB[~vxB_axis]))
    # f_geo_pos = (lambda m: np.where(m, np.square(np.sqrt(np.clip(f_vxB, 0.0, None)) - (np.cos(phi) / np.maximum(np.abs(np.sin(phi)), 1e-6)) * np.sqrt(np.clip(f_vxvxB, 0.0, None))), 0.0))(~vxB_axis)
    # 这里的地磁修正公式在 vxB_axis 区域会出现除以 sin(phi) 的问题，使用 np.maximum 来避免除以零，同时 np.clip 来确保 sqrt 的输入非负，整个表达式在 vxB_axis 区域被设置为 0

    # print("Geometric correction factors (f_geo_pos):", f_geo_pos, flush = True)
    # print(phi, flush = True)

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

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(pos_vxB[:, 1], pos_vxB[:, 0], c = f_geo_pos, cmap = 'viridis', s = 50, alpha = 0.9, edgecolors = 'k')
    cbar = plt.colorbar(sc)
    cbar.set_label('Energy Fluence [eV/m$^2$]', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.title('Shower Plane Energy Fluence Distribution', fontsize=18)
    plt.xlabel(r'v $\times$ B [m]', fontsize=16)
    plt.ylabel(r'v $\times$ (v $\times$ B) [m]', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=0.6, length=4)
    for spine in plt.gca().spines.values(): spine.set_linewidth(1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels(): label.set_fontweight('bold')
    plt.tick_params(axis='x', pad=6)
    plt.savefig(os.path.join(event_dir, 'fluence_map_shower_plane.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #--------------------------------------------------------------------------------------------------

    '''
    dX = positions_all - np.array([rec_x_xmax, rec_y_xmax, rec_z_xmax]).T

    l_ant = np.linalg.norm(dX, axis = 1)

    shower_axis_unit = np.array([-np.sin(np.deg2rad(rec_theta_sph)) * np.cos(np.deg2rad(rec_phi_sph)),
                                 -np.sin(np.deg2rad(rec_theta_sph)) * np.sin(np.deg2rad(rec_phi_sph)),
                                 -np.cos(np.deg2rad(rec_theta_sph))])

    w_ant = np.dot(shower_axis_unit, dX.T) / l_ant

    print("Alternative viewing angles (w_ant in degrees):", np.arccos(w_ant) * 180 / np.pi, flush = True)
    '''
    #-------------------------------------------------------------------------------------------------

    XmaxDis_km = dXmax / 1e3
    dw = 227.958137 / (176.414293 * XmaxDis_km ** 2 + 112.917133 * XmaxDis_km) + 1.351487 # dw 经验参数化公式

    # paras = create_params(amp = max(f_geo_pos), w_c = dict(value = omega_c, vary = True), domega = dict(value = dw, vary = True))

    # paras = create_params(amp = dict(value = max(f_geo_pos), min = max(f_geo_pos) / 100, max = max(f_geo_pos) * 100, vary = True), 
    #                      w_c = dict(value = omega_c, min = 1e-4, max = 1.0, vary = True), 
    #                      domega = dict(value = dw, min = 0.01, max = 20, vary = True))

    #---------------------- 拟合切伦科夫分布 ---------------------------------------------------
    
    omega_c = rec_wc # 使用 ADF 重建得到的切伦科夫角作为初始值，避免拟合陷入局部最小值
    chi2_che = che_fit(w[~vxB_axis], f_geo_pos[~vxB_axis], max(f_geo_pos), omega_c, dw)

    f_che_pars = np.array([chi2_che[0], chi2_che[1], chi2_che[2]])

    chi2_che_err = np.array([chi2_che[4], chi2_che[5], chi2_che[6]])

    logger.info(f"Initial Che parameters: {f_che_pars}, with errors: {chi2_che_err}, Chi2/dof: {chi2_che[3] / (len(w[~vxB_axis]) - 1)}")

    logger.info(f"Asymmetric ADF parameters: {rec_A:.5e} {rec_wc} {rec_dw}")

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
    '''
    E_constructed_EeV = (correction_E_geo / 1e9 / 0.07) ** (1 / 2.03)

    print("Constructed E_em (EeV) from corrected E_geo:", E_constructed_EeV, flush = True)


    E_constructed_EeV_two = (E_em_corrected_two / 1e9 / 0.08) ** (1 / 1.91)

    print("Constructed E_em (EeV) from corrected E_em with asymmetric ADF:", E_constructed_EeV_two, flush = True)
    '''
    #------------------------------ 绘制切伦科夫拟合结果 ------------------------------------------------

    plt.figure(figsize = (8, 6))

    w_line = np.linspace(0, 0.1, 200)
    f_che_line, f_che_line_as = f_Che(f_che_pars, w_line), f_Che(adf_pars, w_line) # 这里的 phi 参数对拟合曲线影响较小，暂时固定为 0 来展示 ADF 曲线

    # plt.plot(w_line * 180 / np.pi, f_che_line, color = 'blue', label = 'Fitted distribution')
    # plt.scatter(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], color = 'blue', label = 'Data points')
    plt.plot(w_line * 180 / np.pi, f_che_line_as, color = 'red')#, label = 'Asymmetric ADF function')
    plt.errorbar(w * 180 / np.pi, Fluence * early_late, yerr = Fluence * early_late * 0.1 + 0.01 * np.max(Fluence * early_late), fmt = 's', color = 'red')
    # plt.errorbar(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], yerr = f_geo_pos[~vxB_axis] * 0.1 + 0.01 * np.max(f_geo_pos[~vxB_axis]), fmt = 'o', color = 'blue', label = 'Data points')
    # plt.errorbar(w[~vxB_axis] * 180 / np.pi, f_geo_pos[~vxB_axis], yerr = np.sqrt(f_geo_pos[~vxB_axis]), fmt = 'o', color = 'blue', label = 'Data points')
    # plt.axvline(x = f_che_pars[1] * 180 / np.pi, color = 'blue', linestyle = '--', label = f'Cherenkov angle: {f_che_pars[1] * 180 / np.pi:.2f} deg')
    plt.axvline(x = rec_wc * 180 / np.pi, color = 'red', linestyle = '--', label = f'ADF Cherenkov angle: {rec_wc * 180 / np.pi:.2f} deg')
    plt.xlabel(r'$\omega$ [deg]', fontsize = 16)
    plt.ylabel(r'Fluence [$\rm eV/m^2$]', fontsize = 16)
    plt.title('Cherenkov Angle Distribution', fontsize = 18)
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
    plt.savefig(os.path.join(event_dir, 'che_fit_figure.png'), dpi = 300)
    plt.close()

    #----------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------

    return energy_result