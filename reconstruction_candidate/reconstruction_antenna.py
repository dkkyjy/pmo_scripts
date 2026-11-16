"""天线响应层：有效长度表、端口旋转与 ADC→Voc 换算。

只依赖 reconstruction_common 与本目录既有的数值/数据模块；不做 ROOT 读取，
也不依赖 parameter_files 之外的任何东西（EFL 表按需懒加载）。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import windows
import scipy.io

from logger_config import logger
from galacticnoise_get import gala_new
from vad2voc_fixed import vad2voc
from interpolation_ import inter

from reconstruction_common import rotation_m, _resolve_data_file, _PARAM_DIR

tukey_wind = windows.tukey(200) # 创建一个长度为200的Tukey窗函数

REfile_grand = _resolve_data_file('RECON_EFL_MAT', 'EFL_data_3.2m_XYZ_20250313.mat')
_EFL_CACHE = {}



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


def _get_efl_all_grand():
    """懒加载 GRAND 天线有效长度表。

    这个 .mat 有 961 MB, loadmat 后常驻约 4 GB —— 而 ADF 用带噪电压做方向拟合时
    完全不需要 (只有电压->电场反演才用)。改成首次调用时才加载, 并行任务可省下这笔内存。
    """
    if 'all' not in _EFL_CACHE:
        if not os.path.exists(REfile_grand):
            raise FileNotFoundError(
                f"未找到 GRAND 天线有效长度数据文件: {REfile_grand}\n"
                f"请将其放入 {_PARAM_DIR}/ , 或用环境变量 RECON_EFL_MAT 指向实际路径。"
            )
        _EFL_CACHE['all'] = scipy.io.loadmat(REfile_grand)['EFL_all']
    return _EFL_CACHE['all']


def _interp_complex_1d(f_src, y_src, f_tar, kind = "cubic"):
    """
    复数一维插值 (实部/虚部分别插值, 与 vad2voc 风格一致).
    范围外的频点置 0 (与频带外置零的既有约定一致).
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


def get_lec_grand_real_MHz(e_theta, e_phi, freq_target_MHz):
    """
    获取 GRAND 天线在指定真实频点上的有效长度 (球面坐标系).

    参数
    ----
    e_theta : float, 天顶角 (度).
    e_phi : float, 方位角 (度).
    freq_target_MHz : array_like
        目标频点数组 (MHz), 可为真实 FFT 频点 (如 0.488 MHz 间隔).
        仅 30-250 MHz 内的频点有效, 范围外置 0.

    返回
    ----
    Lce_sphere_real : (Nfreq_target, 3, 3) complex
        索引 [频点, 球面分量, 端口]:
          分量 0 = theta, 1 = phi, 2 = r;
          端口 0/1/2 = X/Y/Z.
        球面分量/端口索引与 ``Lce_sphere`` 的既有约定一致 ([:, 0, p] 为 theta, [:, 1, p] 为 phi),
        只是频率轴由 1 MHz 网格换成你指定的 freq_target_MHz.
    """
    freq_target_MHz = np.asarray(freq_target_MHz, dtype = float)

    # 1) 取该方向的辐射响应 (与原函数相同)
    e_radiation = inter(_get_efl_all_grand(), e_theta, e_phi)     # (221, 3 ports, 3 comps)

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
