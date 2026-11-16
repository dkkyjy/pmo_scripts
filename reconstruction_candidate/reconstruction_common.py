"""路径解析、路径常量与不依赖其它重建层的工具函数。

本模块只做两件事：把机器相关的路径收敛成常量（可用环境变量覆盖），以及几个纯计算工具。
它不 import 本目录的其它重建模块，也不做任何文件读取——读取与天线响应在各自层里。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from logger_config import logger

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAM_DIR = os.path.join(_HERE, 'parameter_files')
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

def _resolve_path(env_var, default):
    """返回环境变量 ``env_var`` 指定的路径, 未设置时回退到 ``default``。"""
    return os.environ.get(env_var, default)

def _resolve_data_file(env_var, filename):
    """在 parameter_files/ 下解析数据文件, 可用环境变量 ``env_var`` 覆盖。"""
    return _resolve_path(env_var, os.path.join(_PARAM_DIR, filename))

DU_POS_TABLE = _resolve_path('RECON_DU_POS_TABLE', os.path.join(_REPO_ROOT, '_gp65_rtksort.txt'))

DU_ROTATION_FILE = _resolve_data_file('RECON_DU_ROTATION_FILE', '_rtk_fred_rotation_order.txt')

efield_rec_dir = './Reconstruction/' # 结果保存目录

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

def rotation_m(theta, phi):
    # 定义旋转矩阵，用于坐标系转换
    # 输入: theta (天顶角), phi (方位角)
    # 输出: 3x3 旋转矩阵
    return np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
                     [np.cos(theta) * np.sin(phi),  np.cos(phi), np.sin(theta) * np.sin(phi)],
                     [             -np.sin(theta),            0,               np.cos(theta)]])

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
