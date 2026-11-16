import os
import matplotlib.pyplot as plt
import sys
from typing import NamedTuple
import numpy as np
from scipy.fftpack import ifft, fft # 引入快速傅里叶变换和逆变换函数
from scipy.fftpack import fftfreq
from scipy.signal import hilbert # 引入希尔伯特变换
from scipy import integrate # 引入数值积分函数
import h5py # 引入h5py库用于读取.mat文件
import iminuit
from multiprocessing.pool import Pool
import io
import glob
import contextlib
import scienceplots

plt.rcParams.update({"text.usetex": False})
plt.style.use(['science', 'grid', 'no-latex'])
# ── 快速模式: RECON_SKIP_PLOTS=1 时把所有 savefig 置空 (Agg 后端延迟渲染,
#    PNG 光栅化 + 编码 + ceph 写盘是单事件主要耗时来源; 统计只需 summary/
#    shower_output.txt, 不需要诊断图)。未设该环境变量时行为不变。──
if os.environ.get('RECON_SKIP_PLOTS'):
    plt.savefig = lambda *a, **k: None



from scipy.stats import chi2, norm


# 这部分函数为官方可下载函数

#--------------------------------------------------------------------------

# 这部分为自定义函数, 见文件夹python代码文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#--------------------------------------------------------------------------
# 路径层已移到 reconstruction_common：外部资源常量与环境变量覆盖在那里定义
#   - parameter_files/ : EFL / PG_ALL / Z_ant / 30_250galactic 等参数与数据文件
#   - 每个资源常量都可用同名环境变量覆盖, 便于跨站点部署:
#       RECON_EFL_MAT            天线有效长度 (EFL) 数据 (.mat)
#       RECON_DU_POS_TABLE       生产 DU 位置表 (GP65, teventadc 输入用)
#       RECON_DU_ROTATION_FILE   DU 安装转角表
#       RECON_COREAS_DET_POS     CoREAS du_coords.txt (tadc 输入用)
# 本文件只负责导入这些常量并把三个编排串起来。
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------

from logger_config import logger  # 统一日志输出 (loguru), 替代 print
from equivalent import CEL # 引入计算天线有效长度的函数
from galacticnoise_get import gala, gala_new # 引入获取银河噪声的函数
from Time_domain_Shower_Edata_get import time_data_get # 引入获取时域簇射电场数据的函数
import AiresInfoFunctions as aires
import coordinatesystems, helper
import models as atm

# 拟合与方向层已拆到独立模块（单向依赖：本文件 -> reconstruction_direction）。
# 这里只接回本文件仍在解析的符号；不要在这里加转发别名。
from reconstruction_direction import (
    PWF_fit,
    che_fit,
    f_Che,
    f_adf,
    find_core,
    density_and_alpha_correction,
    recons_direction_swf_adf,
    reconstruct_direction_only,
    direction_result_from_tuple,
    magnetic_field_vector,
)

# 重建层已按职责拆分（单向依赖：本文件 -> common / antenna / io / direction）。
from reconstruction_common import (
    DU_ROTATION_FILE,
    _PARAM_DIR,
    efield_rec_dir,
    rotation_m,
    calculate_energy_fluence_and_error,
)
from reconstruction_antenna import (
    ADC_trace_to_Voc_time,
    Voc_fft_real_MHz,
    e_recons_real_MHz,
    get_lec_grand_real_MHz,
    re_calculate_Voc_from_reconstruction,
    rotate_xy_ports,
    read_du_rotation_angles,
)
from reconstruction_io import (
    read_root_file_from_path,
    read_xyz_coordinates,
    causality_filter,
    _load_du_position_for_input,
    _read_du_position_table,
)

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------

# 定义一些全局参数和常量


time_wind = 100 # 定义时间窗口大小


thetaList, phiList = np.linspace(90, 180, 6), np.linspace(0, 360, 6) # 定义初始角度列表, 用于波前拟合的初始猜测


N, f0, f1 = 2000, 1.0, np.linspace(0, 1000, 1001) # 定义频率点数和频率范围, MHz

# magnetic_field_vector = helper.spherical_to_cartesian(np.deg2rad(60.79) + np.pi / 2, 0) # GRAND 站点的地磁场矢量, 单位矢量, 笛卡尔坐标系下
logger.info(f"Magnetic field vector (cartesian): {magnetic_field_vector}")
# ? ? ? ? ? 地磁场矢量方位角 0.36 deg ?

#--------------------------------------------------------------------------

# efield_rec_dir = './Reconstruction_test/' # 结果保存目录, 用于测试筛选事件
# efield_rec_dir = './Reconstruction_colleague/'
# 目录不在这里创建: 只有真正做电场反演(efield_recons_*)时才需要, 由 _event_rec_dir 按事例按需创建
# (exist_ok=True)。ADF/解析 A 只用电压 fluence, 完全不碰这个目录。

#--------------------------------------------------------------------------


#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------

# GRAND 天线有效长度 (EFL) 数据文件, 默认取 parameter_files/ 下的副本, 可用
# 环境变量 RECON_EFL_MAT 覆盖。



#--------------------------------------------------------------------------



# 真实频点版有效长度: 可指定天顶角/方位角/任意频点数组

#--------------------------------------------------------------------------



#--------------------------------------------------------------------------


#--------------------------------------------------------------------------


#--------------------------------------------------------------------------




#--------------------------------------------------------------------------


# 注: "全部 DU 的旋转坐标"(仅用于绘图背景点) 不再作为模块级全局状态, 而是由
# efield_recons_from_efield_PWF / energy_restruction 在选定正确坐标表后按需局部加载,
# 以避免模块导入副作用并支持不同探测器布局。

#--------------------------------------------------------------------------


#--------------------------------------------------------------------------


#--------------------------------------------------------------------------


#--------------------------------------------------------------------------

# DU 转角: RTK 实测角 -> 重建代码方位角约定 -> 把 X/Y 端口旋到标称系



# 每个 DU 的转角表 (已换算成代码的方位角约定: 逆时针为正)。
# 文件缺失时不修正转角 (等价于所有 DU 转角为 0), 保证链路仍可运行。
if os.path.exists(DU_ROTATION_FILE):
    DU_ROTATION_ANGLES = read_du_rotation_angles(DU_ROTATION_FILE)
else:
    logger.warning(f"[WARN] 未找到 DU 转角表 {DU_ROTATION_FILE}, 跳过端口转角修正。"
          f"如需启用, 请将其放入 {_PARAM_DIR}/ 或用环境变量 RECON_DU_ROTATION_FILE 指定。")
    DU_ROTATION_ANGLES = {}

#--------------------------------------------------------------------------


#-------------------------------------------------------------------------


#--------------------------------------------------------------------------

def efield_recons_from_efield_PWF(root_path, event_number_int, det_pos_path = None, valid_du_ids = None, theta = None, phi = None, prior_times = None): # 基于平面波前拟合 (PWF) 的电场重建
    # 可选外部先验 (来自 matched_PWM.yaml 的 PWM 结果):
    #   valid_du_ids: 给定则用 np.isin 直接筛天线, 替换内部 causality_filter 图优化筛选;
    #   prior_times : 给定则用 yaml 的到达时间 (du_id -> ns) 替换 ROOT 的 du_nanoseconds;
    #   theta / phi : yaml 的 zenith / azimuth (deg)。换算为 PWF 内部坐标作单一初值,
    #                 仍执行一次 PWF_fit (不跑网格搜索)。

    logger.info(f"Starting E-field reconstruction using PWF from {root_path} , event:  {int(event_number_int)}")

    filename = os.path.splitext(os.path.basename(root_path))[0] + f"_event_{int(event_number_int)}" # 得到文件名和事件编号

    os.makedirs(efield_rec_dir + filename + '/', exist_ok=True)

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


#--------------------------------------------------------------------------




#--------------------------------------------------------------------------






#--------------------------------------------------------------------------













#--------------------------------------------------------------------------






#--------------------------------------------------------------------------




#--------------------------------------------------------------------------


#--------------------------------------------------------------------------


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

    _dir = reconstruct_direction_only(
        positions_x, positions_y, positions_z, nanosecond_time, Fluence,
        init_input, obs_altitude, p_sigma = Fluence_sigma,
        x_xmax = x_xmax, y_xmax = y_xmax, z_xmax = z_xmax,
        chi_square_external = chi_square_external,
    )
    rec_theta_plane, rec_phi_plane, chi2Plane = _dir.zenith_init, _dir.azimuth_init, _dir.chi2_plane
    rec_x_xmax, rec_y_xmax, rec_z_xmax = _dir.x_max, _dir.y_max, _dir.z_max
    chi2Sph = _dir.chi2_sph
    rec_theta_sph, rec_phi_sph, rec_dw, rec_A = _dir.zenith, _dir.azimuth, _dir.dw, _dir.A
    fmin, rec_wc = _dir.fmin, _dir.wc
    rec_theta_err, rec_phi_err = _dir.zenith_err, _dir.azimuth_err
    rec_dw_err, rec_A_err = _dir.dw_err, _dir.A_err
    rec_x_xmax_err, rec_y_xmax_err, rec_z_xmax_err = _dir.x_max_err, _dir.y_max_err, _dir.z_max_err
    rec_dw_fixed = _dir.dw_fixed


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

    if rec_dw_fixed: 
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
    
    if rec_dw_fixed:
        dE_dw_adf = 0.0          # 与旧行为一致：固定 dw 时不计 dw 不确定度
    else:
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

#--------------------------------------------------------------------------
# ADF 角重建, 但每天线 fluence 直接取自带噪电压 (Voc) 的 ±N 样点窗 —— 跳过 E-field 反演
#--------------------------------------------------------------------------
def adf_angles_from_voc(root_path, event_number_int, det_pos_path = None, fluence_window = 5,
                        x_xmax = None, y_xmax = None, z_xmax = None, chi_square_external = None):
    """用带噪电压直接做 ADF, 不做逐天线电场反演。

    与 ``efield_recons_from_efield_PWF`` + ``energy_restruction`` 的区别:
    不跑 vad2voc 之后的 LS 电场反演, 也不写 ``Reconstruction/`` 目录。每天线 fluence 取

        p = sum_{|t - t_peak| <= N} sum_ch Voc^2

    其中 ``Voc`` 由 :func:`ADC_trace_to_Voc_time` 从 ADC 波形得到 (单位 uV),
    ``t_peak`` 取该 DU 三通道绝对值之和最大的采样点 (逐天线对齐, 避免阵列上脉冲
    到达时间差把窗口切掉)。``fluence_window = 0`` 退化为整条迹求和。

    实测 (CoREAS-AN, 149 分片 / 14773 事例): 整条迹 -> ±5 样点, 角分辨率中位
    0.549 deg -> 0.318 deg; 见 tools/sim_dunhuang/adf_adc_window.md 。

    返回 dict (失败返回 None): azimuth/zenith (ADF 结果), zenith_pwf/azimuth_pwf
    (PWF 初值), xmax (3,), dw, A, chi2, n_du, fluence。
    """
    window = int(fluence_window)
    if window < 0:
        raise ValueError("fluence_window 必须 >= 0")

    event_num, du_ids_all, _gps_time, du_nanoseconds, _LST_time, trace_1, trace_2, trace_3 = \
        read_root_file_from_path(root_path, event_number_int)

    tree_kind, du_pos_path = _load_du_position_for_input(root_path, det_pos_path)
    table = _read_du_position_table(du_pos_path)
    pos_of = {int(row[0]): np.array([row[1], row[2], row[3]], dtype = float) for row in table}

    ids = [int(d) for d in du_ids_all]
    keep = [i for i, d in enumerate(ids) if d in pos_of]
    if len(keep) < 4:
        logger.warning(f"event {event_num}: 只有 {len(keep)} 个 DU 有坐标 (<4), 跳过")
        return None
    ids = [ids[i] for i in keep]
    x = np.array([pos_of[d][0] for d in ids])
    y = np.array([pos_of[d][1] for d in ids])
    z = np.array([pos_of[d][2] for d in ids])
    t_ns = np.asarray(du_nanoseconds, dtype = float)[keep]

    # --- PWF 初值 (与 E-field 路径相同的 6x6 网格) ---
    best = (np.inf, None, None)
    for theta0 in np.linspace(90, 180, 6):
        for phi0 in np.linspace(0, 360, 6):
            try:
                theta_fit, phi_fit, fval, _te, _pe = PWF_fit(x, y, z, t_ns * 1e-9, theta0, phi0)
            except Exception:
                continue
            if fval < best[0]:
                best = (fval, theta_fit, phi_fit)
    if best[1] is None:
        logger.warning(f"event {event_num}: PWF 初值拟合失败, 跳过")
        return None
    chi2_plane, theta_p, phi_p = best
    zenith_ini = 180.0 - theta_p
    azimuth_ini = (phi_p + 180.0) % 360.0

    # --- 每天线 fluence: ±N 样点窗内的 Voc^2 ---
    fluence = np.full(len(ids), np.nan)
    for i, idx in enumerate(keep):
        try:
            Voc_time = ADC_trace_to_Voc_time(trace_1[idx], trace_2[idx], trace_3[idx])[0]
        except Exception as exc:
            logger.debug(f"event {event_num} DU {ids[i]}: Voc 计算失败 ({exc})")
            continue
        v = np.nan_to_num(np.asarray(Voc_time, dtype = float))
        if window > 0:
            peak = int(np.argmax(np.abs(v).sum(axis = 1)))
            lo, hi = max(0, peak - window), min(len(v), peak + window + 1)
            fluence[i] = float(np.sum(v[lo:hi, :] ** 2))
        else:
            fluence[i] = float(np.sum(v ** 2))

    ok = np.isfinite(fluence) & (fluence > 0)
    if ok.sum() < 4:
        logger.warning(f"event {event_num}: 有效 fluence 的 DU 只有 {int(ok.sum())} 个 (<4), 跳过")
        return None

    obs_altitude = float(np.mean(z))
    # 注意: recons_angle 内部的 SWF_fit 用 c[m/s] * t, 所以这里必须传**秒**
    # (energy_restruction 也是这么做的: fluence_data[:, 7] * 1e-9)。
    res = direction_result_from_tuple(
        recons_direction_swf_adf(x, y, z, t_ns * 1e-9, fluence, [zenith_ini, azimuth_ini, chi2_plane],
                                 obs_altitude, None, x_xmax, y_xmax, z_xmax, chi_square_external))
    return dict(
        azimuth = float(res.azimuth), zenith = float(res.zenith),
        azimuth_pwf = azimuth_ini, zenith_pwf = zenith_ini, chi2_pwf = float(chi2_plane),
        xmax = (float(res.x_max), float(res.y_max), float(res.z_max)),
        dw = float(res.dw), A = float(res.A), chi2 = float(res.chi2_sph), wc = float(res.wc),
        n_du = int(ok.sum()), fluence_window = window, fluence = fluence[ok].tolist(),
    )


# 示例 / 命令行入口: 仅在直接运行本文件时执行, 避免被 import 时触发重型计算。
# 通过环境变量 RECON_ROOT 与 RECON_EVENT 指定输入 root 文件与事件号:
#   RECON_ROOT=/path/to/adc_*.root RECON_EVENT=0 python reconstruction_candidate_fluence_error_fixed.py
#   RECON_ADF_VOC_WINDOW=5 同上, 但跳过电场反演、直接用在 ±5 样点窗内的 Voc^2 做 ADF
#   (CoREAS 输入还需 RECON_COREAS_DET_POS=position/du_coords_CoreasDC2.txt)
# 若设置了 RECON_ROOT 但未设置 RECON_EVENT, 默认事件号为 0。
if __name__ == '__main__':
    _demo_root = os.environ.get('RECON_ROOT')
    if _demo_root:
        _demo_event = int(os.environ.get('RECON_EVENT', '0'))
        _voc_window = os.environ.get('RECON_ADF_VOC_WINDOW')
        if _voc_window is not None:
            # 跳过 E-field 反演: 直接用在 ±N 样点窗内的 Voc^2 做 ADF
            logger.info(adf_angles_from_voc(_demo_root, _demo_event,
                                            fluence_window = int(_voc_window)))
        else:
            efield_recons_from_efield_PWF(_demo_root, _demo_event)
            energy_restruction(_demo_root, _demo_event)
    else:
        logger.info("未设置 RECON_ROOT, 跳过内置示例。"
              "调用 efield_recons_from_efield_PWF(root_path, event_number, det_pos_path=None) "
              "与 energy_restruction(root_path, event_number, det_pos_path=None) 即可。")

#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# ADF 拟合, 幅度 A 解析消去 (chi2 对 A 是二次型): 少一个自由参数, 少 3 倍初值
#--------------------------------------------------------------------------








