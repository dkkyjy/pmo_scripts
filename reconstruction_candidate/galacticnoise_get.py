import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from FFT_get import fftget
from complex_expansion import expan
from IFFT_get import ifftget
from scipy.io import loadmat
from scipy.interpolate import interp1d

# 参数/数据文件目录: 相对本文件定位, 可用环境变量 RECON_PARAM_DIR 覆盖。
_PARAM_DIR = os.environ.get('RECON_PARAM_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameter_files'))


def gala(lst, N, f0, f1, show_flag):
    # This Python file uses the following encoding: utf-8

    # = == == == == 本程序作为子程序完成银河噪声的计算及扩展 == == == == =
    #  ----------------------输入 - ---------------------------------- %
    # lst：Select the galactic noise LST at the LST moment
    # N 为扩展长度
    # f0为频率分辨率，f1是单边谱频点
    # % ----------------------输出 - ---------------------------------- %
    # v_complex_double, galactic_v_time

    # 银河噪声谱数据文件, 默认取 parameter_files/ 下的副本, 可用环境变量 RECON_GALACTIC_MAT 覆盖。
    GALAshowFile = os.environ.get('RECON_GALACTIC_MAT', os.path.join(_PARAM_DIR, '30_250galactic.mat'))

    GALAshow = h5py.File(GALAshowFile,"r")
    GALApsd_dbm = np.transpose(GALAshow['psd_narrow_huatu'])
    GALApower_dbm = np.transpose(GALAshow['p_narrow_huatu'])
    GALAvoltage = np.transpose(GALAshow['v_amplitude'])
    GALApower_mag = np.transpose(GALAshow['p_narrow'])
    GALAfreq = GALAshow['freq_all']

    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 3, 1)
        for g in range(3):
            plt.plot(GALAfreq, GALApsd_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.subplot(1, 3, 2)
        for g in range(3):
            plt.plot(GALAfreq, GALApower_dbm[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.subplot(1, 3, 3)
        for g in range(3):
            plt.plot(GALAfreq, GALAvoltage[:, g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    f_start = 30
    f_end = 250
    v_amplitude_double = np.zeros((N, 3), dtype=complex)
    v_amplitude_fft = np.zeros((N, 3), dtype=complex)
    v_complex_double = np.zeros((N, 3), dtype=complex)
    unit_uv = 1e6
    for p in range(3):
        [f, v_amplitude_double[:, p]] = expan(N, f0, f_start, f_end, GALAvoltage[:, p, lst])  # 拓展电压

        v_amplitude_fft[0, p] = v_amplitude_double[0, p] * N
        v_amplitude_fft[1:, p] = v_amplitude_double[1:, p] * N / 2

        # 加入一随机高斯白噪声的相位
        Noise_time = np.random.randn(N, 1) * np.sqrt(np.mean(GALApower_mag[:, p])) * unit_uv  # 产生一个高斯噪声
        [Noise_fft, Noise_fft_m_single, Noise_fft_p_single] = fftget(Noise_time, N, f1, 0)
        Noise_p = np.angle(Noise_fft)  # 相位,deg若为True，返回值采用角度制；若为False（默认），返回值采用弧度制。

        # 生成复数
        v_complex_double[:, p] = v_amplitude_fft[:, p] * np.exp(1j * Noise_p[:, 0])


    # 时域
    [galactic_v_time, galactic_v_m_single, galactic_v_p_single] = ifftget(v_complex_double, N, f1, 2)
    if show_flag==1:
        plt.figure()
        for g in range(3):
            plt.plot(f, v_amplitude_fft[:, g])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage after expansion", fontsize=15)
        plt.tight_layout()
    plt.show()
    return v_complex_double, galactic_v_time


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def gala_new(lst, N_time, seed = None, rng = None):
    """
    读取银河系噪声在指定 LST 时刻、指定频点 freq_target_MHz 上的开路电压 (复数, 单位 uV)。

    参数
    ----
    lst : int 或 (int, int)
        LST 时间。int 表示小时 (0-23, 分钟取 0),
        (hour, minute) 分别表示 0-23 时、0-59 分。
    N_time : int, 1024 or 512
        原始 ADC 、 开路电压的 trace 时域长度。

    seed : 随机种子。None (默认) -> 每次调用都用系统熵生成"全新"的随机相位,
           即各 DU、各事件互不相同; 需要可复现时显式传入 (如 seed = 1000 * event_num + du_index)。
    rng  : 也可以直接传一个已有的 np.random.Generator (给了它就用它, 忽略 seed)。

    返回
    ----
    v : (len(freq_target_MHz), 3) 数组, 30--250 之间MHz 频点有效, 其余为 0)
        开路电压 (复数, 单位 uV; 值 = 每个频点幅度 uV * N_time/2)。
    """

    #-----------------------------------------------------------------------------

    freqs_real = np.fft.rfftfreq(N_time, d = 2e-9) / 1e6
    
    freq_target_MHz = freqs_real[(freqs_real >= 30) & (freqs_real <= 250)]

    grid_1MHz = np.arange(30, 251)  # 1 MHz 频点

    #-----------------------------------------------------------------------------

    if np.ndim(lst) == 0: lst_hour, lst_minute = int(lst), 0
    
    else: lst_hour, lst_minute = int(lst[0]), int(lst[1])

    #-----------------------------------------------------------------------------

    # 全天空功率 (PG) 数据文件, 默认取 parameter_files/ 下的副本, 可用环境变量 RECON_PG_MAT 覆盖。
    PG_MAT_path = os.environ.get('RECON_PG_MAT', os.path.join(_PARAM_DIR, 'PG_ALL_jifen_perminute.mat'))
    if not os.path.exists(PG_MAT_path):
        raise FileNotFoundError(
            f"未找到银河噪声功率数据文件: {PG_MAT_path}\n"
            f"请将其放入 {_PARAM_DIR}/ , 或用环境变量 RECON_PG_MAT 指向实际路径。"
        )
    PG_MAT_file = loadmat(PG_MAT_path)

    key = next(
            k for k, v in PG_MAT_file.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 4)

    PG = np.asarray(PG_MAT_file[key], dtype = np.float64)

    #-----------------------------------------------------------------------------

    # 天线阻抗数据文件, 默认取 parameter_files/ 下的副本, 可用环境变量 RECON_ZANT_CSV 覆盖。
    z_ant_path = os.environ.get('RECON_ZANT_CSV', os.path.join(_PARAM_DIR, 'Z_ant_3.2m.csv'))
    raw = np.loadtxt(z_ant_path, delimiter = ",", skiprows = 1)

    Z_ant_1MHz = np.column_stack([
            raw[:, 1] + 1j * raw[:, 2],   # X 端口
            raw[:, 3] + 1j * raw[:, 4],   # Y 端口
            raw[:, 5] + 1j * raw[:, 6],   # Z 端口
            ]).astype(np.complex128)

    #-----------------------------------------------------------------------------

    psd = PG[:, :, lst_hour, lst_minute]                                    # (221,3) W/Hz
    p_w = psd * 1e6                                                         # 1 MHz 带宽功率 (W)
    # amp_221 = 2.0 * np.sqrt(np.maximum(p_w * np.real(Z_ant), 0.0)) * 1e6  # (221,3) uV
    amp_221 = 2.0 * np.sqrt(p_w * 50.0) * 1e6                               # (221,3) uV

    #-----------------------------------------------------------------------------

    amp = np.empty((freq_target_MHz.size, 3), dtype = float)
    
    for p in range(3):
        amp[:, p] = interp1d(grid_1MHz, amp_221[:, p], kind = "cubic")(freq_target_MHz)

    #-----------------------------------------------------------------------------
    
    coeff = amp * (N_time / 2.0)   # 与 Voc 同口径的未归一化系数
    
    if rng is None:
        rng = np.random.default_rng(seed)
    v = coeff * np.exp(1j * 2.0 * np.pi * rng.random(coeff.shape))

    return freq_target_MHz, v

#---------------------------------------------------------------------------------
