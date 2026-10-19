import numpy as np
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')

_PARAM_DIR = os.path.join(os.path.dirname(__file__), 'parameter_files')


def vad2voc(Vad, target_freq_hz=None):
    """
    从Vad计算Voc的核心函数，对应原MATLAB函数

    参数:
        Vad: 输入的Vad矩阵 (频率数 × 3)
        target_freq_hz: 【新增】目标频率轴, 单位Hz。若提供, 则所有S参数/Z_ant直接
            插值到该频点上计算 (即trace真实FFT频点), 避免"先插到固定1MHz网格
            再二次插值"带来的额外插值误差。若为 None, 则退回旧行为:
            使用固定 30-250MHz、1MHz步长网格 (221点)。
            注意: Vad 的第一维长度必须与 target_freq_hz (或默认221点) 一致。
    返回:
        Voc: 计算得到的Voc矩阵 (频率数 × 3)
    """
    # ===================== 1. 初始化频率参数 =====================
    if target_freq_hz is not None:
        target_freq = np.asarray(target_freq_hz, dtype=np.float64)  # 直接用真实频点 (Hz)
    else:
        freq_ALL = np.arange(0.03, 0.250001, 0.001)  # 旧行为: 0.03到0.25，步长0.001 (GHz)
        target_freq = freq_ALL * 1e9
    freq_num = len(target_freq)

    # ===================== 2. 读取各类参数文件 =====================
    # 读取Z_ant (复阻抗)
    z_ant = np.genfromtxt(os.path.join(_PARAM_DIR, 'Z_ant_3.2m.csv'), delimiter=',', skip_header=1)

    # 读取S参数文件（封装为函数调用），返回 (data, is_ma_format)
    balun_a, balun_a_ma = read_s2p(os.path.join(_PARAM_DIR, 'balun13in20230612.s2p'))
    balun_d, balun_d_ma = read_s2p(os.path.join(_PARAM_DIR, 'balun_before_ad.s2p'))
    lna, lna_ma = read_s2p(os.path.join(_PARAM_DIR, 'LNA-X.s2p'))
    cable, cable_ma = read_s2p(os.path.join(_PARAM_DIR, 'cable+Connector.s2p'))
    vga, vga_ma = read_s2p(os.path.join(_PARAM_DIR, 'feb+amfitler+biast.s2p'))
    ADload, ADload_ma = read_s1p(os.path.join(_PARAM_DIR, 'S_balun_AD.s1p'))
    MN_xy, MN_xy_ma = read_s2p(os.path.join(_PARAM_DIR, 'MatchingNetworkX.s2p'))
    MN_z, MN_z_ma = read_s2p(os.path.join(_PARAM_DIR, 'MatchingNetworkZ.s2p'))

    # ===================== 3. 相位补偿计算 =====================
    f_unit = 1e9  # 频率单位 (Hz per GHz, 用于波长公式)
    lamda = 3e8 / target_freq  # 波长计算 (target_freq 已是 Hz)
    theta1 = 0.3 * np.sqrt(2.2) * 2 * np.pi / lamda  # 相位补偿

    # ===================== 4. dB/角度 转 复数形式 =====================
    def db_angle_to_complex(data, is_ma, is_s1p=False):
        """将S参数的dB/角度转换为复数形式。is_ma=True 表示幅度已是线性值。"""
        if is_s1p:
            mag = data[:, 1] if is_ma else 10 ** (data[:, 1] / 20)
            phase = data[:, 2] * np.pi / 180
            return mag * np.exp(1j * phase)
        else:
            complex_data = np.zeros((len(data), 4), dtype=np.complex128)
            for i in range(4):
                mag_col = 1 + 2 * i
                angle_col = 2 + 2 * i
                mag = data[:, mag_col] if is_ma else 10 ** (data[:, mag_col] / 20)
                phase = data[:, angle_col] * np.pi / 180
                complex_data[:, i] = mag * np.exp(1j * phase)
            return complex_data

    # 转换各类S参数为复数
    lna_complex = db_angle_to_complex(lna, lna_ma)
    cable_complex = db_angle_to_complex(cable, cable_ma)
    vga_complex = db_angle_to_complex(vga, vga_ma)
    balun_a_complex = db_angle_to_complex(balun_a, balun_a_ma)
    balun_d_complex = db_angle_to_complex(balun_d, balun_d_ma)
    MN_xy_complex = db_angle_to_complex(MN_xy, MN_xy_ma)
    MN_z_complex = db_angle_to_complex(MN_z, MN_z_ma)
    load_complex = db_angle_to_complex(ADload, ADload_ma, is_s1p=True)

    # ===================== 5. 频率插值（统一频率轴） =====================
    def interpolate_s_params(s_data, s_complex, target_freq):
        """S参数插值到目标频率轴"""
        source_freq = s_data[:, 0]
        interp_result = np.zeros((len(target_freq), 4), dtype=np.complex128)
        for i in range(4):
            interp_func = interp1d(source_freq, s_complex[:, i], kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
            interp_result[:, i] = interp_func(target_freq)
        return interp_result

    # target_freq 已在函数开头按 target_freq_hz (或默认221点1MHz网格) 定义, 此处直接复用
    s_cable = interpolate_s_params(cable, cable_complex, target_freq)
    s_vga = interpolate_s_params(vga, vga_complex, target_freq)
    s_lna = interpolate_s_params(lna, lna_complex, target_freq)
    s_MN_xy = interpolate_s_params(MN_xy, MN_xy_complex, target_freq)
    s_MN_z = interpolate_s_params(MN_z, MN_z_complex, target_freq)
    s_balun_a = interpolate_s_params(balun_a, balun_a_complex, target_freq)
    s_balun_d = interpolate_s_params(balun_d, balun_d_complex, target_freq)

    # VGA相位补偿
    s_vga[:, 2] = s_vga[:, 2] * np.exp(1j * theta1)
    s_vga[:, 3] = s_vga[:, 3] * np.exp(2j * theta1)

    # Load插值与相位补偿
    interp_load = interp1d(ADload[:, 0], load_complex, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
    s_load = interp_load(target_freq)
    s_load = s_load * np.exp(2j * theta1)

    # 转换Z_ant为复数形式
    # 【修复】原代码 z_ant[:, 2*i] 会把第0列(频率, MHz)错当成通道1阻抗实部, 这里改为
    # 正确列: re(Z11)=col1,im(Z11)=col2; re(Z22)=col3,im(Z22)=col4; re(Z33)=col5,im(Z33)=col6
    z_ant_freq_hz = z_ant[:, 0] * 1e6  # Z_ant 文件自带频率列, 单位 MHz -> Hz
    Z_ant_raw = np.zeros((len(z_ant), 3), dtype=np.complex128)
    for i in range(3):
        Z_ant_raw[:, i] = z_ant[:, 1 + 2 * i] + 1j * z_ant[:, 2 + 2 * i]
    # 【新增】Z_ant 同样插值到 target_freq (真实频点), 而不是假设行数与freq_num一一对应
    Z_ant = np.zeros((freq_num, 3), dtype=np.complex128)
    for i in range(3):
        interp_re = interp1d(z_ant_freq_hz, Z_ant_raw[:, i].real, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_im = interp1d(z_ant_freq_hz, Z_ant_raw[:, i].imag, kind='cubic', bounds_error=False, fill_value='extrapolate')
        Z_ant[:, i] = interp_re(target_freq) + 1j * interp_im(target_freq)

    # ===================== 6. S参数转A参数 =====================
    def s2a(Sdata):
        """S参数转A参数 (S->A矩阵转换)"""
        row = len(Sdata)
        Adata = np.zeros((row, 2, 2), dtype=np.complex128)
        for i in range(row):
            S11, S21, S12, S22 = Sdata[i, :4]
            # A参数计算公式
            A11 = ((1 + S11) * (1 - S22) + S21 * S12) / (2 * S21)
            A12 = ((1 + S11) * (1 + S22) - S21 * S12) / (2 * S21)
            A21 = ((1 - S11) * (1 - S22) - S21 * S12) / (2 * S21)
            A22 = ((1 - S11) * (1 + S22) + S21 * S12) / (2 * S21)
            Adata[i, :, :] = np.array([[A11, A12], [A21, A22]])
            # 去归一化: B *= Z0, C /= Z0 (Z0 = 50Ω)
            Adata[i, 0, 1] *= 50
            Adata[i, 1, 0] /= 50
        return Adata

    def a2s(Adata):
        """A参数转S参数 (A->S矩阵转换)"""
        Freq = len(Adata)
        Sdata = np.zeros((Freq, 4), dtype=np.complex128)
        for i in range(Freq):
            A = Adata[i, :, :]
            A11, A12 = A[0, 0], A[0, 1]
            A21, A22 = A[1, 0], A[1, 1]
            detA = np.linalg.det(A)
            denominator = A11 + A12 + A21 + A22
            S11 = (A11 + A12 - A21 - A22) / denominator
            S21 = 2 / denominator
            S12 = 2 * detA / denominator
            S22 = (-A11 + A12 - A21 + A22) / denominator
            Sdata[i, :] = [S11, S21, S12, S22]
        return Sdata

    # 计算各模块A参数
    a_lna = s2a(s_lna)
    a_MN_xy = s2a(s_MN_xy)
    a_MN_z = s2a(s_MN_z)
    a_balun_a = s2a(s_balun_a)
    a_balun_d = s2a(s_balun_d)
    a_vga = s2a(s_vga)
    a_cable = s2a(s_cable)

    # 计算总A参数 (矩阵级联)
    a_total = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    a_total_z = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    for i in range(freq_num):
        # XY通道: balun_a -> MN_xy -> LNA -> cable -> vga
        a_total[i] = (a_balun_a[i] @ a_MN_xy[i] @ a_lna[i] @ a_cable[i] @ a_vga[i])
        # Z通道: balun_a -> MN_z -> LNA -> cable -> vga
        a_total_z[i] = (a_balun_a[i] @ a_MN_z[i] @ a_lna[i] @ a_cable[i] @ a_vga[i])

    # ===================== 7. 计算Zload和Zad =====================
    Z_load = 50 * (1 + s_load) / (1 - s_load)
    Z_ad = np.zeros(freq_num, dtype=np.complex128)
    for i in range(freq_num):
        A_d = a_balun_d[i]
        Z_ad[i] = (A_d[0, 1] - Z_load[i] * A_d[1, 1]) / (A_d[1, 0] * Z_load[i] - A_d[0, 0])

    # ===================== 8. 计算输入阻抗Zin =====================
    Zin = np.zeros(freq_num, dtype=np.complex128)
    Zin_z = np.zeros(freq_num, dtype=np.complex128)
    Zin_vga = np.zeros(freq_num, dtype=np.complex128)
    S11_vga = np.zeros(freq_num)
    VSWR_vga = np.zeros(freq_num)

    for i in range(freq_num):
        # 计算各通道Zin
        A_t = a_total[i]
        Zin[i] = (A_t[0, 0] * Z_load[i] + A_t[0, 1]) / (A_t[1, 0] * Z_load[i] + A_t[1, 1])

        A_tz = a_total_z[i]
        Zin_z[i] = (A_tz[0, 0] * Z_load[i] + A_tz[0, 1]) / (A_tz[1, 0] * Z_load[i] + A_tz[1, 1])

        A_v = a_vga[i]
        Zin_vga[i] = (A_v[0, 0] * Z_load[i] + A_v[0, 1]) / (A_v[1, 0] * Z_load[i] + A_v[1, 1])

        # 计算S11和VSWR
        s11 = (Zin_vga[i] - 50) / (Zin_vga[i] + 50)
        S11_vga[i] = 20 * np.log10(np.abs(s11))
        VSWR_vga[i] = (1 + np.abs(s11)) / (1 - np.abs(s11))

    # ===================== 9. 计算Voc =====================
    V2 = Vad
    I2 = np.zeros_like(V2, dtype=np.complex128)
    V1 = np.zeros_like(V2, dtype=np.complex128)
    I1 = np.zeros_like(V2, dtype=np.complex128)
    Voc = np.zeros_like(V2, dtype=np.complex128)

    for freq_idx in range(freq_num):
        # 计算I2 = V2 / Z_ad
        I2[freq_idx, :] = V2[freq_idx, :] / Z_ad[freq_idx]

        # 计算总矩阵 (balun_d级联到total)
        matrix = a_total[freq_idx] @ a_balun_d[freq_idx]
        matrix_z = a_total_z[freq_idx] @ a_balun_d[freq_idx]

        # 计算V1 (XY通道和Z通道分开)
        V1[freq_idx, 0:2] = matrix[0, 0] * V2[freq_idx, 0:2] + matrix[0, 1] * I2[freq_idx, 0:2]
        V1[freq_idx, 2] = matrix_z[0, 0] * V2[freq_idx, 2] + matrix_z[0, 1] * I2[freq_idx, 2]

        # 计算I1
        I1[freq_idx, 0:2] = matrix[1, 0] * V2[freq_idx, 0:2] + matrix[1, 1] * I2[freq_idx, 0:2]
        I1[freq_idx, 2] = matrix_z[1, 0] * V2[freq_idx, 2] + matrix_z[1, 1] * I2[freq_idx, 2]

        # 计算Voc
        Voc[freq_idx, 0:2] = I1[freq_idx, 0:2] * (Zin[freq_idx] + Z_ant[freq_idx, 0:2])
        Voc[freq_idx, 2] = I1[freq_idx, 2] * (Zin_z[freq_idx] + Z_ant[freq_idx, 2])

    return Voc


# ===================== 辅助函数：读取S参数文件 =====================
def _detect_touchstone_format(file_path):
    """从 Touchstone 文件头检测是 MA (线性幅度) 还是 DB (dB) 格式。"""
    with open(file_path, 'r') as f:
        for line in f:
            tline = line.strip()
            if not tline or not tline.startswith('#'):
                continue
            parts = tline.lower().split()
            if 'ma' in parts:
                return True   # MA 格式，幅度已是线性值
            if 'db' in parts:
                return False  # DB 格式，需要 10^(dB/20) 转换
    return False  # 默认当作 DB


def read_s2p(file_path):
    """读取S2P文件，返回 (data_array, is_ma_format)"""
    import os
    is_ma = _detect_touchstone_format(file_path)

    k = 0
    fid = None
    while fid is None and k < 10:
        try:
            fid = open(file_path, 'r')
        except:
            import time
            time.sleep(1)
            k += 1

    if fid is None:
        raise FileNotFoundError(f"ERROR,There is no SP file: {file_path}")

    sdata = []
    for line in fid:
        tline = line.strip()
        if not tline:
            continue
        if tline.startswith(('!', '#')):
            continue
        nums = np.array([float(x) for x in tline.split()])
        if len(nums) >= 9:
            sdata.append(nums[:9])

    fid.close()
    return np.array(sdata), is_ma


def read_s1p(file_path):
    """读取S1P文件，返回 (data_array, is_ma_format)"""
    import os
    is_ma = _detect_touchstone_format(file_path)

    k = 0
    fid = None
    while fid is None and k < 10:
        try:
            fid = open(file_path, 'r')
        except:
            import time
            time.sleep(1)
            k += 1

    if fid is None:
        raise FileNotFoundError(f"ERROR,There is no SP file: {file_path}")

    sdata = []
    for line in fid:
        tline = line.strip()
        if not tline:
            continue
        if tline.startswith(('!', '#')):
            continue
        nums = np.array([float(x) for x in tline.split()])
        if len(nums) >= 3:
            sdata.append(nums[:3])

    fid.close()
    return np.array(sdata), is_ma


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # 构造测试用的Vad矩阵 (频率数 × 3)
    freq_ghz = np.arange(0.03, 0.250001, 0.002) * 1e9
    freq_num = len(freq_ghz)
    Vad_test = np.random.rand(freq_num, 3) + 1j * np.random.rand(freq_num, 3)

    # 调用函数计算Voc
    Voc_result = vad2voc(Vad_test, target_freq_hz = freq_ghz)
    print(f"Voc计算完成，结果形状: {Voc_result.shape}")
    print(f"Voc示例值 (第1个频率点): {Voc_result[0]}")
