import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from numpy.ma import log10, abs
from interpolation_ import inter
from scipy import interpolate
from complex_expansion import expan

def CEL(e_theta, e_phi, N, f0, unit, show_flag):
    # This Python file uses the following encoding: utf-8

    # from complex_expansion import expan

    # = == == == == This program is used as a subroutine to complete the calculation and expansion of the 30-250MHz complex equivalent length == == == == =
    #  ----------------------input- ---------------------------------- %
    # filename address, S1P file (delete the previous string of s1p file in advance, put the test results of the three ports in the antennaVSWR folder, and name them 1 2 3 in turn)
    # % show_flag :flag of showing picture
    # N is the extended length
    # e_theta, e_phi is the direction of incidence
    # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
    # f0 is the frequency resolution,
    # % ----------------------output - ---------------------------------- %
    # f frequency sequence, the default unit is MHz
    # Lce_complex_expansion is the equivalent length of a specific incident direction
    # s11_complex is the antenna test data

    # Complex electric field 30-250MHz
    #REfile = ".//Complex_RE.mat"
    REfile = "./data/Complex_RE.mat"
    RE = h5py.File(REfile, 'r')
    RE_zb = np.transpose(RE['data_rE_ALL'])
    re_complex = RE_zb.view('complex')
    f_radiation = np.transpose(RE['f_radiation'])  # mhz
    effective = max(f_radiation.shape[0], f_radiation.shape[1])

    e_radiation = inter(re_complex, e_theta, e_phi)

    # 测试s1p
    s11_complex = np.zeros((effective, 3), dtype=complex)  # 3 ports
    for p in range(3):
        str_p = str(p + 1)
        #filename = ".//antennaVSWR//" + str_p + ".s1p"
        filename = "./data/antennaVSWR/" + str_p + ".s1p"
        freq = np.loadtxt(filename, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            re = np.loadtxt(filename, usecols=1)
            im = np.loadtxt(filename, usecols=2)
            db = 20 * log10(abs(re + 1j * im))
        elif unit == 1:
            db = np.loadtxt(filename, usecols=1)
            deg = np.loadtxt(filename, usecols=2)
            mag = 10 ** (db / 20)
            re = mag * np.cos(deg / 180 * math.pi)
            im = mag * np.sin(deg / 180 * math.pi)
        if p == 0:
            dB = np.zeros((3, len(freq)))
        dB[p] = db

        # Interpolation is a data of 30-250mhz interval 1mhz
        freqnew = np.arange(30, 251, 1)
        f_re = interpolate.interp1d(freq, re, kind="cubic")
        renew = f_re(freqnew)
        f_im = interpolate.interp1d(freq, im, kind="cubic")
        imnew = f_im(freqnew)
        s11_complex[:, p] = renew + 1j * imnew

    # %Reduced current
    z0 = 50
    a1 = 1
    I_complex = 1 / math.sqrt(z0) * (1 - s11_complex) * a1

    # %Denominator
    eta = 120 * math.pi
    c = 3 * 1e8
    f_unit = 1e6
    lamda = c / (f_radiation * f_unit)  # m
    lamda = np.transpose(lamda)
    k = 2 * math.pi / lamda
    fenmu = 1j * (I_complex / (2 * lamda) * eta)

    # Equivalent length
    # Extend the frequency band
    f1 = f_radiation[0][0]
    f2 = f_radiation[0][-1]

    Lce_complex_short = np.zeros((effective, 3, 3), dtype=complex)
    Lce_complex_expansion = np.zeros((N, 3, 3), dtype=complex)
    for i in range(3):  # Polarization i = 1, 2, 3 respectively represent xyz polarization
        for p in range(3):
            # Xyz polarization of a single port
            Lce_complex_short[:, i, p] = e_radiation[:, p, i] / fenmu[:, p]
            [f, Lce_complex_expansion[:, i, p]] = expan(N, f0, f1, f2, Lce_complex_short[:, i, p])
    if show_flag == 1:
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        for p in range(3):
            plt.plot(freq, dB[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right', fontsize=15)
        plt.title(r"$\mathregular{S_{11}}$" + " of Antenna", fontsize=15)
        plt.show()

        string = np.array(['X', 'Y', 'Z'])
        plt.figure(figsize=(9, 3))
        for i in range(3):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, i + 1)
            for j in range(3):
                plt.plot(f, abs(Lce_complex_expansion[:, j, i]))
            plt.legend(["X polarization", "Y polarization", "Z polarization"], loc='upper right', fontsize=9)
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("Equivalent length / m", fontsize=15)
            plt.title(r"$\mathregular{L_e}$" + " in " + string[i] + " direction", fontsize=15)
            plt.xlim(30, 250)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # The smaller the value, the farther away
        plt.show()
    return Lce_complex_expansion, s11_complex


def CEL_old(e_theta, e_phi, N, f0, unit, show_flag):
    # This Python file uses the following encoding: utf-8

    # = == == == == 本程序作为子程序完成30-250MHz复等效长度的计算及扩展 == == == == =
    #  ----------------------输入 - ---------------------------------- %
    # filename地址,S1P文件（提前把s1p文件删除前面字符串，将三个端口的测试结果放在antennaVSWR文件夹下，并依次命名为1 2 3)
    # % show_flag是否生成图形
    # N为扩展长度
    # e_theta,e_phi是入射方向
    # unit为0则测试数据采用实部虚部形式，1为db与相位形式
    # f0为频率分辨率，
    # % ----------------------输出 - ---------------------------------- %
    # f频率序列，默认单位为MHz
    # Lce_complex_expansion为特定入射方向的等效长度
    # s11_complex为天线测试数据

    # 复数电场30-250MHz
    #REfile = "./data/Complex_RE.mat"
    REfile = "/Users/zkw/scripts/voltagecalculate_1125/data/Complex_RE.mat"
    #REfile = "../voltagecalculate_1125/data/Complex_RE.mat"
    RE = h5py.File(REfile,"r")
    RE_zb = np.transpose(RE['data_rE_ALL'])
    re_complex = RE_zb.view('complex')
    f_radiation = np.transpose(RE['f_radiation'])  # mhz
    effective = max(f_radiation.shape[0], f_radiation.shape[1])

    e_radiation = inter(re_complex, e_theta, e_phi)

    # 测试s1p
    s11_complex = np.zeros((effective, 3), dtype=complex)  # 3个端口
    for p in range(3):
        str_p = str(p + 1)
        filename = "/Users/zkw/scripts/voltagecalculate_1125/data/antennaVSWR/" + str_p + ".s1p"
        freq = np.loadtxt(filename, usecols=0) / 1e6  # HZ to MHz
        if unit == 0:
            re = np.loadtxt(filename, usecols=1)
            im = np.loadtxt(filename, usecols=2)
            db = 20 * log10(abs(re + 1j * im))
        elif unit == 1:
            db = np.loadtxt(filename, usecols=1)
            deg = np.loadtxt(filename, usecols=2)
            mag = 10 ** (db / 20)
            re = mag * np.cos(deg / 180 * math.pi)
            im = mag * np.sin(deg / 180 * math.pi)
        if p == 0:
            dB = np.zeros((3, len(freq)))
        dB[p] = db

        # 插值为30-250mhz间隔1mhz一个数据
        freqnew = np.arange(30, 251, 1)
        f_re = interpolate.interp1d(freq, re, kind="cubic")
        renew = f_re(freqnew)
        f_im = interpolate.interp1d(freq, im, kind="cubic")
        imnew = f_im(freqnew)
        s11_complex[:, p] = renew + 1j * imnew

    # %归算电流
    z0 = 50
    a1 = 1
    I_complex = 1 / math.sqrt(z0) * (1 - s11_complex) * a1

    # %分母
    eta = 120 * math.pi
    c = 3 * 1e8
    f_unit = 1e6
    lamda = c / (f_radiation * f_unit)  # m
    lamda = np.transpose(lamda)
    k = 2 * math.pi / lamda
    fenmu = 1j * (I_complex / (2 * lamda) * eta)

    #  等效长度
    # 扩展频带
    f1 = f_radiation[0][0]
    f2 = f_radiation[0][-1]

    Lce_complex_short = np.zeros((effective, 3, 3), dtype=complex)
    Lce_complex_expansion = np.zeros((N, 3, 3), dtype=complex)
    for i in range(3):  # 极化i = 1, 2, 3分别表示xyz极化
        for p in range(3):
            #  单个端口的xyz极化
            Lce_complex_short[:, i, p] = e_radiation[:, p, i] / fenmu[:, p]
            [f, Lce_complex_expansion[:, i, p]] = expan(N, f0, f1, f2, Lce_complex_short[:, i, p])
    if show_flag == 1:
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        for p in range(3):
            plt.plot(freq, dB[p])
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc='lower right', fontsize=15)
        plt.title(r"$\mathregular{S_{11}}$" + " of Antenna", fontsize=15)
        plt.show()

        string = np.array(['X', 'Y', 'Z'])
        plt.figure(figsize=(9, 3))
        for i in range(3):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, i + 1)
            for j in range(3):
                plt.plot(f, abs(Lce_complex_expansion[:, j, i]))
            plt.legend(["X polarization", "Y polarization", "Z polarization"], loc='upper right', fontsize=9)
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("Equivalent length / m", fontsize=15)
            plt.title(r"$\mathregular{L_e}$" + " in " + string[i] + " direction", fontsize=15)
            plt.xlim(30, 250)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 数值越小离得越远
        plt.show()
    return Lce_complex_expansion, s11_complex,Lce_complex_short
