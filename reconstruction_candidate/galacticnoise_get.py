def gala(lst, N, f0, f1, show_flag):
    # This Python file uses the following encoding: utf-8
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    import os
    from FFT_get import fftget
    from complex_expansion import expan
    from IFFT_get import ifftget

    # = == == == == 本程序作为子程序完成银河噪声的计算及扩展 == == == == =
    #  ----------------------输入 - ---------------------------------- %
    # lst：Select the galactic noise LST at the LST moment
    # N为扩展长度
    # f0为频率分辨率，f1是单边谱频点
    # % ----------------------输出 - ---------------------------------- %
    # v_complex_double, galactic_v_time

    _PARAM_DIR = os.path.join(os.path.dirname(__file__), 'parameter_files')
    GALAshowFile = os.path.join(_PARAM_DIR, '30_250galactic.mat')
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
