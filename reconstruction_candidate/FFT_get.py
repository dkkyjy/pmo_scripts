def fftget(data_ori, N, f1, show_flag):
    # This Python file uses the following encoding: utf-8
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    # = == == == == 本程序作为子程序完成数据傅里叶变化，并根据要求生成参数 == == == == =
    #  ----------------------输入 - ---------------------------------- %
    # % data_ori时域数据,矩阵形式
    # % show_flag生成图形
    # % N傅里叶变化点数
    # % f1单边频率
    # % ----------------------输出 - ---------------------------------- %
    # % data_fft频域复数数据
    # % data_fft_m_single频域幅度单边谱
    # % data_fft 频域相位

    lienum = data_ori.shape[1]
    data_fft = np.zeros((N, lienum), dtype=complex)
    data_fft_m = np.zeros((int(N), lienum))
    # data_fft_m_single = np.zeros((int(N/2), lienum))
    # data_fft_p = np.zeros((int(N), lienum))
    # data_fft_p_single = np.zeros((int(N/2), lienum))

    for i in range(lienum):
        data_fft[:, i] = fft(data_ori[:, i])

        data_fft_m[:, i] = abs(data_fft[:, i])*2 / N  # 幅度
        data_fft_m[0] = data_fft_m[0] / 2

        data_fft_m_single = data_fft_m[0: len(f1)]  # 单边

        data_fft_p = np.angle(data_fft, deg=True)  # 相位
        data_fft_p = np.mod(data_fft_p, 2*180)  # -pi到pi转为0到2pi
        # data_fft_p_deg = np.rad2deg(data_fft_p)
        data_fft_p_single = data_fft_p[0: len(f1)]

    string = np.array(['x', 'y', 'z'])
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        for j in range(lienum):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, j+1)
            plt.plot(f1, data_fft_m_single[:, j])
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("E"+string[j]+"(uv/m)", fontsize=15)
            plt.suptitle("Electric Field Spectrum", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)   # 数值越小离得越远
        plt.show()

        plt.figure(figsize=(9, 3))
        for j in range(lienum):
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.subplot(1, 3, j+1)
            plt.plot(f1, data_fft_p_single[:, j])
            plt.xlabel("Frequency(MHz)", fontsize=15)
            plt.ylabel("E"+string[j]+"(deg)", fontsize=15)
            plt.suptitle("Electric Field Phase", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return np.array(data_fft), np.array(data_fft_m_single), np.array(data_fft_p_single)
