def ifftget(data_ori, N, f1, true):
    # This Python file uses the following encoding: utf-8
    import numpy as np
    from scipy.fftpack import ifft

    # %= == == == == 本程序作为子程序完成数据傅里叶变化，并根据要求生成参数 == == == == =
    # % ----------------------输入 - ---------------------------------- %
    # % data_ori频域数据, 复数
    # % true  1表示该复数是合成的即幅值是真实幅值  2表示该复数是傅里叶变换之后得到的；
    # % N傅里叶变化点数
    # % t时间序列
    # ns
    # % ----------------------输出 - ---------------------------------- %
    # % data_ifft 时域数据

    lienum = data_ori.shape[1]

    # %= == == == == == == == == == == == == == == 先画频谱相位 == == == == == ==
    data_ori_m = np.zeros((int(N), lienum))
    data_ori_p = np.zeros((int(N), lienum))
    if true == 1:
        for i in range(lienum):
            data_ori_m[:, i] = abs(data_ori[:, i])  # 幅度
            data_ori_m_single = data_ori_m[0: len(f1)]  # 单边

            data_ori_p[:, i] = np.angle(data_ori[:, i], deg=True)  # 相位
            data_ori_p[:, i] = np.mod(data_ori_p[:, i], 2*180)  # -pi到pi转为0到2pi
            data_ori_p_single = data_ori_p[0: len(f1)]

    elif true == 2:
        for i in range(lienum):
            data_ori_m[:, i] = abs(data_ori[:, i]) * 2 / N
            data_ori_m[0] = data_ori_m[0] / 2

            data_ori_m_single = data_ori_m[0: len(f1)]  # 单边

            data_ori_p = np.angle(data_ori, deg=True)  # 相位
            data_ori_p = np.mod(data_ori_p, 2*180)  # -pi到pi转为0到2pi
            data_ori_p_single = data_ori_p[0: len(f1)]

    # % % 时域
    data_ifft = np.zeros((N, lienum))
    for i in range(lienum):
        data_ifft[:, i] = ifft(data_ori[:, i]).real

    return np.array(data_ifft), np.array(data_ori_m_single), np.array(data_ori_p_single)
