def time_data_get(filename, Ts, show_flag):
    # This Python file uses the following encoding: utf-8
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    # = == == == == 本程序作为子程序完成shower时域信号的读入，并截断取部分信号或扩展信号长度，并根据时域要求生成参数 == == == == =
    #  ----------------------输入 - ---------------------------------- %
    # filename地址
    # Ts时间间隔
    # % show_flag是否生成图形

    # % ----------------------输出 - ---------------------------------- %
    # % t时间序列, 单位ns
    # % E_shower_cut对应于时间序列的三极化分量，默认单位为uv
    # % fs % 采样频率, 默认单位MHZ
    # % f0; % 基频, 即频率分辨率，默认单位为MHZ
    # % f频率序列，默认单位为MHz
    # % f1 单边谱频率序列，默认单位为MHz

    t = np.loadtxt(filename, usecols=(0))
    ex = np.loadtxt(filename, usecols=(1))
    ey = np.loadtxt(filename, usecols=(2))
    ez = np.loadtxt(filename, usecols=(3))

    # = == == == == == == == == == =时频参数生成 == == == == == == == == == == == == == == == == == == == == ==
    # = == == =为了使频率分辨率为1MHz，采样点数 = 采样频率 == == == =

    fs = 1 / Ts * 1000  # 采样频率, MHZ
    N = math.ceil(fs)
    f0 = fs / N  # 基频, 即频率分辨率
    f = np.arange(0, N) * f0  # 频率序列
    f1 = f[0:int(N / 2) + 1]
    # 只取一半, 注意奇数偶数，奇数：第floor(N / 2 + 1)个和floor(N / 2 + 1) + 1共轭；
    # 偶数：第floor(N / 2 + 1) - 1个和floor(N / 2 + 1) + 1共轭；

    # = == == == 把原始信号长度改为与N相同======================
    t_cut = np.zeros((N))
    ex_cut = np.zeros((N))
    ey_cut = np.zeros((N))
    ez_cut = np.zeros((N))

    lt = len(t)
    if N <= lt:
        # ============================为了避免峰值未取到，判断峰值是否在N内 == == == == == == == =
        posx = np.argmax(ex)
        posy = np.argmax(ey)
        posz = np.argmax(ez)
        hang = max(posx, posy, posz)
        if hang >= N:  # 如果不在
            t_cut[0: N - 500] = t[hang - (N - 500): hang]
            t_cut[N - 500: N] = t[hang: hang + 500]

            ex_cut[0: N - 500] = ex[hang - (N - 500): hang]
            ex_cut[N - 500: N] = ex[hang: hang + 500]

            ey_cut[0: N - 500] = ey[hang - (N - 500): hang]
            ey_cut[N - 500: N] = ey[hang: hang + 500]

            ez_cut[0: N - 500] = ez[hang - (N - 500): hang]
            ez_cut[N - 500: N] = ez[hang: hang + 500]
        else:
            t_cut[0:N] = t[0:N]
            ex_cut[0: N] = ex[0: N]
            ey_cut[0: N] = ey[0: N]
            ez_cut[0: N] = ez[0: N]
    else:
        t_cut[0:lt] = t[0:]
        ex_cut[0: lt] = ex[0:]
        ey_cut[0: lt] = ey[0:]
        ez_cut[0: lt] = ez[0:]
        a = t[-1] + Ts
        b = t[-1] + Ts * (N - lt + 2)  # 小数加减精度总出问题，这里+2之后不管小数变成.9999还是.000001都能保证取到前n-lt个数
        add = np.arange(a, b, Ts)
        t_cut[lt:] = add[:(N - lt)]
    if show_flag == 1:
        plt.figure(figsize=(9, 3))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.subplot(1, 3, 1)
        plt.plot(t_cut, ex_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ex(uv/m)", fontsize=15)

        plt.subplot(1, 3, 2)
        plt.plot(t_cut, ey_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ey(uv/m)", fontsize=15)

        plt.subplot(1, 3, 3)
        plt.plot(t_cut, ez_cut)
        plt.xlabel("time(ns)", fontsize=15)
        plt.ylabel("Ez(uv/m)", fontsize=15)

        plt.suptitle("E Fields of Shower in Time Domain", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return np.array(t_cut), np.array(ex_cut), np.array(ey_cut), np.array(ez_cut), fs, f0, f, f1, N
