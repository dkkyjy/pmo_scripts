def expan(N, f0, f1, f2, data):
    # This Python file uses the following encoding: utf-8
    import numpy as np
    from numpy.ma import abs
    # = == == == == 本程序作为子程序完成频谱的拓展 == == == == =
    # % N是频点数，也即需要拓展的频谱
    # % f0是频率步进, MHz
    # % f1是需拓展频谱的起始频率，f2是需拓展频谱的截止频率
    # % 该程序只考虑被拓展数据的长度小于floor(N / 2), 比如N = 10, 被拓展数据长度 <= 5;N = 9, 被拓展数据长度 <= 4
    # data是一维

    f = np.arange(0, N) * f0  # 频率序列
    effective = len(data)
    delta_start = abs(f - f1)  # 与f1的差值
    delta_end = abs(f - f2)  # 与f2的差值
    f_hang_start = np.where(delta_start == min(delta_start))  # 差值最小所在行
    f_hang_start = f_hang_start[0][0]
    f_hang_end = np.where(delta_end == min(delta_end))
    f_hang_end = f_hang_end[0][0]
    data_expansion = np.zeros((N), dtype=complex)
    if f_hang_start == 0:
        data_expansion[0] = data[0]
        add = np.arange(f_hang_end + 1, N - effective + 1, 1)
        duichen = np.arange(N - 1, N - effective + 1 - 1, -1)
        data_expansion[add] = 0
        data_expansion[f_hang_start: f_hang_end + 1] = data
        data_expansion[duichen] = data[1:].conjugate()
    else:
        a1 = np.arange(0, f_hang_start - 1 + 1, 1).tolist()
        a2 = np.arange(f_hang_end + 1, N - f_hang_start - effective + 1, 1).tolist()
        a3 = np.arange(N - f_hang_start + 1, N, 1).tolist()  # 需要补0;
        add = a1 + a2 + a3
        add = np.array(add)
        duichen = np.arange(N - f_hang_start, N - f_hang_start - effective, -1)
        data_expansion[add] = 0
        data_expansion[f_hang_start: f_hang_end + 1] = data[:]
        data_expansion[duichen] = data.conjugate()

    return f, data_expansion
