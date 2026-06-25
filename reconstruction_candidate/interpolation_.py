import numpy as np
from numpy.ma import abs
import math

def inter(data_complex_five,e_theta,e_phi):

    #RK: Interpolation techique used here is similar to the one used in grandlib.

    # =================This subroutine is an interpolation procedure for a five-dimensional function in a specific theta phi direction======================
    # data_complex_five is the original data, 5 dimensions
    # e_theta,e_phi is the incident direction

    #    Four adjacent points
    down_theta = math.floor(e_theta)
    up_theta   = math.ceil(e_theta)
    down_phi   = math.floor(e_phi)
    up_phi     = math.ceil(e_phi)

    a = abs(round(e_theta) - e_theta)
    b = abs(round(e_phi) - e_phi)

    numf=data_complex_five.shape[0]
    #    interpolation
    data_complex = np.zeros((181, 361), dtype=complex)
    data_new = np.zeros((numf, 3, 3), dtype=complex)
    for i in range(numf):
        for j in range(3):
            for k in range(3):
                data_complex[:, :] = data_complex_five[i, j, k, :, :]
                L1 = data_complex[down_theta, down_phi]
                L2 = data_complex[up_theta, down_phi]
                L3 = data_complex[down_theta, up_phi]
                L4 = data_complex[up_theta, up_phi]

                rt1 = (e_theta - down_theta) / 1.
                rt0 = 1.0 - rt1
                rp1 = (e_phi - down_phi) / 1.
                rp0 = 1.0 - rp1

                data_new[i, j, k] = rt0 * rp0 * L1 + rt1 * rp0 * L2 + rt0 * rp1 * L3 + rt1 * rp1* L4

    return np.array(data_new)


def inter_old(data_complex_five,e_theta,e_phi):
    # This Python file uses the following encoding: utf-8

    # =================此子程序是特定theta phi方向五维函数的插值程序======================
    # data_complex_five 是原始数据，5维
    # e_theta,e_phi是入射方向

    #    Four adjacent points
    down_theta = math.floor(e_theta)
    up_theta = math.ceil(e_theta)
    down_phi = math.floor(e_phi)
    up_phi = math.ceil(e_phi)

    a = abs(round(e_theta) - e_theta)
    b = abs(round(e_phi) - e_phi)

    numf=data_complex_five.shape[0]
    #    interpolation
    data_complex = np.zeros((181, 361), dtype=complex)
    data_new = np.zeros((numf, 3, 3), dtype=complex)
    for i in range(numf):
        for j in range(3):
            for k in range(3):
                data_complex[:, :] = data_complex_five[i, j, k, :, :]
                L1 = data_complex[down_theta, down_phi]
                L2 = data_complex[up_theta, down_phi]
                L3 = data_complex[down_theta, up_phi]
                L4 = data_complex[up_theta, up_phi]
                if a < 1e-5:
                    if b < 1e-5:
                        data_current = data_complex[round(e_theta), round(e_phi)]
                    else:
                        ratio = (e_phi - down_phi) / 1
                        data_current = ratio * (
                                data_complex[round(e_theta), down_phi] + data_complex[round(e_theta), up_phi])
                else:
                    if b < 1e-5:
                        ratio = (e_theta - down_theta) / 1
                        data_current = ratio * (
                                data_complex[down_theta, round(e_phi)] + data_complex[up_theta, round(e_phi)])
                    else:
                        ratio = (e_theta - down_theta) * (e_phi - down_phi) / 1
                        data_current = ratio * (L1 + L2 + L3 + L4)
                data_new[i, j, k] = data_current
    return np.array(data_new)
