"""
并行重建调度脚本（YAML 输出版）— fixed 版适配
================================================
与原版 ``run_parallel_reconstruction.py`` 的区别: 本脚本调用
``reconstruction_candidate_fluence_error_fixed`` 中的 fixed 版链路, 其接口与
落盘格式与原版不同, 故需单独适配:

  - ``energy_restruction`` 返回 ``(w, Fluence*early_late, delta_angle)``
    (不再是结构化 dict); 标量结果改从 ``shower_output.txt`` 读取。
  - ``*_reconstruction_results.txt`` 为 16 列 (尾部多 ``Fluence_sigma``)。
  - 输入 tree 依类型自动选 DU 坐标表; CoREAS ``tadc`` 需 ``--det-pos du_coords.txt``。

用法:
    python run_parallel_reconstruction_fixed.py \
        --input input.txt \
        --det-pos /path/to/du_coords.txt \
        --ncores 8 \
        --output summary.yaml

candidates YAML 输入 (``--candidates``) 除 ``file``/``index`` 外, 还支持可选外部
先验字段 (PWM/SWM), 给定后跳过对应内部网格搜索:
  PWM: ``du_id`` (天线子集) / ``zenith`` / ``azimuth`` (平面波方向)
  SWM: ``x`` / ``y`` / ``z`` (Xmax 源点) / ``chi_square`` (外部 chi2)

``shower_output.txt`` 列 (见 fixed 版 energy_restruction 的 shower_output):
  0:rec_x_xmax  1:rec_y_xmax  2:rec_z_xmax  3:chi2Sph
  4:rec_theta_sph  5:rec_theta_err  6:rec_phi_sph  7:rec_phi_err
  8:rec_A  9:rec_A_err  10:rec_wc  11:0  12:rec_dw  13:rec_dw_err  14:fmin(chi2/dof)
  15:height  16:rho  17:alpha  18:corex  19:corey  20:corez  21:dXmax
  22..: chi2_che / 各能量列
"""

import argparse
import re
import traceback
import os
import glob
import sys
from typing import Optional, List, NamedTuple
import numpy as np
import yaml
from multiprocessing import Pool
from scipy.signal import hilbert
from logger_config import logger

# ── 把重建代码所在目录加入 sys.path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coordinatesystems
from reconstruction_candidate_fluence_error_fixed import (
    efield_recons_from_efield_PWF,
    energy_restruction,
    efield_rec_dir,
    helper,
    rotation_m,
    find_core,
    magnetic_field_vector,
)

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{8})")  # 从文件名中提取 YYYYMMDD 日期


class TaskSpec(NamedTuple):
    """单个重建任务的规格，替代裸元组，字段自文档化。

    后 7 个字段为可选外部先验，来自 candidates YAML 的 PWM/SWM 结果；
    为 None 时回落到内部网格搜索（与原行为一致）。
    """
    root_path: str
    event_index: int
    det_pos_path: Optional[str] = None
    resume: bool = True
    valid_du_ids: Optional[List[int]] = None   # PWM: 天线子集
    theta: Optional[float] = None              # PWM: 平面波天顶角 (deg)
    phi: Optional[float] = None                # PWM: 平面波方位角 (deg)
    x_xmax: Optional[float] = None             # SWM: Xmax 源点 x (m)
    y_xmax: Optional[float] = None             # SWM: Xmax 源点 y (m)
    z_xmax: Optional[float] = None             # SWM: Xmax 源点 z (m)
    chi_square_external: Optional[float] = None  # SWM: 外部 chi2


def _extract_date_from_filename(filename):
    """从文件名中提取 8 位日期 (YYYYMMDD)，返回 (year, month, day) 元组。"""
    m = _DATE_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"无法从文件名提取日期 (YYYYMMDD): {filename}")
    ds = m.group(1)
    return ds[:4], ds[4:6], ds[6:8]


def _build_root_path(filename, base_path):
    """根据文件名和 base_path 构造完整 root 文件路径。

    若 filename 已是完整/相对带目录路径则直接返回; 否则自动拼为
    ``base_path/YYYY/MM/DD/filename`` (日期从 filename 提取)。
    """
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    year, month, day = _extract_date_from_filename(filename)
    return os.path.join(base_path, year, month, day, filename)


def _read_efield_txt(root_path, event_number_int):
    """
    读取 efield_recons_from_efield_PWF 落盘的 *reconstruction_results.txt。

    fixed 版列顺序（16 列, 尾部多 Fluence_sigma）：
      0:event_num  1:gps_time  2:LST  3:du_id
      4:theta_plane  5:phi_plane  6:chi2Plane/dof
      7:du_nanoseconds  8:pos_x  9:pos_y  10:pos_z
      11:Fluence  12:Fluence_theta  13:Fluence_phi  14:Fluence_theta_phi
      15:Fluence_sigma
    """
    filename = (
        os.path.splitext(os.path.basename(root_path))[0]
        + f"_event_{int(event_number_int)}"
    )
    event_dir = os.path.join(efield_rec_dir, filename)
    pattern = os.path.join(event_dir, '*reconstruction_results.txt')
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"找不到 reconstruction_results.txt: {pattern}")
    data = np.loadtxt(hits[0])
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def _read_shower_output(root_path, event_number_int):
    """读取 fixed 版 energy_restruction 落盘的 shower_output.txt（一维 35 列）。"""
    filename = (
        os.path.splitext(os.path.basename(root_path))[0]
        + f"_event_{int(event_number_int)}"
    )
    event_dir = os.path.join(efield_rec_dir, filename)
    pattern = os.path.join(event_dir, 'shower_output.txt')
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"找不到 shower_output.txt: {pattern}")
    return np.loadtxt(hits[0])


def _safe(v):
    """将 numpy 标量 / nan 转为 Python 原生类型，便于 YAML 序列化。"""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None          # YAML 里显示为 null
    return v


def _event_rec_dir(root_path, event_index):
    """返回某事件在 efield_rec_dir 下的结果目录 (与主链路命名一致)。"""
    filename = (
        os.path.splitext(os.path.basename(root_path))[0]
        + f"_event_{int(event_index)}"
    )
    return os.path.join(efield_rec_dir, filename)


# status 约定: 0=成功  1=电场失败  2=能量失败  3=已完成(断点续跑跳过)
_STATUS_RESUMED = 3


def _recompute_antenna_fields(fd, so, rec_dir):
    """断点续跑补全: 从落盘文件确定性复现 antennas 的三个派生字段。

    这三个字段是主链路 ``energy_restruction`` 的中间返回值、未随 summary 落盘,
    但对已完整落盘的事件可零拟合地精确复现（公式逐行搬运自主链路
    ``reconstruction_candidate_fluence_error_fixed.energy_restruction``）：

      - ``omega_rad``    : 各天线相对簇射轴的视角 ω (rad)
      - ``energy_flux``  : Fluence × early_late 修正因子
      - ``polarization`` : 测量偏振角与期望偏振角的最小差 (deg)

    输入:
        fd  : ``*_reconstruction_results.txt`` 的 (n_ant, ≥16) 数组
        so  : ``shower_output.txt`` 的一维数组 (≥ 22 列)
        rec_dir : 该事件结果目录 (含 ``*E_rec_time_all_DUs.txt``)

    返回 ``(omega_rad, energy_flux, polarization)`` 三个 list; 任一步失败返回
    ``None``（调用方回落为空列表，仅丢失这三个非关键字段，不影响主结果）。
    """
    try:
        positions_x, positions_y, positions_z = fd[:, 8], fd[:, 9], fd[:, 10]
        Fluence        = fd[:, 11]

        rec_x_xmax, rec_y_xmax, rec_z_xmax = so[0], so[1], so[2]
        rec_theta_sph, rec_phi_sph = so[4], so[6]

        obs_altitude  = np.mean(positions_z)
        positions_all = np.array([positions_x, positions_y, positions_z]).T

        shower_axis  = helper.spherical_to_cartesian(
            np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph))
        core_x, core_y, core_z, dXmax = find_core(
            rec_x_xmax, rec_y_xmax, rec_z_xmax, rec_theta_sph, rec_phi_sph,
            obs_altitude)
        shower_core = np.array([core_x, core_y, core_z])

        cs = coordinatesystems.cstrafo(
            np.deg2rad(rec_theta_sph), np.deg2rad(rec_phi_sph),
            magnetic_field_vector)

        # ── early_late 与 energy_flux ─────────────────────────────────────────
        ant2Xmax_dis = np.sqrt(
            (positions_x - rec_x_xmax) ** 2
            + (positions_y - rec_y_xmax) ** 2
            + (positions_z - rec_z_xmax) ** 2)
        early_late = np.square(ant2Xmax_dis / dXmax)
        energy_flux = Fluence * early_late

        # ── 视角 ω (omega_rad) ────────────────────────────────────────────────
        obs_vec = np.array([rec_x_xmax - positions_x,
                            rec_y_xmax - positions_y,
                            rec_z_xmax - positions_z])
        l_dist  = np.sqrt((positions_x - rec_x_xmax) ** 2
                          + (positions_y - rec_y_xmax) ** 2
                          + (positions_z - rec_z_xmax) ** 2)
        u_ant   = obs_vec / l_dist
        omega_rad = np.arccos(np.dot(shower_axis, u_ant))

        # ── 偏振角差 (polarization / delta_angle) ────────────────────────────
        R_sph2cart = rotation_m(rec_theta_sph * np.pi / 180,
                                rec_phi_sph * np.pi / 180)
        E_all = np.loadtxt(glob.glob(
            os.path.join(rec_dir, '*E_rec_time_all_DUs.txt'))[0])
        n_time_rec = E_all.shape[1] // 2
        E_rec_time_sph = E_all.reshape(E_all.shape[0], n_time_rec, 2)

        pos_vxB = cs.transform_to_vxB_vxvxB(positions_all, shower_core)

        peak_vB_voltage   = np.zeros(len(positions_x))
        peak_vvB_voltage  = np.zeros(len(positions_x))
        for ant in range(len(positions_x)):
            E_sph_t = np.zeros((E_rec_time_sph.shape[1], 3))
            E_sph_t[:, 0] = E_rec_time_sph[ant, :, 0]
            E_sph_t[:, 1] = E_rec_time_sph[ant, :, 1]
            E_cart_t   = np.dot(E_sph_t, R_sph2cart.T)
            E_shower_t = cs.transform_to_vxB_vxvxB(E_cart_t.T).T
            E_vxB_t, E_vxvxB_t = E_shower_t[:, 0], E_shower_t[:, 1]

            env_vxB, env_vxvxB = np.abs(hilbert(E_vxB_t)), np.abs(hilbert(E_vxvxB_t))
            env_tot = np.sqrt(env_vxB ** 2 + env_vxvxB ** 2)
            t_peak, half = np.argmax(env_tot), np.max(env_tot) / 2.0

            left  = np.where(env_tot[:t_peak] < half)[0]
            t_start = left[-1] + 1 if len(left) > 0 else 0
            right = np.where(env_tot[t_peak:] < half)[0]
            t_end = t_peak + right[0] if len(right) > 0 else len(env_tot)

            peak_vB_voltage[ant]  = np.mean(E_vxB_t[t_start:t_end])
            peak_vvB_voltage[ant] = np.mean(E_vxvxB_t[t_start:t_end])

        angle_meas = np.arctan2(peak_vvB_voltage, peak_vB_voltage) * 180 / np.pi
        rr = np.sqrt(pos_vxB[:, 0] ** 2 + pos_vxB[:, 1] ** 2)
        angle_expected = np.arctan2(
            0.10 * pos_vxB[:, 1] / rr, 1.0 + 0.10 * pos_vxB[:, 0] / rr) * 180 / np.pi
        angle_meas, angle_expected = angle_meas % 180, angle_expected % 180
        delta_angle = np.minimum(np.abs(angle_meas - angle_expected),
                                 180 - np.abs(angle_meas - angle_expected))

        return (_safe_list(np.atleast_1d(omega_rad)),
                _safe_list(np.atleast_1d(energy_flux)),
                _safe_list(np.atleast_1d(delta_angle)))
    except Exception:
        logger.warning(
            "续跑补全 antennas 派生字段失败 (将置空):\n" + traceback.format_exc())
        return None


def _load_completed_entry(root_path, event_index):
    """断点续跑: 若某事件的重建产物已完整落盘, 直接从盘上重建 summary entry。

    判据: 该事件目录同时存在 ``shower_output.txt`` 与 ``*_reconstruction_results.txt``,
    且 ``shower_output.txt`` 可解析、列数达到 fmin (索引 14)。

    返回 entry 字典 (``_status=_STATUS_RESUMED``); 不满足条件时返回 ``None``,
    调用方回落到正常重建流程。

    ``antennas`` 的三个派生字段 (``omega_rad / energy_flux / polarization``) 由
    ``_recompute_antenna_fields`` 从落盘文件确定性复现, 使续跑产物与正常路径等价;
    若复现失败则回落为空列表 (仅丢失这三个非关键字段)。
    """
    d = _event_rec_dir(root_path, event_index)
    so_path = os.path.join(d, 'shower_output.txt')
    rr_hits = glob.glob(os.path.join(d, '*reconstruction_results.txt'))
    if not (os.path.isfile(so_path) and rr_hits):
        return None
    try:
        so = np.atleast_1d(np.loadtxt(so_path))
        if so.size < 15:            # 至少需要到 rec_dw/fmin 之前的字段
            return None
        fd = _read_efield_txt(root_path, event_index)
    except Exception:
        return None

    entry = dict(
        event_num=None,
        file_path=root_path,
        event_index=event_index,
        PWF=dict(rec_theta_plane=None, rec_phi_plane=None, chi2_per_dof=None),
        SWF=dict(rec_x_xmax=None, rec_y_xmax=None, rec_z_xmax=None, chi2_per_dof=None),
        ADF=dict(rec_theta_sph=None, rec_phi_sph=None, rec_A=None, rec_wc=None,
                 rec_dw=None, chi2_per_dof=None),
        antennas=dict(du_ids=[], omega_rad=[], energy_flux=[], polarization=[]),
        _status=_STATUS_RESUMED,
    )
    entry['event_num'] = int(fd[0, 0])
    entry['PWF']['rec_theta_plane'] = _safe(fd[0, 4])
    entry['PWF']['rec_phi_plane'] = _safe(fd[0, 5])
    entry['PWF']['chi2_per_dof'] = _safe(fd[0, 6])
    entry['antennas']['du_ids'] = _safe_int_list(fd[:, 3].astype(int))

    # 从落盘文件复现三个派生字段 (omega_rad / energy_flux / polarization)
    recomputed = _recompute_antenna_fields(fd, so, d)
    if recomputed is not None:
        entry['antennas']['omega_rad'] = recomputed[0]
        entry['antennas']['energy_flux'] = recomputed[1]
        entry['antennas']['polarization'] = recomputed[2]

    entry['SWF']['rec_x_xmax'] = _safe(so[0])
    entry['SWF']['rec_y_xmax'] = _safe(so[1])
    entry['SWF']['rec_z_xmax'] = _safe(so[2])
    entry['SWF']['chi2_per_dof'] = _safe(so[3])

    entry['ADF']['rec_theta_sph'] = _safe(so[4])
    entry['ADF']['rec_phi_sph'] = _safe(so[6])
    entry['ADF']['rec_A'] = _safe(so[8])
    entry['ADF']['rec_wc'] = _safe(so[10])
    entry['ADF']['rec_dw'] = _safe(so[12])
    entry['ADF']['chi2_per_dof'] = _safe(so[14])
    return entry


def _optional_float(payload, key):
    """从 YAML payload 中提取可选浮点数，缺失或为 None 时返回 None。"""
    v = payload.get(key)
    return float(v) if v is not None else None


def load_tasks_from_candidates_yaml(candidate_yaml_path, base_path):
    """从 candidates.yaml 中提取 TaskSpec 任务列表。

    除 ``file`` / ``index`` 外, 还读取可选先验字段 (PWM/SWM):
    ``du_id`` / ``zenith`` / ``azimuth`` / ``x`` / ``y`` / ``z`` / ``chi_square``。
    缺失时对应先验为 None, 行为与旧版一致 (内部网格搜索)。
    """
    tasks = []
    with open(candidate_yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"candidates YAML top-level is not a mapping: {candidate_yaml_path}")

    for event_key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        file_path = payload.get("file")
        event_index = payload.get("index")
        du_ids = payload.get("du_id")
        if file_path is None or event_index is None:
            logger.warning(
                f"Missing file or index in candidate: key={event_key} "
                f"file={file_path} index={event_index}"
            )
            continue
        full_path = _build_root_path(str(file_path), base_path)
        tasks.append(TaskSpec(
            root_path=full_path,
            event_index=int(event_index),
            valid_du_ids=[int(d) for d in du_ids] if isinstance(du_ids, list) and du_ids else None,
            theta=_optional_float(payload, "zenith"),
            phi=_optional_float(payload, "azimuth"),
            x_xmax=_optional_float(payload, "x"),
            y_xmax=_optional_float(payload, "y"),
            z_xmax=_optional_float(payload, "z"),
            chi_square_external=_optional_float(payload, "chi_square"),
        ))

    return tasks


def load_tasks_from_txt(txt_path, base_path):
    """从纯文本列表读取 TaskSpec 任务列表。每行: <filename> <event_index>。

    所有可选先验字段固定为 None（txt 模式不跳过内部网格搜索）。
    """
    tasks = []
    with open(txt_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"第 {lineno} 行格式错误，跳过: {line!r}")
                continue
            full_path = _build_root_path(parts[0], base_path)
            tasks.append(TaskSpec(root_path=full_path, event_index=int(parts[1])))
    return tasks


def _safe_list(arr):
    """将 numpy 数组转为 Python 列表（元素为原生类型，float nan→None）。"""
    return [_safe(v) for v in arr]


def _safe_int_list(arr):
    """将整型 numpy 数组转为 Python int 列表（不做 nan 检查）。"""
    return [int(v) for v in arr]


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_one(args):
    """单个事例全流程：电场重建 → 能量重建 → 返回结构化 dict。

    ``args`` 为 ``TaskSpec``（或向后兼容的裸元组）。外部先验
    （valid_du_ids/theta/phi 用于 PWM; x/y/z_xmax/chi_square 用于 SWM）非空时
    透传给重建函数以跳过内部网格搜索。
    """
    if isinstance(args, TaskSpec):
        spec = args
    else:
        # 向后兼容裸元组 (root_path, event_index[, det_pos_path[, resume]])
        root_path, event_index = args[0], int(args[1])
        det_pos_path = args[2] if len(args) > 2 else None
        resume = args[3] if len(args) > 3 else True
        spec = TaskSpec(root_path=root_path, event_index=event_index,
                        det_pos_path=det_pos_path, resume=resume)

    root_path = spec.root_path
    event_index = int(spec.event_index)
    det_pos_path = spec.det_pos_path
    resume = spec.resume

    # ── 断点续跑：产物已完整落盘则直接读回，跳过全部重型计算 ──────────────────
    if resume:
        cached = _load_completed_entry(root_path, event_index)
        if cached is not None:
            logger.info(
                f"[resume] 跳过已完成事件: {os.path.basename(root_path)}"
                f" evt {event_index} (event_num={cached['event_num']})"
            )
            return cached

    # 默认返回结构（重建失败时仍保留文件路径信息）
    entry = dict(
        event_num=None,
        file_path=root_path,
        event_index=event_index,
        PWF=dict(
            rec_theta_plane=None, rec_phi_plane=None, chi2_per_dof=None,
        ),
        SWF=dict(
            rec_x_xmax=None, rec_y_xmax=None, rec_z_xmax=None, chi2_per_dof=None,
        ),
        ADF=dict(
            rec_theta_sph=None, rec_phi_sph=None,
            rec_A=None, rec_wc=None, rec_dw=None, chi2_per_dof=None,
        ),
        antennas=dict(
            du_ids=[], omega_rad=[], energy_flux=[],
        ),
        _status=0,   # 内部字段，写入 YAML 前去掉
    )

    # ── 步骤 1：电场重建 ──────────────────────────────────────────────────────
    try:
        efield_recons_from_efield_PWF(root_path, event_index, det_pos_path,
                                      spec.valid_du_ids, spec.theta, spec.phi)

        fd = _read_efield_txt(root_path, event_index)

        entry['event_num']                  = int(fd[0, 0])
        entry['PWF']['rec_theta_plane'] = _safe(fd[0, 4])
        entry['PWF']['rec_phi_plane']   = _safe(fd[0, 5])
        entry['PWF']['chi2_per_dof']    = _safe(fd[0, 6])
        entry['antennas']['du_ids']         = _safe_int_list(fd[:, 3].astype(int))
        entry['antennas']['energy_flux']    = _safe_list(fd[:, 11])

    except Exception:
        entry['_status'] = 1
        logger.error(
            f"电场重建失败: {os.path.basename(root_path)} evt {event_index}\n"
            + traceback.format_exc()
        )
        return entry

    # ── 步骤 2：能量重建 ──────────────────────────────────────────────────────
    try:
        w, energy_flux_adf, delta_angle = energy_restruction(
            root_path, event_index, det_pos_path,
            spec.x_xmax, spec.y_xmax, spec.z_xmax, spec.chi_square_external)

        # fixed 版标量结果落盘在 shower_output.txt
        so = _read_shower_output(root_path, event_index)

        entry['SWF']['rec_x_xmax']    = _safe(so[0])
        entry['SWF']['rec_y_xmax']    = _safe(so[1])
        entry['SWF']['rec_z_xmax']    = _safe(so[2])
        entry['SWF']['chi2_per_dof']  = _safe(so[3])

        entry['ADF']['rec_theta_sph'] = _safe(so[4])
        entry['ADF']['rec_phi_sph']   = _safe(so[6])
        entry['ADF']['rec_A']         = _safe(so[8])
        entry['ADF']['rec_wc']        = _safe(so[10])
        entry['ADF']['rec_dw']        = _safe(so[12])
        entry['ADF']['chi2_per_dof']  = _safe(so[14])

        # w / energy_flux_adf / delta_angle 均由 energy_restruction 直接返回,
        # 与 reconstruction_results.txt 中的 DU 顺序一致。
        entry['antennas']['omega_rad'] = _safe_list(np.atleast_1d(w))
        entry['antennas']['energy_flux'] = _safe_list(np.atleast_1d(energy_flux_adf))
        entry['antennas']['polarization'] = _safe_list(np.atleast_1d(delta_angle))

    except Exception:
        entry['_status'] = 2
        logger.error(
            f"能量重建失败: {os.path.basename(root_path)} evt {event_index}\n"
            + traceback.format_exc()
        )
        return entry

    # ── 一致性校验 ────────────────────────────────────────────────────────────
    n_du = len(entry['antennas']['du_ids'])
    n_w  = len(entry['antennas']['omega_rad'])
    n_f  = len(entry['antennas']['energy_flux'])
    if not (n_du == n_w == n_f):
        logger.warning(
            f"数组长度不一致: {os.path.basename(root_path)} evt {event_index}"
            f"  du_ids={n_du}  omega={n_w}  energy_flux={n_f}"
        )
    else:
        logger.info(
            f"event_num={entry['event_num']}  N_DU={n_du}"
            f"  θ={entry['ADF']['rec_theta_sph']:.2f}"
            f"  φ={entry['ADF']['rec_phi_sph']:.2f}"
        )
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# 结果汇总
# ─────────────────────────────────────────────────────────────────────────────

def assemble_summary_yaml(results, output_path):
    """将并行重建结果组装为 YAML 并写入文件。

    返回:
        Tuple[dict, List[int]]: (summary字典, statuses列表)
    """
    summary = {}
    statuses = []          # 提前记录 status，避免 pop 后丢失
    for entry in results:
        status = entry.pop('_status')   # 去掉内部状态字段
        statuses.append(status)
        event_key = entry.pop('event_num')

        # event_num 读取失败时用文件名+index作为备用 key，避免覆盖
        if event_key is None:
            event_key = f"unknown_{os.path.basename(entry['file_path'])}_{entry['event_index']}"

        # 若出现重复 event_num（不同文件的同号事例），加后缀区分
        if event_key in summary:
            suffix = 1
            while f"{event_key}_dup{suffix}" in summary:
                suffix += 1
            event_key = f"{event_key}_dup{suffix}"
            logger.warning(f"重复的 event_num，已重命名为 {event_key}")

        summary[event_key] = entry

    with open(output_path, 'w') as fout:
        yaml.dump(summary, fout, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return summary, statuses


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='粒子重建并行调度脚本（fixed 版, YAML 输出）')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        help='候选事例列表 txt，每行: <root_path> <event_number>',
    )
    input_group.add_argument(
        '--candidates',
        help='candidates.yaml 文件路径，自动从中提取 (file, index) 任务列表',
    )
    parser.add_argument('--output', default=None,
                        help='汇总输出文件 (默认: 根据输入文件名自动生成 {stem}_reconstruction_summary.yaml)')
    parser.add_argument('--ncores', type=int, default=20,
                        help='并行进程数 (默认: 20)')
    parser.add_argument('--base-path', default='TD',
                        help='ROOT 文件基础目录，默认: TD。文件会自动定位到 base-path/YYYY/MM/DD/ 下')
    parser.add_argument('--det-pos', default=None,
                        help='DU 坐标表路径。CoREAS 输入需指向 du_coords.txt'
                             '（默认自动在 root 文件同级/上级目录查找 du_coords.txt）')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用断点续跑：默认会跳过产物(shower_output.txt+'
                             'reconstruction_results.txt)已完整落盘的事件，'
                             '只重算缺失的；加此开关则强制全部重算。')
    args = parser.parse_args()

    # 读取输入列表
    if args.candidates is not None:
        logger.info(f"从 candidates YAML 读取任务: {args.candidates}")
        tasks = load_tasks_from_candidates_yaml(args.candidates, args.base_path)
        input_path = args.candidates
    else:
        tasks = load_tasks_from_txt(args.input, args.base_path)
        input_path = args.input

    # 将 det_pos / resume 附加到每个任务，供 worker 传递给重建函数
    # （TaskSpec 保留 YAML 已读入的 PWM/SWM 先验字段）
    resume = not args.no_resume
    tasks = [t._replace(det_pos_path=args.det_pos, resume=resume) for t in tasks]

    # 自动生成输出文件名: {stem}_reconstruction_summary.yaml
    if args.output is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{stem}_reconstruction_summary.yaml"
    else:
        output_path = args.output

    logger.info(f"共读取 {len(tasks)} 个事例，使用 {args.ncores} 个进程并行重建")

    # 并行执行，输出顺序与输入一致
    with Pool(processes=args.ncores) as pool:
        results = pool.map(reconstruct_one, tasks)

    # 组装 YAML 字典并写文件
    _summary, statuses = assemble_summary_yaml(results, output_path)

    n_ok = sum(1 for s in statuses if s == 0)
    n_fail_field = sum(1 for s in statuses if s == 1)
    n_fail_energy = sum(1 for s in statuses if s == 2)
    n_resumed = sum(1 for s in statuses if s == _STATUS_RESUMED)
    logger.info(
        f"完成: {len(results)} 个事例已处理"
        f"（成功 {n_ok}, 电场失败 {n_fail_field}, 能量失败 {n_fail_energy},"
        f" 续跑跳过 {n_resumed}）"
    )
    logger.info(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
