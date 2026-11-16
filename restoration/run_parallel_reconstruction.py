"""
并行重建调度脚本（YAML 输出版）
================================
读取候选事例列表（txt 或 candidates.yaml），
用 N 个进程并行执行电场重建 + 能量重建，
最终将所有事例的结果汇总为一个 YAML 文件。

用法（txt 输入）:
    python run_parallel_reconstruction.py \
        --input  candidate_list.txt \
        --base-path /path/to/TD \
        --ncores 20

    输出文件自动生成为 candidate_list_reconstruction_summary.yaml。
    也可通过 --output 手动指定。

用法（candidates YAML 输入，自动提取 file/index）:
    python run_parallel_reconstruction.py \
        --candidates ../Reco_Dir/Trigger_20260617_RUN10386_candidates.yaml \
        --base-path /path/to/TD \
        --ncores 20

    输出文件自动生成为 Trigger_20260617_RUN10386_candidates_reconstruction_summary.yaml。

    ROOT 文件路径会自动生成为 base-path/YYYY/MM/DD/filename，
    其中 YYYY/MM/DD 从输入文件名中提取。

YAML 结构示例:
    12345:                        # event_num（从落盘 txt 读取）
      file_path: /path/to/file.root
      event_index: 196
      PWF:
        rec_theta_plane: 150.12
        rec_phi_plane:   235.47
        chi2_per_dof:    1.23e+00
      SWF:
        rec_x_xmax:  -12345.6
        rec_y_xmax:   23456.7
        rec_z_xmax:   15000.0
        chi2_per_dof: 0.98
      ADF:
        rec_theta_sph: 150.34
        rec_phi_sph:   235.61
        rec_A:         1.23e+05
        rec_wc:        0.031
        rec_dw:        2.10
        chi2_per_dof:  0.87
      antennas:
        du_ids:      [101, 203, 305, ...]
        omega_rad:   [0.031, 0.028, ...]
        energy_flux: [1.23e+04, 9.87e+03, ...]
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
from logger_config import logger

# ── 把重建代码所在目录加入 sys.path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reconstruction_candidate import (
    efield_recons_from_efield_PWF,
    energy_restruction,
    efield_rec_dir,
    helper,
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
    valid_du_ids: Optional[List[int]] = None   # PWM: 天线子集
    theta: Optional[float] = None              # PWM: 平面波天顶角 (deg)
    phi: Optional[float] = None                # PWM: 平面波方位角 (deg)
    x_xmax: Optional[float] = None             # SWM: Xmax 源点 x (m)
    y_xmax: Optional[float] = None             # SWM: Xmax 源点 y (m)
    z_xmax: Optional[float] = None             # SWM: Xmax 源点 z (m)
    chi_square_external: Optional[float] = None  # SWM: 外部 chi2


def _extract_date_from_filename(filename):
    """从文件名中提取 8 位日期 (YYYYMMDD)，返回 (year, month, day) 元组。

    例如 ``Trigger_20260604_RUN10379_candidates.yaml`` → ``(2026, 06, 04)``。
    """
    m = _DATE_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"无法从文件名提取日期 (YYYYMMDD): {filename}")
    ds = m.group(1)
    return ds[:4], ds[4:6], ds[6:8]


def _build_root_path(filename, base_path):
    """根据文件名和 base_path 构造完整 root 文件路径。

    输入 filename 可以是只有文件名的 ``Trigger_xxx.root``，也可以是完整路径。
    如果是完整路径则直接返回；否则自动拼接为 ``base_path/YYYY/MM/DD/filename``，
    其中 YYYY/MM/DD 从 filename 中提取。
    """
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    year, month, day = _extract_date_from_filename(filename)
    return os.path.join(base_path, year, month, day, filename)


def _read_efield_txt(root_path, event_number_int):
    """
    读取 efield_recons_from_efield_PWF 落盘的 *reconstruction_results.txt。

    列顺序（见原代码 saved_data[i]）：
      0:event_num  1:gps_time  2:LST  3:du_id
      4:theta_plane  5:phi_plane  6:chi2Plane/dof
      7:du_nanoseconds  8:pos_x  9:pos_y  10:pos_z
      11:Fluence  12:Fluence_theta  13:Fluence_phi  14:Fluence_theta_phi
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


def _safe(v):
    """将 numpy 标量 / nan 转为 Python 原生类型，便于 YAML 序列化。"""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None          # YAML 里显示为 null
    return v


def _optional_float(payload, key):
    """从 YAML payload 中提取可选浮点数，缺失或为 None 时返回 None。"""
    v = payload.get(key)
    return float(v) if v is not None else None


def load_tasks_from_candidates_yaml(candidate_yaml_path, base_path):
    """从 candidates.yaml 中提取 TaskSpec 任务列表。

    candidates YAML 结构（顶层 key 为事件标识，payload 包含 file/index）::

        event_key_1:
          event_number: 12345
          file: Trigger_xxx.root
          index: 196
          ...

    除 ``file``/``index`` 外，还读取可选先验字段 (PWM/SWM):
    ``du_id``/``zenith``/``azimuth``/``x``/``y``/``z``/``chi_square``；
    缺失时对应先验为 None，行为与旧版一致（内部网格搜索）。

    file 字段如只有文件名，会自动补全为 ``base_path/YYYY/MM/DD/file``。

    返回:
        List[TaskSpec]
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
    """从纯文本列表读取 TaskSpec 任务列表。

    每行格式: ``<filename> <event_index>``，支持 # 注释和空行。
    第一列如只有文件名，会自动补全为 ``base_path/YYYY/MM/DD/filename``。
    所有可选先验字段固定为 None（txt 模式不跳过内部网格搜索）。

    返回:
        List[TaskSpec]
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
        # 向后兼容裸元组 (root_path, event_index[, det_pos_path])
        root_path, event_index = args[0], int(args[1])
        det_pos_path = args[2] if len(args) > 2 else None
        spec = TaskSpec(root_path=root_path, event_index=event_index, det_pos_path=det_pos_path)

    root_path = spec.root_path
    event_index = int(spec.event_index)
    det_pos_path = spec.det_pos_path

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

        _pos_x, _pos_y, _pos_z = fd[:, 8], fd[:, 9], fd[:, 10]

    except Exception:
        entry['_status'] = 1
        logger.error(
            f"电场重建失败: {os.path.basename(root_path)} evt {event_index}\n"
            + traceback.format_exc()
        )
        return entry

    # ── 步骤 2：能量重建 ──────────────────────────────────────────────────────
    try:
        energy_result, delta_angle = energy_restruction(
            root_path, event_index, det_pos_path,
            spec.x_xmax, spec.y_xmax, spec.z_xmax, spec.chi_square_external)

        entry['SWF']['rec_x_xmax']    = _safe(energy_result['rec_x_xmax'])
        entry['SWF']['rec_y_xmax']    = _safe(energy_result['rec_y_xmax'])
        entry['SWF']['rec_z_xmax']    = _safe(energy_result['rec_z_xmax'])
        entry['SWF']['chi2_per_dof']  = _safe(energy_result['chi2Sph'])

        entry['ADF']['rec_theta_sph'] = _safe(energy_result['rec_theta_sph'])
        entry['ADF']['rec_phi_sph']   = _safe(energy_result['rec_phi_sph'])
        entry['ADF']['rec_A']         = _safe(energy_result['rec_A'])
        entry['ADF']['rec_wc']        = _safe(energy_result['rec_wc'])
        entry['ADF']['rec_dw']        = _safe(energy_result['rec_dw'])
        entry['ADF']['chi2_per_dof']  = _safe(energy_result['chi2_per_dof'])

        # 重算视角 w，与 energy_restruction 内公式完全一致
        rx, ry, rz     = energy_result['rec_x_xmax'], energy_result['rec_y_xmax'], energy_result['rec_z_xmax']
        theta_s, phi_s = energy_result['rec_theta_sph'], energy_result['rec_phi_sph']
        if not any(np.isnan([rx, ry, rz, theta_s, phi_s])):
            shower_axis = helper.spherical_to_cartesian(
                np.deg2rad(theta_s), np.deg2rad(phi_s)
            )
            obs    = np.array([rx - _pos_x, ry - _pos_y, rz - _pos_z])
            l_dist = np.linalg.norm(obs, axis=0)
            u_ant  = obs / l_dist
            w      = np.arccos(np.clip(np.dot(shower_axis, u_ant), -1.0, 1.0))
            entry['antennas']['omega_rad'] = _safe_list(w)

        entry['antennas']['polarization'] = _safe_list(delta_angle)

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
    parser = argparse.ArgumentParser(description='粒子重建并行调度脚本（YAML 输出）')
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
    args = parser.parse_args()

    # 读取输入列表
    if args.candidates is not None:
        logger.info(f"从 candidates YAML 读取任务: {args.candidates}")
        tasks = load_tasks_from_candidates_yaml(args.candidates, args.base_path)
        input_path = args.candidates
    else:
        tasks = load_tasks_from_txt(args.input, args.base_path)
        input_path = args.input

    # 将 det_pos 附加到每个任务，供 worker 传递给重建函数
    # （TaskSpec 保留 YAML 已读入的 PWM/SWM 先验字段）
    if args.det_pos is not None:
        tasks = [t._replace(det_pos_path=args.det_pos) for t in tasks]

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
    logger.info(
        f"完成: {len(results)} 个事例已处理"
        f"（成功 {n_ok}, 电场失败 {n_fail_field}, 能量失败 {n_fail_energy}）"
    )
    logger.info(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
