"""
定制化筛选子程序：shower footprint 形状判据 + d_max ring cut。

从 Analysis_pipeline 移植（_Integrate_script_dataclean_remove_reco.py 的
filter_by_shower_footprint 与 _analyze_zenith_dispersion.py 的 ring cut），
作为独立子程序加入 pmo_scripts 生产线：

- 默认在 main.py 的 PWM 与 SWM 阶段之间启用（不改变任何现有阶段逻辑）；
- 可用 --skip-footprint / --skip-ringcut 分别关闭；
- 仅过滤内存中的 state，不新增缓存文件后缀，不改动既有 YAML 约定；
- 亦支持独立运行：
    python -m find_event.shower_footprint_filter Trigger_x_PWM.yaml --det-pos _gp65_rtksort.txt

判据 1（filter_by_shower_footprint，宇宙线 shower footprint 几何，optimized 版）：
  触发 DU 沿入射方位角方向应拉长（observed_ratio > min_observed_ratio），
  且 expected/observed 扁率比不超过该天顶角处 p90（secant 拟合）的 shape_relax 倍，
  大多数 DU 落在放大椭圆内（min_du_fraction_in_ellipse），
  触发时间残差 RMS 不超过天顶角依赖阈值
  （time residual p90 的 secant 拟合 + time_offset_ns 后乘 time_relax）。

判据 2（filter_by_ring_cut，触发天线弥散度，天顶角依赖）：
  仅对 n_du < n_du_max 的事例：
      d_max <= dmax_cut_threshold(zenith_pwm)  且  d_mean < d_mean_max
  其中 d_max/d_mean 为水平几何中心到各触发 DU 距离的最大/平均值。

作者注：阈值参数为 Analysis_pipeline 当前版本的原样移植
（base=800 m 即 2000-1200，d_mean_max=700 m，n_du_max=6），
如需调整请通过函数参数传入，勿改默认值。
"""

import os
import numpy as np
from logger_config import logger

C_LIGHT_NS = 299792458.0 / 1e9 / 1.000259  # m/ns


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _incident_angles(direction):
    """入射天顶角(0-90)与方位角(atan2(-y,-x)，与 SWM 图一致)。"""
    v = np.array(direction, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.nan, np.nan
    v = v / n
    zenith = np.degrees(np.arccos(np.clip(abs(v[2]), -1.0, 1.0)))
    azimuth = (np.degrees(np.arctan2(-v[1], -v[0])) + 360.0) % 360.0
    return zenith, azimuth


def _positions_as_int_key(positions):
    """统一 DU 坐标字典为 int 键。"""
    return {int(k): np.asarray(v, dtype=float) for k, v in positions.items()}


def _event_matches(event_times, positions_int):
    """把 {du_id: t_ns} 转成 (ids, times, pos) 数组；缺坐标的 DU 跳过。"""
    ids, times = [], []
    for raw_id, raw_t in event_times.items():
        du_id = int(raw_id)
        if du_id not in positions_int:
            logger.warning(f"DU {du_id} missing position; skipped in customized cuts")
            continue
        ids.append(du_id)
        times.append(float(raw_t))
    if not ids:
        return None
    pos = np.array([positions_int[d] for d in ids], dtype=float)
    return ids, np.array(times, dtype=float), pos


# ---------------------------------------------------------------------------
# 判据 1：shower footprint 形状筛选
# ---------------------------------------------------------------------------
def filter_by_shower_footprint(
    event_times,
    directions,
    positions,
    output_prefix=None,
    ellipse_ratio_tol=2.0,          # 保留兼容（新判据不再使用）
    max_time_residual_ns=200.0,     # 保留兼容（新判据不再使用）
    min_du_fraction_in_ellipse=0.8,
    ellipse_scale=1.8,
    cone_half_angle_deg=2.0,
    min_observed_ratio=1.2,
    max_expected_ratio=15.0,
    shape_relax=1.5,
    time_offset_ns=80.0,
    time_relax=2.0,
    verbose=True,
):
    """
    宇宙线 shower footprint 几何筛选（移植自 Analysis_pipeline）。

    参数：
      event_times : {event_key: {du_id: t_ns}}
      directions  : {event_key: [x, y, z]}（PWM 传播方向，与 state["directions"] 相同）
      positions   : {du_id: [x, y, z]}
      output_prefix : 若非 None，写出诊断 CSV（_footprint_filter_diagnostics.csv）
                      与诊断散点图（_footprint_ratio_diagnostic.png）

    返回：保留的 event_key 列表（保持输入顺序）。
    """
    positions_int = _positions_as_int_key(positions)
    keys = list(directions.keys())

    kept_keys = []
    diag_lines = [
        "key,n_du,zenith,azimuth,expected_ratio,observed_ratio,"
        "spatial_ok,frac_in_ellipse,ellipse_ok,time_residual_rms,time_ok,kept"
    ]
    diag_zenith, diag_azimuth = [], []
    diag_observed, diag_expected = [], []
    diag_kept, diag_rms, diag_ndu = [], [], []

    for key in keys:
        norm = _event_matches(event_times.get(key, {}), positions_int)
        if norm is None:
            continue
        ids, times, pos_xyz = norm
        n_du = len(ids)
        if n_du < 4:
            continue

        zenith, azimuth = _incident_angles(directions[key])
        if not np.isfinite(zenith):
            continue

        pos_xy = pos_xyz[:, :2].copy()
        pos_z = pos_xyz[:, 2]

        # 高度校正投影到平均高度平面（与传播方向一致；|v_z| 过小时退化为原始坐标）
        v = np.array(directions[key], dtype=float)
        v = v / np.linalg.norm(v)
        if abs(v[2]) > 0.05:
            z_ref = np.mean(pos_z)
            s_proj = (z_ref - pos_z) / v[2]
            pos_xy = pos_xy + np.outer(s_proj, v[:2])

        center = np.mean(pos_xy, axis=0)
        rel = pos_xy - center

        az_rad = np.deg2rad(azimuth)
        u_major = np.array([np.cos(az_rad), np.sin(az_rad)])
        u_minor = np.array([-np.sin(az_rad), np.cos(az_rad)])
        s_major = rel @ u_major
        s_minor = rel @ u_minor

        spread_major = np.percentile(s_major, 95) - np.percentile(s_major, 5)
        spread_minor = np.percentile(s_minor, 95) - np.percentile(s_minor, 5)

        cos_a = np.cos(np.deg2rad(cone_half_angle_deg))
        sin_t = np.sin(np.deg2rad(zenith))
        denom_sq = max(cos_a ** 2 - sin_t ** 2, 1e-12)
        expected_ratio = min(cos_a / np.sqrt(denom_sq), max_expected_ratio)
        observed_ratio = spread_major / max(spread_minor, 1e-6)

        # 形状判据（与 Analysis_pipeline 当前 optimized 版一致）：
        # expected/observed 不超过该天顶角处 p90(secant 拟合) 的 shape_relax 倍，
        # 且 footprint 必须拉长（注意：与 Analysis 版相同，此处硬编码 1.0，
        # min_observed_ratio 参数保留但不再参与该判据）
        sec = 1.0 / np.cos(np.deg2rad(zenith))
        ratio_exp2obs = expected_ratio / observed_ratio
        shape_limit = (0.9236 + 0.2798 * sec + 0.0542 * sec ** 2.0) * shape_relax
        spatial_ok = (ratio_exp2obs < shape_limit) and (observed_ratio > 1.0)

        semi_major = ellipse_scale * spread_major / 2.0
        semi_minor = ellipse_scale * spread_minor / 2.0
        ellipse_dist = np.sqrt(
            (s_major / max(semi_major, 1e-6)) ** 2
            + (s_minor / max(semi_minor, 1e-6)) ** 2
        )
        frac_in_ellipse = np.mean(ellipse_dist <= 1.0)
        ellipse_ok = frac_in_ellipse >= min_du_fraction_in_ellipse

        if spread_major > 1.0:
            coeff = np.polyfit(s_major, times, 1)
            fit = np.polyval(coeff, s_major)
            residual_rms = float(np.sqrt(np.mean((times - fit) ** 2)))
        else:
            residual_rms = np.inf
        # 时间判据（与 Analysis_pipeline 当前 optimized 版一致）：天顶角依赖阈值
        # time_residual_p90(sec) 的 secant 拟合 (-60.321 + 41.1506*sec + 2.3754*sec^2)，
        # (+80) 后乘 2 倍余量
        optimized_time_residual_ns = (
            -60.321 + 41.1506 * sec + 2.3754 * sec ** 2.0 + time_offset_ns
        ) * time_relax
        time_ok = residual_rms <= optimized_time_residual_ns

        is_good = bool(spatial_ok and ellipse_ok and time_ok)

        diag_lines.append(
            f"{key},{n_du},{zenith:.2f},{azimuth:.2f},"
            f"{expected_ratio:.3f},{observed_ratio:.3f},"
            f"{spatial_ok},{frac_in_ellipse:.3f},{ellipse_ok},"
            f"{residual_rms:.3f},{time_ok},{is_good}"
        )
        diag_zenith.append(zenith)
        diag_azimuth.append(azimuth)
        diag_observed.append(observed_ratio)
        diag_expected.append(expected_ratio)
        diag_kept.append(is_good)
        diag_rms.append(residual_rms if np.isfinite(residual_rms) else np.nan)
        diag_ndu.append(n_du)

        if is_good:
            kept_keys.append(key)

    if verbose:
        logger.info(
            f"[Footprint filter] kept {len(kept_keys)}/{len(keys)} events "
            f"(min_du_fraction_in_ellipse={min_du_fraction_in_ellipse}, "
            f"cone_half_angle_deg={cone_half_angle_deg}, "
            f"shape_relax={shape_relax}, time_offset_ns={time_offset_ns}, time_relax={time_relax})"
        )

    if output_prefix:
        diag_file = f"{output_prefix}_footprint_filter_diagnostics.csv"
        try:
            with open(diag_file, "w") as f:
                f.write("\n".join(diag_lines) + "\n")
            logger.info(f"[Footprint filter] diagnostics saved to {diag_file}")
        except OSError as exc:
            logger.warning(f"[Footprint filter] failed to write diagnostics: {exc}")
        _plot_footprint_diagnostics(
            diag_zenith, diag_azimuth, diag_observed, diag_expected,
            diag_kept, diag_rms, diag_ndu, output_prefix,
        )

    return kept_keys


def _plot_footprint_diagnostics(zeniths, azimuths, observed, expected,
                                kept, rms, ndus, output_prefix):
    """诊断散点图（2x3）：扁率/时间残差/天线数 vs 角度，风格与主程序一致。"""
    if not zeniths:
        return
    try:
        import matplotlib.pyplot as plt
        try:
            import scienceplots  # noqa: F401
            plt.style.use(["science", "grid", "notebook"])
        except Exception:
            pass
    except ImportError:
        logger.warning("[Footprint filter] matplotlib unavailable, skip diagnostic plot")
        return

    z = np.array(zeniths)
    a = np.array(azimuths)
    obs = np.array(observed)
    obs = np.where(obs > 29.5, 30.0, obs)
    exp = np.array(expected)
    kept = np.array(kept, dtype=bool)
    rms = np.array(rms, dtype=float)
    ndu = np.array(ndus, dtype=float)
    colors = np.where(kept, "dodgerblue", "crimson")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        (axes[0, 0], z, obs, "Zenith (deg)", "Observed ratio"),
        (axes[0, 1], z, exp, "Zenith (deg)", "Expected ratio"),
        (axes[1, 0], a, obs, "Azimuth (deg)", "Observed ratio"),
        (axes[1, 1], a, exp, "Azimuth (deg)", "Expected ratio"),
        (axes[0, 2], ndu, rms, "Number of antennas", "Time residual RMS (ns)"),
        (axes[1, 2], z, rms, "Zenith (deg)", "Time residual RMS (ns)"),
    ]
    for ax, x, y, xlabel, ylabel in panels:
        ax.scatter(x, y, c=colors, s=25, alpha=0.75, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", color="gray", alpha=0.5)
        ax.scatter([], [], c="dodgerblue", s=40, label="kept")
        ax.scatter([], [], c="crimson", s=40, label="rejected")
        ax.legend(fontsize=10, loc="best")
        if "Azimuth" in xlabel:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 45))

    plt.tight_layout()
    fname = f"_{os.path.basename(output_prefix)}_footprint_ratio_diagnostic.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Footprint filter] saved diagnostic plot: {fname}")


# ---------------------------------------------------------------------------
# 判据 2：d_max ring cut（天顶角依赖的弥散度筛选）
# ---------------------------------------------------------------------------
def dmax_cut_threshold(zenith_deg, base=800.0, slope=300.0, zenith_flat=75.0):
    """
    天顶角依赖的 d_max 阈值（移植自 Analysis_pipeline 当前版本）：
      theta < 75 deg          -> base (800 m，即 2000-1200)
      75 <= theta <= 90 deg   -> (theta - 75) * slope + base
      theta > 90 deg          -> 按 90 deg 处理
    """
    if zenith_deg < zenith_flat:
        return base
    theta = min(zenith_deg, 90.0)
    return (theta - zenith_flat) * slope + base


def filter_by_ring_cut(
    event_times,
    directions,
    positions,
    n_du_max=6,
    d_mean_max=700.0,
    base=800.0,
    slope=300.0,
    zenith_flat=75.0,
    verbose=True,
):
    """
    触发天线弥散度筛选（仅对 n_du < n_du_max 的事例；其余直接保留）。

    判据：d_max <= dmax_cut_threshold(zenith) 且 d_mean < d_mean_max。
    方向退化（zenith 非有限）的事例保守拒绝。

    返回：保留的 event_key 列表（保持输入顺序）。
    """
    positions_int = _positions_as_int_key(positions)
    keys = list(directions.keys())

    kept_keys = []
    n_small = 0
    for key in keys:
        norm = _event_matches(event_times.get(key, {}), positions_int)
        if norm is None:
            continue
        ids, _, pos_xyz = norm
        if len(ids) >= n_du_max:
            kept_keys.append(key)
            continue
        n_small += 1
        xy = pos_xyz[:, :2]
        center = xy.mean(axis=0)
        d = np.linalg.norm(xy - center, axis=1)
        d_max = float(d.max())
        d_mean = float(d.mean())
        zen, _ = _incident_angles(directions[key])
        if not np.isfinite(zen):
            continue
        thr = dmax_cut_threshold(zen, base=base, slope=slope, zenith_flat=zenith_flat)
        if d_max <= thr and d_mean < d_mean_max:
            kept_keys.append(key)

    if verbose:
        logger.info(
            f"[Ring cut] kept {len(kept_keys)}/{len(keys)} events "
            f"(n_du<{n_du_max}: {sum(1 for k in kept_keys if len(event_times.get(k, {})) < n_du_max)}/{n_small}, "
            f"d_mean_max={d_mean_max} m, base={base} m, slope={slope} m/deg)"
        )
    return kept_keys


# ---------------------------------------------------------------------------
# 组合入口：供 main.py 在 PWM 与 SWM 之间调用（默认启用）
# ---------------------------------------------------------------------------
def apply_customized_cuts(state, positions, output_prefix,
                          run_footprint=True, run_ring=True):
    """
    在内存 state 上依次应用 footprint 筛选与 ring cut，并裁剪全部 state 字段。

    参数：
      state        : main.py 的阶段状态字典（各字段为 {event_key: value}）
      positions    : DU 坐标 {du_id: [x, y, z]}
      output_prefix: 诊断文件前缀（footprint 诊断写出用）
      run_footprint/run_ring : 分别启用两个判据

    返回：裁剪后的 state。
    """
    event_times = state.get("times", {})
    directions = state.get("directions", {})
    keys = list(directions.keys())

    kept = set(keys)
    if run_footprint:
        kept &= set(filter_by_shower_footprint(
            event_times, directions, positions, output_prefix=output_prefix))
    if run_ring:
        kept &= set(filter_by_ring_cut(event_times, directions, positions))

    removed = [k for k in keys if k not in kept]
    for field, value in list(state.items()):
        if isinstance(value, dict):
            state[field] = {k: v for k, v in value.items() if k in kept}

    logger.info(
        f"[Customized cuts] {len(keys)} events in, {len(kept)} kept, {len(removed)} removed "
        f"(footprint={'on' if run_footprint else 'off'}, ring={'on' if run_ring else 'off'})"
    )
    return state


# ---------------------------------------------------------------------------
# 独立运行入口：python -m find_event.shower_footprint_filter <file>_PWM.yaml
# ---------------------------------------------------------------------------
def _standalone_main(argv=None):
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Standalone customized cuts on a *_PWM.yaml file"
    )
    parser.add_argument("pwm_file", help="Path to *_PWM.yaml produced by main.py")
    parser.add_argument("--det-pos", default="_gp65_rtksort_2002_DU7.txt")
    parser.add_argument("--skip-footprint", action="store_true")
    parser.add_argument("--skip-ringcut", action="store_true")
    args = parser.parse_args(argv)

    from find_event.io import load_data_from_file
    positions = load_data_from_file(args.det_pos)

    with open(args.pwm_file) as f:
        payload = yaml.safe_load(f) or {}

    event_times, directions = {}, {}
    for key, result in payload.items():
        times = result.get("time") or {}
        direction = [result.get("x"), result.get("y"), result.get("z")]
        if any(v is None for v in direction):
            continue
        event_times[key] = times
        directions[key] = np.array(direction, dtype=float)

    out_prefix = args.pwm_file.replace(".yaml", "")
    kept = set(directions.keys())
    if not args.skip_footprint:
        kept &= set(filter_by_shower_footprint(
            event_times, directions, positions, output_prefix=out_prefix))
    if not args.skip_ringcut:
        kept &= set(filter_by_ring_cut(event_times, directions, positions))

    out_file = args.pwm_file.replace(".yaml", "_cut.yaml")
    with open(out_file, "w") as f:
        yaml.safe_dump({k: payload[k] for k in payload if k in kept}, f)
    logger.info(f"Standalone cuts: kept {len(kept)}/{len(directions)} events -> {out_file}")


if __name__ == "__main__":
    _standalone_main()
