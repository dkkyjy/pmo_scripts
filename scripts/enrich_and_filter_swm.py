#!/usr/bin/env python3
"""Enrich SWM YAML events with DU signal and filter slope candidates.

Process a single SWM file and:
1. Inject DU-level signal values from the corresponding XY merged YAML file.
2. Fit signal as a linear function of 1/distance for each event.
3. Filter events with positive slope, low SWM chi-square, and sufficient DUs,
   writing a candidate YAML and slope/reduced-chi-square diagnostic plots.

For batch processing by date, see ``run_enrich_and_filter.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure

from logger_config import logger


SWM_FILENAME_PATTERN = re.compile(
    r"^Trigger_(?P<date>\d{8})\d{0,6}_RUN(?P<run>\d+)_(?P<middle>.*)_SWM\.yaml$"
)


YamlDict = Dict[str, Any]
DetectorPositionMap = Dict[str, Tuple[float, float, float]]
SWM_CHI_SQUARE_MAX = 100.0
XMAX_DISTANCE_MIN_KM = 10.0
XMAX_DISTANCE_MAX_KM = 190.0


def _apply_plot_style() -> None:
    """Apply plotting style with scienceplots when available."""
    try:
        import scienceplots  # type: ignore  # noqa: F401

        plt.style.use(["science", "grid", "notebook"])
    except Exception as exc:
        logger.warning(
            "scienceplots not available, fallback to default matplotlib style: {}",
            exc,
        )


def load_detector_positions(det_pos_file: Path) -> DetectorPositionMap:
    """Load DU detector positions from text file.

    Expected file format (whitespace-separated):
        du_id x y z
    """
    positions: DetectorPositionMap = {}
    with det_pos_file.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                logger.debug(
                    "Skip detector position line {} (not enough columns): {}",
                    line_number,
                    line,
                )
                continue

            du_id = str(parts[0])
            try:
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                z_coord = float(parts[3])
            except ValueError:
                logger.debug(
                    "Skip detector position line {} (float parse failed): {}",
                    line_number,
                    line,
                )
                continue

            positions[du_id] = (x_coord, y_coord, z_coord)

    return positions


def _signal_value_to_scalar(signal_value: Any) -> Optional[float]:
    """Convert one signal entry to scalar value for plotting."""
    if isinstance(signal_value, (int, float)):
        return float(signal_value)

    if isinstance(signal_value, list):
        numeric_values: List[float] = []
        for item in signal_value:
            if isinstance(item, (int, float)):
                numeric_values.append(float(item))
        if numeric_values:
            return max(numeric_values)
        return None

    return None


def _fit_linear_relation(
    inv_distances: List[float],
    signals: List[float],
) -> Optional[Tuple[float, float, float]]:
    """Fit signal as a linear function of 1/distance.

    Note: Uses unweighted least-squares (``np.polyfit``, degree 1).
    The returned ``reduced_chi_square`` is actually MSE (mean squared error)
    since measurement uncertainties are not propagated.
    """
    if len(inv_distances) < 2 or len(signals) < 2:
        return None

    x_values = np.asarray(inv_distances, dtype=float)
    y_values = np.asarray(signals, dtype=float)
    if np.allclose(x_values, x_values[0]):
        return None

    slope, intercept = np.polyfit(x_values, y_values, 1)
    fitted_values = slope * x_values + intercept
    residuals = y_values - fitted_values
    degrees_of_freedom = len(x_values) - 2
    if degrees_of_freedom > 0:
        reduced_chi_square = float(
            np.sum(residuals ** 2) / float(degrees_of_freedom)
        )
    else:
        reduced_chi_square = float("nan")

    return float(slope), float(intercept), reduced_chi_square


def _plot_zenith_azimuth_polar_scatter(
    fig: Figure,
    axis: Any,
    zenith_azimuth_points: List[Tuple[float, float, float]],
) -> None:
    """Plot zenith/azimuth points with reduced-chi-square color coding."""
    azimuth_rad = np.deg2rad([item[1] for item in zenith_azimuth_points])
    zenith_deg = [item[0] for item in zenith_azimuth_points]
    zenith_azimuth_scatter = axis.scatter(
        azimuth_rad,
        zenith_deg,
        c=np.log10([item[2] for item in zenith_azimuth_points]),
        cmap="viridis",
        s=14,
        alpha=0.8,
        edgecolors="none",
    )
    zenith_azimuth_color_bar = fig.colorbar(
        zenith_azimuth_scatter,
        ax=axis,
    )
    zenith_azimuth_color_bar.set_label(r"$\log_{10}$(reduced $\chi^2$)")
    axis.set_theta_zero_location("N")
    axis.set_xticks(np.deg2rad([0, 315, 270, 225, 180, 135, 90, 45]))
    axis.set_xticklabels(
        ["N", "315°", "E", "225°", "S", "135°", "W", "45°"],
        fontsize=12,
    )
    axis.set_rlabel_position(135)
    axis.set_ylabel("zenith [deg]")
    axis.set_title("zenith vs azimuth (polar)")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for single-file SWM signal enrichment and slope filtering."""
    parser = argparse.ArgumentParser(
        description=(
            "Read one SWM merged YAML file, inject DU signal data from "
            "the corresponding XY merged YAML file, fit signal vs 1/distance, "
            "and filter slope > 0 candidates."
        )
    )
    parser.add_argument(
        "swm_file",
        help="Path to the SWM YAML file to process.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for output files. Defaults to the same directory as "
            "the source SWM file."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_with_signal",
        help="Suffix appended before .yaml for output files.",
    )
    parser.add_argument(
        "--det-pos-file",
        default="_gp65_rtksort_2002_DU7.txt",
        help="Detector position file path (du_id x y z).",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "Directory for distance-signal plots. Defaults to the same directory "
            "as output YAML."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable all plotting.",
    )
    parser.add_argument(
        "--chi-square-threshold",
        type=float,
        default=100.0,
        help=(
            "Threshold used to split chi_square bins in origin-distance "
            "histogram (default: 100)."
        ),
    )
    parser.add_argument(
        "--swm-chi-square-max",
        type=float,
        default=SWM_CHI_SQUARE_MAX,
        help=(
            "Maximum SWM chi-square used for cut lines and event-selection "
            "cuts in overview plots."
        ),
    )
    parser.add_argument(
        "--xmax-min-km",
        type=float,
        default=XMAX_DISTANCE_MIN_KM,
        help="Minimum xmax distance in km used by the overview-plot cuts.",
    )
    parser.add_argument(
        "--xmax-max-km",
        type=float,
        default=XMAX_DISTANCE_MAX_KM,
        help="Maximum xmax distance in km used by the overview-plot cuts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without writing files.",
    )
    # ── slope candidate filter options ──────────────────────────────────
    parser.add_argument(
        "--no-slope-filter",
        action="store_true",
        help="Skip slope candidate filtering and slope diagnostic plots.",
    )
    parser.add_argument(
        "--candidate-yaml",
        type=Path,
        default=None,
        help=(
            "Save slope > 0 candidate events as YAML. "
            "Default: auto-derived from swm-file."
        ),
    )
    parser.add_argument(
        "--min-du-count",
        type=int,
        default=6,
        help="Minimum number of DUs required for slope candidate (default: 6).",
    )
    return parser


def load_yaml_dict(file_path: Path) -> YamlDict:
    """Load one YAML mapping file and normalize null payload to empty dict."""
    with file_path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.load(file_obj, Loader=yaml.FullLoader) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML top-level is not a mapping: {file_path}")
    return payload


def write_yaml_dict(file_path: Path, payload: Mapping[str, Any]) -> None:
    """Write mapping payload to YAML file."""
    with file_path.open("w", encoding="utf-8") as file_obj:
        yaml.dump(dict(payload), file_obj, sort_keys=False)


def output_path_for(
    swm_file: Path,
    output_dir: Optional[Path],
    suffix: str,
) -> Path:
    """Resolve output path for one SWM input file."""
    stem = swm_file.stem
    file_name = f"{stem}{suffix}.yaml"
    if output_dir is None:
        return swm_file.with_name(file_name)
    return output_dir / file_name


def expected_xy_file_for_swm(swm_file: Path) -> Path:
    """Build corresponding XY merged YAML file path for one SWM file.

    Derives the XY filename from the SWM filename by replacing ``_SWM.yaml``
    with ``_XY_merged.yaml`` (or ``_XY.yaml`` as fallback).

    Also handles already-enriched ``*_SWM_with_signal.yaml`` files by
    stripping the ``_with_signal`` suffix first.
    """
    swm_name = swm_file.name

    # Strip _with_signal suffix if present (already enriched files)
    if swm_name.endswith("_SWM_with_signal.yaml"):
        swm_name = swm_name[: -len("_with_signal.yaml")] + ".yaml"

    if swm_name.endswith("_matched_SWM.yaml"):
        xy_name = swm_name[: -len("_matched_SWM.yaml")] + "_XY.yaml"
        xy_file = swm_file.with_name(xy_name)
        if xy_file.exists():
            return xy_file
    if swm_name.endswith("_matched_fingerprint_SWM.yaml"):
        xy_name = swm_name[: -len("_matched_fingerprint_SWM.yaml")] + "_XY.yaml"
        xy_file = swm_file.with_name(xy_name)
        if xy_file.exists():
            return xy_file
    if swm_name.endswith("_SWM.yaml"):
        xy_name = swm_name[: -len("_SWM.yaml")] + "_XY.yaml"
        xy_file = swm_file.with_name(xy_name)
        if xy_file.exists():
            return xy_file
        # Fallback: try _XY_merged.yaml
        xy_name_fallback = swm_name[: -len("_SWM.yaml")] + "_XY_merged.yaml"
        return swm_file.with_name(xy_name_fallback)

    # Fallback to regex for non-standard names
    matched = SWM_FILENAME_PATTERN.match(swm_name)
    if matched is None:
        raise ValueError(f"Unsupported SWM filename format: {swm_file.name}")

    date_text = matched.group("date")
    run_text = matched.group("run")
    middle_text = matched.group("middle") or ""
    xy_name = f"Trigger_{date_text}_RUN{run_text}{middle_text}_XY_merged.yaml"
    return swm_file.with_name(xy_name)


def _event_number_index(events: Mapping[str, Any]) -> Dict[int, str]:
    """Build event_number -> event key index for fallback event matching."""
    number_to_key: Dict[int, str] = {}
    for key, payload in events.items():
        if not isinstance(payload, dict):
            continue
        event_number = payload.get("event_number")
        if isinstance(event_number, int):
            if event_number in number_to_key:
                logger.warning(
                    "Duplicate event_number={} in XY events: keys={}, {}",
                    event_number,
                    number_to_key[event_number],
                    key,
                )
            number_to_key[event_number] = str(key)
    return number_to_key


def _normalize_du_ids(du_ids: Any) -> List[str]:
    """Normalize DU IDs to string list for consistent YAML-key lookups."""
    if isinstance(du_ids, list):
        return [str(item) for item in du_ids]
    if du_ids is None:
        return []
    return [str(du_ids)]


def _xy_event_for_swm_event(
    swm_key: str,
    swm_event: Mapping[str, Any],
    xy_events: Mapping[str, Any],
    xy_event_number_to_key: Mapping[int, str],
) -> Optional[Mapping[str, Any]]:
    """Find matching XY event by key first and event_number second."""
    xy_event = xy_events.get(swm_key)
    if isinstance(xy_event, dict):
        return xy_event

    event_number = swm_event.get("event_number")
    if not isinstance(event_number, int):
        return None

    xy_key = xy_event_number_to_key.get(event_number)
    if xy_key is None:
        return None

    fallback_event = xy_events.get(xy_key)
    if isinstance(fallback_event, dict):
        return fallback_event
    return None


def enrich_swm_events_with_signal(
    swm_events: MutableMapping[str, Any],
    xy_events: Mapping[str, Any],
) -> Dict[str, int]:
    """Inject per-DU signal from XY events into SWM events.

    Returns:
        A small counter dict for reporting.
    """
    xy_event_number_to_key = _event_number_index(xy_events)
    stats = {
        "total_events": 0,
        "matched_events": 0,
        "missing_xy_events": 0,
        "missing_du_signals": 0,
        "events_with_missing_du_signal": 0,
        "missing_xy_event_keys_sample": [],
        "missing_du_signal_event_keys_sample": [],
    }

    for swm_key, swm_payload in swm_events.items():
        if not isinstance(swm_payload, dict):
            continue

        stats["total_events"] += 1
        swm_event = swm_payload

        xy_event = _xy_event_for_swm_event(
            str(swm_key),
            swm_event,
            xy_events,
            xy_event_number_to_key,
        )

        if xy_event is None:
            stats["missing_xy_events"] += 1
            if len(stats["missing_xy_event_keys_sample"]) < 5:
                stats["missing_xy_event_keys_sample"].append(str(swm_key))
            swm_event["signal"] = {}
            swm_event["missing_signal_du_id"] = _normalize_du_ids(
                swm_event.get("du_id")
            )
            continue

        stats["matched_events"] += 1
        xy_signal = xy_event.get("signal", {})
        if not isinstance(xy_signal, dict):
            xy_signal = {}

        event_signal: Dict[str, Any] = {}
        missing_du_ids: List[str] = []
        for du_id in _normalize_du_ids(swm_event.get("du_id")):
            if du_id in xy_signal:
                event_signal[du_id] = xy_signal[du_id]
            else:
                missing_du_ids.append(du_id)

        stats["missing_du_signals"] += len(missing_du_ids)
        if missing_du_ids:
            stats["events_with_missing_du_signal"] += 1
            if len(stats["missing_du_signal_event_keys_sample"]) < 5:
                stats["missing_du_signal_event_keys_sample"].append(str(swm_key))
        swm_event["signal"] = event_signal
        swm_event["missing_signal_du_id"] = missing_du_ids

    return stats


# ── slope candidate filter (merged from stats_slope_candidates.py) ──────────

def _extract_du_count(payload: Dict[str, Any]) -> int:
    """Extract DU count from event payload's ``du_id`` list."""
    du_id = payload.get("du_id")
    if isinstance(du_id, list):
        return len(du_id)
    return 0


def _extract_slope_from_fit(payload: Dict[str, Any]) -> Optional[float]:
    """Extract slope from ``signal_distance_linear_fit``, returning None if unavailable."""
    fit_payload = payload.get("signal_distance_linear_fit")
    if not isinstance(fit_payload, dict):
        return None
    slope = fit_payload.get("slope")
    if slope is None:
        return None
    try:
        return float(slope)
    except (TypeError, ValueError):
        return None


def _extract_rchi2_from_fit(payload: Dict[str, Any]) -> Optional[float]:
    """Extract reduced_chi_square from ``signal_distance_linear_fit``."""
    fit_payload = payload.get("signal_distance_linear_fit")
    if not isinstance(fit_payload, dict):
        return None
    rchi2 = fit_payload.get("reduced_chi_square")
    if rchi2 is None:
        return None
    try:
        return float(rchi2)
    except (TypeError, ValueError):
        return None


def _extract_chi_square(payload: Dict[str, Any]) -> Optional[float]:
    """Extract SWM chi_square from event payload, returning None if unavailable."""
    chi_square = payload.get("chi_square")
    if chi_square is None:
        return None
    try:
        return float(chi_square)
    except (TypeError, ValueError):
        return None


def filter_slope_candidates(
    swm_events: Mapping[str, Any],
    chi_square_max: float,
    min_du_count: int,
) -> Tuple[Dict[str, Any], int, int, int, int, int, List[float], List[float]]:
    """Filter events with slope > 0 and collect slope/rchi2 arrays for plotting.

    Returns:
        (qualified_events, total, slope_gt_0, slope_le_0, no_fit, filtered,
         slopes, rchi2s)
    """
    total = 0
    slope_gt_0 = 0
    slope_le_0 = 0
    no_fit = 0
    filtered = 0  # chi_square >= max or du_count < min

    qualified_events: Dict[str, Any] = {}
    slopes: List[float] = []
    rchi2s: List[float] = []

    for event_key, payload in swm_events.items():
        if not isinstance(payload, dict):
            continue
        total += 1

        chi_square = _extract_chi_square(payload)
        if chi_square is None or chi_square >= chi_square_max:
            filtered += 1
            continue

        if _extract_du_count(payload) < min_du_count:
            filtered += 1
            continue

        slope = _extract_slope_from_fit(payload)
        rchi2 = _extract_rchi2_from_fit(payload)

        if slope is not None and np.isfinite(slope):
            if rchi2 is not None and np.isfinite(rchi2):
                slopes.append(slope)
                rchi2s.append(rchi2)
            if slope > 0:
                slope_gt_0 += 1
                qualified_events[event_key] = payload
            else:
                slope_le_0 += 1
        else:
            no_fit += 1

    logger.info(
        "Slope filter: total={} slope>0={} slope<=0={} no_fit={} filtered(chi2/DU)={}",
        total,
        slope_gt_0,
        slope_le_0,
        no_fit,
        filtered,
    )

    return qualified_events, total, slope_gt_0, slope_le_0, no_fit, filtered, slopes, rchi2s


def plot_slope_histogram(
    output_path: Path,
    slopes: List[float],
    title: str,
    reduced_chi_squares: Optional[List[float]] = None,
) -> None:
    """Plot slope distribution histogram with reduced chi_square subplots.

    When ``reduced_chi_squares`` is provided, a 3-panel figure is produced:
      1. Slope histogram (top)
      2. Reduced chi_square histogram (middle)
      3. Slope vs reduced chi_square scatter (bottom)

    Otherwise only the slope histogram is drawn.
    """
    if not slopes:
        logger.warning("No slope data; skip histogram: {}", output_path)
        return

    slope_array = np.asarray(slopes, dtype=float)
    finite_mask = np.isfinite(slope_array)
    slope_array = slope_array[finite_mask]
    if len(slope_array) == 0:
        logger.warning("No finite slope values; skip histogram: {}", output_path)
        return

    has_rchi2 = (
        reduced_chi_squares is not None
        and len(reduced_chi_squares) == len(slopes)
    )
    if has_rchi2:
        rchi2_array = np.asarray(reduced_chi_squares, dtype=float)[finite_mask]
        rchi2_finite_mask = np.isfinite(rchi2_array)
        scat_slopes = slope_array[rchi2_finite_mask]
        scat_rchi2 = rchi2_array[rchi2_finite_mask]
        rchi2_hist = rchi2_array[np.isfinite(rchi2_array)]
        nrows = 3
    else:
        nrows = 1

    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows), squeeze=False)

    bins = 60

    # ---- Panel 1: slope histogram ----
    ax_slope = axes[0][0]
    counts, bin_edges, _ = ax_slope.hist(
        slope_array, bins=bins, alpha=0.7, color="tab:blue", edgecolor="black",
    )
    positive_mask = slope_array > 0
    if positive_mask.any():
        ax_slope.hist(
            slope_array[positive_mask],
            bins=bin_edges,
            alpha=0.7,
            color="tab:red",
            edgecolor="black",
            label=f"slope > 0 (n={positive_mask.sum()})",
        )
    ax_slope.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_slope.set_xlabel("slope")
    ax_slope.set_ylabel("Event count")
    ax_slope.set_title(title)
    ax_slope.legend()
    ax_slope.grid(True, linestyle=":", alpha=0.6)

    if has_rchi2:
        # ---- Panel 2: reduced chi_square histogram ----
        ax_rchi2 = axes[1][0]
        if len(rchi2_hist) > 0:
            rchi2_bins = min(bins, max(10, int(len(rchi2_hist) ** 0.5)))
            rchi2_counts, rchi2_edges, _ = ax_rchi2.hist(
                rchi2_hist, bins=rchi2_bins, alpha=0.7, color="tab:green",
                edgecolor="black",
            )
            good_mask = rchi2_hist < 10
            if good_mask.any():
                ax_rchi2.hist(
                    rchi2_hist[good_mask],
                    bins=rchi2_edges,
                    alpha=0.7,
                    color="tab:olive",
                    edgecolor="black",
                    label=f"rchi2 < 10 (n={good_mask.sum()})",
                )
        ax_rchi2.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.7,
                         label="rchi2 = 1")
        ax_rchi2.set_xlabel("reduced chi_square")
        ax_rchi2.set_ylabel("Event count")
        ax_rchi2.set_title(f"Reduced chi_square distribution — {title}")
        ax_rchi2.legend()
        ax_rchi2.grid(True, linestyle=":", alpha=0.6)

        # ---- Panel 3: slope vs reduced chi_square scatter ----
        ax_scat = axes[2][0]
        if len(scat_slopes) > 0:
            ax_scat.scatter(scat_rchi2, scat_slopes, alpha=0.5, s=8,
                            c="tab:blue", edgecolors="none")
            pos_mask = scat_slopes > 0
            if pos_mask.any():
                ax_scat.scatter(scat_rchi2[pos_mask], scat_slopes[pos_mask],
                                alpha=0.6, s=8, c="tab:red", edgecolors="none",
                                label=f"slope > 0 (n={pos_mask.sum()})")
        ax_scat.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax_scat.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax_scat.set_xlabel("reduced chi_square")
        ax_scat.set_ylabel("slope")
        ax_scat.set_title(f"Slope vs reduced chi_square — {title}")
        ax_scat.legend()
        ax_scat.grid(True, linestyle=":", alpha=0.6)
        if len(scat_rchi2) > 0:
            rchi2_max = np.percentile(scat_rchi2, 99) if len(scat_rchi2) > 0 else 1
            if rchi2_max > 100:
                ax_scat.set_xscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Slope histogram saved: {}", output_path)


# ── end slope candidate filter ─────────────────────────────────────────────


def process_one_file(
    swm_file: Path,
    output_dir: Optional[Path],
    plot_dir: Optional[Path],
    det_positions: DetectorPositionMap,
    suffix: str,
    overwrite: bool,
    dry_run: bool,
    no_plot: bool,
    chi_square_threshold: float,
    swm_chi_square_max: float,
    xmax_min_km: float,
    xmax_max_km: float,
    # slope filter args
    no_slope_filter: bool = False,
    candidate_yaml: Optional[Path] = None,
    min_du_count: int = 6,
) -> bool:
    """Process one SWM file: enrich with signal, fit, and optionally filter slope candidates."""
    output_path = output_path_for(swm_file, output_dir, suffix)
    replot_only = output_path.exists() and not overwrite

    # If the input is already an enriched file, skip enrichment and use it directly.
    input_is_enriched = swm_file.name.endswith("_SWM_with_signal.yaml")

    if replot_only:
        logger.info(
            "Output already exists and --overwrite is not set; "
            "reuse existing enriched YAML for plotting: {}",
            output_path,
        )
        swm_events = load_yaml_dict(output_path)
        stats = {
            "total_events": sum(
                1 for payload in swm_events.values() if isinstance(payload, dict)
            ),
            "matched_events": 0,
            "missing_xy_events": 0,
            "missing_du_signals": 0,
            "events_with_missing_du_signal": 0,
            "missing_xy_event_keys_sample": [],
            "missing_du_signal_event_keys_sample": [],
        }
    elif input_is_enriched:
        logger.info(
            "Input is already an enriched file; skipping XY enrichment: {}",
            swm_file,
        )
        swm_events = load_yaml_dict(swm_file)
        stats = {
            "total_events": sum(
                1 for payload in swm_events.values() if isinstance(payload, dict)
            ),
            "matched_events": 0,
            "missing_xy_events": 0,
            "missing_du_signals": 0,
            "events_with_missing_du_signal": 0,
            "missing_xy_event_keys_sample": [],
            "missing_du_signal_event_keys_sample": [],
        }
    else:
        xy_file = expected_xy_file_for_swm(swm_file)
        logger.info("Start processing swm_file={} xy_file={}", swm_file, xy_file)
        if not xy_file.exists():
            logger.warning("Skip {}, missing XY file: {}", swm_file, xy_file)
            return True  # not an error — just nothing to enrich

        swm_events = load_yaml_dict(swm_file)
        xy_events = load_yaml_dict(xy_file)
        logger.info(
            "Loaded yaml entries swm_events={} xy_events={}",
            len(swm_events),
            len(xy_events),
        )
        stats = enrich_swm_events_with_signal(swm_events, xy_events)
        logger.info(
            "Enrich stats total={} matched={} missing_xy={} missing_du_signal={} "
            "events_with_missing_du_signal={}",
            stats["total_events"],
            stats["matched_events"],
            stats["missing_xy_events"],
            stats["missing_du_signals"],
            stats["events_with_missing_du_signal"],
        )
        if stats["missing_xy_event_keys_sample"]:
            logger.info(
                "Sample missing XY event keys: {}",
                stats["missing_xy_event_keys_sample"],
            )
        if stats["missing_du_signal_event_keys_sample"]:
            logger.info(
                "Sample events with missing DU signal: {}",
                stats["missing_du_signal_event_keys_sample"],
            )

    event_fit_results: List[
        Tuple[str, float, float, float, int, float, float, float, float, float]
    ] = []
    distance_signal_points: List[Tuple[float, float, float]] = []
    swm_xmax_chi_points: List[Tuple[float, float]] = []
    zenith_azimuth_points: List[Tuple[float, float, float]] = []
    origin_distances: List[float] = []
    origin_distance_chi_squares: List[float] = []
    missing_position_du_ids: set[str] = set()
    missing_signal_events = 0
    for event_key, swm_payload in swm_events.items():
        if not isinstance(swm_payload, dict):
            continue

        swm_payload["signal_distance_linear_fit"] = None
        swm_payload["xmax_distance_km"] = None
        swm_payload["antenna_distances_km"] = {}

        x_value = swm_payload.get("x")
        y_value = swm_payload.get("y")
        z_value = swm_payload.get("z")
        if not isinstance(x_value, (int, float)):
            continue
        if not isinstance(y_value, (int, float)):
            continue
        if not isinstance(z_value, (int, float)):
            continue
        x_coord = float(x_value)
        y_coord = float(y_value)
        z_coord = float(z_value)

        origin_distance = math.sqrt(
            x_coord ** 2 + y_coord ** 2 + z_coord ** 2
        ) / 1000.0  # convert to km
        swm_payload["xmax_distance_km"] = origin_distance
        origin_distances.append(origin_distance)
        chi_square_value = swm_payload.get("chi_square")
        event_swm_chi_square: Optional[float] = None
        if isinstance(chi_square_value, (int, float)):
            chi_square_float = float(chi_square_value)
            event_swm_chi_square = chi_square_float
            origin_distance_chi_squares.append(chi_square_float)
            swm_xmax_chi_points.append((origin_distance, chi_square_float))
        else:
            origin_distance_chi_squares.append(float("nan"))

        zenith_value = swm_payload.get("zenith")
        azimuth_value = swm_payload.get("azimuth")
        event_zenith_azimuth: Tuple[float, float] | None = None
        if (
            isinstance(zenith_value, (int, float))
            and isinstance(azimuth_value, (int, float))
            and isinstance(chi_square_value, (int, float))
            and 0 < float(chi_square_value) <= chi_square_threshold
            and xmax_min_km < origin_distance < xmax_max_km
        ):
            event_zenith_azimuth = (
                float(zenith_value),
                float(azimuth_value),
            )

        signal_map = swm_payload.get("signal", {})
        if not isinstance(signal_map, dict):
            missing_signal_events += 1
            continue

        event_distance_signal_pairs: List[Tuple[float, float]] = []
        event_inv_distance_signal_pairs: List[Tuple[float, float]] = []
        event_distance_signal_chi_squares: List[float] = []
        for du_id in _normalize_du_ids(swm_payload.get("du_id")):
            detector_pos = det_positions.get(du_id)
            if detector_pos is None:
                missing_position_du_ids.add(du_id)
                continue

            signal_scalar = _signal_value_to_scalar(signal_map.get(du_id))
            if signal_scalar is None:
                continue

            distance = math.sqrt(
                (x_coord - detector_pos[0]) ** 2
                + (y_coord - detector_pos[1]) ** 2
                + (z_coord - detector_pos[2]) ** 2
            ) / 1000.0  # convert to km
            swm_payload["antenna_distances_km"][str(du_id)] = distance
            inv_distance = 1.0 / distance if distance > 0 else float("inf")
            event_distance_signal_pairs.append((distance, signal_scalar))
            event_inv_distance_signal_pairs.append((inv_distance, signal_scalar))
            if isinstance(chi_square_value, (int, float)):
                event_distance_signal_chi_squares.append(float(chi_square_value))
            else:
                event_distance_signal_chi_squares.append(float("nan"))

        filtered_event_pairs = [
            (x, y) for x, y in event_inv_distance_signal_pairs
            if math.isfinite(x)
        ]

        fit_result = _fit_linear_relation(
            [item[0] for item in filtered_event_pairs],
            [item[1] for item in filtered_event_pairs],
        )
        if fit_result is None:
            logger.debug(
                "Skip per-event fit event_key={} valid_points={}",
                event_key,
                len(filtered_event_pairs),
            )
        else:
            if event_swm_chi_square is None:
                continue
            slope, intercept, reduced_chi_square = fit_result
            x_min = min(item[0] for item in filtered_event_pairs)
            x_max = max(item[0] for item in filtered_event_pairs)
            # Also compute original distance range for reference
            distances_km = [1.0 / x for x, _ in filtered_event_pairs if x > 0]
            dist_min_km = min(distances_km) if distances_km else 0.0
            dist_max_km = max(distances_km) if distances_km else 0.0
            swm_payload["signal_distance_linear_fit"] = {
                "slope": slope,
                "intercept": intercept,
                "reduced_chi_square": reduced_chi_square,
                "point_count": len(filtered_event_pairs),
                "inv_distance_min": x_min,
                "inv_distance_max": x_max,
                "distance_min_km": dist_min_km,
                "distance_max_km": dist_max_km,
            }
            if not math.isnan(reduced_chi_square) and reduced_chi_square > 0:
                for (distance, signal), chi_square in zip(
                    event_distance_signal_pairs,
                    event_distance_signal_chi_squares,
                ):
                    if (
                        not math.isnan(chi_square)
                        and chi_square <= chi_square_threshold
                        and xmax_min_km < origin_distance < xmax_max_km
                    ):
                        distance_signal_points.append(
                            (distance, signal, reduced_chi_square)
                        )
                if event_zenith_azimuth is not None:
                    zenith_azimuth_points.append(
                        (
                            event_zenith_azimuth[0],
                            event_zenith_azimuth[1],
                            reduced_chi_square,
                        )
                    )
            if (
                event_swm_chi_square > swm_chi_square_max
                or origin_distance <= xmax_min_km
                or origin_distance >= xmax_max_km
            ):
                continue
            event_fit_results.append(
                (
                    str(event_key),
                    slope,
                    intercept,
                    reduced_chi_square,
                    len(filtered_event_pairs),
                    x_min,
                    x_max,
                    dist_min_km,
                    dist_max_km,
                    event_swm_chi_square,
                )
            )

    logger.info(
        "Per-event fit results={} missing_position_du_ids={} "
        "events_with_invalid_signal_map={} origin_distances={}",
        len(event_fit_results),
        len(missing_position_du_ids),
        missing_signal_events,
        len(origin_distances),
    )
    if missing_position_du_ids:
        sample_du_ids = sorted(missing_position_du_ids)[:8]
        logger.info("Sample missing detector positions du_id={}", sample_du_ids)

    valid_origin_records = [
        (distance, chi_square)
        for distance, chi_square in zip(
            origin_distances,
            origin_distance_chi_squares,
        )
        if not math.isnan(chi_square)
    ]
    origin_low_bin = [
        distance
        for distance, chi_square in valid_origin_records
        if chi_square <= chi_square_threshold
    ]
    origin_high_bin = [
        distance
        for distance, chi_square in valid_origin_records
        if chi_square > chi_square_threshold
    ]
    logger.info(
        "Origin-distance chi_square bins threshold={} low_bin={} high_bin={} "
        "missing_chi_square={}",
        chi_square_threshold,
        len(origin_low_bin),
        len(origin_high_bin),
        len(origin_distances) - len(valid_origin_records),
    )
    logger.info(
        "Scatter cuts applied chi_square_threshold={} xmax_cut_range=({} km, {} km)",
        chi_square_threshold,
        xmax_min_km,
        xmax_max_km,
    )

    output_base_dir = output_path.parent
    resolved_plot_dir = plot_dir or output_base_dir
    overview_file = resolved_plot_dir / f"{swm_file.stem}_overview_6panel.png"

    if not no_plot and not dry_run:
        resolved_plot_dir.mkdir(parents=True, exist_ok=True)
        _apply_plot_style()

        swm_chi_square_xmax_mid = [
            chi_square
            for distance, chi_square in valid_origin_records
            if xmax_min_km < distance < xmax_max_km
        ]
        swm_chi_square_xmax_other = [
            chi_square
            for distance, chi_square in valid_origin_records
            if not (xmax_min_km < distance < xmax_max_km)
        ]
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        fig.delaxes(axes[0, 1])
        ax_zenith_azimuth = fig.add_subplot(3, 2, 2, projection="polar")

        # Subplot 1: SWM chi-square histogram
        ax_swm_chi = axes[0, 0]
        if swm_chi_square_xmax_mid or swm_chi_square_xmax_other:
            ax_swm_chi.hist(
                [
                    np.log10(swm_chi_square_xmax_mid),
                    np.log10(swm_chi_square_xmax_other),
                ],
                bins=30,
                stacked=True,
                alpha=0.85,
                color=["steelblue", "gray"],
                edgecolor="white",
                label=[
                    f"{xmax_min_km:g} < xmax < {xmax_max_km:g} km",
                    "other xmax",
                ],
            )
            ax_swm_chi.axvline(
                np.log10(swm_chi_square_max),
                color="crimson",
                linestyle="-",
                linewidth=2,
                label=f"SWM $\\chi^2$={swm_chi_square_max:g}",
            )
            ax_swm_chi.set_xlabel("log$_{10}$ SWM $\\chi^2$")
            ax_swm_chi.set_ylabel("Event count")
            ax_swm_chi.set_title("SWM $\\chi^2$ histogram")
            ax_swm_chi.legend()
        else:
            ax_swm_chi.text(0.5, 0.5, "No valid SWM $\\chi^2$", ha="center", va="center")
            ax_swm_chi.set_title("SWM $\\chi^2$ histogram")

        # Subplot 2: zenith/azimuth polar scatter
        if zenith_azimuth_points:
            _plot_zenith_azimuth_polar_scatter(
                fig=fig,
                axis=ax_zenith_azimuth,
                zenith_azimuth_points=zenith_azimuth_points,
            )
        else:
            ax_zenith_azimuth.text(
                0.5,
                0.5,
                "No zenith/azimuth data",
                ha="center",
                va="center",
                transform=ax_zenith_azimuth.transAxes,
            )
            ax_zenith_azimuth.set_title("zenith vs azimuth (polar)")

        # Subplot 3: xmax distance histogram
        ax_xmax = axes[1, 0]
        if origin_distances:
            ax_xmax.hist(
                [origin_low_bin, origin_high_bin],
                bins=30,
                stacked=True,
                alpha=0.75,
                color=["steelblue", "gray"],
                edgecolor="white",
                label=[
                    f"SWM $\\chi^2$ <= {chi_square_threshold:g}",
                    f"SWM $\\chi^2$ > {chi_square_threshold:g}",
                ],
            )
            ax_xmax.set_xlabel("xmax distance from origin [km]")
            ax_xmax.set_ylabel("Event count")
            ax_xmax.set_title("xmax distance histogram")
            ax_xmax.axvline(
                xmax_min_km,
                color="crimson",
                linestyle=":",
                linewidth=2,
                label=f"xmax = {xmax_min_km:g} km",
            )
            ax_xmax.axvline(
                xmax_max_km,
                color="crimson",
                linestyle="--",
                linewidth=2,
                label=f"xmax = {xmax_max_km:g} km",
            )
            ax_xmax.legend()
        else:
            ax_xmax.text(0.5, 0.5, "No xmax distance data", ha="center", va="center")
            ax_xmax.set_title("xmax distance histogram")

        # Subplot 4: distance-signal scatter with 1/distance linear-fit overlays
        ax_distance_signal = axes[1, 1]
        if distance_signal_points and event_fit_results:
            distances = [item[0] for item in distance_signal_points]
            signals = [item[1] for item in distance_signal_points]
            reduced_chi_squares = [item[2] for item in distance_signal_points]
            logger.info(
                "Distance-signal points={} per-event fits={}",
                len(distance_signal_points),
                len(event_fit_results),
            )

            scatter_distance_signal = ax_distance_signal.scatter(
                distances,
                signals,
                c=np.log10(reduced_chi_squares),
                cmap="viridis",
                s=10,
                alpha=0.8,
            )
            color_bar_distance_signal = fig.colorbar(
                scatter_distance_signal,
                ax=ax_distance_signal,
            )
            color_bar_distance_signal.set_label(
                r"$\log_{10}$(reduced $\chi^2$)"
            )

            for _, slope, intercept, _, _, _, _, dist_min, dist_max, _ in event_fit_results:
                x_fit = np.linspace(dist_min, dist_max, 100)
                ax_distance_signal.plot(
                    x_fit,
                    slope / x_fit + intercept,
                    linewidth=1.2,
                    alpha=0.7,
                )

            ax_distance_signal.set_xlabel(
                "Distance from reconstructed xmax to DU position [km]"
            )
            ax_distance_signal.set_ylabel("Signal amplitude")
            ax_distance_signal.set_title("Distance vs Signal with 1/distance linear fits")
        else:
            ax_distance_signal.text(
                0.5,
                0.5,
                "No distance-signal or fit data",
                ha="center",
                va="center",
            )
            ax_distance_signal.set_title("Distance vs Signal with 1/distance linear fits")

        # Subplot 5: SWM chi-square vs xmax-distance scatter
        ax_swm_xmax_scatter = axes[2, 0]
        if swm_xmax_chi_points:
            ax_swm_xmax_scatter.scatter(
                [item[0] for item in swm_xmax_chi_points],
                [item[1] for item in swm_xmax_chi_points],
                s=12,
                alpha=0.75,
                edgecolors="none",
                color="steelblue",
            )
            ax_swm_xmax_scatter.set_yscale("log")
            ax_swm_xmax_scatter.set_xlabel("xmax distance from origin [km]")
            ax_swm_xmax_scatter.set_ylabel("SWM $\\chi^2$")
            ax_swm_xmax_scatter.set_title("SWM $\\chi^2$ vs xmax distance")
            ax_swm_xmax_scatter.axvline(
                xmax_min_km,
                color="crimson",
                linestyle=":",
                linewidth=2,
                label=f"xmax = {xmax_min_km:g} km",
            )
            ax_swm_xmax_scatter.axvline(
                xmax_max_km,
                color="crimson",
                linestyle="--",
                linewidth=2,
                label=f"xmax = {xmax_max_km:g} km",
            )
            ax_swm_xmax_scatter.axhline(
                SWM_CHI_SQUARE_MAX,
                color="crimson",
                linestyle="-",
                linewidth=2,
                label=f"SWM $\\chi^2$ = {swm_chi_square_max:g}",
            )
            ax_swm_xmax_scatter.legend()
        else:
            ax_swm_xmax_scatter.text(
                0.5,
                0.5,
                "No SWM $\\chi^2$/xmax data",
                ha="center",
                va="center",
            )
            ax_swm_xmax_scatter.set_title("SWM $\\chi^2$ vs xmax distance")

        # Subplot 6: fit-parameter scatter
        ax_fit = axes[2, 1]
        if event_fit_results:
            slopes = [item[1] for item in event_fit_results]
            intercepts = [item[2] for item in event_fit_results]
            reduced_chi_squares = [item[3] for item in event_fit_results]
            finite_reduced_chi_squares = [
                value for value in reduced_chi_squares if not math.isnan(value)
            ]
            finite_positive_reduced_chi_squares = [
                value
                for value in reduced_chi_squares
                if not math.isnan(value) and value > 0
            ]
            logger.info(
                "Collected per-event linear fits count={} slope_range=[{}, {}] intercept_range=[{}, {}] finite_reduced_chi_square_count={}",
                len(event_fit_results),
                min(slopes),
                max(slopes),
                min(intercepts),
                max(intercepts),
                len(finite_reduced_chi_squares),
            )

            if finite_positive_reduced_chi_squares:
                colored_fit_results = [
                    item
                    for item in event_fit_results
                    if not math.isnan(item[3]) and item[3] > 0
                ]
                scatter_fit = ax_fit.scatter(
                    [item[1] for item in colored_fit_results],
                    [item[2] for item in colored_fit_results],
                    c=np.log10([item[3] for item in colored_fit_results]),
                    cmap="viridis",
                    s=28,
                    alpha=0.85,
                    edgecolors="none",
                )
                color_bar_fit = fig.colorbar(scatter_fit, ax=ax_fit)
                color_bar_fit.set_label(r"$\log_{10}$(reduced $\chi^2$)")
                ax_fit.set_xlabel("Slope (1/distance fit)")
                ax_fit.set_ylabel("Intercept")
                ax_fit.set_title("Signal vs 1/distance fit parameters")
            else:
                ax_fit.text(
                    0.5,
                    0.5,
                    "No positive reduced $\\chi^2$",
                    ha="center",
                    va="center",
                )
                ax_fit.set_title("Signal vs 1/distance fit parameters")
        else:
            ax_fit.text(0.5, 0.5, "No fit results", ha="center", va="center")
            ax_fit.set_title("Signal vs 1/distance fit parameters")

        fig.suptitle(f"SWM with signal overview: {swm_file.stem}", fontsize=20)
        fig.tight_layout()
        fig.savefig(str(overview_file), dpi=180)
        plt.close(fig)
        logger.info("Saved 6-panel overview plot: {}", overview_file)

    # ── slope candidate filter ─────────────────────────────────────────
    if not no_slope_filter and not dry_run:
        # Resolve candidate YAML path
        if candidate_yaml is not None:
            candidate_yaml_path = candidate_yaml
        else:
            # For SWM_with_signal input files: replace _with_signal with _candidates
            # For raw SWM files: use stem + _candidates
            stem = output_path.stem
            if stem.endswith("_with_signal"):
                candidate_stem = stem[: -len("_with_signal")]
            else:
                candidate_stem = swm_file.stem
            candidate_yaml_path = output_path.with_name(
                f"{candidate_stem}_candidates.yaml"
            )

        (
            qualified_events,
            slope_total,
            slope_gt_0,
            slope_le_0,
            no_fit,
            slope_filtered,
            slope_values,
            rchi2_values,
        ) = filter_slope_candidates(
            swm_events,
            chi_square_max=swm_chi_square_max,
            min_du_count=min_du_count,
        )

        # Write candidate YAML
        candidate_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_yaml_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(qualified_events, fp, default_flow_style=False, sort_keys=False)
        logger.info(
            "Candidate YAML written: {} ({} events)",
            candidate_yaml_path,
            len(qualified_events),
        )

        # Slope diagnostic plot
        if not no_plot and slope_values:
            resolved_plot_dir.mkdir(parents=True, exist_ok=True)
            slope_plot_path = resolved_plot_dir / f"{candidate_stem}_slope_hist.png"
            plot_slope_histogram(
                slope_plot_path,
                slope_values,
                title=f"Slope distribution: {swm_file.name}",
                reduced_chi_squares=rchi2_values if rchi2_values else None,
            )

    if dry_run:
        logger.info(
            "[dry-run] {} -> {} | total={} matched={} missing_xy={} "
            "missing_du_signal={}",
            swm_file.name,
            output_path.name,
            stats["total_events"],
            stats["matched_events"],
            stats["missing_xy_events"],
            stats["missing_du_signals"],
        )
        return True

    if replot_only or input_is_enriched:
        if input_is_enriched:
            logger.info(
                "Skip writing YAML because input is already enriched: {}",
                swm_file,
            )
        else:
            logger.info(
                "Skip writing YAML because --overwrite is not set and file exists: {}",
                output_path,
            )
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml_dict(output_path, swm_events)
    logger.info(
        "Wrote {} | total={} matched={} missing_xy={} missing_du_signal={}",
        output_path,
        stats["total_events"],
        stats["matched_events"],
        stats["missing_xy_events"],
        stats["missing_du_signals"],
    )
    return True


def main() -> int:
    """CLI entrypoint — process a single SWM file with enrichment and slope filtering."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info(
        "CLI args swm_file={} output_dir={} "
        "suffix={} det_pos_file={} plot_dir={} no_plot={} overwrite={} "
        "dry_run={} chi_square_threshold={} swm_chi_square_max={} "
        "xmax_min_km={} xmax_max_km={} no_slope_filter={} min_du_count={}",
        args.swm_file,
        args.output_dir,
        args.suffix,
        args.det_pos_file,
        args.plot_dir,
        args.no_plot,
        args.overwrite,
        args.dry_run,
        args.chi_square_threshold,
        args.swm_chi_square_max,
        args.xmax_min_km,
        args.xmax_max_km,
        args.no_slope_filter,
        args.min_du_count,
    )

    if args.swm_chi_square_max <= 0:
        parser.error("--swm-chi-square-max must be > 0")
    if args.xmax_min_km >= args.xmax_max_km:
        parser.error(
            "--xmax-min-km must be smaller than "
            "--xmax-max-km"
        )

    output_dir: Optional[Path] = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()

    plot_dir: Optional[Path] = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir).resolve()

    det_pos_file = Path(args.det_pos_file).resolve()
    if not det_pos_file.exists():
        parser.error(f"Detector position file does not exist: {det_pos_file}")
    det_positions = load_detector_positions(det_pos_file)
    logger.info(
        "Loaded detector positions count={} from {}",
        len(det_positions),
        det_pos_file,
    )
    if not det_positions:
        parser.error(f"No valid detector positions parsed from: {det_pos_file}")

    swm_file = Path(args.swm_file).resolve()

    try:
        ok = process_one_file(
            swm_file=swm_file,
            output_dir=output_dir,
            plot_dir=plot_dir,
            det_positions=det_positions,
            suffix=args.suffix,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            no_plot=args.no_plot,
            chi_square_threshold=args.chi_square_threshold,
            swm_chi_square_max=args.swm_chi_square_max,
            xmax_min_km=args.xmax_min_km,
            xmax_max_km=args.xmax_max_km,
            no_slope_filter=args.no_slope_filter,
            candidate_yaml=args.candidate_yaml,
            min_du_count=args.min_du_count,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to process {}: {}", swm_file, exc)
        return 2

    logger.info("Done. success={}", ok)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
