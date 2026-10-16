#!/usr/bin/env python3
"""Plot overview distributions from SWM_with_signal event files.

The script scans all ``*_SWM_with_signal.yaml`` files under ``--reco-dir`` and
aggregates event-level values for:
1) ``chi_square``
2) ``xmax distance``
3) ``theta/phi`` scatter views colored by ``log10(chi_square)``
4) linear-fit parameters ``slope`` and ``intercept`` colored by
    ``reduced_chi_square``

For xmax distance, it first tries the configured distance field in each event
payload. If that field is missing, it falls back to Euclidean distance from
origin using ``x, y, z``: ``sqrt(x^2 + y^2 + z^2)``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import yaml

from logger_config import logger


SWM_FILE_PATTERN = re.compile(
    r"^Trigger_(?P<date>\d{8})_RUN(?P<run>\d+).*_SWM_with_signal\.yaml$"
)


FitRecord = Tuple[float, float, float, float, float]
AngleRecord = Tuple[float, float, float, float, float]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate CLI arguments and fail fast with parser errors."""
    if args.bins < 1:
        parser.error("--bins must be >= 1")
    if re.fullmatch(r"\d{8}", str(args.start_date)) is None:
        parser.error("--start-date must match YYYYMMDD")
    if re.fullmatch(r"\d{8}", str(args.end_date)) is None:
        parser.error("--end-date must match YYYYMMDD")
    if args.start_date > args.end_date:
        parser.error("--start-date must be <= --end-date")
    if args.xmax_min_km < 0:
        parser.error("--xmax-min-km must be >= 0")
    if args.xmax_max_km < 0:
        parser.error("--xmax-max-km must be >= 0")
    if args.xmax_min_km > args.xmax_max_km:
        parser.error("--xmax-min-km must be <= --xmax-max-km")
    if args.chi_square_fit_max <= 0:
        parser.error("--chi-square-fit-max must be > 0")


def _prepare_theta_phi_color_data(
    records: List[AngleRecord],
    color_mode: str,
) -> Tuple[List[AngleRecord], np.ndarray | List[float], str]:
    """Prepare plotted records, color values, and colorbar label."""
    if color_mode == "xmax":
        plotted_records = records
        color_values: np.ndarray | List[float] = [
            record[0] for record in plotted_records
        ]
        color_bar_label = "xmax distance [km]"
        return plotted_records, color_values, color_bar_label

    if color_mode == "log_swm":
        plotted_records = [record for record in records if record[3] > 0]
        if len(plotted_records) < len(records):
            logger.info(
                "log_swm: filtered out {} records with chi_square <= 0",
                len(records) - len(plotted_records),
            )
        color_values = np.log10([record[3] for record in plotted_records])
        color_bar_label = r"log$_{10}$ SWM $\chi^2$"
        return plotted_records, color_values, color_bar_label

    plotted_records = [
        record
        for record in records
        if not math.isnan(record[4]) and record[4] > 0
    ]
    if len(plotted_records) < len(records):
        logger.info(
            "log_reduced: filtered out {} records with reduced_chi_square <= 0 or NaN",
            len(records) - len(plotted_records),
        )
    color_values = np.log10([record[4] for record in plotted_records])
    color_bar_label = r"log$_{10}$ reduced $\chi^2$"
    return plotted_records, color_values, color_bar_label


def _plot_fit_panel(
    fig: Figure,
    axis: plt.Axes,
    records: List[FitRecord],
    color_values: List[float] | np.ndarray,
    color_bar_label: str,
    title: str,
) -> None:
    """Plot one slope/intercept scatter panel with colorbar."""
    scatter = axis.scatter(
        [record[2] for record in records],
        [record[3] for record in records],
        c=color_values,
        cmap="viridis",
        s=18,
        alpha=0.85,
        edgecolors="none",
    )
    axis.set_title(title)
    axis.set_xlabel("slope")
    axis.set_ylabel("intercept")
    color_bar = fig.colorbar(scatter, ax=axis)
    color_bar.set_label(color_bar_label)


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line parser for histogram plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Read all SWM YAML files and plot chi_square and xmax distance "
            "histogram distributions plus fit-parameter scatter."
        )
    )
    parser.add_argument(
        "--swm-file",
        default=None,
        help="Process one SWM_with_signal YAML file directly.",
    )
    parser.add_argument(
        "--reco-dir",
        default="../Reco_Dir",
        help="Directory containing Trigger_*_SWM_with_signal.yaml files.",
    )
    parser.add_argument(
        "--output",
        default="../Reco_Dir/swm_with_signal_overview.png",
        help="Output PNG file path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Histogram bins for chi_square and xmax subplots (default: 60).",
    )
    parser.add_argument(
        "--xmax-field",
        default="xmax_distance_km",
        help="Preferred event field name for xmax distance.",
    )
    parser.add_argument(
        "--theta-field",
        default="zenith",
        help="Event field used as theta in theta/phi scatter subplots.",
    )
    parser.add_argument(
        "--phi-field",
        default="azimuth",
        help="Event field used as phi in theta/phi scatter subplots.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit of SWM files for quick test.",
    )
    parser.add_argument(
        "--log-every-files",
        type=int,
        default=20,
        help="Log progress every N files during collection (default: 20).",
    )
    parser.add_argument(
        "--start-date",
        default="20250701",
        help="Start date in YYYYMMDD (inclusive). Default: 20250701.",
    )
    parser.add_argument(
        "--end-date",
        default="20260401",
        help="End date in YYYYMMDD (inclusive). Default: 20260401.",
    )
    parser.add_argument(
        "--xmax-min-km",
        type=float,
        default=10.0,
        help="Lower bound (inclusive) of xmax distance cut in km.",
    )
    parser.add_argument(
        "--xmax-max-km",
        type=float,
        default=100.0,
        help="Upper bound (inclusive) of xmax distance cut in km.",
    )
    parser.add_argument(
        "--chi-square-fit-max",
        type=float,
        default=100.0,
        help="Upper chi_square cut for fit-parameter distributions.",
    )
    return parser


def _iter_swm_files(
    reco_dir: Path,
    max_files: int | None,
    start_date: str,
    end_date: str,
) -> Iterable[Path]:
    """Yield SWM files from reco directory filtered by date range."""
    scanned_count = 0
    matched_name_count = 0
    in_range_count = 0
    files: List[Path] = []
    for file_path in sorted(reco_dir.glob("Trigger_*_SWM_with_signal.yaml")):
        scanned_count += 1
        matched = SWM_FILE_PATTERN.match(file_path.name)
        if matched is None:
            continue
        matched_name_count += 1
        date_text = matched.group("date")
        if not (start_date <= date_text <= end_date):
            continue

        files.append(file_path)
        in_range_count += 1
        if max_files is not None and len(files) >= max_files:
            break

    logger.info(
        "SWM file filter scanned={} matched_pattern={} in_date_range={} "
        "selected={}",
        scanned_count,
        matched_name_count,
        in_range_count,
        len(files),
    )
    yield from files


def _to_float(value: Any) -> float | None:
    """Convert numeric-like value to float, returning None on failure."""
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_xmax_distance(event_payload: Dict[str, Any], xmax_field: str) -> float | None:
    """Extract xmax distance from payload using preferred field or xyz fallback."""
    direct_value = _to_float(event_payload.get(xmax_field))
    if direct_value is not None:
        return direct_value

    x_value = _to_float(event_payload.get("x"))
    y_value = _to_float(event_payload.get("y"))
    z_value = _to_float(event_payload.get("z"))
    if x_value is None or y_value is None or z_value is None:
        return None
    return math.sqrt(x_value**2 + y_value**2 + z_value**2) / 1000.0  # convert to km


def _extract_fit_triplet(
    event_payload: Dict[str, Any],
) -> Tuple[float, float, float] | None:
    """Extract slope, intercept, and reduced chi-square from event payload."""
    fit_payload = event_payload.get("signal_distance_linear_fit")
    if not isinstance(fit_payload, dict):
        return None

    slope = _to_float(fit_payload.get("slope"))
    intercept = _to_float(fit_payload.get("intercept"))
    reduced_chi_square = _to_float(fit_payload.get("reduced_chi_square"))
    if slope is None or intercept is None or reduced_chi_square is None:
        return None

    return slope, intercept, reduced_chi_square


def _extract_angle_pair(
    event_payload: Dict[str, Any],
    theta_field: str,
    phi_field: str,
) -> Tuple[float, float] | None:
    """Extract theta and phi values from one event payload."""
    theta_value = _to_float(event_payload.get(theta_field))
    phi_value = _to_float(event_payload.get(phi_field))
    if theta_value is None or phi_value is None:
        return None
    return theta_value, phi_value


PerFileData = Tuple[
    List[Tuple[float, float]], List[FitRecord], List[AngleRecord]
]


def _collect_values(
    swm_files: List[Path],
    xmax_field: str,
    theta_field: str,
    phi_field: str,
    log_every_files: int,
) -> Tuple[
    List[Tuple[float, float]],
    List[FitRecord],
    List[AngleRecord],
    List[Tuple[Path, PerFileData]],
    int,
    int,
]:
    """Collect chi_square and xmax distance values from SWM files.

    Returns:
        paired_values, fit_records, angle_records, per_file_data,
        file_count, event_count
    """
    paired_values: List[Tuple[float, float]] = []
    fit_records: List[FitRecord] = []
    angle_records: List[AngleRecord] = []
    per_file_data: List[Tuple[Path, PerFileData]] = []
    file_count = 0
    event_count = 0

    for swm_file in swm_files:
        file_count += 1
        file_paired: List[Tuple[float, float]] = []
        file_fit: List[FitRecord] = []
        file_angle: List[AngleRecord] = []
        file_event_count = 0
        file_chi_square_count = 0
        file_xmax_count = 0
        file_fit_count = 0
        file_angle_count = 0
        with swm_file.open("r", encoding="utf-8") as file_obj:
            payload = yaml.load(file_obj, Loader=yaml.FullLoader) or {}

        if not isinstance(payload, dict):
            logger.warning("Skip non-mapping SWM payload: {}", swm_file)
            continue

        for _, event_payload in payload.items():
            if not isinstance(event_payload, dict):
                continue

            event_count += 1
            file_event_count += 1
            chi_square = _to_float(event_payload.get("chi_square"))
            xmax_distance = _extract_xmax_distance(event_payload, xmax_field)
            fit_triplet = _extract_fit_triplet(event_payload)
            angle_pair = _extract_angle_pair(
                event_payload,
                theta_field=theta_field,
                phi_field=phi_field,
            )

            if chi_square is not None:
                file_chi_square_count += 1
            if xmax_distance is not None:
                file_xmax_count += 1

            chi_valid = chi_square is not None and chi_square > 0
            xmax_valid = xmax_distance is not None
            angle_valid = angle_pair is not None
            fit_valid = fit_triplet is not None

            if chi_valid and xmax_valid:
                pair = (chi_square, xmax_distance)
                paired_values.append(pair)
                file_paired.append(pair)

            if chi_valid and xmax_valid and fit_valid:
                file_fit_count += 1
                fit_record = (
                    chi_square,
                    xmax_distance,
                    fit_triplet[0],
                    fit_triplet[1],
                    fit_triplet[2],
                )
                fit_records.append(fit_record)
                file_fit.append(fit_record)

            if chi_valid and xmax_valid and angle_valid:
                file_angle_count += 1
                angle_record = (
                    xmax_distance,
                    angle_pair[0],
                    angle_pair[1],
                    chi_square,
                    fit_triplet[2] if fit_valid else float("nan"),
                )
                angle_records.append(angle_record)
                file_angle.append(angle_record)

        per_file_data.append((swm_file, (file_paired, file_fit, file_angle)))

        logger.info(
            "Parsed file {}/{} {} events={} chi_square_valid={} xmax_valid={} fit_valid={} angle_valid={}",
            file_count,
            len(swm_files),
            swm_file.name,
            file_event_count,
            file_chi_square_count,
            file_xmax_count,
            file_fit_count,
            file_angle_count,
        )

        if log_every_files > 0 and file_count % log_every_files == 0:
            logger.info(
                "Progress files_processed={} cumulative_events={} "
                "cumulative_valid_pairs={} cumulative_fit_values={} "
                "cumulative_angle_values={}",
                file_count,
                event_count,
                len(paired_values),
                len(fit_records),
                len(angle_records),
            )

    return paired_values, fit_records, angle_records, per_file_data, file_count, event_count


def _plot_overview(
    chi_square_selected_pairs: List[Tuple[float, float]],
    xmax_selected_pairs: List[Tuple[float, float]],
    both_selected_pairs: List[Tuple[float, float]],
    fit_records: List[FitRecord],
    angle_records: List[AngleRecord],
    bins: int,
    chi_square_fit_max: float,
    xmax_min_km: float,
    xmax_max_km: float,
    theta_field: str,
    phi_field: str,
    output_path: Path,
) -> None:
    """Plot a 3x3 overview with histograms, theta/phi, and fit distributions."""
    _apply_plot_style()

    fig = plt.figure(figsize=(24, 18))
    grid_spec = fig.add_gridspec(3, 3)

    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            projection = "polar" if row == 1 else None
            axes[row, col] = fig.add_subplot(grid_spec[row, col], projection=projection)

    if chi_square_selected_pairs:
        axes[0, 0].scatter(
            [pair[1] for pair in chi_square_selected_pairs],
            [pair[0] for pair in chi_square_selected_pairs],
            s=10,
            alpha=0.6,
            edgecolors="none",
        )
    else:
        axes[0, 0].text(0.5, 0.5, "No events", ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title(rf"xmax vs SWM $\chi^2$ for $\chi^2$ < {chi_square_fit_max:g}")
    axes[0, 0].set_xlabel("xmax distance [km]")
    axes[0, 0].set_ylabel(r"SWM $\chi^2$")
    axes[0, 0].set_yscale("log")

    if xmax_selected_pairs:
        axes[0, 1].scatter(
            [pair[1] for pair in xmax_selected_pairs],
            [pair[0] for pair in xmax_selected_pairs],
            s=10,
            alpha=0.6,
            edgecolors="none",
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No events", ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title(
        rf"xmax vs SWM $\chi^2$ for {xmax_min_km:g} <= xmax <= {xmax_max_km:g} km"
    )
    axes[0, 1].set_xlabel("xmax distance [km]")
    axes[0, 1].set_ylabel(r"SWM $\chi^2$")
    axes[0, 1].set_yscale("log")

    if both_selected_pairs:
        axes[0, 2].scatter(
            [pair[1] for pair in both_selected_pairs],
            [pair[0] for pair in both_selected_pairs],
            s=10,
            alpha=0.6,
            edgecolors="none",
        )
    else:
        axes[0, 2].text(0.5, 0.5, "No events", ha="center", va="center", transform=axes[0, 2].transAxes)
    axes[0, 2].set_title(
        rf"xmax vs SWM $\chi^2$ for $\chi^2$ < {chi_square_fit_max:g} and {xmax_min_km:g} <= xmax <= {xmax_max_km:g} km"
    )
    axes[0, 2].set_xlabel("xmax distance [km]")
    axes[0, 2].set_ylabel(r"SWM $\chi^2$")
    axes[0, 2].set_yscale("log")

    chi_square_records = [
        record for record in angle_records if record[3] < chi_square_fit_max
    ]
    xmax_records = [
        record
        for record in angle_records
        if xmax_min_km <= record[0] <= xmax_max_km
    ]
    both_theta_phi_records = [
        record
        for record in angle_records
        if record[3] < chi_square_fit_max and xmax_min_km <= record[0] <= xmax_max_km
    ]

    theta_phi_specs = [
        (
            axes[1, 0],
            chi_square_records,
            rf"$\chi^2$ < {chi_square_fit_max:g}",
            "xmax",
        ),
        (
            axes[1, 1],
            xmax_records,
            rf"{xmax_min_km:g} <= xmax <= {xmax_max_km:g} km",
            "log_swm",
        ),
        (
            axes[1, 2],
            both_theta_phi_records,
            rf"$\chi^2$ < {chi_square_fit_max:g} and {xmax_min_km:g} <= xmax <= {xmax_max_km:g} km",
            "log_reduced",
        ),
    ]
    for axis, records, title_text, color_mode in theta_phi_specs:
        plotted_records, color_values, color_bar_label = (
            _prepare_theta_phi_color_data(records, color_mode)
        )

        if plotted_records:
            theta_phi_scatter = axis.scatter(
                np.deg2rad([record[2] for record in plotted_records]),
                [record[1] for record in plotted_records],
                c=color_values,
                cmap="viridis",
                s=16,
                alpha=0.8,
                edgecolors="none",
            )
            axis.set_theta_zero_location("N")
            axis.set_xticks(np.deg2rad([0, 315, 270, 225, 180, 135, 90, 45]))
            axis.set_xticklabels(
                ["N", "315°", "E", "225°", "S", "135°", "W", "45°"],
                fontsize=12,
            )
            theta_phi_color_bar = fig.colorbar(theta_phi_scatter, ax=axis)
            theta_phi_color_bar.set_label(color_bar_label)
        else:
            axis.text(
                0.5,
                0.5,
                "No events",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(title_text)
        axis.set_xlabel(f"{phi_field} [deg]")
        axis.set_ylabel(f"{theta_field} [deg]")

    chi_square_selected_records = [
        record for record in fit_records if record[0] < chi_square_fit_max
    ]
    _plot_fit_panel(
        fig=fig,
        axis=axes[2, 0],
        records=chi_square_selected_records,
        color_values=[record[1] for record in chi_square_selected_records],
        color_bar_label="xmax distance [km]",
        title=rf"Slope vs Intercept for $\chi^2$ < {chi_square_fit_max:g}",
    )

    xmax_selected_records = [
        record
        for record in fit_records
        if xmax_min_km <= record[1] <= xmax_max_km
    ]
    _plot_fit_panel(
        fig=fig,
        axis=axes[2, 1],
        records=xmax_selected_records,
        color_values=np.log10([record[0] for record in xmax_selected_records]),
        color_bar_label=r"log$_{10}$ SWM $\chi^2$",
        title=(
            rf"Slope vs Intercept for {xmax_min_km:g} <= xmax <= "
            rf"{xmax_max_km:g} km"
        ),
    )

    chi_square_xmax_selected_records = [
        record
        for record in fit_records
        if record[0] < chi_square_fit_max
        and xmax_min_km <= record[1] <= xmax_max_km
    ]
    _plot_fit_panel(
        fig=fig,
        axis=axes[2, 2],
        records=chi_square_xmax_selected_records,
        color_values=np.log10(
            [record[4] for record in chi_square_xmax_selected_records]
        ),
        color_bar_label=r"log$_{10}$ reduced $\chi^2$",
        title=(
            rf"Slope vs Intercept for $\chi^2$ < {chi_square_fit_max:g} and "
            rf"{xmax_min_km:g} <= xmax <= {xmax_max_km:g} km"
        ),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def main() -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    output_path = Path(args.output).resolve()

    if args.swm_file is not None:
        swm_file = Path(args.swm_file).resolve()
        if not swm_file.is_file():
            parser.error(f"SWM file does not exist: {swm_file}")
        swm_files = [swm_file]
        logger.info("Input mode=swm-file file={}", swm_file)
    else:
        reco_dir = Path(args.reco_dir).resolve()
        if not reco_dir.is_dir():
            parser.error(f"Reco directory does not exist: {reco_dir}")

        swm_files = list(
            _iter_swm_files(
                reco_dir,
                args.max_files,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        )
        logger.info(
            "Input mode=scan reco_dir={} date_range={}..{}",
            reco_dir,
            args.start_date,
            args.end_date,
        )

        if not swm_files:
            logger.warning(
                "No SWM files found under {} in date range {}..{}",
                args.reco_dir,
                args.start_date,
                args.end_date,
            )
            return 1

        logger.info(
            "Start collecting values from swm_files={} reco_dir={} xmax_field={} "
            "date_range={}..{}",
            len(swm_files),
            args.reco_dir,
            args.xmax_field,
            args.start_date,
            args.end_date,
        )

    (
        paired_values,
        fit_records,
        angle_records,
        per_file_data,
        file_count,
        event_count,
    ) = _collect_values(
        swm_files,
        xmax_field=args.xmax_field,
        theta_field=args.theta_field,
        phi_field=args.phi_field,
        log_every_files=args.log_every_files,
    )

    plot_xmax_min_km = args.xmax_min_km
    plot_xmax_max_km = args.xmax_max_km

    def _make_selected(
        pairs: List[Tuple[float, float]],
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
    ]:
        """Return (chi_square_selected, xmax_selected, both_selected) in one pass."""
        chi_sel: List[Tuple[float, float]] = []
        xmax_sel: List[Tuple[float, float]] = []
        both_sel: List[Tuple[float, float]] = []
        for p in pairs:
            chi_ok = p[0] < args.chi_square_fit_max
            xmax_ok = plot_xmax_min_km <= p[1] <= plot_xmax_max_km
            if chi_ok:
                chi_sel.append(p)
            if xmax_ok:
                xmax_sel.append(p)
            if chi_ok and xmax_ok:
                both_sel.append(p)
        return chi_sel, xmax_sel, both_sel

    chi_square_selected_pairs, xmax_selected_pairs, both_selected_pairs = (
        _make_selected(paired_values)
    )

    logger.info(
        "Collection done files={} events={} valid_pairs={} valid_fits={} valid_angles={} chi_square_fit_max={} xmax_range=[{}, {}]",
        file_count,
        event_count,
        len(paired_values),
        len(fit_records),
        len(angle_records),
        args.chi_square_fit_max,
        plot_xmax_min_km,
        plot_xmax_max_km,
    )
    logger.info(
        "Plot cuts applied chi_square_selected_pairs={} xmax_selected_pairs={} both_selected_pairs={} from_valid_pairs={} with chi_square<{} xmax_in_range=[{}, {}]",
        len(chi_square_selected_pairs),
        len(xmax_selected_pairs),
        len(both_selected_pairs),
        len(paired_values),
        args.chi_square_fit_max,
        plot_xmax_min_km,
        plot_xmax_max_km,
    )

    # ── Per-file overview plots ─────────────────────────────────────────
    output_dir = output_path.parent
    for swm_file, (file_paired, file_fit, file_angle) in per_file_data:
        if not file_paired or not file_fit or not file_angle:
            logger.warning(
                "Skip per-file plot for {}: pairs={} fits={} angles={}",
                swm_file.name,
                len(file_paired),
                len(file_fit),
                len(file_angle),
            )
            continue

        fc, fx, fb = _make_selected(file_paired)
        per_file_png = output_dir / f"{swm_file.stem}_overview.png"
        _plot_overview(
            chi_square_selected_pairs=fc,
            xmax_selected_pairs=fx,
            both_selected_pairs=fb,
            fit_records=file_fit,
            angle_records=file_angle,
            bins=args.bins,
            chi_square_fit_max=args.chi_square_fit_max,
            xmax_min_km=plot_xmax_min_km,
            xmax_max_km=plot_xmax_max_km,
            theta_field=args.theta_field,
            phi_field=args.phi_field,
            output_path=per_file_png,
        )
        logger.info("Saved per-file overview: {}", per_file_png)

    # ── Combined overview plot ──────────────────────────────────────────
    if not paired_values:
        logger.error("No valid chi_square/xmax pairs found.")
        return 2
    if not fit_records:
        logger.error("No valid linear-fit values found.")
        return 2
    if not angle_records:
        logger.error("No valid theta/phi values found.")
        return 2

    _plot_overview(
        chi_square_selected_pairs=chi_square_selected_pairs,
        xmax_selected_pairs=xmax_selected_pairs,
        both_selected_pairs=both_selected_pairs,
        fit_records=fit_records,
        angle_records=angle_records,
        bins=args.bins,
        chi_square_fit_max=args.chi_square_fit_max,
        xmax_min_km=plot_xmax_min_km,
        xmax_max_km=plot_xmax_max_km,
        theta_field=args.theta_field,
        phi_field=args.phi_field,
        output_path=output_path,
    )
    logger.info("Saved combined overview figure: {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())