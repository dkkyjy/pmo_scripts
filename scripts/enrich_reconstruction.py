#!/usr/bin/env python3
"""Merge reconstruction_summary.yaml info into SWM-with-signal YAML files.

For each SWM-with-signal YAML, this script reads
``reconstruction_summary.yaml`` and merges per-event reconstruction fields
(PWF_fit, SWF, ADF, antennas) into matching events, writing output to
``{swm_file.stem}_reconstruction.yaml``.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import yaml

from logger_config import logger

plt.style.use(["science", "notebook", "grid"])


SWM_FILE_PATTERN = re.compile(
    r"^Trigger_(?P<date>\d{8})_RUN(?P<run>\d+)_merged.*_SWM_with_signal\.yaml$"
)

RECON_FIELDS = ("PWF_fit", "SWF", "ADF", "antennas")


def load_reconstruction_summary(
    recon_path: Path,
) -> Dict[int, Dict[str, Any]]:
    """Load reconstruction_summary.yaml and index by event_number (int key)."""
    with recon_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if not isinstance(data, dict):
        logger.error("reconstruction_summary top-level is not a dict")
        return {}

    # Keys are already event_numbers (int), but may be str after yaml load
    indexed: Dict[int, Dict[str, Any]] = {}
    for key, payload in data.items():
        if not isinstance(payload, dict):
            continue
        try:
            event_number = int(key)
        except (TypeError, ValueError):
            continue
        indexed[event_number] = payload

    logger.info(
        "Loaded reconstruction_summary: {} events",
        len(indexed),
    )
    return indexed


def _iter_swm_files(
    reco_dir: Path,
    start_date: str,
    end_date: str,
    max_files: Optional[int],
    swm_file: Optional[Path],
) -> List[Path]:
    """Collect SWM-with-signal files from reco directory."""
    if swm_file is not None:
        if not swm_file.exists():
            logger.error("SWM file does not exist: {}", swm_file)
            return []
        return [swm_file]

    files: List[Path] = []
    for file_path in sorted(reco_dir.glob("*_SWM_with_signal.yaml")):
        matched = SWM_FILE_PATTERN.match(file_path.name)
        if matched is None:
            continue
        date_text = matched.group("date")
        if not (start_date <= date_text <= end_date):
            continue
        files.append(file_path)

    if max_files is not None:
        files = files[:max_files]

    logger.info(
        "SWM file filter: in_date_range={} selected={}",
        len(files),
        len(files),
    )
    return files


def _extract_energy_flux_omega(
    output: Dict[Any, Dict[str, Any]],
) -> tuple:
    """Extract energy_flux and omega_rad per DU across all events.

    energy_flux is normalized per event (divided by the event's max energy_flux).
    """
    energy_flux_all: List[float] = []
    omega_rad_all: List[float] = []
    for _event_key, payload in output.items():
        antennas = payload.get("antennas")
        if not isinstance(antennas, dict):
            continue
        ef = antennas.get("energy_flux")
        om = antennas.get("omega_rad")
        if not (isinstance(ef, list) and isinstance(om, list)):
            continue
        if len(ef) != len(om) or len(ef) == 0:
            continue
        ef_max = max(e for e in ef if e is not None)
        if ef_max == 0:
            continue
        for e, o in zip(ef, om):
            if e is not None and o is not None:
                energy_flux_all.append(e / ef_max)
                omega_rad_all.append(o)
    return np.array(energy_flux_all), np.array(omega_rad_all)


def _plot_energy_flux_vs_omega(
    output_path: Path,
    energy_flux: np.ndarray,
    omega_rad: np.ndarray,
    n_events: int,
) -> None:
    """Plot energy_flux vs omega_rad scatter for all DUs."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        omega_rad,
        energy_flux,
        s=12,
        alpha=0.6,
        edgecolors="none",
    )
    ax.set_xlabel(r"$\omega_{\rm rad}$")
    ax.set_ylabel("Energy flux (normalized per event)")
    ax.set_title(
        f"{output_path.stem}\n"
        f"({n_events} events, {len(energy_flux)} DUs)"
    )
    fig.tight_layout()
    png_path = output_path.with_name(
        f"{output_path.stem}_energy_flux_vs_omega.png"
    )
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved -> {}", png_path)


def merge_one_file(
    swm_path: Path,
    recon_index: Dict[int, Dict[str, Any]],
    no_plot: bool = False,
) -> Optional[Path]:
    """Merge reconstruction data into one SWM YAML file.

    Only events that exist in reconstruction_summary.yaml are kept in the output.
    Returns the output path on success, None on failure.
    """
    with swm_path.open("r", encoding="utf-8") as fp:
        swm_data = yaml.safe_load(fp)

    if not isinstance(swm_data, dict):
        logger.warning("SWM top-level is not a dict: {}", swm_path)
        return None

    # Build output: only keep events that are in reconstruction_summary.yaml
    output: Dict[Any, Dict[str, Any]] = {}
    merged_count = 0
    for event_key, payload in swm_data.items():
        if not isinstance(payload, dict):
            continue
        event_number = payload.get("event_number")
        if event_number is None:
            continue
        try:
            en = int(event_number)
        except (TypeError, ValueError):
            continue

        recon_payload = recon_index.get(en)
        if recon_payload is None:
            continue

        # Merge reconstruction fields
        for field in RECON_FIELDS:
            if field in recon_payload:
                payload[field] = recon_payload[field]
        output[event_key] = payload
        merged_count += 1

    if merged_count == 0:
        logger.warning(
            "No reconstruction events found for: {}",
            swm_path.name,
        )
        return None

    output_path = swm_path.with_name(f"{swm_path.stem}_reconstruction.yaml")
    with output_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(
            output,
            fp,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info(
        "Kept {}/{} events -> {}",
        merged_count,
        len(swm_data),
        output_path,
    )

    # Plot energy_flux vs omega_rad scatter
    if not no_plot:
        energy_flux, omega_rad = _extract_energy_flux_omega(output)
        if len(energy_flux) > 0:
            _plot_energy_flux_vs_omega(
                output_path, energy_flux, omega_rad, merged_count,
            )
        else:
            logger.warning(
                "No energy_flux/omega_rad data to plot for: {}",
                swm_path.name,
            )

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge reconstruction_summary.yaml into SWM-with-signal YAML files."
        )
    )
    parser.add_argument(
        "reco_dir",
        nargs="?",
        default="../Reco_Dir",
        help="Directory containing *_SWM_with_signal.yaml files (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--recon-summary",
        type=Path,
        default=None,
        help=(
            "Path to reconstruction_summary.yaml. "
            "Default: <repo_root>/reconstruction_summary.yaml"
        ),
    )
    parser.add_argument(
        "--swm-file",
        type=Path,
        default=None,
        help="Process a single SWM-with-signal YAML file directly.",
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
        "--max-files",
        type=int,
        default=None,
        help="Limit number of SWM files to process.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating energy_flux vs omega_rad scatter plot.",
    )
    return parser


def main() -> int:
    """Main workflow."""
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.recon_summary is not None:
        recon_path = args.recon_summary
    else:
        recon_path = repo_root / "reconstruction_summary.yaml"

    if not recon_path.exists():
        logger.error(
            "reconstruction_summary.yaml not found: {}",
            recon_path,
        )
        return 2

    reco_dir = Path(args.reco_dir)
    if not reco_dir.is_dir():
        logger.error("Reco directory does not exist: {}", reco_dir)
        return 2

    if args.swm_file is not None and not args.swm_file.exists():
        logger.error("SWM file does not exist: {}", args.swm_file)
        return 2

    if re.fullmatch(r"\d{8}", str(args.start_date)) is None:
        parser.error("--start-date must match YYYYMMDD")
    if re.fullmatch(r"\d{8}", str(args.end_date)) is None:
        parser.error("--end-date must match YYYYMMDD")
    if args.start_date > args.end_date:
        parser.error("--start-date must be <= --end-date")

    # Load reconstruction summary once
    recon_index = load_reconstruction_summary(recon_path)
    if not recon_index:
        logger.error("No reconstruction data loaded.")
        return 1

    # Collect SWM files
    swm_files = _iter_swm_files(
        reco_dir,
        args.start_date,
        args.end_date,
        args.max_files,
        args.swm_file,
    )

    if not swm_files:
        logger.error("No SWM-with-signal files found in: {}", reco_dir)
        return 1

    success_count = 0
    fail_count = 0
    for swm_path in swm_files:
        result = merge_one_file(swm_path, recon_index, no_plot=args.no_plot)
        if result is not None:
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        "Done. total_files={} success={} fail={}",
        len(swm_files),
        success_count,
        fail_count,
    )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
