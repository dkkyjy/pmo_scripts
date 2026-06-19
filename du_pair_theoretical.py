import csv
import itertools
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Iterable, Any
import numpy as np
from collections import namedtuple

from logger_config import logger
from find_event.io import load_data_from_file

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0 / 1.000259 # m/s in air at STP, adjusted for refractive index
NS_PER_SECOND = 1e9

EXPECTED_PAIR_DELTA_FIELDS = (
    "du_a",
    "du_b",
    "distance_m",
    "theoretical_delta_ns",
)

ExpectedPairDelta = namedtuple(
    "ExpectedPairDelta", ["du_a", "du_b", "distance_m", "theoretical_delta_ns"]
)


def make_named_row(fields: Iterable[str], **values: Any):
    """Create one named row based on a predefined field schema using namedtuple."""
    tuple_type = {
        EXPECTED_PAIR_DELTA_FIELDS: ExpectedPairDelta,
    }.get(fields)
    return tuple_type(**values)


def build_expected_pair_deltas(
    detector_positions: Dict[str, np.ndarray],
) -> List[ExpectedPairDelta]:
    """Build expected DU-pair deltas from geometry as ``distance / c``."""
    usable_ids = sorted(detector_positions.keys())

    rows: List[ExpectedPairDelta] = []
    for du_a, du_b in itertools.combinations(usable_ids, 2):
        pos_a = detector_positions[du_a]
        pos_b = detector_positions[du_b]
        distance_m = float(np.linalg.norm(pos_b - pos_a))
        expected_delta_ns = distance_m / SPEED_OF_LIGHT_M_PER_S * NS_PER_SECOND
        rows.append(
            make_named_row(
                EXPECTED_PAIR_DELTA_FIELDS,
                du_a=du_a,
                du_b=du_b,
                distance_m=distance_m,
                theoretical_delta_ns=expected_delta_ns,
            )
        )

    rows.sort(
        key=lambda row: (
            str(row.du_a),
            str(row.du_b),
        )
    )
    return rows


def theoretical_cache_path(det_pos_path: Path) -> Path:
    """Return shared cache CSV path for theoretical DU-pair deltas."""
    return det_pos_path.with_name(f"{det_pos_path.stem}_du_pair_theoretical.csv")


def write_theoretical_cache(path: Path, rows: Sequence[ExpectedPairDelta]) -> None:
    """Write shared theoretical DU-pair cache CSV."""
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ExpectedPairDelta._fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row._asdict())


def read_theoretical_cache(path: Path) -> List[ExpectedPairDelta]:
    """Read shared theoretical DU-pair cache CSV."""
    rows: List[ExpectedPairDelta] = []
    invalid_row_count = 0
    required_fields = {"du_a", "du_b", "distance_m", "theoretical_delta_ns"}

    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        header_fields = set(reader.fieldnames or [])
        if not required_fields.issubset(header_fields):
            missing_fields = sorted(required_fields - header_fields)  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "Theoretical cache missing required columns: "
                f"{','.join(missing_fields)}"
            )

        for row_index, row in enumerate(reader, start=2):
            try:
                du_a = str(row["du_a"])
                du_b = str(row["du_b"])
                distance_m = float(row["distance_m"])
                theoretical_delta_ns = float(row["theoretical_delta_ns"])
            except (TypeError, ValueError, KeyError) as exc:
                invalid_row_count += 1
                logger.warning(
                    "Skip invalid theoretical cache row at line {}: {}",
                    row_index,
                    exc,
                )
                continue

            rows.append(
                make_named_row(
                    EXPECTED_PAIR_DELTA_FIELDS,
                    du_a=du_a,
                    du_b=du_b,
                    distance_m=distance_m,
                    theoretical_delta_ns=theoretical_delta_ns,
                )
            )

    if invalid_row_count > 0:
        logger.warning(
            "Skipped {} invalid rows while reading theoretical cache: {}",
            invalid_row_count,
            path,
        )

    return rows


def load_or_build_theoretical_rows(
    det_pos_path: Path,
    force_recompute: bool = False,
) -> List[ExpectedPairDelta]:
    """Load shared theoretical cache if exists; otherwise build and persist."""
    cache_path = theoretical_cache_path(det_pos_path)
    if not force_recompute and cache_path.exists():
        logger.info("Using theoretical DU-pair cache: {}", cache_path)
        return read_theoretical_cache(cache_path)

    if force_recompute and cache_path.exists():
        logger.info("Force recompute enabled, ignoring cache: {}", cache_path)

    logger.info("Loaded detector positions from: {}", det_pos_path)
    detector_positions = load_data_from_file(det_pos_path)
    logger.info(f"Loaded {len(detector_positions)} detector positions")
    rows = build_expected_pair_deltas(detector_positions)
    write_theoretical_cache(cache_path, rows)
    logger.info("Wrote theoretical DU-pair cache: {}", cache_path)
    return rows


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build theoretical DU-pair deltas from geometry."
    )
    parser.add_argument(
        "det_pos",
        type=str,
        nargs="?",
        default="_gp65_rtksort_2002_DU7.txt",
        help="Path to detector positions file (default: _gp65_rtksort_2002_DU7.txt)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute theoretical deltas even if cache exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    det_pos_path = Path(args.det_pos)
    rows = load_or_build_theoretical_rows(det_pos_path, force_recompute=args.force_recompute)
