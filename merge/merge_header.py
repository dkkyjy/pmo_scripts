#!/usr/bin/env python3
"""Merge Trigger YAML files by event_number.

Behavior:
- Match only `Trigger*.yaml` under the input directory.
- Always merge by `event_number` (append mode is not supported).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from logger_config import logger
from merge.common import (
    build_traceability_header,
    find_files,
    parse_date_dir as _parse_date_dir,
    resolve_input_dir,
)

PATTERN = "Trigger*.yaml"
YAML_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def event_key_sort_value(event_key: str) -> Any:
    """Sort event keys numerically when possible, else lexicographically."""
    return int(event_key) if event_key.isdigit() else event_key


def build_trigger_pattern(run_number: Optional[int]) -> str:
    """Build a coarse glob pattern for Trigger YAML files.

    When run_number is provided, use a narrower glob and then rely on
    regex filtering for strict numeric boundaries.
    """
    if run_number is None:
        return PATTERN
    return f"Trigger_*_RUN{run_number}_*.yaml"


def _filter_files_by_run_number(files: List[Path], run_number: Optional[int]) -> List[Path]:
    """Filter files by exact RUN segment while avoiding RUN10/RUN100 collisions."""
    if run_number is None:
        return files

    # Match RUN<run_number> where next character is not a digit.
    run_pattern = re.compile(rf"RUN{run_number}(?!\d)")
    return [path for path in files if run_pattern.search(path.name)]


def _to_scalar_or_list(value: Any) -> List[Any]:
    """Normalize one DU value into scalar-or-list form.

    - Lists are preserved.
    - Scalars are wrapped to list only when concatenation is needed by caller.
    """
    if isinstance(value, list):
        return value
    return [value]


def merge_sample_map(
    base_map: Dict[str, Any],
    incoming_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge DU sample maps and preserve multiple trigger samples per DU.

    Supports both legacy scalar values and new list values.
    When a DU appears in both maps, samples are concatenated in order.
    """
    merged: Dict[str, Any] = dict(base_map)
    for du_id, incoming_value in incoming_map.items():
        du_id_str = str(du_id)
        if du_id_str not in merged:
            merged[du_id_str] = incoming_value
            continue

        existing_list = _to_scalar_or_list(merged[du_id_str])
        incoming_list = _to_scalar_or_list(incoming_value)
        merged[du_id_str] = [*existing_list, *incoming_list]

    return merged


def merge_du_id_list(
    base_ids: Optional[List[Any]],
    incoming_ids: List[Any],
) -> List[str]:
    """Merge DU id lists and keep insertion order with string values."""
    existing_ids = base_ids if isinstance(base_ids, list) else []
    merged_ids = [str(item) for item in existing_ids]
    seen = set(merged_ids)

    for item in incoming_ids:
        item_str = str(item)
        if item_str not in seen:
            merged_ids.append(item_str)
            seen.add(item_str)

    return merged_ids


def merge_event_payload(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two event payload dicts for the same event_number."""
    merged = dict(base)
    logger.debug(
        "Merging payloads for event_number={} with incoming fields={}",
        incoming.get("event_number"),
        list(incoming.keys()),
    )

    for field, value in incoming.items():
        logger.debug("Processing field='{}' (type={})", field, type(value).__name__)

        if field in {"time", "signal"} and isinstance(value, dict):
            existing = merged.get(field)
            if not isinstance(existing, dict):
                existing = {}
            before_count = len(existing)
            merged[field] = merge_sample_map(existing, value)
            logger.debug(
                "Merged dict field='{}': before_keys={}, incoming_keys={}, after_keys={}",
                field,
                before_count,
                len(value),
                len(merged[field]),
            )
            continue

        if field == "du_id" and isinstance(value, list):
            existing_ids = merged.get("du_id")
            before_count = len(existing_ids) if isinstance(existing_ids, list) else 0
            merged["du_id"] = merge_du_id_list(existing_ids, value)
            logger.debug(
                "Merged du_id list: before_count={}, incoming_count={}, after_count={}",
                before_count,
                len(value),
                len(merged["du_id"]),
            )
            continue

        if field in {"file", "index"}:
            existing_value = merged.get(field)
            if existing_value is None:
                merged[field] = value
                logger.debug("Set field='{}' from incoming payload", field)
                continue

            existing_values = (
                existing_value if isinstance(existing_value, list) else [existing_value]
            )
            incoming_values = value if isinstance(value, list) else [value]

            combined_values = list(existing_values)
            for item in incoming_values:
                if item not in combined_values:
                    combined_values.append(item)

            merged[field] = combined_values
            logger.debug(
                "Merged field='{}' history values: count={}",
                field,
                len(combined_values),
            )
            continue

        if field not in merged or merged[field] is None:
            merged[field] = value
            logger.debug("Set field='{}' from incoming payload", field)
        else:
            logger.debug("Kept existing field='{}' and ignored incoming value", field)

    return merged


def load_yaml_dict(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load one YAML file and ensure top-level is dict."""
    logger.debug(f"Loading YAML file: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        content = yaml.safe_load(file_obj)
    if content is None:
        logger.warning(f"YAML file is empty: {path}")
        return {}
    if not isinstance(content, dict):
        logger.warning(f"Skipping YAML with non-dict top-level: {path}")
        return {}
    logger.debug(f"Loaded {len(content)} entries from {path}")
    return content


def merge_yaml_by_event_number(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Merge entries by event_number with adjacent-file comparison only.

    For file N, duplicates are merged only against file N-1. This avoids
    full-history comparisons while preserving per-event payload merge rules.
    """
    logger.info(f"Merging by event_number from {len(files)} files")
    logger.info("Using adjacent-window merge mode (current file vs previous file)")

    grouped: Dict[str, Dict[str, Any]] = {}
    previous_file_events: Dict[str, Dict[str, Any]] = {}
    merged_event_keys = set()

    merged_count = 0
    skipped_count = 0

    total_files = len(files)
    for index, path in enumerate(files, start=1):
        content = load_yaml_dict(path)
        logger.info(
            "Processing file [{}/{}] {} with {} top-level entries",
            index,
            total_files,
            path,
            len(content),
        )

        current_file_events: Dict[str, Dict[str, Any]] = {}
        file_skipped = 0
        for _, payload in content.items():
            if not isinstance(payload, dict):
                skipped_count += 1
                file_skipped += 1
                continue

            event_number = payload.get("event_number")
            if event_number is None:
                logger.warning(f"File {path} has entry without event_number; skipped")
                skipped_count += 1
                file_skipped += 1
                continue

            grouped_key = str(event_number)
            if grouped_key not in current_file_events:
                current_file_events[grouped_key] = payload
            else:
                current_file_events[grouped_key] = merge_event_payload(
                    current_file_events[grouped_key],
                    payload,
                )

        file_inserted = 0
        file_merged = 0
        current_processed: Dict[str, Dict[str, Any]] = {}
        for grouped_key, payload in current_file_events.items():
            if grouped_key in previous_file_events:
                original_payload = previous_file_events[grouped_key]
                logger.debug(
                    "Merging adjacent duplicate event_number={} from file={}",
                    grouped_key,
                    path,
                )
                logger.opt(lazy=True).debug(
                    "Previous-file record before merge for event_number={}:\n{}",
                    lambda: grouped_key,
                    lambda: original_payload,
                )
                logger.opt(lazy=True).debug(
                    "Incoming record for event_number={}:\n{}",
                    lambda: grouped_key,
                    lambda: payload,
                )

                merged_payload = merge_event_payload(original_payload, payload)
                grouped[grouped_key] = merged_payload
                current_processed[grouped_key] = merged_payload
                merged_count += 1
                file_merged += 1
                merged_event_keys.add(grouped_key)
                logger.opt(lazy=True).debug(
                    "Merged result for event_number={}:\n{}",
                    lambda: grouped_key,
                    lambda: merged_payload,
                )
                continue

            grouped[grouped_key] = payload
            current_processed[grouped_key] = payload
            file_inserted += 1

        previous_file_events = current_processed

        logger.info(
            "File summary [{}/{}] {}: inserted={}, merged={}, skipped={}",
            index,
            total_files,
            path,
            file_inserted,
            file_merged,
            file_skipped,
        )

    logger.info(
        "event_number merge completed: unique_events={}, merged_collisions={}, skipped_entries={}",
        len(grouped),
        merged_count,
        skipped_count,
    )
    if merged_event_keys:
        merged_event_list = sorted(
            merged_event_keys,
            key=event_key_sort_value,
        )
        logger.info(
            "Merged event_number list ({}): {}",
            len(merged_event_list),
            ",".join(merged_event_list),
        )
    else:
        logger.info("No duplicate event_number entries were merged")

    return dict(sorted(grouped.items(), key=lambda item: event_key_sort_value(item[0])))


def write_merged_yaml(outpath: Path, files: List[Path], merged_data: Dict[str, Dict[str, Any]]) -> None:
    """Write merged YAML with traceability header."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Writing merged YAML: {outpath}")
    serialize_start = time.perf_counter()

    header_text = build_traceability_header(
        outpath.name,
        files,
        include_time=True,
    )
    yaml_text = yaml.dump(
        merged_data,
        Dumper=YAML_DUMPER,
        allow_unicode=True,
        sort_keys=False,
    )

    serialize_ms = (time.perf_counter() - serialize_start) * 1000
    write_start = time.perf_counter()
    with outpath.open("w", encoding="utf-8") as file_obj:
        file_obj.write(header_text)
        file_obj.write(yaml_text)
    write_ms = (time.perf_counter() - write_start) * 1000
    logger.info(
        "Merged YAML written: {} (records={}, serialize_ms={:.1f}, write_ms={:.1f})",
        outpath,
        len(merged_data),
        serialize_ms,
        write_ms,
    )


def merge_files_for_pattern(
    dirpath: Path,
    pattern: str,
    outpath: Path,
    run_number: Optional[int] = None,
) -> Tuple[int, str]:
    """Merge files matching pattern and write to outpath."""
    logger.info(f"Searching files in {dirpath} with pattern '{pattern}'")
    code, message, files = find_files(dirpath, pattern)
    if code != 0:
        return code, message

    files = _filter_files_by_run_number(files, run_number)
    if run_number is not None and not files:
        return (
            1,
            f"No files matching RUN{run_number} in {dirpath} after boundary filtering",
        )

    logger.info(f"Found {len(files)} input files for merge")
    logger.debug("Input files:\n{}", "\n".join(str(file_path) for file_path in files))

    merged_data = merge_yaml_by_event_number(files)
    write_merged_yaml(outpath, files, merged_data)
    return (
        0,
        f"Wrote {len(files)} files -> {outpath} "
        f"(records={len(merged_data)}, mode=event-number, run_number={run_number})",
    )


def merge_trigger_files(
    dirpath: Path,
    ymd: str,
    outdir: Path,
    run_number: Optional[int] = None,
) -> int:
    """Merge Trigger*.yaml in one date directory into one output file."""
    if run_number is None:
        logger.warning(
            "Deprecated usage: --run-number is not provided; "
            "falling back to full-day Trigger*.yaml merge"
        )
        outpath = outdir / f"Trigger_{ymd}_merged.yaml"
    else:
        outpath = outdir / f"Trigger_{ymd}_RUN{run_number}_merged.yaml"

    pattern = build_trigger_pattern(run_number)
    logger.info(
        "Start merge workflow: input_dir={}, output_dir={}, output_file={}, run_number={}",
        dirpath,
        outdir,
        outpath,
        run_number,
    )
    code, msg = merge_files_for_pattern(dirpath, pattern, outpath, run_number=run_number)
    if code != 0:
        logger.error(msg)
        return code
    logger.info(msg)
    return 0


def parse_date_dir(date_dir: str) -> Tuple[Path, str]:
    """Parse yyyy/mm/dd directory string and return path + yyyymmdd."""
    return _parse_date_dir(date_dir)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge Trigger*.yaml files in a date directory by event_number"
    )
    parser.add_argument("dir", help="Date directory path like yyyy/mm/dd")
    parser.add_argument(
        "-o",
        "--output",
        default="../Reco_Dir",
        help="Output directory for merged files (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help=(
            "Optional RUN number for file filtering and output naming. "
            "When omitted, script keeps legacy full-day merge behavior."
        ),
    )
    args = parser.parse_args(argv)
    outdir = Path(args.output)

    logger.info(
        "CLI arguments: dir={}, output={}, run_number={}",
        args.dir,
        outdir,
        args.run_number,
    )

    input_dir, ymd = resolve_input_dir(args.dir, outdir)
    logger.info(f"Resolved date={ymd}, input_dir={input_dir}")

    return merge_trigger_files(input_dir, ymd, outdir, run_number=args.run_number)


if __name__ == "__main__":
    raise SystemExit(main())
