#!/usr/bin/env python3
"""Merge Trigger YAML files by event_number.

Behavior:
- Match only `Trigger*.yaml` under the input directory.
- Always merge by `event_number` (append mode is not supported).
"""

from __future__ import annotations

import argparse
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

TYPE_PATTERNS = {
    "F": "Trigger*_F.yaml",
    "X": "Trigger*_X.yaml",
    "Y": "Trigger*_Y.yaml",
    "Z": "Trigger*_Z.yaml",
    "XY": "Trigger*_XY.yaml",
}


def payload_to_log_text(payload: Dict[str, Any]) -> str:
    """Convert one event payload into readable YAML text for logs."""
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()


def _to_scalar_or_list(value: Any) -> List[Any]:
    """Normalize one DU value into a list for safe concatenation."""
    if isinstance(value, list):
        return value
    return [value]


def merge_du_value_map(base_map: Dict[str, Any], incoming_map: Dict[str, Any]) -> Dict[str, Any]:
    """Merge DU-value maps and preserve repeated trigger samples per DU.

    Supports both legacy scalar values and list values. If the same DU exists
    in both maps, values are concatenated in order.
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
            merged[field] = merge_du_value_map(existing, value)
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
            if not isinstance(existing_ids, list):
                existing_ids = []
            before_count = len(existing_ids)
            seen = set(str(item) for item in existing_ids)
            for item in value:
                item_str = str(item)
                if item_str not in seen:
                    existing_ids.append(item_str)
                    seen.add(item_str)
            merged["du_id"] = existing_ids
            logger.debug(
                "Merged du_id list: before_count={}, incoming_count={}, after_count={}",
                before_count,
                len(value),
                len(existing_ids),
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
    """Merge entries by event_number across all files."""
    logger.info(f"Merging by event_number from {len(files)} files")
    grouped: Dict[str, Dict[str, Any]] = {}
    merged_event_keys = set()

    merged_count = 0
    skipped_count = 0

    for path in files:
        content = load_yaml_dict(path)
        logger.debug("Processing file {} with {} top-level entries", path, len(content))

        file_inserted = 0
        file_merged = 0
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

            event_key = str(event_number)
            if event_key not in grouped:
                grouped[event_key] = payload
                file_inserted += 1
                logger.debug("Inserted new event key={} from file={}", event_key, path)
            else:
                original_payload = grouped[event_key]
                logger.info(
                    "Merging duplicate event_number={} from file={}",
                    event_key,
                    path,
                )
                logger.info(
                    "Original record before merge for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(original_payload),
                )
                logger.info(
                    "Incoming record for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(payload),
                )

                merged_payload = merge_event_payload(original_payload, payload)
                grouped[event_key] = merged_payload
                merged_count += 1
                file_merged += 1
                merged_event_keys.add(event_key)
                logger.info(
                    "Merged result for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(merged_payload),
                )

        logger.debug(
            "File summary {}: inserted={}, merged={}, skipped={}",
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
            key=lambda item: int(item) if item.isdigit() else item,
        )
        logger.info(
            "Merged event_number list ({}): {}",
            len(merged_event_list),
            ",".join(merged_event_list),
        )
    else:
        logger.info("No duplicate event_number entries were merged")

    return dict(sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def write_merged_yaml(outpath: Path, files: List[Path], merged_data: Dict[str, Dict[str, Any]]) -> None:
    """Write merged YAML with traceability header."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Writing merged YAML: {outpath}")

    header_text = build_traceability_header(
        outpath.name,
        files,
        include_time=True,
    )
    yaml_text = yaml.safe_dump(merged_data, allow_unicode=True, sort_keys=False)
    with outpath.open("w", encoding="utf-8") as file_obj:
        file_obj.write(header_text)
        file_obj.write(yaml_text)
    logger.info(f"Merged YAML written: {outpath} (records={len(merged_data)})")


def merge_files_for_pattern(
    dirpath: Path,
    pattern: str,
    outpath: Path,
) -> Tuple[int, str]:
    """Merge files matching pattern and write to outpath."""
    logger.info(f"Searching files in {dirpath} with pattern '{pattern}'")
    code, message, files = find_files(dirpath, pattern)
    if code != 0:
        return code, message

    logger.info(f"Found {len(files)} input files for merge")
    logger.debug("Input files:\n{}", "\n".join(str(file_path) for file_path in files))

    merged_data = merge_yaml_by_event_number(files)
    write_merged_yaml(outpath, files, merged_data)
    return 0, f"Wrote {len(files)} files -> {outpath} (records={len(merged_data)}, mode=event-number)"


def merge_trigger_files(dirpath: Path, ymd: str, outdir: Path) -> int:
    """Merge Trigger*.yaml in one date directory into one output file."""
    for trace_type, pattern in TYPE_PATTERNS.items():
        outpath = outdir / f"Trigger_{ymd}_{trace_type}_merged.yaml"
        logger.info(
            "Start merge workflow: input_dir={}, output_dir={}, output_file={}",
            dirpath,
            outdir,
            outpath,
        )
        code, msg = merge_files_for_pattern(dirpath, pattern, outpath)
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
    args = parser.parse_args(argv)
    outdir = Path(args.output)

    logger.info(f"CLI arguments: dir={args.dir}, output={outdir}")

    input_dir, ymd = resolve_input_dir(args.dir, outdir)
    logger.info(f"Resolved date={ymd}, input_dir={input_dir}")

    return merge_trigger_files(input_dir, ymd, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
