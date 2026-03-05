#!/usr/bin/env python3
"""Merge Trigger YAML files by event_number.

Behavior:
- Match only `Trigger*.yaml` under the input directory.
- Always merge by `event_number` (append mode is not supported).
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from logger_config import logger


PATTERN = "Trigger*.yaml"


def event_key_sort_value(event_key: str) -> Any:
    """Sort event keys numerically when possible, else lexicographically."""
    return int(event_key) if event_key.isdigit() else event_key

def payload_to_log_text(payload: Dict[str, Any]) -> str:
    """Convert one event payload into readable YAML text for logs."""
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip()


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
        if field in {"du_ns", "du_vs"} and isinstance(value, dict):
            existing = merged.get(field)
            if not isinstance(existing, dict):
                existing = {}
            before_count = len(existing)
            existing.update(value)
            merged[field] = existing
            logger.debug(
                "Merged dict field='{}': before_keys={}, incoming_keys={}, after_keys={}",
                field,
                before_count,
                len(value),
                len(existing),
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
        logger.info("Processing file {} with {} top-level entries", path, len(content))

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
                logger.debug(
                    "Merging duplicate event_number={} from file={}",
                    event_key,
                    path,
                )
                logger.debug(
                    "Original record before merge for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(original_payload),
                )
                logger.debug(
                    "Incoming record for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(payload),
                )

                merged_payload = merge_event_payload(original_payload, payload)
                grouped[event_key] = merged_payload
                merged_count += 1
                file_merged += 1
                merged_event_keys.add(event_key)
                logger.debug(
                    "Merged result for event_number={}:\n{}",
                    event_key,
                    payload_to_log_text(merged_payload),
                )

        logger.info(
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

    header_lines = [
        f"# Merged: {outpath.name}",
        f"# Time: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "# Source files:",
    ]
    header_lines.extend(f"#   - {path}" for path in files)
    header_text = "\n".join(header_lines) + "\n\n"

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
    if not dirpath.exists() or not dirpath.is_dir():
        return 2, f"Directory not found: {dirpath}"

    files = sorted(dirpath.glob(pattern))
    if not files:
        return 1, f"No files matching '{pattern}' in {dirpath}"

    logger.info(f"Found {len(files)} input files for merge")
    logger.debug("Input files:\n{}", "\n".join(str(file_path) for file_path in files))

    merged_data = merge_yaml_by_event_number(files)
    write_merged_yaml(outpath, files, merged_data)
    return 0, f"Wrote {len(files)} files -> {outpath} (records={len(merged_data)}, mode=event-number)"


def merge_trigger_files(dirpath: Path, ymd: str, outdir: Path) -> int:
    """Merge Trigger*.yaml in one date directory into one output file."""
    outpath = outdir / f"Trigger_{ymd}_merged.yaml"
    logger.info(
        "Start merge workflow: input_dir={}, output_dir={}, output_file={}",
        dirpath,
        outdir,
        outpath,
    )
    code, msg = merge_files_for_pattern(dirpath, PATTERN, outpath)
    if code != 0:
        logger.error(msg)
        return code
    logger.info(msg)
    return 0


def parse_date_dir(date_dir: str) -> Tuple[Path, str]:
    """Parse yyyy/mm/dd directory string and return path + yyyymmdd."""
    logger.debug(f"Parsing date directory: {date_dir}")
    date_obj = datetime.datetime.strptime(date_dir, "%Y/%m/%d")
    return Path(date_dir), date_obj.strftime("%Y%m%d")


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

    date_path, ymd = parse_date_dir(args.dir)
    input_dir = outdir / date_path
    logger.info(f"Resolved date={ymd}, input_dir={input_dir}")

    return merge_trigger_files(input_dir, ymd, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
