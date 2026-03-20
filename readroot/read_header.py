"""
ROOT file data reading and processing module.

This module reads a single ROOT file (teventadc) containing "Trigger",
extracts du_id, gps_time, and du_nanoseconds.

Main functions:
1. Parse command line arguments
2. Read and process ROOT file data
3. Generate matching file

Usage: python read_header.py <file_path> <base_path> <out_dir_base>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import uproot

from logger_config import logger
from readroot.common import (
    DATETIME_FORMAT,
    EventPayload,
    YamlData,
    build_time_map_and_du_ids,
    find_teventadc_key,
    format_event_datetime,
    mkdir as shared_mkdir,
    write_yaml,
)

RunNumberList = List[int]
EventNumberList = List[int]
DuIdList = List[List[int]]
GpsTimeList = List[List[int]]
DuNanosecondList = List[List[int]]
ReadHeaderResult = Tuple[
    RunNumberList,
    EventNumberList,
    DuIdList,
    GpsTimeList,
    DuNanosecondList,
]
EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]

def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command line arguments and perform basic validation."""
    parser = argparse.ArgumentParser(
        description="Read ROOT file and generate du_ns matching file"
    )
    parser.add_argument("file_path", help="Path to the ROOT file to process")
    parser.add_argument(
        "--date",
        default="",
        help="ROOT file base directory (default out_dir_base path)",
    )
    parser.add_argument(
        "--out_dir_base",
        "--out-dir-base",
        "-o",
        dest="out_dir_base",
        default="../Reco_Dir",
        help="Output directory base path (default: ../Reco_Dir)",
    )
    return parser.parse_args(argv[1:])

def read_file_du_time_ns(file_name: str) -> ReadHeaderResult:
    """Read event fields from teventadc TTree and return lists."""
    root_file = uproot.open(file_name)
    key_name = find_teventadc_key(list(root_file.keys()), file_name)
    events = root_file[key_name]
    data = events.arrays(
        ["run_number", "event_number", "du_id", "gps_time", "du_nanoseconds"],
        library="np",
    )

    run_number_list = data["run_number"].tolist()
    event_number_list = data["event_number"].tolist()
    du_id_list = data["du_id"].tolist()
    gps_time_list = data["gps_time"].tolist()
    du_nanosecond_list = data["du_nanoseconds"].tolist()
    return (
        run_number_list,
        event_number_list,
        du_id_list,
        gps_time_list,
        du_nanosecond_list,
    )


def build_event_payload(
    run_number: int,
    event_number: int,
    du_ids: List[int],
    gps_times: List[int],
    du_nanoseconds: List[int],
    index: int,
    file_path: str,
) -> EventPayload:
    """Build one event payload in the YAML output schema."""
    time, list_du_id = build_time_map_and_du_ids(du_ids, du_nanoseconds)
    gps_time = gps_times[2]
    datetime_str = format_event_datetime(gps_times)

    return {
        "run_number": run_number,
        "event_number": event_number,
        "datetime": datetime_str,
        "gps_time": int(gps_time),
        "time": time,
        "du_id": list_du_id,
        "file": file_path,
        "index": index,
    }


def cal_dict_du_ns(
    run_number_list: RunNumberList,
    event_number_list: EventNumberList,
    du_id_list: DuIdList,
    gps_time_list: GpsTimeList,
    du_nanosecond_list: DuNanosecondList,
    data_dict: YamlData,
    file_path: str = "",
) -> None:
    """Build output payloads for each event and write to ``data_dict``."""
    for index, du_ids in enumerate(du_id_list):
        if len(du_ids) == 0:
            continue

        payload = build_event_payload(
            run_number=int(run_number_list[index]),
            event_number=int(event_number_list[index]),
            du_ids=du_ids,
            gps_times=gps_time_list[index],
            du_nanoseconds=du_nanosecond_list[index],
            index=index,
            file_path=file_path,
        )
        data_dict[str(payload["event_number"])] = payload


def process_single_file(file_path: str, result_dict: YamlData) -> None:
    """Process one ROOT file and append parsed events into ``result_dict``."""
    try:
        run_number, event_number, du_ids, gps_times, du_ns = read_file_du_time_ns(
            file_path
        )
        file_name = os.path.basename(file_path)
        cal_dict_du_ns(
            run_number,
            event_number,
            du_ids,
            gps_times,
            du_ns,
            result_dict,
            file_name,
        )
    except Exception as exc:
        logger.warning(f"Error processing file {file_path}: {exc}, skipping")


def mkdir(path: str) -> None:
    """Create directory if it does not exist."""
    already_exists = os.path.exists(path)
    shared_mkdir(path)
    if already_exists:
        logger.debug(f"Directory already exists: {path}")


def write_output(data_dict: YamlData, out_dir: str, file_name: str) -> str:
    """Write YAML results to ``out_dir/file_name`` and return file path."""
    file_path = os.path.join(out_dir, file_name)
    write_yaml(file_path, data_dict)
    return file_path


def process_root_file(file_path: str, date: str, out_dir_base: str) -> bool:
    """Process one ROOT file and write one YAML output file."""
    try:
        out_dir = os.path.join(out_dir_base, date)
        mkdir(out_dir)

        logger.info(f"Processing file: {file_path}")

        result: YamlData = {}
        process_single_file(file_path, result)
        logger.info(f"Processed events: {len(result)}")

        output_filename = f"{os.path.basename(file_path).replace('.root', '.yaml')}"
        out_path = write_output(result, out_dir, output_filename)
        logger.info(f"Written to: {out_path}")
        return True
    except Exception as exc:
        logger.error(f"Processing failed: {exc}")
        return False


def main(argv: List[str]) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    success = process_root_file(args.file_path, args.date, args.out_dir_base)
    return 0 if success else 2


if __name__ == "__main__":
    main(sys.argv)
