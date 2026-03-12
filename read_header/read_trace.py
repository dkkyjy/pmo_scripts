"""
ROOT file data reading and processing module.

This module reads a single ROOT file (teventadc) containing "Trigger",
extracts du_id, gps_time, du_nanoseconds, and trace data,
calculates the maximum ADC values and XY combined amplitude maximum values
within the interval [left:right] for each channel, and outputs matching files.

Main functions:
1. Parse command line arguments
2. Read and process ROOT file data
3. Generate matching files

Usage: python read_trace.py <file_path> <left> <right> <base_path> <out_dir_base>
left/right are optional, default 0,512
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import uproot

from logger_config import logger
from read_header.common import (
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
TraceMaxValueList = List[List[int]]
ReadTraceResult = Tuple[
    RunNumberList,
    EventNumberList,
    DuIdList,
    GpsTimeList,
    DuNanosecondList,
    TraceMaxValueList,
    TraceMaxValueList,
    TraceMaxValueList,
    TraceMaxValueList,
    TraceMaxValueList,
]
EventPayload = Dict[str, Any]
YamlData = Dict[str, EventPayload]


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command line arguments and perform basic validation."""
    parser = argparse.ArgumentParser(
        description="Read ROOT trace and generate matching files (XY/FilterY/X)"
    )
    parser.add_argument("file_path", help="Path to the ROOT file to process")
    parser.add_argument(
        "--left",
        nargs="?",
        default=0,
        type=int,
        help="Left boundary of clipping interval (default 0)",
    )
    parser.add_argument(
        "--right",
        nargs="?",
        default=None,
        type=int,
        help="Right boundary of clipping interval (default None)",
    )
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


def calculate_trace_max_values(
    trace_list: List[List[Any]], left: int, right: int
) -> List[List[int]]:
    """Calculate max absolute trace values for each channel in [left:right]."""
    max_values_list: List[List[int]] = []
    for trace in trace_list:
        max_values: List[int] = []
        for channel in trace:
            if len(channel) == 0:
                max_values.append(10)
            else:
                channel_array = np.array(list(channel))
                max_values.append(int(np.max(np.abs(channel_array[left:right]))))
        max_values_list.append(max_values)
    return max_values_list


def calculate_xy_combined_max_values(
    trace_x_list: List[List[Any]],
    trace_y_list: List[List[Any]],
    left: int,
    right: int,
) -> List[List[int]]:
    """Calculate max combined XY amplitude values for each channel."""
    combined_max_list: List[List[int]] = []
    for trace_x, trace_y in zip(trace_x_list, trace_y_list):
        channel_max = [
            int(
                np.max(
                    np.hypot(
                        np.array(list(x_channel)[left:right]),
                        np.array(list(y_channel)[left:right]),
                    )
                )
            )
            if len(x_channel) > 0 and len(y_channel) > 0
            else 10
            for x_channel, y_channel in zip(trace_x, trace_y)
        ]
        combined_max_list.append(channel_max)
    return combined_max_list

def read_file_du_time_ns(file_name: str, left: int, right: int) -> ReadTraceResult:
    """Read fields from teventadc and compute per-channel max value lists."""
    root_file = uproot.open(file_name)
    key_name = find_teventadc_key(list(root_file.keys()), file_name)
    events = root_file[key_name]
    data = events.arrays(
        [
            "run_number",
            "event_number",
            "du_id",
            "gps_time",
            "du_nanoseconds",
            "trace_0",
            "trace_1",
            "trace_2",
            "trace_3",
        ],
        library="np",
    )

    run_number_list = data["run_number"].tolist()
    event_number_list = data["event_number"].tolist()
    du_id_list = data["du_id"].tolist()
    gps_time_list = data["gps_time"].tolist()
    du_nanosecond_list = data["du_nanoseconds"].tolist()
    trace_f_list = data["trace_0"].tolist()
    trace_x_list = data["trace_1"].tolist()
    trace_y_list = data["trace_2"].tolist()
    trace_z_list = data["trace_3"].tolist()

    ftrace_adc_maxvalue_list = calculate_trace_max_values(trace_f_list, left, right)
    xtrace_adc_maxvalue_list = calculate_trace_max_values(trace_x_list, left, right)
    ytrace_adc_maxvalue_list = calculate_trace_max_values(trace_y_list, left, right)
    ztrace_adc_maxvalue_list = calculate_trace_max_values(trace_z_list, left, right)
    squared_sum_max_list = calculate_xy_combined_max_values(
        trace_x_list,
        trace_y_list,
        left,
        right,
    )

    return (
        run_number_list,
        event_number_list,
        du_id_list,
        gps_time_list,
        du_nanosecond_list,
        ftrace_adc_maxvalue_list,
        xtrace_adc_maxvalue_list,
        ytrace_adc_maxvalue_list,
        ztrace_adc_maxvalue_list,
        squared_sum_max_list,
    )


def build_event_payload(
    run_number: int,
    event_number: int,
    du_ids: List[int],
    gps_times: List[int],
    du_nanoseconds: List[int],
    trace_adc_maxvalue: List[int],
    index: int,
    file_path: str,
) -> EventPayload:
    """Build one event payload with multi-trigger DU list values."""
    time_map, list_du_id = build_time_map_and_du_ids(du_ids, du_nanoseconds)
    signal_map: Dict[str, List[int]] = {}

    for item_index, du_id_str in enumerate(list_du_id):
        max_value = trace_adc_maxvalue[item_index] if item_index < len(trace_adc_maxvalue) else 10
        signal_map.setdefault(du_id_str, []).append(int(max_value))

    gps_time = gps_times[2]
    datetime_str = format_event_datetime(gps_times)

    return {
        "run_number": run_number,
        "event_number": event_number,
        "datetime": datetime_str,
        "gps_time": int(gps_time),
        "time": time_map,
        "signal": signal_map,
        "du_id": list_du_id,
        "file": file_path,
        "index": int(index),
    }


def cal_dict_du_ns(
    run_number_list: RunNumberList,
    event_number_list: EventNumberList,
    du_id_list: DuIdList,
    gps_time_list: GpsTimeList,
    du_nanosecond_list: DuNanosecondList,
    data_dict: YamlData,
    trace_adc_maxvalue_list: TraceMaxValueList,
    file_path: str = "",
) -> None:
    """Pack DU nanoseconds and amplitudes for each event into ``data_dict``."""
    for index, du_ids in enumerate(du_id_list):
        du_nanoseconds = du_nanosecond_list[index]
        if len(du_nanoseconds) > 0 and du_nanoseconds[0] > 0:
            payload = build_event_payload(
                run_number=int(run_number_list[index]),
                event_number=int(event_number_list[index]),
                du_ids=du_ids,
                gps_times=gps_time_list[index],
                du_nanoseconds=du_nanoseconds,
                trace_adc_maxvalue=trace_adc_maxvalue_list[index],
                index=index,
                file_path=file_path,
            )
            data_dict[str(payload["event_number"])] = payload


def process_single_file(
    file_path: str,
    left: int,
    right: int,
    dict_list_f: YamlData,
    dict_list_x: YamlData,
    dict_list_y: YamlData,
    dict_list_z: YamlData,
    dict_list_xy: YamlData,
) -> None:
    """Process one ROOT file and fill F/X/Y/Z/XY result dictionaries."""
    logger.info(f"Processing file: {file_path}")
    try:
        (
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            ftrace_adc_maxvalue_list,
            xtrace_adc_maxvalue_list,
            ytrace_adc_maxvalue_list,
            ztrace_adc_maxvalue_list,
            xytrace_adc_maxvalue_list,
        ) = read_file_du_time_ns(file_path, left, right)

        file_name = os.path.basename(file_path)
        channel_targets = [
            (dict_list_f, ftrace_adc_maxvalue_list),
            (dict_list_x, xtrace_adc_maxvalue_list),
            (dict_list_y, ytrace_adc_maxvalue_list),
            (dict_list_z, ztrace_adc_maxvalue_list),
            (dict_list_xy, xytrace_adc_maxvalue_list),
        ]
        for target_dict, maxvalue_list in channel_targets:
            cal_dict_du_ns(
                run_number_list,
                event_number_list,
                du_id_list,
                gps_time_list,
                du_nanosecond_list,
                target_dict,
                maxvalue_list,
                file_name,
            )
    except Exception as exc:
        logger.warning(f"Error processing file {file_path}, skipping: {exc}")


def mkdir(path: str) -> None:
    """Create directory if it does not exist."""
    already_exists = os.path.exists(path)
    shared_mkdir(path)
    if already_exists:
        logger.debug(f"Directory already exists, no need to create: {path}")


def write_single_output(data_dict: YamlData, file_path: str) -> None:
    """Write one dictionary payload to YAML file."""
    write_yaml(file_path, data_dict)


def write_outputs(
    data_dict_f: YamlData,
    data_dict_x: YamlData,
    data_dict_y: YamlData,
    data_dict_z: YamlData,
    data_dict_xy: YamlData,
    out_dir: str,
    file_name: str,
) -> None:
    """Write F/X/Y/Z/XY dictionaries to corresponding output files."""
    output_configs = [
        (data_dict_f, f"{file_name}_F.yaml"),
        (data_dict_x, f"{file_name}_X.yaml"),
        (data_dict_y, f"{file_name}_Y.yaml"),
        (data_dict_z, f"{file_name}_Z.yaml"),
        (data_dict_xy, f"{file_name}_XY.yaml"),
    ]

    for data_dict, filename in output_configs:
        file_path = os.path.join(out_dir, filename)
        write_single_output(data_dict, file_path)

    logger.info(f"Results written to directory: {out_dir}")


def process_root_file(
    file_path: str,
    left: int,
    right: int,
    date: str,
    out_dir_base: str,
) -> bool:
    """Main function to process one ROOT file into five channel YAML outputs."""
    try:
        out_dir = os.path.join(out_dir_base, date)
        mkdir(out_dir)

        dict_list_f: YamlData = {}
        dict_list_x: YamlData = {}
        dict_list_y: YamlData = {}
        dict_list_z: YamlData = {}
        dict_list_xy: YamlData = {}

        process_single_file(
            file_path,
            left,
            right,
            dict_list_f,
            dict_list_x,
            dict_list_y,
            dict_list_z,
            dict_list_xy,
        )
        logger.info(
            f"Processed file events: {len(dict_list_f)}, {len(dict_list_x)}, {len(dict_list_y)}, {len(dict_list_z)}, {len(dict_list_xy)}"
        )

        output_filename = os.path.splitext(os.path.basename(file_path))[0]
        write_outputs(
            dict_list_f,
            dict_list_x,
            dict_list_y,
            dict_list_z,
            dict_list_xy,
            out_dir,
            output_filename,
        )
        return True
    except Exception as exc:
        logger.error(f"Processing failed: {exc}")
        return False


def main(argv: List[str]) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    success = process_root_file(
        args.file_path,
        args.left,
        args.right,
        args.date,
        args.out_dir_base,
    )
    return 0 if success else 2


if __name__ == "__main__":
    main(sys.argv)
