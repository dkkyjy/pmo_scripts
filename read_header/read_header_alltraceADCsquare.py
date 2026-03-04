"""
ROOT file data reading and processing module

This module reads a single ROOT file (teventadc) containing "Trigger",
extracts du_id, gps_time, du_nanoseconds, and trace data,
calculates the maximum ADC values and XY combined amplitude maximum values within the specified interval [left:right] for each channel,
constructs keys based on gps_time and minimum nanoseconds, and outputs matching files.

Main functions:
1. Parse command line arguments
2. Read and process ROOT file data
3. Generate matching files

Usage: python read_header_alltraceADCsquare.py <file_path> <left> <right> <base_path> <out_dir_base>
left/right are optional, default 0,512
"""

import uproot
import numpy as np
import sys
import os
import yaml
import argparse
from logger_config import logger
from typing import List, Dict, Tuple, Any


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
    """
    Calculate the maximum absolute value of trace data within the specified interval [left:right].

    Parameters:
        trace_list: Three-dimensional list representing trace data
        left: Left boundary of clipping interval
        right: Right boundary of clipping interval

    Returns:
        Two-dimensional list, each element is the maximum value of the corresponding trace
    """
    max_values_list = []
    for trace in trace_list:
        max_values = []
        for channel in trace:
            if len(channel) == 0:
                max_values.append(10)
            else:
                # Convert to numpy array properly
                channel_array = np.array(list(channel))
                max_values.append(int(np.max(np.abs(channel_array[left:right]))))
        max_values_list.append(max_values)
    return max_values_list


def calculate_xy_combined_max_values(
    trace_x_list: List[List[Any]], trace_y_list: List[List[Any]], left: int, right: int
) -> List[List[int]]:
    """
    Calculate the maximum amplitude value after combining X and Y channels.

    Parameters:
        trace_x_list: X channel trace data
        trace_y_list: Y channel trace data
        left: Left boundary of clipping interval
        right: Right boundary of clipping interval

    Returns:
        Two-dimensional list, each element is the maximum value of the combined channels
    """
    combined_max_list = []
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


def read_file_du_time_ns(file_name: str, left: int, right: int):
    """
    Read fields from the teventadc TTree of a single ROOT file:
    Returns du_id_list, gps_time_list, du_nanosecond_list,
           ftrace_adc_maxvalue_list, xtrace_adc_maxvalue_list, ytrace_adc_maxvalue_list, ztrace_adc_maxvalue_list, squared_sum_max_list
    Note: Slice the trace data and calculate the maximum absolute value or combined amplitude maximum value within the interval, returning 10 for empty channels.
    """
    f = uproot.open(file_name)
    keys = list(f.keys())
    idx = None
    for i, k in enumerate(keys):
        if "teventadc" in str(k):
            idx = i
            break
    if idx is None:
        raise KeyError(f"teventadc TTree not found in file: {file_name}")
    key_name = keys[idx]
    events = f[key_name]
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

    # Use reconstructed functions to calculate maximum values for each channel
    ftrace_adc_maxvalue_list = calculate_trace_max_values(trace_f_list, left, right)
    xtrace_adc_maxvalue_list = calculate_trace_max_values(trace_x_list, left, right)
    ytrace_adc_maxvalue_list = calculate_trace_max_values(trace_y_list, left, right)
    ztrace_adc_maxvalue_list = calculate_trace_max_values(trace_z_list, left, right)

    # Calculate XY combined amplitude maximum value
    squared_sum_max_list = calculate_xy_combined_max_values(
        trace_x_list, trace_y_list, left, right
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


def cal_dict_du_ns(
    run_number_list: List[List[int]],
    event_number_list: List[List[int]],
    du_id_list: List[List[int]],
    gps_time_list: List[List[int]],
    du_nanosecond_list: List[List[int]],
    data_dict: Dict[str, Dict],
    trace_adc_maxvalue_list: List[List[int]],
    file_path: str = "",
) -> None:
    """
    Pack the du_id and corresponding maximum value for each event, with key constructed using gps_time (index 2 or minimum value) and minimum nanoseconds.
    data_dict is directly modified to add entries, with value as [dict_du_ns].
    """
    for i in range(len(du_id_list)):
        run_number = run_number_list[i]
        event_number = event_number_list[i]
        du_ids = du_id_list[i]
        gps_times = gps_time_list[i]
        du_nanoseconds = du_nanosecond_list[i]
        trace_adc_maxvalue = trace_adc_maxvalue_list[i]
        if len(du_nanoseconds) > 0 and du_nanoseconds[0] > 0:
            list_du_id = []
            dict_du_ns = {}
            dict_du_amp = {}
            for j in range(len(du_ids)):
                # Get the corresponding maximum value, using default value 10 if index is out of range
                max_value = trace_adc_maxvalue[j] if j < len(trace_adc_maxvalue) else 10
                dict_du_ns[str(du_ids[j])] = int(du_nanoseconds[j])
                dict_du_amp[str(du_ids[j])] = int(max_value)
                list_du_id.append(str(du_ids[j]))

            # extract gps_times
            date = gps_times[0]
            time = gps_times[1]
            gps_time = gps_times[2]
            # Add information to key
            data_dict[str(event_number)] = {
                "run_number": run_number,
                "event_number": event_number,
                "date": str(date),
                "time": f'{time:0<6}',
                "gps_time": int(gps_time),
                "du_ns": dict_du_ns,
                "du_vs": dict_du_amp,
                "du_id": list_du_id,
                "file": file_path,
                "index": int(i),
            }


def process_single_file(
    file_path: str,
    left: int,
    right: int,
    dict_list_f: Dict,
    dict_list_x: Dict,
    dict_list_y: Dict,
    dict_list_z: Dict,
    dict_list_xy: Dict,
) -> None:
    """
    Process a single file and add results to the corresponding dictionaries.

    Parameters:
        file_path: File path
        left: Left boundary of clipping interval
        right: Right boundary of clipping interval
        dict_list_f: F channel result dictionary
        dict_list_x: X channel result dictionary
        dict_list_y: Y channel result dictionary
        dict_list_z: Z channel result dictionary
        dict_list_xy: XY combined result dictionary
    """
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
        # Extract filename for key generation
        file_name = os.path.basename(file_path)
        cal_dict_du_ns(
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            dict_list_f,
            ftrace_adc_maxvalue_list,
            file_name,
        )
        cal_dict_du_ns(
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            dict_list_x,
            xtrace_adc_maxvalue_list,
            file_name,
        )
        cal_dict_du_ns(
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            dict_list_y,
            ytrace_adc_maxvalue_list,
            file_name,
        )
        cal_dict_du_ns(
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            dict_list_z,
            ztrace_adc_maxvalue_list,
            file_name,
        )
        cal_dict_du_ns(
            run_number_list,
            event_number_list,
            du_id_list,
            gps_time_list,
            du_nanosecond_list,
            dict_list_xy,
            xytrace_adc_maxvalue_list,
            file_name,
        )
    except Exception as e:
        logger.warning(f"Error processing file {file_path}, skipping: {e}")


def mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logger.debug(f"Directory already exists, no need to create: {path}")


def write_single_output(data_dict: Dict, file_path: str) -> None:
    """
    Write a single dictionary to a file.

    Parameters:
        data_dict: Data dictionary to write
        file_path: Output file path
    """
    with open(file_path, "w") as fw:
        yaml.dump(data_dict, fw)


def write_outputs(
    data_dict_f: Dict,
    data_dict_x: Dict,
    data_dict_y: Dict,
    data_dict_z: Dict,
    data_dict_xy: Dict,
    out_dir: str,
    file_name: str,
) -> None:
    """Write multiple dictionaries to corresponding files."""

    # Define output file configurations
    output_configs = [
        (data_dict_f, f"{file_name}_F.yaml"),
        (data_dict_x, f"{file_name}_X.yaml"),
        (data_dict_y, f"{file_name}_Y.yaml"),
        (data_dict_z, f"{file_name}_Z.yaml"),
        (data_dict_xy, f"{file_name}_XY.yaml"),
    ]

    # Write each file
    for data_dict, filename in output_configs:
        file_path = os.path.join(out_dir, filename)
        write_single_output(data_dict, file_path)

    logger.info(f"Results written to directory: {out_dir}")


def process_root_file(
    file_path: str, left: int, right: int, date: str, out_dir_base: str
) -> bool:
    """
    Main function to process a ROOT file.

    Parameters:
        file_path: Path to the ROOT file to process
        left: Left boundary of clipping interval
        right: Right boundary of clipping interval
        base_path: ROOT file base directory
        out_dir_base: Output directory base path

    Returns:
        Whether processing was successful
    """
    try:
        out_dir = os.path.join(out_dir_base, date)
        mkdir(out_dir)

        # Process data
        dict_list_f = {}
        dict_list_x = {}
        dict_list_y = {}
        dict_list_z = {}
        dict_list_xy = {}

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

        # Write output files
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
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Process ROOT file
    success = process_root_file(
        args.file_path, args.left, args.right, args.date, args.out_dir_base
    )

    return 0 if success else 2


if __name__ == "__main__":
    main(sys.argv)
