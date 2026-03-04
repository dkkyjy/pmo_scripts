"""
ROOT file data reading and processing module

This module reads a single ROOT file (teventadc) containing "Trigger",
extracts du_id, gps_time, du_nanoseconds,

Main functions:
1. Parse command line arguments
2. Read and process ROOT file data
3. Generate matching file

Usage: python _read_header.py <file_path> <base_path> <out_dir_base>
"""

import uproot
import numpy as np
import sys
import os
import yaml
import argparse
from logger_config import logger
from typing import List, Dict, Tuple


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


def read_file_du_time_ns(
    file_name: str,
) -> Tuple[
    List[List[int]], List[List[int]], List[List[int]], List[List[int]], List[List[int]]
]:
    """
    Read du_id, gps_time, and du_nanoseconds fields from the teventadc TTree of a single ROOT file and return as lists.
    Returns three lists (one sublist per event).
    """
    f = uproot.open(file_name)
    # Find the key containing teventadc (compatible with different naming versions)
    keys = list(f.keys())
    idx = None
    for i, k in enumerate(keys):
        if "teventadc" in str(k):
            idx = i
            break
    if idx is None:
        raise KeyError(f"teventadc TTree not found in file: {file_name}")
    events = f[keys[idx]]
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


def cal_dict_du_ns(
    run_number_list: List[List[int]],
    event_number_list: List[List[int]],
    du_id_list: List[List[int]],
    gps_time_list: List[List[int]],
    du_nanosecond_list: List[List[int]],
    data_dict,
    file_path: str = "",
) -> None:
    """
    Combine du_id and du_nanoseconds from events, construct key using gps_time and minimum nanoseconds, and write results to data_dict.
    Directly modifies the passed data_dict.
    """
    for i in range(len(du_id_list)):
        run_number = run_number_list[i]
        event_number = event_number_list[i]
        du_ids = du_id_list[i]
        gps_times = gps_time_list[i]
        du_nanoseconds = du_nanosecond_list[i]
        if len(du_ids) > 0:
            dict_du_ns = {}
            list_du_id = []
            for j in range(len(du_ids)):
                list_du_id.append(str(du_ids[j]))
                dict_du_ns[str(du_ids[j])] = int(du_nanoseconds[j])

            # extract gps_times
            date = gps_times[0]
            time = gps_times[1]
            gps_time = gps_times[2]
            
            # Add information to data_dict
            data_dict[str(event_number)] = {
                "run_number": run_number,
                "event_number": event_number,
                "date": str(date),
                "time": f'{time:0>6}',
                "gps_time": int(gps_time),
                "du_ns": dict_du_ns,
                "du_id": list_du_id,
                "file": file_path,
                "index": int(i),
            }


def process_single_file(file_path: str, result_dict) -> None:
    """
    Process a single file and add results to the result dictionary.

    Parameters:
        file_path: File path
        result_dict: Result dictionary
    """
    try:
        run_number, event_number, du_ids, gps_times, du_ns = read_file_du_time_ns(
            file_path
        )
        # Extract filename for key generation
        file_name = os.path.basename(file_path)
        cal_dict_du_ns(run_number, event_number, du_ids, gps_times, du_ns, result_dict, file_name)
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}, skipping")


def mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logger.debug(f"Directory already exists: {path}")


def write_output(
    data_dict: Dict[str, List[Tuple[int, int]]], out_dir: str, file_name: str
) -> str:
    """Write results to out_dir and return the actual written file path."""
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, "w") as f:
        yaml.dump(data_dict, f)
    return file_path


def process_root_file(file_path: str, date: str, out_dir_base: str) -> bool:
    """
    Main function to process a ROOT file.

    Parameters:
        file_path: Path to the ROOT file to process
        base_path: ROOT file base directory
        out_dir_base: Output directory base path

    Returns:
        Whether processing was successful
    """
    try:
        # Ensure output directory exists
        # Extract date from file path for directory structure

        out_dir = os.path.join(out_dir_base, date)
        mkdir(out_dir)

        logger.info(f"Processing file: {file_path}")

        # Process data
        result: Dict[str, List[Tuple[int, int]]] = {}
        process_single_file(file_path, result)
        logger.info(f"Processed events: {len(result)}")

        # Write output file
        output_filename = f"{os.path.basename(file_path).replace('.root', '_N.yaml')}"
        out_path = write_output(result, out_dir, output_filename)
        logger.info(f"Written to: {out_path}")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Process ROOT file
    success = process_root_file(args.file_path, args.date, args.out_dir_base)

    return 0 if success else 2


if __name__ == "__main__":
    main(sys.argv)
