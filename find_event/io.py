import numpy as np
from collections import defaultdict
from .utils import gps_to_utc, calculate_azimuth, calculate_zenith
from logger_config import logger


def load_data_from_file(filename):
    """
    Load detector (DU) positions from a text file.

    File format convention (one detector per line, lines starting with # are treated as comments and skipped):
        ID x y z
    Where ID is a string (usually detector number), and x/y/z are floating point coordinates in meters.

    Parameters:
        - filename: Detector coordinate file path (string)

    Returns:
        - dict: { id_str: np.array([x, y, z]) }

    Implementation details:
        - Ignore empty lines and comment lines starting with '#'
        - Print warning and skip the line when coordinates in a row cannot be converted to float
        - Raise FileNotFoundError exception if file does not exist
    """
    data = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines
                if line and not line.startswith("#"):
                    parts = line.strip().split()
                    # Expect at least 4 columns: ID, x, y, z
                    if len(parts) >= 4:
                        duid = parts[0]
                        try:
                            coords = np.array(
                                [float(parts[1]), float(parts[2]), float(parts[3])]
                            )
                            data[duid] = coords
                        except ValueError:
                            # Print warning and continue if a column cannot be parsed as float
                            logger.warning(f"Could not parse coordinates for DU {duid}")
    except FileNotFoundError:
        logger.error(f"Detector position file not found: {filename}")
        raise
    return data


def filter_and_write_to_file(directions_array, times, chi_squares, output_file):
    """
    Filter events based on direction, time, and chi-square values and write results to a text file.

    Parameters:
        - directions_array: Iterable list of direction vectors (each as np.array([x,y,z]))
        - times: List of GPS times corresponding to directions_array
        - chi_squares: List of chi-square values corresponding to directions_array
        - output_file: Output file path (string), opened in write mode and will overwrite existing content

    Line format (example):
        <gps_time> <utc_time> <zenith> <azimuth> Chi <chi_value>

    Filtering criteria (current implementation):
        - chi_fit < 5e2
        - 45 < zenith_angles < 85

    Notes:
        - This function assumes directions_array/times/chi_squares have the same length and correspond one-to-one.
        - Uses `calculate_azimuth`, `calculate_zenith`, `gps_to_utc` to calculate required dimensions and time strings.
    """
    with open(output_file, "w") as f:
        index = 0
        for direction, gps_time, chi_fit in zip(directions_array, times, chi_squares):
            index += 1
            azimuth = calculate_azimuth(direction)
            zenith_angles = calculate_zenith(direction)
            utc_times = gps_to_utc(gps_time)
            # Filter based on thresholds and write to file
            if chi_fit < 5e2 and 45 < zenith_angles < 85:
                f.write(
                    f"{gps_time} {utc_times} {zenith_angles:.1f} {azimuth:.1f} Chi {chi_fit:.1f}\n"
                )