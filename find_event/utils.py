import numpy as np
from datetime import datetime, timedelta

# GPS and UTC offset (seconds), keeping the original script constant for compatibility
GPS_UTC_OFFSET = 18


"""
Time and coordinate related utility functions

This file provides several small utility functions for other modules (io.py, estimation.py, plotting.py) to call:
- gps_to_utc: Convert GPS seconds to local UTC+8 datetime
- gps2utc1970: Convert GPS seconds to hours (UTC+8, floating point format)
- rotate_coordinates: Apply offset/rotation to coordinates consistent with the original script (maintain compatibility)
- calculate_azimuth / calculate_zenith: Calculate azimuth and zenith angles from direction vectors
"""


def gps_to_utc(gps_time):
        """
        Convert GPS time (seconds) to local time (UTC+8) datetime object.

        Parameters:
            - gps_time: GPS time in seconds (float or int)

        Returns:
            - datetime: Corresponding datetime object in UTC+8 timezone

        Notes:
            - This uses a simple offset conversion: based on 1970-01-01 epoch, plus (gps_time - GPS_UTC_OFFSET + 8*3600) seconds.
            - This conversion is approximate when dealing with special cases like leap seconds, but maintains compatibility with the original script.
        """
        gps_epoch = datetime(1970, 1, 1)
        utc_time = gps_epoch + timedelta(seconds=(gps_time - GPS_UTC_OFFSET + 8 * 3600))
        return utc_time


def gps2utc1970(Tgps):
        """
        Convert GPS time to UTC+8 hours (decimal format), e.g. 13.5 represents 13:30.

        Parameters:
            - Tgps: GPS time (seconds)

        Returns:
            - float: UTC+8 hours (0-24) in floating point representation

        Notes:
            - This function does simple branch processing for leap seconds (based on time threshold), then uses datetime conversion and adds 8 hours.
            - This is a simplified treatment of historical leap seconds, maintaining consistency with the original code logic.
        """
        if Tgps < 1483228818.0:
                leapsecond = 36
        else:
                leapsecond = 37
        leapsecond = leapsecond - 19
        curDate = datetime.fromtimestamp(Tgps - leapsecond)
        utc_plus_8 = curDate + timedelta(hours=8)
        return utc_plus_8.hour + utc_plus_8.minute / 60 + utc_plus_8.second / 3600


def rotate_coordinates(coord):
        """
        Apply rotation and offset to given coordinates consistent with the original script.

        Parameters:
            - coord: Iterable three-dimensional coordinates [x, y, z]

        Returns:
            - np.array: Rotated/offset coordinate array

        Notes:
            - Currently, rotation_matrix is an identity matrix, the actual effect is to apply constant offsets to coordinates,
                these constants come from the original script, used to convert the coordinate system to the reference frame required by the project.
        """
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotated = np.dot(rotation_matrix, coord)
        # The following three offsets maintain consistency with the original script (note the original offset notation was subtracting negative numbers)
        rotated[0] -= -3455.6213993713604
        rotated[1] -= -1277.747
        rotated[2] -= 24.02
        return rotated


def calculate_azimuth(direction_vector):
        """
        Calculate the azimuth angle of a direction vector (in degrees, range 0-360).

        Definition:
            - Azimuth is defined as the angle measured from the x-axis (0°), increasing counterclockwise in the x-y plane.

        Parameters:
            - direction_vector: Length-3 vector (x, y, z)

        Returns:
            - float: Azimuth angle in degrees

        Notes:
            - This convention matches standard mathematical definitions and is important for interpreting the physical meaning of the angle.
        """
        x, y, z = direction_vector
        azimuth = np.degrees(np.arctan2(y, x))
        if azimuth < 0:
            azimuth += 360
        return azimuth


def calculate_zenith(directionVector):
        """
        Calculate the zenith angle of a direction vector (in degrees).

        Parameters:
            - directionVector: Length-3 vector (x, y, z)

        Returns:
            - float: Zenith angle in degrees (0 represents directly overhead)
        """
        x, y, z = directionVector
        epsilon = 1e-12
        zenith = np.arccos(z / (x ** 2 + y ** 2 + z ** 2 + epsilon) ** 0.5) * 180 / np.pi
        return zenith


def calculate_distance_to_axis(detector_positions, source_position, direction):
    """
    Calculate the perpendicular distance from each detector to the incident axis (the incident axis is defined as: the line connecting the intersection of the reverse extension of the signal incidence direction with the Z=0 plane and the signal source)

    Parameters:
        detector_positions: Dictionary {detector_id: position_array}
        source_position: Signal source position (x,y,z) (obtained through spherical wave fitting)
        direction: Incident direction vector (dx,dy,dz) (pointing towards the signal source)

    Returns:
        Dictionary {detector_id: distance_to_axis}
    """
    distances = {}
    direction_norm = direction / np.linalg.norm(direction)  # Normalize incident direction

    # 1. Calculate the intersection point of the reverse extension of the incident direction with the Z=0 plane (ground projection point)
    # Parametric equation: p = source_position + t * (-direction_norm)
    # Find the intersection point when p_z = 0
    if direction_norm[2] == 0:
        # If the incident direction is parallel to the Z=0 plane, directly use the signal source position as a point on the axis
        ground_point = source_position
    else:
        t = source_position[2] / direction_norm[2]  # Step size from reverse extension to Z=0
        ground_point = source_position - t * direction_norm

    # 2. Define the direction vector of the incident axis (from ground projection point to signal source)
    axis_direction = source_position - ground_point
    axis_direction_norm = axis_direction / np.linalg.norm(axis_direction)  # Normalize

    # 3. Calculate the perpendicular distance from each detector to the incident axis
    for det_id, pos in detector_positions.items():
        # Vector from detector to a point on the axis (ground projection point)
        vec = pos - ground_point
        # Cross product to calculate perpendicular distance
        cross_prod = np.cross(vec, axis_direction_norm)
        distance = np.linalg.norm(cross_prod)
        distances[det_id] = distance

    return distances

def calculate_distance_to_source(detector_positions, source_position):
    """

    Parameters:
        detector_positions: Dictionary {detector_id: position_array}
        source_position: Signal source position (x,y,z) (obtained through spherical wave fitting)
        direction: Incident direction vector (dx,dy,dz) (pointing towards the signal source)

    Returns:
        Dictionary {detector_id: distance_to_axis}
    """
    distances = {}

    # 3. Calculate the perpendicular distance from each detector to the incident axis
    for det_id, pos in detector_positions.items():
        distance = np.sqrt( (source_position[0] - pos[0])**2 + (source_position[1] - pos[1])**2 + (source_position[2] - pos[2])**2 )
        distances[det_id] = distance

    return distances

