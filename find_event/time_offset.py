"""
Time offset configuration module

This module contains time offset configuration data for detectors.
"""

# Currently used time offset configuration
time_offset = [
    # Beacon GP35 point5 wrt 1023
    (1018, 40.46),
    (1023, 0.00),
    (1035, 34.51),
    (1038, 15.69),
    (1040, 30.24),
    (1043, 62.32),
    (1045, 56.49),
    (1048, 44.84),
    (1058, 38.72),
    (1059, 56.55),
    (1065, 56.61),
    
    # Beacon GP35 point 4 wrt DU1011
    (1011, 0.00),
    (1013, -0.28),
    (1019, 11.42),
    (1020, 27.06),
    (1029, 15.81),
    (103, 18.05),
    (1032, 5.92),
    (1031, 0.00),
    (1033, 0.00),
    (1042, -3.65),
    (1044, 9.60),
    (1046, 16.34),
    (1047, 27.38),
    (1049, 8.79),
    (1051, 36.03),
    (1052, 26.22),
    (1055, -3.10),
    (1057, 29.99),
    (1074, 14.97),
    (1075, 3.81),
    (1077, -6.91),
    (1078, 35.12),
    (1079, 3.85),
    (1081, 24.10),
    (1082, 10.66),
    (1083, 13.47),
    (1084, 7.21),
    (1085, -11.08),
    (1086, -0.00),
    (1088, 33.63),
    (1089, -10.83),
    (109, 0.0000),
    (1092, 15.45),
    (1093, 28.28),
    (1066, 0.00),
    (1014, 0.00),
    (1072, 0.00),
    (1010, 0.00),
    (1024, 0.00),
    (1073, 0.00),
    (1016, 0.00), 
    (1017, 0.00),
    (1022, 0.00),
    (1024, 0.00),
    (1030, 0.00),
    (1037, 0.00),
    (1039, 0.00),
    (1041, 0.00),
    (1050, 0.00),
    (1056, 0.00),
    (1066, 0.00),
    (1071, 0.00),
    (1072, 0.00),
    (1073, 0.00),
    (1076, 0.00),
    (1087, 0.00),
    (1090, 0.00),
    (1091, 0.00),
    (1012, 0.00),
    (1094, -1.44)
]

id_time_dict = dict(time_offset)

def get_time_offset(id):
    """Return the time offset for the given detector ID (return 0 if not configured)."""
    time_dif = id_time_dict.get(id, None)
    if time_dif is None:
        return 0 
    return time_dif


def load_time_data(file_path):
    """Load all data from file and return dictionary"""
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                id_val = int(parts[0])
                data_dict[id_val] = {
                    'mean': float(parts[1]),
                    'dispersion': float(parts[2])
                }
    return data_dict

# Usage example
#time_data = load_time_data('2025-06-22_offset_allruns_beacon.txt')
time_data = load_time_data('2025-10-28_beacon_25Hz_offset.txt')
# =========================