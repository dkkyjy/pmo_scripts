from .utils import gps_to_utc
from logger_config import logger
import numpy as np
import yaml

c_val = 299792458.0/1e9/1.000259

def parse_match_line(match_line):
    """Concise version"""
    parts = [part.strip().strip("'\"") for part in match_line.split(',')]
    detector_id = parts[0]
    nanoseconds = int(parts[1])
    
    if len(parts) >= 3:
        signal = int(parts[2])
        return (detector_id, nanoseconds), (detector_id, signal)
    else:
        return (detector_id, nanoseconds)
    
    
def optimized_read_matching_times(matching_file, detector_positions, min_detectors=4, speed_of_light_tolerance=1.01):
    """
    Optimized matching time reading function:
    1. Eliminate duplicate triggers from the same DU (keep earliest time).
    2. Strictly validate speed of light propagation limits, eliminate detectors that exceed limits.
    3. Only keep events with at least min_detectors valid detectors.

    Parameters:
        matching_file: Data file path
        detector_positions: Dictionary in format {detector_id: (x, y, z)}
        min_detectors: Minimum number of valid detectors (default 4)
        speed_of_light_tolerance: Speed of light validation tolerance (default 1.01, i.e., 1%)
    """
    matching_times = {}
    matching_signals = {}
    matching_du_ids = {}
    matching_event_numbers = {}
    matching_index = {}

    with open(matching_file, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        for gps_time_str, data in data_dict.items():
            matches = data['du_ns']
            signals = data.get('du_vs', None)
            event_number = data['event_number']
            index = data['index']
            try:
                if len(matches) < min_detectors:
                    continue  # Skip events with insufficient detectors directly

                # Step 1: Process multiple triggers from the same DU (keep the earliest)
                unique_matches = {}
                
                for det_id, ns in matches.items():
                    if det_id not in unique_matches or ns < unique_matches[det_id]:
                        unique_matches[det_id] = ns

                # Sort by trigger time
                sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1])

                #*********************
                # Step 2: Global speed of light consistency check (iteratively eliminate outliers)
                valid_matches = dict(sorted_matches)  # Temporarily use dictionary for convenience

                while len(valid_matches) >= min_detectors:
                    det_ids = list(valid_matches.keys())
                    times = list(valid_matches.values())
                    violations = {det_id: 0 for det_id in det_ids}
                
                    # Pairwise comparison, count violations for each detector
                    for i in range(len(det_ids)):
                        for j in range(i+1, len(det_ids)):
                            di, dj = det_ids[i], det_ids[j]
                            ti, tj = times[i], times[j]
                            if di not in detector_positions or dj not in detector_positions:
                                continue
                            pos_i, pos_j = detector_positions[di], detector_positions[dj]
                            distance = np.linalg.norm(pos_i - pos_j)
                            max_time_diff = distance / c_val
                            actual_diff = abs(ti - tj)
                            #print(di,dj,ti,tj,distance,actual_diff,max_time_diff)
                            if actual_diff > max_time_diff * speed_of_light_tolerance:
                                logger.debug(f"{di} {dj} {distance} {actual_diff} {max_time_diff}")
                                violations[di] += 1
                                violations[dj] += 1
                
                    # If no violations, end
                    if all(v == 0 for v in violations.values()):
                        break
                
                    # Otherwise, eliminate the detector with the most violations
                    worst = max(violations, key=violations.get)
                    logger.warning(f"{gps_time_str}: Eliminating outlier detector {worst}, violations={violations[worst]}")
                    valid_matches.pop(worst)
                
                # Final result
                logger.info(f"length of matching {len(valid_matches)}")
                if len(valid_matches) >= min_detectors:
                    matching_times[gps_time_str] = valid_matches
                    matching_du_ids[gps_time_str] = list(valid_matches.keys())
                    if signals is not None:
                        matching_signals[gps_time_str] = {key:signals[key] for key in valid_matches.keys()}
                    else:
                        matching_signals[gps_time_str] = None
                    matching_event_numbers[gps_time_str] = event_number
                    matching_index[gps_time_str] = index

            except Exception as e:
                logger.exception(f"Error parsing: {e}")
                continue
    return matching_times, matching_signals, matching_du_ids, matching_event_numbers, matching_index

def optimized_read_matching_times_with_signal(file_path, detector_positions, min_detectors=4, speed_of_light_tolerance=1.01):
    """
    Optimized reading function specifically for the following format:
    GPS time:{'time': {detector ID: time,...}, 'signal': {detector ID: signal strength,...}}
    """
    matching_times = {}
    matching_signal = {}
    matching_du_ids = {}
    matching_event_numbers = {}
    matching_index = {}

    with open(file_path, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        for gps_time_str, data in data_dict.items():
            time_matches = data['du_ns']
            signal_matches = data['du_vs']
            event_number = data['event_number']
            index = data['index']
            try:
                # Check data consistency
                if len(time_matches) != len(signal_matches):
                    continue
                    
                # Merge valid data
                valid_data = []
                for (det_id_t, ns), (det_id_s, amp) in zip(time_matches.items(), signal_matches.items()):
                    if det_id_t == det_id_s and det_id_t in detector_positions:
                        valid_data.append((det_id_t, ns, amp))
                
                if len(valid_data) < min_detectors:
                    continue

                # Sort by time
                valid_data.sort(key=lambda x: x[1])  # Sort by ns
                
                # Validate speed of light limits
                final_matches = {}
                final_signals = {}
                for i, (det_id, ns, amp) in enumerate(valid_data):
                    pos_i = detector_positions[det_id]
                    valid = True
                    
                    for j, (other_id, other_ns, _) in enumerate(valid_data[:i]):
                        pos_j = detector_positions[other_id]
                        distance = np.linalg.norm(pos_i - pos_j)
                        max_time_diff = distance / c_val
                        actual_diff = abs(ns - other_ns)
                        
                        if actual_diff > max_time_diff * speed_of_light_tolerance:
                            valid = False
                            break
                    
                    if valid:
                        final_matches[det_id] = ns
                        final_signals[det_id] = amp
                
                if len(final_matches) >= min_detectors:
                    matching_times[gps_time_str] = final_matches
                    matching_signal[gps_time_str] = final_signals
                    matching_du_ids[gps_time_str] = list(final_matches.keys())
                    matching_event_numbers[gps_time_str] = event_number
                    matching_index[gps_time_str] = index
                    
            except Exception as e:
                logger.error(f"Error parsing: {str(e)}")
                continue

    return matching_times, matching_signal, matching_du_ids, matching_event_numbers, matching_index