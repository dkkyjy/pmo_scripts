import numpy as np
from .utils import gps_to_utc, calculate_distance_to_axis, calculate_distance_to_source
from .plotting import plot_detector_positions, plot_detector_positions_with_signal, plot_signal_fit_with_signal
from logger_config import logger


def background_reject(detector_positions, current_du_ids_dict, times_dict, signals_dict, chi_square_dict, azimuth_dict, zenith_dict, source_position_dict, gps_time_dict, index_dict, save_name, output_dir, with_signal):
    du_ids_filtered = {}
    times_filtered = {}
    signals_filtered = {}
    chi_squares_filtered = {}
    azimuths_filtered = {}
    zeniths_filtered = {}
    source_directions_filtered = {}
    gps_time_filtered = {}
    for i, key in enumerate(times_dict.keys()):
        current_du_ids = current_du_ids_dict[key]
        times = times_dict[key]
        signals = signals_dict[key]
        chi_square = chi_square_dict[key]
        azimuth = azimuth_dict[key]
        zenith = zenith_dict[key]
        source_position = source_position_dict[key]
        gps_time = gps_time_dict[key]
        index = index_dict[key]
        if (chi_square < 5e2 and 50 < zenith < 85 and ( (abs(source_position[2]/1e3 - 9) > 1.5 and 40 <= azimuth <= 225) or (azimuth < 40 or azimuth > 225)) ):
            du_ids_filtered[key] = current_du_ids
            times_filtered[key] = times
            signals_filtered[key] = signals
            chi_squares_filtered[key] = chi_square
            azimuths_filtered[key] = azimuth
            zeniths_filtered[key] = zenith
            source_directions_filtered[key] = source_position
            gps_time_filtered[key] = gps_time
            
            logger.info(f"Event {index} with zenith {zenith:.2f} and azimuth {azimuth:.2f} satisfies the condition.")
            # Delegate plotting to helper in plotting.py
            if with_signal:
                print('', times)
                print('', signals)
                print('', current_du_ids)
                try:
                    saved = plot_detector_positions_with_signal(
                        detector_positions,
                        current_du_ids,
                        times,
                        signals,
                        chi_square,
                        index,
                        azimuth,
                        zenith,
                        'SWM',
                        save_name,
                        gps_time,
                        output_dir=None,
                    )
                    if saved:
                        logger.info(f"Saved detector plot for SWM Event {index}/{len(times)}: {saved}")
                except Exception as e:
                    logger.exception(f"Failed to save detector plot for event {index}/{len(times)}: {e}")

                distances_to_axis = calculate_distance_to_axis(detector_positions, source_position, 
                                                             source_position/np.linalg.norm(source_position))
                distances_to_source = calculate_distance_to_source(detector_positions, source_position)

                # Prepare plotting data
                x_data = []
                y_data = []
                det_labels = []
                event_time = gps_to_utc(gps_time)
                for det_id in current_du_ids:
                    if det_id in distances_to_source and det_id in signals:
                        x_data.append(distances_to_source[det_id])
                        # y_data.append(signals[det_id] * (np.linalg.norm(detector_positions[det_id] - source_position))**2)
                        y_data.append(signals[det_id] * (np.linalg.norm(detector_positions[det_id] - source_position))**0)
                        det_labels.append(det_id)
                try:
                    save_path = plot_signal_fit_with_signal(save_name, index, x_data, y_data, det_labels, event_time, output_dir=None)
                    if save_path:
                        logger.info(f"Saved signal LDF fit plot for event {index}/{len(times)}: {save_path}")
                except Exception as e:
                    logger.exception(f"Error generating Gaussian fit plot for event {index}/{len(times)}: {e}")
            else:      
                try:
                    saved = plot_detector_positions(
                        detector_positions,
                        current_du_ids,
                        times,
                        chi_square,
                        index,
                        azimuth,
                        zenith,
                        'SWM',
                        save_name,
                        gps_time,
                        output_dir=None,
                    )
                    if saved:
                        logger.info(f"Saved detector plot for SWM Event {index}/{len(times)}: {saved}")
                except Exception as e:
                    logger.exception(f"Failed to save detector plot for event {index}/{len(times)}: {e}")
        else:
            logger.info(f"Event {index} with zenith {zenith:.2f} and azimuth {azimuth:.2f} rejected by background criteria.")
            continue
    return du_ids_filtered, times_filtered, signals_filtered, chi_squares_filtered, azimuths_filtered, zeniths_filtered, source_directions_filtered, gps_time_filtered