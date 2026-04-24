"""
Plotting module

This module contains functions for data visualization and result presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import scienceplots
from matplotlib.markers import MarkerStyle
plt.style.use(['science', 'notebook', 'grid'])

from datetime import datetime
from .utils import calculate_azimuth, calculate_zenith
from .io import filter_and_write_to_file
from pathlib import Path
from logger_config import logger


def _ensure_datetime(val):
    """Helper to ensure a value is a datetime object if it is an ISO string."""
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            return val
    return val


def plot_histograms(time_wrt1023, fig_name):
    """
    Plot histograms of time differences relative to a reference detector (example: 1023) grouped by detector.
    Parameters:
      - time_wrt1023: List, each item is (detector_id, time_difference)
      - fig_name: Output file name prefix
    """
    grouped_data = defaultdict(list)
    # Collect time differences for each detector (excluding abnormally large differences)
    for detector_id, time_difference in time_wrt1023:
        if time_difference < 100:
            grouped_data[detector_id].append(time_difference)

    fig, ax = plt.subplots()
    # Plot histogram for each detector and annotate standard deviation on the plot
    for detector_id, times in grouped_data.items():
        ax.hist(times, bins=40, alpha=0.5, label=f'DU {detector_id}')
        mean = np.mean(times)
        sigma = np.std(times)
        ax.annotate(r'$\sigma: {:.2f}$'.format(sigma), xy=(mean + sigma, 0), xytext=(mean + sigma, 10),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Time Difference (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Time Differences Relative to ID=1023')
    ax.set_ylim(0, 240)
    ax.legend()
    figure = plt.gcf()
    figure.savefig(fig_name + '_Difference.pdf')
    figure.savefig(fig_name + '_Difference.png')


def plot_du_frequencies(du_ids, fig_prefix):
    """
    Count and plot the number of occurrences of each DU under the chi-square threshold condition, and the distribution of the number of DUs contained in each event.
    - gps_times, directions, chi_squares, du_ids: Parallel arrays corresponding to the event list
    - threshold: Only count events with chi_squares less than this threshold
    """

    # Collect all DU ids in these events (each event may contain multiple DUs) and flatten
    flattened_du_ids = [du for sublist in du_ids.values() for du in sublist]

    # Count the number of occurrences of each DU
    du_counts = Counter(flattened_du_ids)
    sorted_du_counts = dict(sorted(du_counts.items()))

    # Count how many DUs each event contains and plot the distribution
    du_id_counts = [len(du_id) for du_id in du_ids.values()]
    unique, counts = np.unique(du_id_counts, return_counts=True)

    # Plot: Left chart shows contribution count for each DU, right chart shows distribution of DU count per event
    ids = list(sorted_du_counts.keys())
    du_counts_values = list(sorted_du_counts.values())
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Left bar chart: DU occurrence count
    axs[0].bar(ids, du_counts_values, color='skyblue')
    axs[0].set_xlabel('DU ID', fontsize=20)
    axs[0].set_ylabel('Count', fontsize=20)
    axs[0].set_title('DU counts', fontsize=23)
    axs[0].tick_params(axis='x', rotation=65, labelsize=16, length=6, direction='in')
    axs[0].tick_params(axis='y', labelsize=20, length=6, direction='in')
    axs[0].grid(True, axis='y')

    # Right bar chart: Distribution of DU count per event
    axs[1].bar(unique, counts, color='lightcoral')
    axs[1].set_xlabel('Number of DU IDs per event', fontsize=23)
    axs[1].set_ylabel('Counts', fontsize=23)
    axs[1].tick_params(axis='x', rotation=65, labelsize=20, length=6, direction='in')
    axs[1].tick_params(axis='y', labelsize=20, length=6, direction='in')
    axs[1].set_title('Multiplicities per CD Event', fontsize=24)
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_DU_contribution.png')



def plot_fitting_parameters_PWM(datetimes, directions, chi_squares, model_name, save_name):
    """
    Plot fitting parameters varying over time: zenith angle, azimuth angle, and chi-square values.
    - datetimes: Dict of datetime strings or objects
    - directions: List of direction vectors for each event (Nx3)
    - chi_squares: Chi-square values for each event
    """
    utc_times = [_ensure_datetime(time) for time in datetimes.values()]
    process_chi = np.array(list(chi_squares.values()))
    directions = np.array(list(directions.values()))

    # Extract direction components and calculate angles
    x_dir = directions[:, 0]
    y_dir = directions[:, 1]
    z_dir = directions[:, 2]
    epsilon = 1e-12
    zenith_angles = np.arccos(z_dir / (epsilon + np.linalg.norm(directions, axis=1))) * 180 / np.pi
    azimuth_angles = np.arctan2(y_dir, x_dir) * 180 / np.pi
    azimuth_angles = np.where(azimuth_angles < 0, azimuth_angles + 360, azimuth_angles)

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    # Beautify axis tick styles
    for ax in axs:
        ax.tick_params(axis='both', direction='in', length=10, width=1.5, labelsize=22)
        ax.tick_params(axis='x', labelrotation=30, direction='in', length=10, width=1.5, labelsize=22)

    # Zenith angle over time
    axs[0].plot(utc_times, zenith_angles, marker='o', ms=0.6, linestyle='none', color='b')
    axs[0].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[0].set_ylabel('Zenith Angle (degrees)', fontsize=22)
    axs[0].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[0].set_title(f'{model_name}: Zenith Angle Over Time', fontsize=22)
    
    # Azimuth angle over time
    axs[1].plot(utc_times, azimuth_angles, marker='o', ms=0.6, linestyle='none', color='r')
    axs[1].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[1].set_ylabel('Azimuth Angle (degrees)', fontsize=22)
    axs[1].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[1].set_title(f'{model_name}: Azimuth Angle Over Time', fontsize=22)

    # Chi-square values over time (log scale)
    axs[2].plot(utc_times, process_chi + 0.01, marker='o', ms=0.6, linestyle='none', color='g')
    axs[2].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[2].set_ylabel('Chi-square', fontsize=22)
    axs[2].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[2].set_title(f'{model_name}: Chi-square Over Time', fontsize=22)
    axs[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_name}_overtime.png')


def plot_fitting_parameters_SWM(datetimes, directions, chi_squares, model_name, save_name):
    """
    Plot fitting parameters varying over time: zenith angle, azimuth angle, and chi-square values.
    - datetimes: Dict of datetime strings or objects
    - directions: List of direction vectors for each event (Nx3)
    - chi_squares: Chi-square values for each event
    """
    utc_times = [_ensure_datetime(time) for time in datetimes.values()]
    process_chi = np.array(list(chi_squares.values()))
    directions = np.array(list(directions.values()))

    # Extract direction components and calculate angles
    x_dir = directions[:, 0]
    y_dir = directions[:, 1]
    z_dir = directions[:, 2]

    fig, axs = plt.subplots(4, 1, figsize=(15, 10))
    # Beautify axis tick styles
    for ax in axs:
        ax.tick_params(axis='both', direction='in', length=10, width=1.5, labelsize=22)
        ax.tick_params(axis='x', labelrotation=30, direction='in', length=10, width=1.5, labelsize=22)

    # Zenith angle over time
    axs[0].plot(utc_times, x_dir/1e3, marker='o', ms=0.6, linestyle='none', color='b')
    axs[0].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[0].set_ylabel('X Direction (km)', fontsize=22)
    axs[0].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[0].set_title(f'{model_name}: X Direction Over Time', fontsize=22)
    
    # Azimuth angle over time
    axs[1].plot(utc_times, y_dir/1e3, marker='o', ms=0.6, linestyle='none', color='r')
    axs[1].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[1].set_ylabel('Y Direction (km)', fontsize=22)
    axs[1].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[1].set_title(f'{model_name}: Y Direction Over Time', fontsize=22)

    # Azimuth angle over time
    axs[2].plot(utc_times, z_dir/1e3, marker='o', ms=0.6, linestyle='none', color='r')
    axs[2].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[2].set_ylabel('Z Direction (km)', fontsize=22)
    axs[2].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[2].set_title(f'{model_name}: Z Direction Over Time', fontsize=22)


    # Chi-square values over time (log scale)
    axs[3].plot(utc_times, process_chi + 0.01, marker='o', ms=0.6, linestyle='none', color='g')
    axs[3].set_xlabel('Time (UTC+8)', fontsize=22)
    axs[3].set_ylabel('Chi-square', fontsize=22)
    axs[3].grid(True, linestyle=':', color='gray', alpha=0.7)
    axs[3].set_title(f'{model_name}: Chi-square Over Time', fontsize=22)
    axs[3].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_name}_overtime.png')


def plot_reconstructed_positions_PWM(datetimes, directions, chi_squares, save_name):
    """
    Plot visualization combination chart of reconstructed signal source direction/position over time:
      - Top left: 3D direction scatter plot (color represents time)
      - Top right: XY or XZ projection plot (depending on whether SWM)
      - Bottom left: Polar coordinate plot (azimuth vs zenith)
      - Bottom right: Chi-square distribution (log scale)

    Parameters:
      - datetimes: Dict of datetime strings or objects
      - chi_squares: Array of chi-square values for each event
      - save_name: Save prefix name
      - output_file: File path for writing filtered results (optional)
      - zenith_angle_range: Zenith angle range included in visualization
    """

    # If there are no points to plot, return directly
    if len(directions) == 0:
        logger.warning(f"Warning: directions array is empty, unable to calculate zenith angle.")
        return

    # Convert datetimes to UTC objects and numeric timestamps for coloring
    utc_times = [_ensure_datetime(time) for time in datetimes.values()]
    numeric_times = []
    for dt in utc_times:
        if hasattr(dt, 'timestamp'):
            numeric_times.append(dt.timestamp())
        else:
            numeric_times.append(0)
    numeric_times = np.array(numeric_times)
    print(f"Debug: numeric_times = {len(numeric_times)}")
    
    directions = np.array(list(directions.values()))
    chi_squares = np.array(list(chi_squares.values()))

    # Calculate zenith and azimuth angles (degrees)
    epsilon = 1e-12
    zenith_angles = np.arccos(directions[:, 2] / (np.linalg.norm(directions, axis=1) + epsilon)) * 180 / np.pi
    azimuth_angles = np.arctan2(directions[:, 1], directions[:, 0]) * 180 / np.pi
    azimuth_angles = np.where(azimuth_angles < 0, azimuth_angles + 360, azimuth_angles)

    # Start plotting layout
    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(231)
    sc = ax1.hist(zenith_angles, bins=50, histtype='step', linewidth=2, color='black')
    ax1.set_xlabel('Zenith Angle (degrees)', fontsize=22)

    # Top right subplot: Draw different projections based on whether SWM (XY or XZ)
    ax2 = fig.add_subplot(232)
    sc2 = ax2.scatter(directions[:, 0], directions[:, 1], s=0.65, c=numeric_times, cmap='viridis')
    ax2.set_xlabel(r'D_X', fontsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_ylabel(r'D_Y', fontsize=20)
    # cbar2 = plt.colorbar(sc2, ax=ax2)
    # cbar2.ax.tick_params(axis='y', rotation=15, pad=0.1)
    # cbar2.set_ticks(selected_times)
    # cbar2.set_ticklabels([utc_time.strftime("%Y-%m-%d %H:%M:%S") for utc_time in selected_utc_times])

    # 3D scatter plot (top left): Color represents time
    ax3 = fig.add_subplot(233, projection='3d')
    sc3 = ax3.scatter(directions[:, 0], directions[:, 1], directions[:, 2], s=0.45, c=numeric_times, cmap='viridis')
    ax3.tick_params(axis='both', labelsize=20)
    ax3.set_xlabel(r'D_X', fontsize=22)
    ax3.set_ylabel(r'D_Y', fontsize=22)
    ax3.set_zlabel(r'D_Z', fontsize=22)
    # cbar3 = plt.colorbar(sc3, ax=ax3)
    # selected_utc_times = [min(utc_times), max(utc_times)]
    # selected_times = [min(gps_times), max(gps_times)]
    # cbar3.set_ticks(selected_times)
    # cbar3.ax.tick_params(axis='y', rotation=15, pad=0.1)
    # cbar3.set_ticklabels([utc_time.strftime("%Y-%m-%d %H:%M:%S") for utc_time in selected_utc_times])

    ax4 = fig.add_subplot(234)
    sc4 = ax4.hist(azimuth_angles, bins=50, histtype='step', linewidth=2, color='black')
    ax4.set_xlabel('Azimuth Angle (degrees)', fontsize=22)


    # Bottom left: Polar coordinate plot (azimuth vs zenith angle)
    ax5 = fig.add_subplot(235, projection='polar')
    sc5 = ax5.scatter(np.deg2rad(azimuth_angles), zenith_angles, s=0.65, c=numeric_times, cmap='viridis')
    ax5.set_theta_zero_location('N')
    ax5.set_xticks(np.deg2rad([0, 315, 270, 225, 180, 135, 90, 45]))
    ax5.set_xticklabels(['N', '315°', 'E', '225°', 'S', '135°', 'W', '45°'], fontsize=22)
    # cbar5 = plt.colorbar(sc5, ax=ax5)
    # cbar5.set_ticks(selected_times)
    # cbar5.set_ticklabels([utc_time.strftime("%Y-%m-%d %H:%M:%S") for utc_time in selected_utc_times])
    # cbar5.ax.tick_params(axis='y', rotation=15, pad=0.1)

    # Bottom right: Chi-square distribution (log scale)
    ax6 = fig.add_subplot(236)
    if len(chi_squares) > 1:
        bins = np.logspace(np.log10(min(chi_squares) + 0.01), np.log10(max(chi_squares)), 100)
    else:
        bins = np.array([chi_squares[0], chi_squares[0] + 1e-6])
    ax6.hist(chi_squares, bins=bins, histtype='step', linewidth=2, color='black')
    ax6.set_xlabel(r'Reduced $\chi^2/ndf$', fontsize=20)
    ax6.set_xscale('log')
    ax6.set_ylabel('# of event', fontsize=20)
    ax6.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22,
                    length=6, direction='in', width=1.5)

    plt.savefig(f'{save_name}_rec.png')



def plot_reconstructed_positions_SWM(datetimes, directions, chi_squares, save_name):
    """
    Plot visualization combination chart of reconstructed signal source direction/position over time:
      - Top left: 3D direction scatter plot (color represents time)
      - Top right: XY or XZ projection plot (depending on whether SWM)
      - Bottom left: Polar coordinate plot (azimuth vs zenith)
      - Bottom right: Chi-square distribution (log scale)

    Parameters:
      - datetimes: Dict of datetime strings or objects
      - chi_squares: Array of chi-square values for each event
      - save_name: Save prefix name
      - output_file: File path for writing filtered results (optional)
      - zenith_angle_range: Zenith angle range included in visualization
    """

    # If there are no points to plot, return directly
    if len(directions) == 0:
        logger.warning(f"Warning: directions array is empty, unable to calculate zenith angle.")
        return

    # Convert datetimes to UTC objects and numeric timestamps for coloring
    utc_times = [_ensure_datetime(time) for time in datetimes.values()]
    numeric_times = []
    for dt in utc_times:
        if hasattr(dt, 'timestamp'):
            numeric_times.append(dt.timestamp())
        else:
            numeric_times.append(0)
    numeric_times = np.array(numeric_times)

    directions = np.array(list(directions.values()))
    chi_squares = np.array(list(chi_squares.values()))

    # Start plotting layout
    fig = plt.figure(figsize=(20, 12))

    # 3D scatter plot (top left): Color represents time
    ax1 = fig.add_subplot(231)
    sc1 = ax1.hist(directions[:, 0]/1e3, bins=50, histtype='step', linewidth=2, color='black')
    ax1.set_xlabel(r'D_X (km)', fontsize=22)
    
    ax2 = fig.add_subplot(232)
    sc2 = ax2.hist(directions[:, 1]/1e3, bins=50, histtype='step', linewidth=2, color='black')
    ax2.set_xlabel(r'D_Y (km)', fontsize=22)
    
    ax3 = fig.add_subplot(233)
    sc3 = ax3.hist(directions[:, 2]/1e3, bins=50, histtype='step', linewidth=2, color='black')
    ax3.set_xlabel(r'D_Z (km)', fontsize=22)
    
    # Top right subplot: Draw different projections based on whether SWM (XY or XZ)
    ax4 = fig.add_subplot(234)
    sc4 = ax4.scatter(directions[:, 0] / 1e3, directions[:, 2] / 1e3, s=0.65, c=numeric_times, cmap='viridis')
    ax4.set_xlabel(r'Position X (km)', fontsize=20)
    ax4.set_ylabel(r'Position Z (km)', fontsize=20)
    ax4.tick_params(axis='both', labelsize=20)
    # ax4.set_title('DX vs DZ, SWM', fontsize=20)
    # cbar4 = plt.colorbar(sc4, ax=ax4)
    # cbar4.ax.tick_params(axis='y', rotation=15, pad=0.1)
    # selected_utc_times = [min(utc_times), max(utc_times)]
    # selected_times = [min(gps_times), max(gps_times)]
    # cbar4.set_ticks(selected_times)
    # cbar4.set_ticklabels([utc_time.strftime("%Y-%m-%d %H:%M:%S") for utc_time in selected_utc_times])
    
    # Top right subplot: Draw different projections based on whether SWM (XY or XZ)
    ax5 = fig.add_subplot(235)
    sc5 = ax5.scatter(directions[:, 1] / 1e3, directions[:, 2] / 1e3, s=0.65, c=numeric_times, cmap='viridis')
    ax5.set_xlabel(r'Position Y (km)', fontsize=20)
    ax5.set_ylabel(r'Position Z (km)', fontsize=20)
    ax5.tick_params(axis='both', labelsize=20)
    # ax5.set_title('DY vs DZ, SWM', fontsize=20)
    # cbar5 = plt.colorbar(sc5, ax=ax5)
    # cbar5.ax.tick_params(axis='y', rotation=15, pad=0.1)
    # cbar5.set_ticks(selected_times)
    # cbar5.set_ticklabels([utc_time.strftime("%Y-%m-%d %H:%M:%S") for utc_time in selected_utc_times])

    # Bottom right: Chi-square distribution (log scale)
    ax6 = fig.add_subplot(236)
    if len(chi_squares) > 1:
        bins = np.logspace(np.log10(min(chi_squares) + 0.01), np.log10(max(chi_squares)), 100)
    else:
        bins = np.array([chi_squares[0], chi_squares[0] + 1e-6])
    ax6.hist(chi_squares, bins=bins, histtype='step', linewidth=2, color='black')
    ax6.set_xlabel(r'Reduced $\chi^2/ndf$', fontsize=20)
    ax6.set_xscale('log')
    ax6.set_ylabel('# of event', fontsize=20)
    ax6.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22,
                    length=6, direction='in', width=1.5)

    plt.savefig(f'{save_name}_rec.png')


def plot_detector_positions(detector_positions, current_du_ids, times, chi_square, index, azimuth, zenith, model_name, save_name, event_datetime, output_dir):
    """Plot detector positions for a single SWM event and save the figure.

    Parameters:
      - detector_positions: dict {det_id: np.array([x,y,z])}
      - current_du_ids: list of detector ids participating in this event
      - times: dict {det_id: time_ns} (used to color points)
      - chi: chi of the event
      - index: event index
      - azimuth, zenith: floats
      - save_name: base name used previously
      - event_datetime: datetime string or object used to produce label
      - output_dir: optional dir to save the image into; if provided it will be created

    Returns the saved file path (str) on success, or None on failure.
    """
    pos_list = []
    colors = []
    labels = []
    all_pos_list = []
    all_labels = []

    for det_id, pos in detector_positions.items():
        all_pos_list.append(pos[:2])
        all_labels.append(str(det_id))

    for det_id in current_du_ids:
        pos = detector_positions[str(det_id)][:2]
        pos_list.append(pos)
        colors.append(times[str(det_id)] - np.mean(list(times.values())))
        labels.append(str(det_id))

    pos_array = np.array(pos_list)
    all_pos_array = np.array(all_pos_list)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(-1 * all_pos_array[:, 1], all_pos_array[:, 0], c='lightgray', marker=MarkerStyle('o'), s=50, label='All DUs')
    sc = ax.scatter(-1 * pos_array[:, 1], pos_array[:, 0], c=colors, cmap="viridis", marker=MarkerStyle('o'), s=100, label=f"Coincident DUs")
    for j, label in enumerate(labels):
        ax.text(-1 * pos_array[j, 1], pos_array[j, 0], label, fontsize=12, ha='left', va='bottom', color='black')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Relative Time (ns)', fontsize=12)

    arrow_length = 500
    arrow_dx = -1 * arrow_length * np.sin(np.deg2rad(azimuth))
    arrow_dy = arrow_length * np.cos(np.deg2rad(azimuth))
    ax.arrow(0, 0, arrow_dx, arrow_dy, head_width=100, head_length=100, fc='red', ec='red', label=f'Azimuth {azimuth:.1f}°;Zenith {zenith:.1f}°')

    event_time = _ensure_datetime(event_datetime)
    ax.set_xlabel('W-E [m]')#, fontsize=12)
    ax.set_ylabel('S-N [m]')#, fontsize=12)
    ax.set_title(rf'{model_name}: DU Positions for Event {index} ( A {azimuth:.1f}°, Z {zenith:.1f}°), $\chi^2$={chi_square:.2f} at {event_time}')#, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)

    # Determine save path
    filename = f"{save_name}_No.{index}_detector_positions.png"
    if output_dir is not None:
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        save_path = outp / filename
    else:
        save_path = Path(filename)

    try:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        return str(save_path)
    except Exception:
        plt.close()
        logger.exception(f"Failed to save detector plot for event {index}")
        return None



def plot_detector_positions_with_signal(detector_positions, current_du_ids, times, signal_amps, chi_square, index, azimuth, zenith, model_name, save_name, event_datetime, output_dir=None):
    """
    Plot detector positions for the deep-seek SWM event and save the figure.

    Parameters:
      - detector_positions: dict {det_id: np.array([x,y,z])}
      - current_du_ids: list of detector ids participating in this event
      - times: dict {det_id: time_ns} (used to color points)
      - signal_amps: dict {det_id: amplitude} used to size markers
      - chi_square: chi of the event
      - index: event index
      - azimuth, zenith: floats
      - save_name: base name used for saving files
      - event_datetime: datetime string or object for title
      - output_dir: optional directory to save into
      - cmap: colormap

    Returns saved file path (str) or None on failure.
    """
    # Prepare position lists
    pos_list = []
    colors = []
    labels = []
    marker_size = []
    all_pos_list = []
    all_labels = []

    for det_id, pos in detector_positions.items():
        all_pos_list.append(pos[:2])
        all_labels.append(str(det_id))

    # Signal amplitude stats for marker scaling
    try:
        signal_values = np.array(list(signal_amps.values())) if signal_amps else np.array([0.0])
        S_min = float(np.min(signal_values))
        S_max = float(np.max(signal_values))
        if S_max == S_min:
            S_max = S_min + 1.0
    except Exception:
        S_min, S_max = 0.0, 1.0

    for det_id in current_du_ids:
        pos = detector_positions[det_id][:2]
        pos_list.append(pos)
        colors.append(times[det_id] - np.mean(list(times.values())))
        labels.append(str(det_id))
        amp = float(signal_amps.get(det_id, signal_amps.get(str(det_id), S_min)))
        # scale marker size safely
        marker_size.append(float((amp - S_min + 10) / (S_max - S_min)) * 400)

    pos_array = np.array(pos_list)
    all_pos_array = np.array(all_pos_list)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(-1 * all_pos_array[:, 1], all_pos_array[:, 0], c='lightgray', marker=MarkerStyle('o'), s=50, label='All DUs')
    sc = ax.scatter(-1 * pos_array[:, 1], pos_array[:, 0], c=colors, cmap="viridis", marker=MarkerStyle('o'), s=marker_size, label=f"Coincident DUs")
    for j, label in enumerate(labels):
        ax.text(-1 * pos_array[j, 1], pos_array[j, 0], label, fontsize=12, ha='left', va='bottom', color='black')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Relative Time (ns)', fontsize=12)

    arrow_length = 500
    arrow_dx = -1 * arrow_length * np.sin(np.deg2rad(azimuth))
    arrow_dy = arrow_length * np.cos(np.deg2rad(azimuth))
    ax.arrow(0, 0, arrow_dx, arrow_dy, head_width=100, head_length=100, fc='red', ec='red', label=f'Azimuth {azimuth:.1f}°;Zenith {zenith:.1f}°')

    event_time = _ensure_datetime(event_datetime)
    ax.set_xlabel('W-E [m]')#, fontsize=12)
    ax.set_ylabel('S-N [m]')#, fontsize=12)
    ax.set_title(rf'{model_name}: DU Positions for Event {index} ( A {azimuth:.1f}°, Z {zenith:.1f}°), $\chi^2$={chi_square:.2f} at {event_time}')#, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True)

    # Determine save path
    filename = f"{save_name}_No.{index}_detector_positions_with_signal.png"
    if output_dir is not None:
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        save_path = outp / filename
    else:
        save_path = Path(filename)

    try:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved detector plot for SWM Event {index}: {save_path}")
        return str(save_path)
    except Exception:
        plt.close()
        logger.exception(f"Failed to save detector plot for event {index}")
        return None


def plot_signal_fit_with_signal(save_name: str, index: int, x_data, y_data, det_labels, event_time, output_dir=None):
    """
    Fit a Gaussian to (x_data, y_data) and plot the result with annotations.

    Returns saved file path (str) or (None, None) on failure. Also returns popt if fit succeeded.
    """
    try:
        from scipy.optimize import curve_fit

        def linear(x, m, b):
            return m * x + b

        x = np.asarray(x_data)
        y = np.asarray(y_data)
        if x.size == 0 or y.size == 0:
            return None

        # p0 = [float(np.max(y)), float(np.median(x)), float(np.std(x) if np.std(x) != 0 else 1.0)]
        popt, pcov = curve_fit(linear, x, y)
        logger.info(f"Linear fit params for event {index}: slope={popt[0]:.2f}, intercept={popt[1]:.1f}")
        # logger.info(f"Gaussian fit params for event {i}: center={popt[1]:.1f}m, width={popt[2]:.1f}m")

        # plt.figure(figsize=(20, 16))
        plt.figure()
        # print('det_labels:', det_labels)
        for j, label in enumerate(det_labels):
            # print(label, x[j], y[j])
            plt.text(x[j], y[j], str(label), ha='center',# fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

        x_fit = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(x_fit, linear(x_fit, *popt), 'r-', label=f'Linear fit (slope={popt[0]:.4f}, intercept={popt[1]:.1f})')
        plt.xlabel('Distance to Xmax [m]')#, fontsize=20)
        plt.ylabel('Signal [ADC]')#, fontsize=20)
        plt.title(f'Event {index} at {event_time}')#, fontsize=20)
        plt.xlim(np.min(x) * 0.99, np.max(x) * 1.01)
        plt.ylim(np.min(y) * 0.99, np.max(y) * 1.01)
        plt.legend()

        filename = f"{save_name}_No.{index}_signal_fit_with_signal.png"
        if output_dir is not None:
            outp = Path(output_dir)
            outp.mkdir(parents=True, exist_ok=True)
            save_path = outp / filename
        else:
            save_path = Path(filename)

        plt.savefig(str(save_path))
        plt.close()
        logger.info(f"Saved signal LDF fit plot for event {index}: {save_path}")
        return str(save_path)
    except Exception as e:
        logger.exception(f"Failed to fit/plot signal LDF for event {index}: {e}")
        try:
            plt.close()
        except Exception:
            pass
        return None

