import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.utils.visualization import plot_rd_map
import os
import sys
import logging  

from src.data.data_utils import apply_rd_processing


def analyze_rd_processing(radar_cube, clean_data, frame_idx=0):
    """
    Analyze Range-Doppler processing and visualize results
    """

    # DEBUG
    print("\nAnalyzing Range-Doppler Processing...")

    # # Process data
    # rd_clean = apply_rd_processing(clean_data, frame_idx=frame_idx)
    # rd_interference = apply_rd_processing(radar_cube, frame_idx=frame_idx)

    # # Calculate range and Doppler axes
    # num_samples, num_chirps = clean_data.shape[0:2]
    # range_res = 0.25
    # range_axis = np.arange(num_samples) * range_res  # Configured range resolution: 0.15m

    # v_res = 0.98  # Velocity resolution (m/s)
    # num_chirps = 128
    # # Calculate Doppler axis
    # max_velocity = v_res * (num_chirps / 2)  # Maximum unambiguous velocity
    # doppler_axis = np.linspace(-max_velocity, max_velocity, num_chirps)

    # # Create visualization
    # plt.figure(figsize=(15, 10))

    # Process data
    rd_clean = apply_rd_processing(clean_data)
    rd_interference = apply_rd_processing(radar_cube)

    # Calculate axes
    num_samples, num_chirps = clean_data.shape[0:2]
    range_res = 0.25  # range resolution in meters
    range_axis = np.arange(num_samples) * range_res
    velocity_res = 0.98  # velocity resolution in m/s
    doppler_axis = np.linspace(-velocity_res*num_chirps/2, 
                              velocity_res*num_chirps/2, 
                              num_chirps)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot RD maps
    plot_rd_map(rd_clean, doppler_axis, range_axis, 'Clean RD Map', 231)
    plot_rd_map(rd_interference, doppler_axis, range_axis, 'Interfered RD Map', 232)

    # Plot Doppler profiles
    plt.subplot(233)
    clean_doppler = np.max(np.abs(rd_clean), axis=0)
    interference_doppler = np.max(np.abs(rd_interference), axis=0)
    plt.plot(doppler_axis, 20 * np.log10(clean_doppler / np.max(clean_doppler)),
             label='Clean')
    plt.plot(doppler_axis, 20 * np.log10(interference_doppler / np.max(interference_doppler)),
             label='Interfered', alpha=0.7)
    plt.title('Doppler Profile')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-80, 0])

    # Plot Range profiles
    plt.subplot(234)
    clean_range = np.max(np.abs(rd_clean), axis=1)
    interference_range = np.max(np.abs(rd_interference), axis=1)
    plt.plot(range_axis, 20 * np.log10(clean_range / np.max(clean_range)),
             label='Clean')
    plt.plot(range_axis, 20 * np.log10(interference_range / np.max(interference_range)),
             label='Interfered', alpha=0.7)
    plt.title('Range Profile')
    plt.xlabel('Range (m)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-80, 0])

    # Add signal analysis
    plt.subplot(235)
    plt.plot(np.abs(clean_data[:, 0, frame_idx]), label='Clean')
    plt.plot(np.abs(radar_cube[:, 0, frame_idx]), label='Interfered', alpha=0.7)
    plt.title('First Chirp Signal')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()

    # Add phase analysis
    plt.subplot(236)
    plt.plot(np.angle(clean_data[:, 0, frame_idx]), label='Clean')
    plt.plot(np.angle(radar_cube[:, 0, frame_idx]), label='Interfered', alpha=0.7)
    plt.title('Phase Information')
    plt.xlabel('Sample')
    plt.ylabel('Phase (rad)')
    plt.grid(True)
    plt.legend()

    
    # Analyze peaks in Doppler profile
    clean_doppler = np.max(np.abs(rd_clean), axis=0)
    doppler_peaks, _ = find_peaks(clean_doppler, height=np.max(clean_doppler) * 0.1)
    
    # Analyze peaks in Range profile
    clean_range = np.max(np.abs(rd_clean), axis=1)
    range_peaks, _ = find_peaks(clean_range, height=np.max(clean_range) * 0.1)

    print("\nPeak Analysis:")
    print("Doppler peaks:")
    for peak in doppler_peaks:
        print(f"Velocity: {doppler_axis[peak]:.2f} m/s")
    
    print("\nRange peaks:")
    for peak in range_peaks:
        print(f"Range: {range_axis[peak]:.2f} m")

    # Optional: Add peak markers to profiles
    plt.subplot(233)  # Doppler profile
    plt.plot(doppler_axis[doppler_peaks], 
            20 * np.log10(clean_doppler[doppler_peaks] / np.max(clean_doppler)),
            'r^', label='Peaks')

    plt.subplot(234)  # Range profile
    plt.plot(range_axis[range_peaks], 
            20 * np.log10(clean_range[range_peaks] / np.max(clean_range)),
            'r^', label='Peaks')


    # # Print detailed analysis
    # print("\nSignal Analysis:")
    # print(
    #     f"Clean Signal Dynamic Range: {20 * np.log10(np.max(np.abs(rd_clean)) / np.min(np.abs(rd_clean)[np.abs(rd_clean) > 0])):.1f} dB")
    # print(
    #     f"Interfered Signal Dynamic Range: {20 * np.log10(np.max(np.abs(rd_interference)) / np.min(np.abs(rd_interference)[np.abs(rd_interference) > 0])):.1f} dB")

    # # Analyze peaks in Doppler profile
    # peaks, _ = find_peaks(clean_doppler, height=np.max(clean_doppler) * 0.1)
    # print(f"\nNumber of detected peaks in clean Doppler profile: {len(peaks)}")
    # if len(peaks) > 0:
    #     print("Peak velocities:", doppler_axis[peaks])
    
    plt.tight_layout()
    plt.show()

    
def analyze_scenario_data(radar_cube, clean_data, scenario_info):
    """
    Analyze and print detailed information about the scenarios
    """
    print("\nData Analysis Summary:")
    print(f"Total number of scenarios: {len(scenario_info)}")
    print(f"Total number of frames: {radar_cube.shape[2]}")
    print(f"Data dimensions: {radar_cube.shape}")

    # Calculate average signal and interference power per scenario
    total_frames = 0
    for info in scenario_info:
        start_frame = total_frames
        end_frame = total_frames + info['num_frames']

        scenario_radar = radar_cube[:, :, start_frame:end_frame]
        scenario_clean = clean_data[:, :, start_frame:end_frame]

        # Calculate scenario metrics
        interference_power = np.mean(np.abs(scenario_radar - scenario_clean) ** 2)
        clean_power = np.mean(np.abs(scenario_clean) ** 2)
        sir = 10 * np.log10(clean_power / interference_power)

        print(f"\nScenario {info['scenario_id']}:")
        print(f"Frames: {info['num_frames']}")
        print(f"Input SIR: {sir:.2f} dB")

        if 'target_info' in info:
            print("Targets:")
            print(info['target_info'])

        total_frames += info['num_frames']





def compare_metrics(train_metrics, test_metrics):
    """
    Compare and display metrics between training and test datasets with improved formatting

    Args:
        train_metrics: Dictionary containing training metrics
        test_metrics: Dictionary containing test metrics
    """
    # Define metrics to compare with their display names and formats
    metrics_config = {
        'mse': {'name': 'Mean Square Error', 'format': '.2e'},
        'psnr': {'name': 'PSNR', 'format': '.2f', 'unit': 'dB'},
        'sir_improvement': {'name': 'SIR Improvement', 'format': '.2f', 'unit': 'dB'},
        'detection_probability': {'name': 'Detection Probability', 'format': '.3f', 'unit': '%'},
        'false_alarm_rate': {'name': 'False Alarm Rate', 'format': '.3f', 'unit': '%'},
        'true_positives': {'name': 'True Positives', 'format': '.0f'},
        'false_positives': {'name': 'False Positives', 'format': '.0f'},
        'false_negatives': {'name': 'False Negatives', 'format': '.0f'}
    }

    print("\nPerformance Metrics Comparison:")
    print("-" * 80)
    header = f"{'Metric':<35} {'Training':<15} {'Test':<15} {'Difference':<15}"
    print(header)
    print("-" * 80)

    for metric, config in metrics_config.items():
        if metric not in train_metrics or metric not in test_metrics:
            continue

        train_value = train_metrics[metric]
        test_value = test_metrics[metric]
        diff = test_value - train_value

        # Format metric name with unit if present
        metric_name = config['name']
        if 'unit' in config:
            metric_name = f"{metric_name} ({config['unit']})"

        # Format values according to their type
        format_str = "{:" + config['format'] + "}"
        train_str = format_str.format(train_value)
        test_str = format_str.format(test_value)
        diff_str = format_str.format(diff)
        if diff > 0:
            diff_str = f"+{diff_str}"

        # Print formatted line
        print(f"{metric_name:<35} {train_str:<15} {test_str:<15} {diff_str:<15}")

    print("-" * 80)

    # Additional statistics section if available
    if all(key in train_metrics for key in ['true_positives', 'false_positives', 'false_negatives']):
        print("\nDetection Statistics:")
        print("-" * 80)
        
        for dataset, metrics in [("Training", train_metrics), ("Test", test_metrics)]:
            tp = metrics['true_positives']
            fp = metrics['false_positives']
            fn = metrics['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{dataset} Set Additional Metrics:")
            print(f"{'Precision':<35} {precision:.3f}")
            print(f"{'Recall':<35} {recall:.3f}")
            print(f"{'F1-Score':<35} {f1_score:.3f}")
            print("-" * 80)


