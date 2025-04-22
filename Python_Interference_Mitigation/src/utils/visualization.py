import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.data.data_utils import apply_rd_processing, apply_rd_processing_torch, apply_stft_processing, inverse_rd_processing, inverse_rd_processing_torch


def plot_training_history(history):
    """Enhanced training history visualization"""
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 2, 2)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    # Plot loss in log scale
    plt.subplot(2, 2, 3)
    plt.semilogy(history['train_loss'], label='Training Loss')
    plt.semilogy(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epoch (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)

    # Plot loss difference
    plt.subplot(2, 2, 4)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    plt.plot(loss_diff, label='Val Loss - Train Loss')
    plt.title('Loss Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Validation - Training Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    plt.show()



def plot_rd_map(rd_data, doppler_axis, range_axis, title, subplot_pos):
   """
   Common RD plot function for consistent visualization
   
   Args:
       rd_data: Range-Doppler map data
       doppler_axis: Velocity axis values
       range_axis: Range axis values
       title: Plot title
       subplot_pos: Subplot position
   """
   plt.subplot(subplot_pos)
   
   # Normalize and convert to dB
   rd_db = 20 * np.log10(np.abs(rd_data) / np.max(np.abs(rd_data)))
   
   # Apply better visualization settings
   plt.imshow(rd_db,
              aspect='auto',
              extent=[doppler_axis[0], doppler_axis[-1], 
                     range_axis[0], range_axis[-1]],
              origin='lower',  # Set origin to lower left
              cmap='viridis',
              interpolation='bilinear')
   
   # Add debug info
   if False:  # Set to True when debugging
       print(f"\nRD Map Debug Info for {title}:")
       print(f"RD shape: {rd_data.shape}")
       print(f"Range axis: {range_axis[0]:.1f} to {range_axis[-1]:.1f}")
       print(f"Doppler axis: {doppler_axis[0]:.1f} to {doppler_axis[-1]:.1f}")
       
       # Check for peaks
       from scipy.signal import find_peaks
       rd_profile = np.max(np.abs(rd_data), axis=1)
       peaks, _ = find_peaks(rd_profile, height=np.max(rd_profile)*0.1)
       print(f"Range peaks at: {range_axis[peaks]}")
   
   plt.title(title)
   plt.xlabel('Velocity (m/s)')
   plt.ylabel('Range (m)')
   cbar = plt.colorbar(label='dB')
   plt.clim([-60, 0])
   
   return rd_db  # Return normalized data for potential analysis


def plot_radar_result(clean_data, interfered_data, reconstructed_data, save_path=0, frame_idx=0, dynamic_range=80):
    """
    Plot RD maps, profiles and complex signals in 2x3 layout
    """
    # Process all three versions using same function
    rd_clean = apply_rd_processing(clean_data, frame_idx=frame_idx)
    rd_interference = apply_rd_processing(interfered_data, frame_idx=frame_idx)
    rd_reconstructed = apply_rd_processing(reconstructed_data, frame_idx=frame_idx)

    # Calculate range and Doppler axes
    num_samples, num_chirps = clean_data.shape[0:2]
    print(f'number of samples: {num_samples}')
    print(f'number of chirps: {num_chirps}')

    # Calculate range and Doppler axes
    num_samples, num_chirps = clean_data.shape[0:2]
    range_res = 0.25
    range_axis = np.arange(num_samples) * range_res  # Configured range resolution: 0.15m

    v_res = 0.98  # Velocity resolution (m/s)
    num_chirps = 128
    # Calculate Doppler axis
    max_velocity = v_res * (num_chirps / 2)  # Maximum unambiguous velocity
    doppler_axis = np.linspace(-max_velocity, max_velocity, num_chirps)

    # Create figure with adjusted layout
    plt.figure(figsize=(15, 10))

    # First row: 3 RD maps
    # Plot RD maps using common function
    plot_rd_map(rd_clean, doppler_axis, range_axis, 'Clean RD Map', 231)
    plot_rd_map(rd_interference, doppler_axis, range_axis, 'Interfered RD Map', 232)
    plot_rd_map(rd_reconstructed, doppler_axis, range_axis, 'Reconstructed RD Map', 233)

    # Second row: Range and Doppler profiles
    # Plot Doppler profiles
    plt.subplot(234)
    clean_doppler = np.max(np.abs(rd_clean), axis=0)
    interference_doppler = np.max(np.abs(rd_interference), axis=0)
    reconstructed_doppler = np.max(np.abs(rd_reconstructed), axis=0)

    plt.plot(doppler_axis, 20 * np.log10(clean_doppler / np.max(clean_doppler)),
             label='Clean')
    plt.plot(doppler_axis, 20 * np.log10(interference_doppler / np.max(interference_doppler)),
             label='Interfered', alpha=0.7)
    plt.plot(doppler_axis, 20 * np.log10(reconstructed_doppler / np.max(reconstructed_doppler)),
             label='Reconstructed', alpha=0.7)
    plt.title('Doppler Profile')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-dynamic_range, 0])

    # Plot Range profiles
    plt.subplot(235)
    clean_range = np.max(np.abs(rd_clean), axis=1)
    interference_range = np.max(np.abs(rd_interference), axis=1)
    reconstructed_range = np.max(np.abs(rd_reconstructed), axis=1)

    plt.plot(range_axis, 20 * np.log10(clean_range / np.max(clean_range)),
             label='Clean')
    plt.plot(range_axis, 20 * np.log10(interference_range / np.max(interference_range)),
             label='Interfered', alpha=0.7)
    plt.plot(range_axis, 20 * np.log10(reconstructed_range / np.max(reconstructed_range)),
             label='Reconstructed', alpha=0.7)
    plt.title('Range Profile')
    plt.xlabel('Range (m)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-dynamic_range, 0])

    # Third row: Real and Imaginary signal parts
    # Plot first chirp signals for real part
    plt.subplot(236)
    plt.plot(np.abs(clean_data[:, 0, frame_idx]), label='Clean')
    plt.plot(np.abs(interfered_data[:, 0, frame_idx]), label='Interfered', alpha=0.7)
    # plt.plot(np.abs(reconstructed_data[:, 0]), label='Reconstructed', alpha=0.7)
    plt.plot(np.abs(reconstructed_data[:, 0, frame_idx]), label='Reconstructed', alpha=0.7)
    plt.title('Real Part of First Chirp Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # # Imaginary part
    # plt.subplot(337)
    # plt.plot(clean_data[:, 1, frame_idx], label='Clean')
    # plt.plot(interfered_data[:, 1, frame_idx], label='Interfered', alpha=0.7)
    # plt.plot(reconstructed_data[:, 1, frame_idx], label='Reconstructed', alpha=0.7)
    # plt.title('Imaginary Part of First Chirp')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.legend()

    # # Threshold value: original value 0.1, adjusted to 0.3 for tighter constraint
    # Analyze peaks in Doppler profile
    reconstructed_doppler = np.max(np.abs(rd_reconstructed), axis=0)
    doppler_peaks, _ = find_peaks(reconstructed_doppler, height=np.max(reconstructed_doppler) * 0.2)
    
    # Analyze peaks in Range profile
    reconstructed_range = np.max(np.abs(rd_reconstructed), axis=1)
    range_peaks, _ = find_peaks(reconstructed_range, height=np.max(reconstructed_range) * 0.2)

    print("\nPeak Analysis:")
    print("Doppler peaks:")
    for peak in doppler_peaks:
        print(f"Velocity: {doppler_axis[peak]:.2f} m/s")
    
    print("\nRange peaks:")
    for peak in range_peaks:
        print(f"Range: {range_axis[peak]:.2f} m")

    # Optional: Add peak markers to profiles
    plt.subplot(234)  # Doppler profile
    plt.plot(doppler_axis[doppler_peaks], 
            20 * np.log10(reconstructed_doppler[doppler_peaks] / np.max(reconstructed_doppler)),
            'r^', label='Peaks')

    plt.subplot(235)  # Range profile
    plt.plot(range_axis[range_peaks], 
            20 * np.log10(reconstructed_range[range_peaks] / np.max(reconstructed_range)),
            'r^', label='Peaks')

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Figure saved to {save_path}")

    plt.show()


def plot_radar_result_rd(rd_clean, rd_interference, rd_reconstructed, v_res=0.98, range_res=0.25, save_path=0, dynamic_range=80):
    """
    Plot RD maps and profiles for data already in RD domain

    """
    num_samples, num_chirps = rd_clean.shape
    print(f'RD domain: range bin: {num_samples}, doppler bins: {num_chirps}')
    
    range_axis = np.arange(num_samples) * range_res
    
    max_velocity = v_res * (num_chirps / 2)  # max unambigious v
    doppler_axis = np.linspace(-max_velocity, max_velocity, num_chirps)
    
    plt.figure(figsize=(15, 10))
    
    # first rd map
    plot_rd_map(rd_clean, doppler_axis, range_axis, 'Clean RD Map', 231)
    plot_rd_map(rd_interference, doppler_axis, range_axis, 'Interfered RD Map', 232)
    plot_rd_map(rd_reconstructed, doppler_axis, range_axis, 'Reconstructed RD Map', 233)
    
    # doppler profile
    plt.subplot(234)
    clean_doppler = np.max(np.abs(rd_clean), axis=0)
    interference_doppler = np.max(np.abs(rd_interference), axis=0)
    reconstructed_doppler = np.max(np.abs(rd_reconstructed), axis=0)
    
    plt.plot(doppler_axis, 20 * np.log10(clean_doppler / np.max(clean_doppler)),
             label='Clean')
    plt.plot(doppler_axis, 20 * np.log10(interference_doppler / np.max(interference_doppler)),
             label='Interfered', alpha=0.7)
    plt.plot(doppler_axis, 20 * np.log10(reconstructed_doppler / np.max(reconstructed_doppler)),
             label='Reconstructed', alpha=0.7)
    plt.title('Doppler Profile')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-dynamic_range, 0])
    
    # range profile
    plt.subplot(235)
    clean_range = np.max(np.abs(rd_clean), axis=1)
    interference_range = np.max(np.abs(rd_interference), axis=1)
    reconstructed_range = np.max(np.abs(rd_reconstructed), axis=1)
    
    plt.plot(range_axis, 20 * np.log10(clean_range / np.max(clean_range)),
             label='Clean')
    plt.plot(range_axis, 20 * np.log10(interference_range / np.max(interference_range)),
             label='Interfered', alpha=0.7)
    plt.plot(range_axis, 20 * np.log10(reconstructed_range / np.max(reconstructed_range)),
             label='Reconstructed', alpha=0.7)
    plt.title('Range Profile')
    plt.xlabel('Range (m)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim([-dynamic_range, 0])

    # time domain signal
    plt.subplot(236)
    
    # from rd back to time
    time_clean = inverse_rd_processing(rd_clean)
    time_interference = inverse_rd_processing(rd_interference)
    time_reconstructed = inverse_rd_processing(rd_reconstructed)
    
    chirp_idx = 0
    plt.plot(np.abs(time_clean[:, chirp_idx]), label='Clean')
    plt.plot(np.abs(time_interference[:, chirp_idx]), label='Interfered', alpha=0.7)
    plt.plot(np.abs(time_reconstructed[:, chirp_idx]), label='Reconstructed', alpha=0.7)
    
    plt.title('Time Domain Signal (First Chirp)')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    
    # peak analysis
    reconstructed_doppler = np.max(np.abs(rd_reconstructed), axis=0)
    doppler_peaks, _ = find_peaks(reconstructed_doppler, height=np.max(reconstructed_doppler) * 0.2)
    
    reconstructed_range = np.max(np.abs(rd_reconstructed), axis=1)
    range_peaks, _ = find_peaks(reconstructed_range, height=np.max(reconstructed_range) * 0.2)
    
    print("\nPeak Analysis:")
    print("Doppler Peaks:")
    for peak in doppler_peaks:
        print(f"Velocity: {doppler_axis[peak]:.2f} m/s")
    
    print("\nRang Peaks:")
    for peak in range_peaks:
        print(f"Range: {range_axis[peak]:.2f} m")
    
    # peak markers
    plt.subplot(234) 
    plt.plot(doppler_axis[doppler_peaks], 
            20 * np.log10(reconstructed_doppler[doppler_peaks] / np.max(reconstructed_doppler)),
            'r^', label='Peaks')
    
    plt.subplot(235) 
    plt.plot(range_axis[range_peaks], 
            20 * np.log10(reconstructed_range[range_peaks] / np.max(reconstructed_range)),
            'r^', label='Peaks')
    
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Figures saved to {save_path}")
    
    # plt.show()
    
    return {
        'doppler_peaks': doppler_axis[doppler_peaks],
        'range_peaks': range_axis[range_peaks]
    }



def find_eer(fpr, tpr):
    """calculate equal error rate (EER)"""
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[eer_idx] + fnr[eer_idx]) / 2



def plot_stft_result(f, t, Zxx, bandwidth, title="STFT Time-Frequency Representation"):

    print("\nplot stft result v3")
    c = 3e8  
    range_res = c / (2 * bandwidth) 
    range_axis = f * range_res  # Convert frequency to distance

    magnitude = np.abs(Zxx)
    
    # Plot the magnitude as a spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, range_axis, 20 * np.log10(magnitude), shading='gouraud', cmap='jet')
    plt.colorbar(label='Magnitude (dB)')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.grid(True)
    plt.show()
