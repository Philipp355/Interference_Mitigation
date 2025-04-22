import numpy as np
import mat73
from scipy.signal import decimate
from scipy.signal import find_peaks
import torch
from scipy.signal import stft


def load_multiple_scenarios(file_paths):
    """
    Load multiple scenario data files and return concatenated data
    """
    all_radar_cubes = []
    all_clean_data = []
    scenario_info = []

    for idx, file_path in enumerate(file_paths):
        try:
            data = mat73.loadmat(file_path)
            radar_cube = data['radar_cube']
            clean_data = data['clean_signals']

            # Store scenario information if available in the mat file
            info = {
                'scenario_id': idx + 1,
                'num_frames': radar_cube.shape[2],
                'shape': radar_cube.shape,
            }

            # If additional info is provided in the mat file
            if 'target_info' in data:
                info['target_info'] = data['target_info']

            scenario_info.append(info)
            all_radar_cubes.append(radar_cube)
            all_clean_data.append(clean_data)

            print(f"\nScenario {idx + 1} loaded:")
            print(f"Shape: {radar_cube.shape}")
            print(f"Number of frames: {radar_cube.shape[2]}")
            if 'target_info' in data:
                print("Target Information:")
                print(data['target_info'])

        except Exception as e:
            print(f"Error loading scenario {idx + 1}: {str(e)}")

    # check the structure of scenario info
    print(scenario_info)

    # Concatenate all scenarios along the frame dimension
    combined_radar = np.concatenate(all_radar_cubes, axis=2)
    combined_clean = np.concatenate(all_clean_data, axis=2)

    return combined_radar, combined_clean, scenario_info




def apply_rd_processing(data, frame_idx=0, window=True):
    if data.ndim == 3:
        frame = data[:, :, frame_idx]
    else:
        frame = data
    
    # window func
    if window:
        range_window = np.hamming(frame.shape[0])[:, np.newaxis]
        doppler_window = np.hamming(frame.shape[1])[np.newaxis, :]
        frame = frame * range_window * doppler_window
    
    # range fft without shifting
    range_fft = np.fft.fft(frame, axis=0)
    
    # center 0 speed
    rd_map = np.zeros_like(range_fft, dtype=complex)
    for i in range(frame.shape[0]):
        rd_map[i,:] = np.fft.fftshift(np.fft.fft(range_fft[i,:]))
    
    return rd_map



def apply_rd_processing_torch(data, window=True):
    """
    transform signal into rd domain as mini batch with tensor
    """
    batch_size, _, samples, chirps = data.shape
    
    # chnage to complex data
    complex_data = data[:, 0, :, :] + 1j * data[:, 1, :, :]
    
    if window:
        range_window = torch.hamming_window(samples, device=data.device)
        doppler_window = torch.hamming_window(chirps, device=data.device)
        
        # range_window = range_window.view(1, -1, 1)     # [1, samples, 1]
        # doppler_window = doppler_window.view(1, 1, -1)  # [1, 1, chirps]

        range_window = range_window.view(-1, 1)     # [samples, 1]
        doppler_window = doppler_window.view(1, -1)  # [1, chirps]
                
        complex_data = complex_data * (range_window * doppler_window)
    
    range_fft = torch.fft.fft(complex_data, dim=1)  # along samples
    
    rd_maps = torch.fft.fft(range_fft, dim=2)  
    rd_maps = torch.fft.fftshift(rd_maps, dim=2)  
    
    # seperate real and imaginary parts
    rd_real = torch.real(rd_maps)
    rd_imag = torch.imag(rd_maps)
    
    return torch.stack([rd_real, rd_imag], dim=1)



def inverse_rd_processing(rd_map, window=True):
    """
    transform rd domain signal back to time domain
    """
    samples, chirps = rd_map.shape
    
    # inverse shift to recover original FFT output format
    ifftshift_rd_map = np.zeros_like(rd_map, dtype=complex)
    for i in range(samples):
        ifftshift_rd_map[i,:] = np.fft.ifftshift(rd_map[i,:])
    
    # inverse FFT along chirps dimension
    range_fft = np.zeros_like(ifftshift_rd_map, dtype=complex)
    for i in range(samples):
        range_fft[i,:] = np.fft.ifft(ifftshift_rd_map[i,:])
    
    # inverse FFT along samples dimension
    time_signal = np.fft.ifft(range_fft, axis=0)
    
    # compensate for window function if applied
    if window:
        range_window = np.hamming(samples)[:, np.newaxis]
        doppler_window = np.hamming(chirps)[np.newaxis, :]
        
        window_weights = range_window * doppler_window
        threshold = 1e-10 
        window_weights = np.where(window_weights < threshold, threshold, window_weights)
        
        time_signal = time_signal / window_weights
    
    return time_signal



def inverse_rd_processing_torch(rd_maps, window=True):
    """
    PyTorch implementation of inverse RD processing
    """
    batch_size, _, samples, chirps = rd_maps.shape
    
    rd_complex = rd_maps[:, 0, :, :] + 1j * rd_maps[:, 1, :, :]
    
    rd_complex = torch.fft.ifftshift(rd_complex, dim=2)  
    
    range_fft = torch.fft.ifft(rd_complex, dim=2)  
    
    time_signal = torch.fft.ifft(range_fft, dim=1) 
    
    if window:
        range_window = torch.hamming_window(samples, device=rd_maps.device)
        doppler_window = torch.hamming_window(chirps, device=rd_maps.device)
        
        range_window = range_window.view(-1, 1)     # [samples, 1]
        doppler_window = doppler_window.view(1, -1)  # [1, chirps]
        
        window_weights = range_window * doppler_window
        threshold = 1e-10  
        window_weights = torch.where(window_weights < threshold, 
                                     torch.ones_like(window_weights) * threshold, 
                                     window_weights)
        
        # compensate for window function
        time_signal = time_signal / window_weights
    

    time_real = torch.real(time_signal)
    time_imag = torch.imag(time_signal)
    
    return torch.stack([time_real, time_imag], dim=1)



def calculate_detection_metrics(rd_clean, rd_reconstructed, threshold_factor=0.1):
    """
    Calculate detection metrics from RD maps
    """
    clean_doppler = np.max(np.abs(rd_clean), axis=0)
    reconstructed_doppler = np.max(np.abs(rd_reconstructed), axis=0)

    clean_doppler = clean_doppler / np.max(clean_doppler)
    reconstructed_doppler = reconstructed_doppler / np.max(reconstructed_doppler)

    # find peaks with normalized threshold
    doppler_peaks_clean, _ = find_peaks(clean_doppler, height=threshold_factor)
    doppler_peaks_reconstructed, _ = find_peaks(reconstructed_doppler, height=threshold_factor)

    clean_range = np.max(np.abs(rd_clean), axis=1)
    reconstructed_range = np.max(np.abs(rd_reconstructed), axis=1)

    # normalize profiles
    clean_range = clean_range / np.max(clean_range)
    reconstructed_range = reconstructed_range / np.max(reconstructed_range)

    # find peaks
    range_peaks_clean, _ = find_peaks(clean_range, height=threshold_factor)
    range_peaks_reconstructed, _ = find_peaks(reconstructed_range, height=threshold_factor)

    # calculate metrics
    detection_metrics = {
        'doppler': {
            'detection_probability': 0,
            'false_alarm_rate': 0,
            'clean_peaks': doppler_peaks_clean.tolist(),
            'reconstructed_peaks': doppler_peaks_reconstructed.tolist(),
            'num_clean_peaks': len(doppler_peaks_clean),
            'num_reconstructed_peaks': len(doppler_peaks_reconstructed)
        },
        'range': {
            'detection_probability': 0,
            'false_alarm_rate': 0,
            'clean_peaks': range_peaks_clean.tolist(),
            'reconstructed_peaks': range_peaks_reconstructed.tolist(),
            'num_clean_peaks': len(range_peaks_clean),
            'num_reconstructed_peaks': len(range_peaks_reconstructed)
        }
    }

    # Calculate Doppler metrics
    if len(doppler_peaks_clean) > 0:
        doppler_true_positives = len(set(doppler_peaks_clean) & set(doppler_peaks_reconstructed))
        detection_metrics['doppler']['detection_probability'] = doppler_true_positives / len(doppler_peaks_clean)

    if len(doppler_peaks_reconstructed) > 0:
        doppler_false_positives = len(set(doppler_peaks_reconstructed) - set(doppler_peaks_clean))
        detection_metrics['doppler']['false_alarm_rate'] = doppler_false_positives / len(doppler_peaks_reconstructed)

    # Calculate Range metrics
    if len(range_peaks_clean) > 0:
        range_true_positives = len(set(range_peaks_clean) & set(range_peaks_reconstructed))
        detection_metrics['range']['detection_probability'] = range_true_positives / len(range_peaks_clean)

    if len(range_peaks_reconstructed) > 0:
        range_false_positives = len(set(range_peaks_reconstructed) - set(range_peaks_clean))
        detection_metrics['range']['false_alarm_rate'] = range_false_positives / len(range_peaks_reconstructed)

    return detection_metrics






def apply_stft_processing(data, frame_idx=0, chirp_idx=0, fs=40e6, nperseg=128, noverlap=64, window=True):
    """
    Apply STFT to a single chirp of radar data along the range dimension
    
    f (numpy.ndarray): Array of sample frequencies
    t (numpy.ndarray): Array of segment times
    Zxx (numpy.ndarray): STFT result (complex values)
    """
    # Extract the specific chirp and frame
    if data.ndim == 3:
        chirp_data = data[:, chirp_idx, frame_idx]
    else:
        raise ValueError("Input data must have 3 dimensions: [samples, chirps, frames]")

    # windowing
    if window:
        chirp_data = chirp_data * np.hamming(chirp_data.shape[0])

    # STFT along range dimension (axis 0)
    f, t, Zxx = stft(chirp_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    return f, t, Zxx

