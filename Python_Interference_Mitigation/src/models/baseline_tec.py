import numpy as np
from scipy import signal


def basic_zeroing(chirp_signal, threshold):

    cleaned_signal = chirp_signal.copy()
    
    interference_mask = np.abs(chirp_signal) > threshold
    
    cleaned_signal[interference_mask] = 0
    
    return cleaned_signal


def fft_based_zeroing(chirp_signal, threshold):

    fft_signal = np.fft.fft(chirp_signal)
    
    interference_mask = np.abs(fft_signal) > threshold
    
    fft_clean = fft_signal.copy()
    fft_clean[interference_mask] = 0
    
    cleaned_signal = np.fft.ifft(fft_clean)
    
    return cleaned_signal




def phase_corrected_stft_zeroing(chirp_signal, factor=2.5, fs=20e6, nperseg=256, noverlap=192):

    signal_mean = np.mean(np.abs(chirp_signal))
    signal_std = np.std(np.abs(chirp_signal))
    extreme_threshold = signal_mean + 5.0 * signal_std
    
    chirp_signal_preprocessed = chirp_signal.copy()
    extreme_mask = np.abs(chirp_signal) > extreme_threshold
    if np.any(extreme_mask):
        phase = np.angle(chirp_signal_preprocessed[extreme_mask])
        chirp_signal_preprocessed[extreme_mask] = signal_mean * np.exp(1j * phase)
    
    # 计算 STFT
    f, t, Zxx = signal.stft(chirp_signal_preprocessed, fs=fs, window='hamming', 
                           nperseg=nperseg, noverlap=noverlap)

    freq_bins = 4  
    freq_per_bin = len(f) // freq_bins if freq_bins > 0 else len(f)
    
    cleaned_Zxx = Zxx.copy()
    
    for i in range(freq_bins):
        start_idx = i * freq_per_bin
        end_idx = (i+1) * freq_per_bin if i < freq_bins-1 else len(f)
        
        region_magnitude = np.abs(Zxx[start_idx:end_idx, :])
        region_mean = np.mean(region_magnitude, axis=1, keepdims=True)
        region_std = np.std(region_magnitude, axis=1, keepdims=True)
        
        region_threshold = region_mean + factor * region_std
        region_mask = region_magnitude > region_threshold
        
        if np.any(region_mask):
            interference_strength = region_magnitude[region_mask] / region_threshold[region_mask]
            min_attenuation = 0.05  
            adaptive_attenuation = np.clip(1.0 / (1.0 + interference_strength), min_attenuation, 0.3)
            
            mask_indices = np.where(region_mask)
            for j in range(len(mask_indices[0])):
                row, col = mask_indices[0][j], mask_indices[1][j]
                orig_phase = np.angle(Zxx[start_idx + row, col])
                orig_mag = np.abs(Zxx[start_idx + row, col])
                attenuated_mag = orig_mag * adaptive_attenuation[j]
                cleaned_Zxx[start_idx + row, col] = attenuated_mag * np.exp(1j * orig_phase)
    
    _, cleaned_signal = signal.istft(cleaned_Zxx, fs=fs, window='hamming', 
                                    nperseg=nperseg, noverlap=noverlap)
    
    if len(cleaned_signal) < len(chirp_signal):
        cleaned_signal = np.pad(cleaned_signal, (0, len(chirp_signal) - len(cleaned_signal)))
    elif len(cleaned_signal) > len(chirp_signal):
        cleaned_signal = cleaned_signal[:len(chirp_signal)]
    
    return cleaned_signal



def improved_apply_zeroing(radar_data, method='phase_corrected', time_factor=3.0, freq_factor=2.5, fs=20e6, nperseg=256, noverlap=192):



    frames, chirps, samples = radar_data.shape
    cleaned_data = np.zeros_like(radar_data, dtype=complex if np.iscomplexobj(radar_data) else float)
    

    total_items = frames * chirps
    processed_items = 0
    print_interval = max(1, total_items // 10)  
    

    for f in range(frames):
        for c in range(chirps):
            if method == 'phase_corrected':
                cleaned_data[f, c] = phase_corrected_stft_zeroing(
                    radar_data[f, c], 
                    factor=freq_factor, 
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
            else:
                cleaned_data[f, c] = phase_corrected_stft_zeroing(
                    radar_data[f, c], 
                    factor=freq_factor, 
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
            

            processed_items += 1
            if processed_items % print_interval == 0 or processed_items == total_items:
                progress = (processed_items / total_items) * 100
                print(f"progress: {progress:.1f}% ({processed_items}/{total_items})")
    
    print("done!")
    return cleaned_data


