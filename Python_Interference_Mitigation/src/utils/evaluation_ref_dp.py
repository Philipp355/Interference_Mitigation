import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
from src.data.data_utils import apply_rd_processing, apply_rd_processing_torch, calculate_detection_metrics, inverse_rd_processing, inverse_rd_processing_torch
from src.data.dataset import RadarDataset
from src.utils.visualization import plot_radar_result, plot_radar_result_rd



def evaluate_refdual_reconstruction(model, test_data, clean_data, frame_idx=0):
    """
    evaluate the reconstruction performance of the ref dp autoencoder
    """
    model.eval()

    dataset = RadarDataset(test_data, clean_data)
    interfered_frame = torch.FloatTensor(dataset[frame_idx][0]).unsqueeze(0)
    clean_frame = torch.FloatTensor(dataset[frame_idx][1]).unsqueeze(0)

    rd_maps_input = apply_rd_processing_torch(interfered_frame)
    rd_maps_target = apply_rd_processing_torch(clean_frame)

    with torch.no_grad():
        rd_maps_output = model(interfered_frame, rd_maps_input)
        
        rd_mse = torch.nn.functional.mse_loss(rd_maps_output, rd_maps_target).item()
        
        reconstructed_complex_rd = rd_maps_output[0, 0].cpu().numpy() + 1j * rd_maps_output[0, 1].cpu().numpy()
        target_complex_rd = rd_maps_target[0, 0].cpu().numpy() + 1j * rd_maps_target[0, 1].cpu().numpy()
        test_complex_rd = rd_maps_input[0, 0].cpu().numpy() + 1j * rd_maps_input[0, 1].cpu().numpy()

        mse = np.mean(np.abs(reconstructed_complex_rd - target_complex_rd) ** 2)

        phase_target = np.angle(target_complex_rd)
        phase_reconstructed = np.angle(reconstructed_complex_rd)
        
        amp_threshold = np.max(np.abs(target_complex_rd)) * 0.05
        significant_points = np.abs(target_complex_rd) > amp_threshold
        
        if np.sum(significant_points) > 0:
            phase_diff = np.abs(np.angle(np.exp(1j * (phase_reconstructed - phase_target))))
            mean_phase_error = np.mean(phase_diff[significant_points])
            mean_phase_error_deg = np.degrees(mean_phase_error)
            
            valid_target = target_complex_rd[significant_points]
            valid_reconstructed = reconstructed_complex_rd[significant_points]
            complex_correlation = np.abs(np.sum(valid_target * np.conj(valid_reconstructed))) / \
                                (np.sqrt(np.sum(np.abs(valid_target)**2) * np.sum(np.abs(valid_reconstructed)**2)))
            
            phase_preservation_score = 1 - (mean_phase_error / np.pi)
        else:
            mean_phase_error_deg = np.nan
            complex_correlation = np.nan
            phase_preservation_score = np.nan
        
    
        print("\nphase preservation:")
        print(f"MPE: {mean_phase_error_deg:.2f}°")
        print(f"complex correlation: {complex_correlation:.4f}")
        print(f"score: {phase_preservation_score:.4f}")

        max_val = np.max(np.abs(target_complex_rd))
        psnr = 20 * np.log10(max_val / np.sqrt(mse))


        interference_power_before = np.mean(np.abs(test_complex_rd - target_complex_rd) ** 2)
        interference_power_after = np.mean(np.abs(reconstructed_complex_rd - target_complex_rd) ** 2)
        clean_power = np.mean(np.abs(target_complex_rd) ** 2)

        sir_before = 10 * np.log10(clean_power / interference_power_before)
        sir_after = 10 * np.log10(clean_power / interference_power_after)
        sir_improvement = sir_after - sir_before

        output_mag = torch.sqrt(rd_maps_output[:, 0] ** 2 + rd_maps_output[:, 1] ** 2)
        target_mag = torch.sqrt(rd_maps_target[:, 0] ** 2 + rd_maps_target[:, 1] ** 2)
        
        
        
        v_res = 0.78  
        range_res = 0.25  
        
        plot_radar_result_rd(
            target_complex_rd,
            test_complex_rd,
            reconstructed_complex_rd,
            v_res=v_res,
            range_res=range_res
        )

        metrics = {
            'rd_mse': mse,
            'psnr': psnr,
            'sir_improvement': sir_improvement,
            'phase_error': mean_phase_error_deg,
            'phase_preservation_score': phase_preservation_score,
            'complex_correlation': complex_correlation
        }

        return metrics, reconstructed_complex_rd