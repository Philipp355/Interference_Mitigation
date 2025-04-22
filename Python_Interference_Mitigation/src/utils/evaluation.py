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


def evaluate_reconstruction(model, test_data, clean_data, frame_idx=0):
    model.eval()

    dataset = RadarDataset(test_data, clean_data)
    interfered_frame = torch.FloatTensor(dataset[frame_idx][0]).unsqueeze(0)
    clean_frame = torch.FloatTensor(dataset[frame_idx][1]).unsqueeze(0)

    # Calculate RD maps
    rd_maps_input = apply_rd_processing_torch(interfered_frame)
    rd_maps_target = apply_rd_processing_torch(clean_frame)

    with torch.no_grad():
        output = model(interfered_frame, rd_maps_input)
        rd_maps_output = apply_rd_processing_torch(output)

        # Convert to complex format for traditional metrics
        reconstructed_complex = output[0, 0, :, :].cpu().numpy() + 1j * output[0, 1, :, :].cpu().numpy()
        clean_frame_complex = clean_data[:, :, frame_idx]
        test_frame_complex = test_data[:, :, frame_idx]

        mse = np.mean(np.abs(reconstructed_complex - clean_frame_complex) ** 2)

        max_val = np.max(np.abs(clean_frame_complex))
        psnr = 20 * np.log10(max_val / np.sqrt(mse))

        interference_power_before = np.mean(np.abs(test_frame_complex - clean_frame_complex) ** 2)
        interference_power_after = np.mean(np.abs(reconstructed_complex - clean_frame_complex) ** 2)
        clean_power = np.mean(np.abs(clean_frame_complex) ** 2)

        sir_before = 10 * np.log10(clean_power / interference_power_before)
        sir_after = 10 * np.log10(clean_power / interference_power_after)
        sir_improvement = sir_after - sir_before

        # Detection metrics calculation (your existing code)
        output_mag = torch.sqrt(rd_maps_output[:, 0] ** 2 + rd_maps_output[:, 1] ** 2)
        target_mag = torch.sqrt(rd_maps_target[:, 0] ** 2 + rd_maps_target[:, 1] ** 2)

        threshold_db = -20
        output_db = 20 * torch.log10(output_mag / torch.max(output_mag))
        target_db = 20 * torch.log10(target_mag / torch.max(target_mag))

        output_detections = output_db > threshold_db
        target_detections = target_db > threshold_db

        true_positives = torch.sum(output_detections & target_detections).item()
        false_positives = torch.sum(output_detections & ~target_detections).item()
        false_negatives = torch.sum(~output_detections & target_detections).item()

        detection_prob = true_positives / (true_positives + false_negatives + 1e-10)
        false_alarm_rate = false_positives / (torch.sum(output_detections).item() + 1e-10)


        # Print all metrics
        print("\nPerformance Metrics:")
        print(f"MSE: {mse:.2e}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SIR Improvement: {sir_improvement:.2f} dB")
        print("\nDetection Performance:")
        print(f"Detection Probability: {detection_prob:.3f}")
        print(f"False Alarm Rate: {false_alarm_rate:.3f}")


        # Create reconstructed cube with same dimensions as input
        reconstructed_cube = np.zeros_like(test_data)
        reconstructed_complex = output[0, 0, :, :].cpu().numpy() + 1j * output[0, 1, :, :].cpu().numpy()
        reconstructed_cube[:, :, frame_idx] = reconstructed_complex

        # Plot results
        plot_radar_result(clean_data, test_data, reconstructed_cube)
        # plot_radar_result(clean_frame_complex, test_frame_complex, reconstructed_complex)


        metrics = {
            'mse': mse,
            'psnr': psnr,
            'sir_improvement': sir_improvement,
            'detection_probability': detection_prob,
            'false_alarm_rate': false_alarm_rate,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
        }

        return metrics, reconstructed_complex


def evaluate_ref_reconstruction(model, test_data, clean_data, frame_idx=0):
    """
    evaluate reconstruction of ref_autoencoder
    """
    model.eval()

    # single frame
    dataset = RadarDataset(test_data, clean_data)
    interfered_frame = torch.FloatTensor(dataset[frame_idx][0]).unsqueeze(0)
    clean_frame = torch.FloatTensor(dataset[frame_idx][1]).unsqueeze(0)

    # convert to rd maps
    rd_maps_input = apply_rd_processing_torch(interfered_frame)
    rd_maps_target = apply_rd_processing_torch(clean_frame)

    with torch.no_grad():
        rd_maps_output = model(rd_maps_input)

        rd_mse = torch.nn.functional.mse_loss(rd_maps_output, rd_maps_target).item()
        

        reconstructed_complex_rd = rd_maps_output[0, 0].cpu().numpy() + 1j * rd_maps_output[0, 1].cpu().numpy()
        target_complex_rd = rd_maps_target[0, 0].cpu().numpy() + 1j * rd_maps_target[0, 1].cpu().numpy()
        test_complex_rd = rd_maps_input[0, 0].cpu().numpy() + 1j * rd_maps_input[0, 1].cpu().numpy()

        mse = np.mean(np.abs(reconstructed_complex_rd - target_complex_rd) ** 2)

        ####################################################################
        # calculate phase preservation metrics
        phase_target = np.angle(target_complex_rd)
        phase_reconstructed = np.angle(reconstructed_complex_rd)
        
        # phase preservation metrics
        amp_threshold = np.max(np.abs(target_complex_rd)) * 0.005  # max amplitude 5%
        significant_points = np.abs(target_complex_rd) > amp_threshold
        
        if np.sum(significant_points) > 0:
            # calculate periodic phase difference
            phase_diff = np.abs(np.angle(np.exp(1j * (phase_reconstructed - phase_target))))
            
            mean_phase_error = np.mean(phase_diff[significant_points])
            mean_phase_error_deg = np.degrees(mean_phase_error)  
            
            valid_target = target_complex_rd[significant_points]
            valid_reconstructed = reconstructed_complex_rd[significant_points]
            complex_correlation = np.abs(np.sum(valid_target * np.conj(valid_reconstructed))) / \
                                  (np.sqrt(np.sum(np.abs(valid_target)**2) * np.sum(np.abs(valid_reconstructed)**2)))

            phase_preservation_score = 1 - (mean_phase_error / np.pi)
        else:
            mean_phase_error = np.nan
            mean_phase_error_deg = np.nan
            complex_correlation = np.nan
            phase_preservation_score = np.nan

        print("\nphase preservation:")
        print(f"MPE: {mean_phase_error_deg:.2f}°")
        print(f"Complex correlation: {complex_correlation:.4f}")
        print(f"Score: {phase_preservation_score:.4f}")
        ####################################################################

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

        # Edit 20250119
        # scan the threshold from -40dB to 0dB, step 5dB
        thresholds = np.arange(-40, 0, 5) 
        for thresh in thresholds:
            target_detect = 20 * torch.log10(target_mag / torch.max(target_mag)) > thresh
            output_detect = 20 * torch.log10(output_mag / torch.max(output_mag)) > thresh
            
            num_targets = torch.sum(target_detect).item()
            tp = torch.sum(output_detect & target_detect).item()
            fp = torch.sum(output_detect & ~target_detect).item()
            fn = torch.sum(~output_detect & target_detect).item()
            
            pd = tp / (tp + fn + 1e-10)
            fa = fp / (torch.sum(output_detect).item() + 1e-10)
            
            print(f"Threshold {thresh}dB: {num_targets} targets, PD={pd:.3f}, FA={fa:.3f}, TP={tp}, FP={fp}, FN={fn}")
        
        
        threshold_db = -40    # changed from -20 to -40
        output_db = 20 * torch.log10(output_mag / torch.max(output_mag))
        target_db = 20 * torch.log10(target_mag / torch.max(target_mag))

        output_detections = output_db > threshold_db
        target_detections = target_db > threshold_db

        true_positives = torch.sum(output_detections & target_detections).item()
        false_positives = torch.sum(output_detections & ~target_detections).item()
        false_negatives = torch.sum(~output_detections & target_detections).item()

        detection_prob = true_positives / (true_positives + false_negatives + 1e-10)
        false_alarm_rate = false_positives / (torch.sum(output_detections).item() + 1e-10)


        print("\nperformance:")
        print(f"RD MSE: {mse:.2e}")  
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SIR改善: {sir_improvement:.2f} dB")
        print("\ndetection:")
        print(f"Pd: {detection_prob:.3f}")
        print(f"FAR: {false_alarm_rate:.3f}")
        print(f"No.TP: {true_positives:.3f}")
        print(f"No.FP: {false_positives:.3f}")
        print(f"No.FN: {false_negatives:.3f}")
        
        
        # rd back to time domain
        time_reconstructed = inverse_rd_processing(reconstructed_complex_rd)
        time_target = inverse_rd_processing(target_complex_rd)
        time_test = inverse_rd_processing(test_complex_rd)
        
        time_mse = np.mean(np.abs(time_reconstructed - time_target) ** 2)
        
        time_max_val = np.max(np.abs(time_target))
        time_psnr = 20 * np.log10(time_max_val / np.sqrt(time_mse))
        
        time_interference_power_before = np.mean(np.abs(time_test - time_target) ** 2)
        time_interference_power_after = np.mean(np.abs(time_reconstructed - time_target) ** 2)
        time_clean_power = np.mean(np.abs(time_target) ** 2)
        
        time_sir_before = 10 * np.log10(time_clean_power / time_interference_power_before)
        time_sir_after = 10 * np.log10(time_clean_power / time_interference_power_after)
        time_sir_improvement = time_sir_after - time_sir_before
        
        print("\nmetrics in time domain:")
        print(f"time MSE: {time_mse:.2e}")
        print(f"time PSNR: {time_psnr:.2f} dB")
        print(f"time SIR improvement: {time_sir_improvement:.2f} dB")

        v_res = 0.78  
        range_res = 0.25  
        
        # plot results in rd
        plot_radar_result_rd(
            target_complex_rd,  
            test_complex_rd,   
            reconstructed_complex_rd,
            v_res=v_res,
            range_res=range_res
        )

        metrics = {
            'rd_mse': rd_mse,
            'rd_psnr': psnr,
            'rd_sir_improvement': sir_improvement,
            'time_mse': time_mse,
            'time_psnr': time_psnr,
            'time_sir_improvement': time_sir_improvement,
            'detection_probability': detection_prob,
            'false_alarm_rate': false_alarm_rate,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'phase_preservation_score': phase_preservation_score,
            'mean_phase_error_deg': mean_phase_error_deg,
            'complex_correlation': complex_correlation
        }

        return metrics, reconstructed_complex_rd



def evaluate_and_save_results(model, test_data, clean_data,
                              training_memory=None, training_time=None,
                              is_test=False, save_path=None):
    print("\nStarting evaluation...")
    model.eval()
    start_time = time.time()

    metrics = {
        'mse': [],
        'psnr': [],
        'sir_improvement': [],
        'detection': {
            'doppler': {'detection_prob': [], 'false_alarm': []},
            'range': {'detection_prob': [], 'false_alarm': []}
        }
    }

    # Get model info
    model_info = []
    # model_info = get_model_info(model)
    # print("\nModel Architecture:")
    # print(f"Type: {model_info['model_type']}")
    # print(f"Parameters: {model_info['total_params']:,}")

    num_samples, num_chirps, num_frames = test_data.shape
    reconstructed_cube = np.zeros_like(test_data)

    # data handling
    dataset = RadarDataset(test_data, clean_data)

    try:
        for frame_idx in range(num_frames):
            # print(f"\nProcessing frame {frame_idx + 1}/{num_frames}")
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}")

            # Get frame data
            interfered_frame = torch.FloatTensor(dataset[frame_idx][0]).unsqueeze(0)

            # Reconstruct frame
            with torch.no_grad():
                if torch.cuda.is_available():
                    interfered_frame = interfered_frame.cuda()
                reconstructed_frame = model(interfered_frame)

            # Convert to complex
            reconstructed_complex = reconstructed_frame[0, 0, :, :].cpu().numpy() + \
                                    1j * reconstructed_frame[0, 1, :, :].cpu().numpy()
            reconstructed_cube[:, :, frame_idx] = reconstructed_complex

            clean_frame = clean_data[:, :, frame_idx]
            interfered_frame = test_data[:, :, frame_idx]
            mse = np.mean(np.abs(reconstructed_complex - clean_frame) ** 2)
            metrics['mse'].append(mse)
            max_val = np.max(np.abs(clean_frame))
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
            metrics['psnr'].append(psnr)

            interference_power_before = np.mean(np.abs(interfered_frame - clean_frame) ** 2)
            interference_power_after = np.mean(np.abs(reconstructed_complex - clean_frame) ** 2)
            clean_power = np.mean(np.abs(clean_frame) ** 2)

            metrics['input_sir'] = 10 * np.log10(clean_power / interference_power_before)
            metrics['output_sir'] = 10 * np.log10(clean_power / interference_power_after)
            metrics['sir_improvement'] = metrics['output_sir'] - metrics['input_sir']

            if is_test:
                rd_clean = apply_rd_processing(clean_frame)
                rd_reconstructed = apply_rd_processing(reconstructed_complex)

 
                frame_detection_metrics = calculate_detection_metrics(rd_clean, rd_reconstructed)

                for dim in ['doppler', 'range']:
                    metrics['detection'][dim]['detection_prob'].append(
                        frame_detection_metrics[dim]['detection_probability'])
                    metrics['detection'][dim]['false_alarm'].append(
                        frame_detection_metrics[dim]['false_alarm_rate'])

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    processing_time = time.time() - start_time

    print(f'shape of clean data: {clean_data.shape}')
    print(f'shape of reconstructed data: {reconstructed_cube.shape}')

    plot_radar_result(clean_data, test_data, reconstructed_cube, frame_idx=0)

    results = {
        'reconstructed_data': reconstructed_cube,
        'metrics': {
            'mean_mse': np.mean(metrics['mse']),
            'mean_psnr': np.mean(metrics['psnr']),
            'mean_sir_improvement': np.mean(metrics['sir_improvement'])
        },
        'performance_metrics': {
            'processing_time': processing_time,
            'processing_time_per_frame': processing_time / num_frames
        }
    }

    if training_memory is not None and training_time is not None:
        results['performance_metrics'].update({
            'training_time': training_time,
            'training_memory': training_memory
        })

    if is_test:
        results['metrics'].update({
            'detection_metrics': metrics['detection'],
            'frame_metrics': {
                'mse': metrics['mse'],
                'psnr': metrics['psnr'],
                'sir_improvement': metrics['sir_improvement']
            }
        })
        results['model_info'] = model_info

        # Display all metrics
        print("\nDetection Performance:")
        print(metrics['detection'])

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Mean MSE: {results['metrics']['mean_mse']:.2e}")
    print(f"Mean PSNR: {results['metrics']['mean_psnr']:.2f} dB")
    print(f"Mean SIR Improvement: {results['metrics']['mean_sir_improvement']:.2f} dB")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Time per frame: {processing_time / num_frames:.4f}s")

    return reconstructed_cube, results


def verify_model_robustness(model, new_radar_cube, new_clean_data, original_metrics=None):
    """
    evaluate model performance on new dataset and compare with original results

    """
    metrics = {}
    model.eval()

    # Process the entire dataset
    dataset = RadarDataset(new_radar_cube, new_clean_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Collection arrays for statistical analysis
    mse_values = []
    sir_values = []
    psnr_values = []
    peak_preservation_ratios = []

    print("\nPerforming Statistical Analysis...")

    with torch.no_grad():
        for batch_idx, (interference, clean) in enumerate(dataloader):
            if torch.cuda.is_available():
                interference = interference.cuda()
                clean = clean.cuda()

            reconstructed = model(interference)

            # Convert to numpy for analysis
            if torch.cuda.is_available():
                reconstructed = reconstructed.cpu()
            reconstructed = reconstructed.numpy()
            interference = interference.cpu().numpy()
            clean = clean.cpu().numpy()

            for i in range(interference.shape[0]):
                reconstructed_complex = reconstructed[i, 0] + 1j * reconstructed[i, 1]
                clean_complex = clean[i, 0] + 1j * clean[i, 1]
                interference_complex = interference[i, 0] + 1j * interference[i, 1]

                mse = np.mean(np.abs(reconstructed_complex - clean_complex) ** 2)
                mse_values.append(mse)

                clean_power = np.mean(np.abs(clean_complex) ** 2)
                interference_power = np.mean(np.abs(interference_complex - clean_complex) ** 2)
                residual_power = np.mean(np.abs(reconstructed_complex - clean_complex) ** 2)

                input_sir = 10 * np.log10(clean_power / interference_power)
                output_sir = 10 * np.log10(clean_power / residual_power)
                sir_values.append(output_sir - input_sir) 

                max_val = np.max(np.abs(clean_complex))
                psnr = 20 * np.log10(max_val / np.sqrt(mse))
                psnr_values.append(psnr)

                rd_clean = np.fft.fftshift(np.fft.fft2(clean_complex))
                rd_reconstructed = np.fft.fftshift(np.fft.fft2(reconstructed_complex))

                clean_peaks = len(find_significant_peaks(rd_clean))
                reconstructed_peaks = len(find_significant_peaks(rd_reconstructed))
                peak_preservation_ratios.append(reconstructed_peaks / clean_peaks if clean_peaks > 0 else 0)

    # Calculate statistical metrics
    metrics['new_dataset'] = {
        'mse': {
            'mean': np.mean(mse_values),
            'std': np.std(mse_values),
            'min': np.min(mse_values),
            'max': np.max(mse_values)
        },
        'sir_improvement': {
            'mean': np.mean(sir_values),
            'std': np.std(sir_values),
            'min': np.min(sir_values),
            'max': np.max(sir_values)
        },
        'psnr': {
            'mean': np.mean(psnr_values),
            'std': np.std(psnr_values),
            'min': np.min(psnr_values),
            'max': np.max(psnr_values)
        },
        'peak_preservation': {
            'mean': np.mean(peak_preservation_ratios),
            'std': np.std(peak_preservation_ratios),
            'min': np.min(peak_preservation_ratios),
            'max': np.max(peak_preservation_ratios)
        }
    }

    print("\nRobustness Analysis Results:")
    print("\nNew Dataset Statistics:")
    for metric, values in metrics['new_dataset'].items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {values['mean']:.3f} ± {values['std']:.3f}")
        print(f"  Range: [{values['min']:.3f}, {values['max']:.3f}]")

    if original_metrics is not None:
        print("\nComparison with Original Dataset:")
        metric_mapping = {
            'mse': 'mse',
            'psnr': 'psnr',
            'sir_improvement': 'suppression_ratio'  # Changed to match the name used in evaluate_reconstruction
        }

        for new_key, orig_key in metric_mapping.items():
            if orig_key in original_metrics:
                new_value = metrics['new_dataset'][new_key]['mean']
                orig_value = original_metrics[orig_key]
                diff = new_value - orig_value
                print(f"\n{new_key.replace('_', ' ').title()} Change:")
                print(f"  Original: {orig_value:.3f}")
                print(f"  New: {new_value:.3f}")
                print(f"  Absolute Change: {diff:.3f}")
                print(f"  Relative Change: {(diff / orig_value) * 100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metric Distributions for New Dataset')

    # MSE histogram
    axes[0, 0].hist(mse_values, bins=30)
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].set_xlabel('MSE')

    # PSNR histogram
    axes[0, 1].hist(psnr_values, bins=30)
    axes[0, 1].set_title('PSNR Distribution')
    axes[0, 1].set_xlabel('PSNR (dB)')

    # SIR improvement histogram
    axes[1, 0].hist(sir_values, bins=30)
    axes[1, 0].set_title('SIR Improvement Distribution')
    axes[1, 0].set_xlabel('SIR Improvement (dB)')

    # Peak preservation histogram
    axes[1, 1].hist(peak_preservation_ratios, bins=30)
    axes[1, 1].set_title('Peak Preservation Ratio Distribution')
    axes[1, 1].set_xlabel('Ratio')

    plt.tight_layout()
    plt.show()

    return metrics


def find_significant_peaks(rd_map, threshold_db=-20):
    """Helper function to find significant peaks in RD map"""
    rd_db = 20 * np.log10(np.abs(rd_map) / np.max(np.abs(rd_map)))
    peaks = np.where(rd_db > threshold_db)
    return list(zip(peaks[0], peaks[1]))


