import os
import time
import psutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.data_utils import apply_rd_processing_torch

def train_refdual_model(model, train_loader, val_loader, num_epochs=100,
                        learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    train_refdual_model - for training the reference dual-path autoencoder model.
    
    Args:
        model: RefDualPathAutoencoder
        train_loader: training data loader
        val_loader: validation data loader
        num_epochs
        learning_rate
        device
    """
    print("\nStart to train the model...")
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # create save directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    save_dir = os.path.join(project_root, 'outputs', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # move device to GPU or CPU
    model = model.to(device)
    print(f"model moved to {device} device")
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.8,
        patience=5,
        min_lr=1e-6
    )
    
    # training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'rd_metrics': []
    }
    
    # early stopping parameters
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 12
    
    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_losses = []
        
        for batch_idx, (interference, clean) in enumerate(train_loader):
            interference = interference.to(device)
            clean = clean.to(device)
            
            rd_maps_input = apply_rd_processing_torch(interference)
            rd_maps_target = apply_rd_processing_torch(clean)
            

            if batch_idx == 0:
                print(f"\nepoch {epoch + 1} - first batch:")
                print(f"input shape: time domain={interference.shape}, RD={rd_maps_input.shape}")
                print(f"range of value - interference: [{interference.min():.3f}, {interference.max():.3f}]")
            
            optimizer.zero_grad()
            rd_maps_output = model(interference, rd_maps_input)
            
    
            loss = rd_domain_loss(rd_maps_output, rd_maps_target)

            loss = simple_mse_loss(rd_maps_output, rd_maps_target)
            
            loss = amplitude_phase_loss(rd_maps_output, rd_maps_target)
            
            loss = progressive_loss(rd_maps_output, rd_maps_target, epoch, num_epochs)
            
            loss = multiscale_contrast_loss(rd_maps_output, rd_maps_target)

            loss = interference_suppression_loss(rd_maps_output, rd_maps_target, rd_maps_input)
            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        rd_metrics = {'psnr': [], 'sir_improvement': [], 'phase_error': []}
        
        with torch.no_grad():
            for val_interference, val_clean in val_loader:
                val_interference = val_interference.to(device)
                val_clean = val_clean.to(device)
                
                rd_maps_input = apply_rd_processing_torch(val_interference)
                rd_maps_target = apply_rd_processing_torch(val_clean)
                
                rd_maps_output = model(val_interference, rd_maps_input)
                
                val_loss = rd_domain_loss(rd_maps_output, rd_maps_target)
                val_losses.append(val_loss.item())
                
                batch_metrics = calculate_rd_metrics(rd_maps_output, rd_maps_target, rd_maps_input)
                rd_metrics['psnr'].append(batch_metrics['psnr'])
                rd_metrics['sir_improvement'].append(batch_metrics['sir_improvement'])
                rd_metrics['phase_error'].append(batch_metrics['phase_error'])
        
        # calculate average losses
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # update learning rate
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # record metrics
        avg_metrics = {
            'psnr': np.mean(rd_metrics['psnr']),
            'sir_improvement': np.mean(rd_metrics['sir_improvement']),
            'phase_error': np.mean(rd_metrics['phase_error'])
        }
        history['rd_metrics'].append(avg_metrics)
        
        print(f'epoch [{epoch + 1}/{num_epochs}] '
              f'training loss: {train_loss:.6f} '
              f'validation loss: {val_loss:.6f} '
              f'PSNR: {avg_metrics["psnr"]:.2f}dB '
              f'SIR improvement: {avg_metrics["sir_improvement"]:.2f}dB '
              f'phase error: {avg_metrics["phase_error"]:.2f}°')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f'update best loss: {best_val_loss:.6f}')
            
            model_filename = os.path.join(save_dir, 'best_refdual_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': avg_metrics
            }, model_filename)
            print(f'model saved to: {model_filename}')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"early stop triggered，stopped at {epoch + 1} epoch")
                break
    
    training_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_usage = final_memory - initial_memory
    
    print(f"\ndone with training")
    print(f"time: {training_time/60:.2f} min")
    print(f"memories: {memory_usage:.1f} MB")
    
    return model, history, memory_usage, training_time


def simple_mse_loss(rd_output, rd_target):
    """simple MSE loss for debugging""" 
    return F.mse_loss(rd_output, rd_target)


def amplitude_phase_loss(rd_output, rd_target, amp_weight=0.7, phase_weight=0.3):


    real_output, imag_output = rd_output[:, 0], rd_output[:, 1]
    real_target, imag_target = rd_target[:, 0], rd_target[:, 1]
    

    output_mag = torch.sqrt(real_output**2 + imag_output**2 + 1e-10)
    target_mag = torch.sqrt(real_target**2 + imag_target**2 + 1e-10)

    magnitude_loss = F.mse_loss(output_mag, target_mag)
    
    output_complex = torch.complex(real_output, imag_output)
    target_complex = torch.complex(real_target, imag_target)

    output_unit = output_complex / (torch.abs(output_complex) + 1e-10)
    target_unit = target_complex / (torch.abs(target_complex) + 1e-10)

    mask = target_mag > torch.mean(target_mag) * 0.1
    
    if torch.any(mask):
        phase_diff = torch.abs(output_unit - target_unit)[mask]
        phase_loss = torch.mean(torch.abs(phase_diff)**2)
    else:
        phase_loss = torch.tensor(0.0, device=rd_output.device)
    
    total_loss = amp_weight * magnitude_loss + phase_weight * phase_loss
    
    # print output every once in a while
    if torch.rand(1) < 0.01:
        print(f"amplitude loss: {magnitude_loss.item():.6f}, phase loss: {phase_loss.item():.6f}")
        
    return total_loss

def rd_domain_loss(rd_output, rd_target):
    """
    loss function for RD domain
    """

    real_output, imag_output = rd_output[:, 0], rd_output[:, 1]
    real_target, imag_target = rd_target[:, 0], rd_target[:, 1]
    
    output_mag = torch.sqrt(real_output**2 + imag_output**2)
    target_mag = torch.sqrt(real_target**2 + imag_target**2)

    mse_loss = F.mse_loss(rd_output, rd_target)

    mag_loss = F.mse_loss(output_mag, target_mag)

    output_complex = torch.complex(real_output, imag_output)
    target_complex = torch.complex(real_target, imag_target)

    significant_mask = target_mag > (torch.mean(target_mag) * 0.1)

    output_phase = torch.angle(output_complex)
    target_phase = torch.angle(target_complex)
    phase_diff = torch.abs(torch.angle(torch.exp(1j * (output_phase - target_phase))))
    
    if torch.any(significant_mask):
        phase_loss = torch.mean(phase_diff[significant_mask])
    else:
        phase_loss = torch.tensor(0.0, device=rd_output.device)

    peak_mask = target_mag > (torch.max(target_mag) * 0.5)
    if torch.any(peak_mask):
        peak_loss = F.mse_loss(output_mag[peak_mask], target_mag[peak_mask])
    else:
        peak_loss = torch.tensor(0.0, device=rd_output.device)
    
    # combine losses
    # total_loss = 0.6 * mse_loss + 0.3 * mag_loss + 0.1 * phase_loss
    total_loss = F.mse_loss(rd_output, rd_target)

    
    return total_loss


def calculate_rd_metrics(rd_output, rd_target, rd_input):
    """
    rd metrics
    """
    # convert to numpy
    output = rd_output.detach().cpu().numpy()
    target = rd_target.detach().cpu().numpy()
    input_rd = rd_input.detach().cpu().numpy()
    
    # to complex format
    output_complex = output[:, 0] + 1j * output[:, 1]
    target_complex = target[:, 0] + 1j * target[:, 1]
    input_complex = input_rd[:, 0] + 1j * input_rd[:, 1]
    

    output_complex = np.mean(output_complex, axis=0)
    target_complex = np.mean(target_complex, axis=0)
    input_complex = np.mean(input_complex, axis=0)
    
    mse = np.mean(np.abs(output_complex - target_complex) ** 2)
    
    max_val = np.max(np.abs(target_complex))
    psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-10))

    interference_power_before = np.mean(np.abs(input_complex - target_complex) ** 2)
    interference_power_after = np.mean(np.abs(output_complex - target_complex) ** 2)
    clean_power = np.mean(np.abs(target_complex) ** 2)
    
    sir_before = 10 * np.log10(clean_power / (interference_power_before + 1e-10))
    sir_after = 10 * np.log10(clean_power / (interference_power_after + 1e-10))
    sir_improvement = sir_after - sir_before

    phase_target = np.angle(target_complex)
    phase_output = np.angle(output_complex)

    significant_points = np.abs(target_complex) > (np.max(np.abs(target_complex)) * 0.1)
    if np.any(significant_points):
        phase_diff = np.abs(np.angle(np.exp(1j * (phase_output - phase_target))))
        mean_phase_error = np.mean(phase_diff[significant_points])
        mean_phase_error_deg = np.degrees(mean_phase_error)
    else:
        mean_phase_error_deg = 0.0
    
    return {
        'psnr': psnr,
        'sir_improvement': sir_improvement,
        'phase_error': mean_phase_error_deg
    }


def progressive_loss(rd_output, rd_target, epoch, max_epochs=100):


    mse_loss = F.mse_loss(rd_output, rd_target)
    

    if epoch < 10:
        return mse_loss
    

    real_output, imag_output = rd_output[:, 0], rd_output[:, 1]
    real_target, imag_target = rd_target[:, 0], rd_target[:, 1]
    

    output_mag = torch.sqrt(real_output**2 + imag_output**2 + 1e-10)
    target_mag = torch.sqrt(real_target**2 + imag_target**2 + 1e-10)
    

    magnitude_loss = F.mse_loss(output_mag, target_mag)

    amp_weight = min(0.5, (epoch - 10) / (max_epochs - 10) * 0.5)
    

    total_loss = (1 - amp_weight) * mse_loss + amp_weight * magnitude_loss
    

    if epoch >= 20:

        peak_mask = target_mag > (torch.max(target_mag) * 0.3)
        if torch.any(peak_mask):
            peak_loss = F.mse_loss(output_mag[peak_mask], target_mag[peak_mask])

            peak_weight = min(0.3, (epoch - 20) / (max_epochs - 20) * 0.3)
            total_loss = total_loss * (1 - peak_weight) + peak_loss * peak_weight
    
    return total_loss


def multiscale_contrast_loss(rd_output, rd_target):


    real_output, imag_output = rd_output[:, 0], rd_output[:, 1]
    real_target, imag_target = rd_target[:, 0], rd_target[:, 1]
    

    output_mag = torch.sqrt(real_output**2 + imag_output**2 + 1e-10)
    target_mag = torch.sqrt(real_target**2 + imag_target**2 + 1e-10)
    

    basic_mse = F.mse_loss(rd_output, rd_target)
    

    global_mag_loss = F.mse_loss(output_mag, target_mag)
    

    high_power_mask = target_mag > torch.quantile(target_mag, 0.9) 
    if torch.any(high_power_mask):
        high_power_loss = F.l1_loss(output_mag[high_power_mask], target_mag[high_power_mask])
    else:
        high_power_loss = torch.tensor(0.0, device=rd_output.device)
    

    mid_power_mask = (target_mag > torch.quantile(target_mag, 0.5)) & (target_mag <= torch.quantile(target_mag, 0.9))
    if torch.any(mid_power_mask):
        mid_power_loss = F.mse_loss(output_mag[mid_power_mask], target_mag[mid_power_mask])
    else:
        mid_power_loss = torch.tensor(0.0, device=rd_output.device)
    

    log_output_mag = torch.log10(output_mag + 1e-10)
    log_target_mag = torch.log10(target_mag + 1e-10)
    log_domain_loss = F.mse_loss(log_output_mag, log_target_mag)
    

    total_loss = 0.4 * basic_mse + 0.2 * global_mag_loss + \
                 0.2 * high_power_loss + 0.1 * mid_power_loss + \
                 0.1 * log_domain_loss
    
    return total_loss


def interference_suppression_loss(rd_output, rd_target, rd_input):

    real_output, imag_output = rd_output[:, 0], rd_output[:, 1]
    real_target, imag_target = rd_target[:, 0], rd_target[:, 1]
    real_input, imag_input = rd_input[:, 0], rd_input[:, 1]
    

    output_mag = torch.sqrt(real_output**2 + imag_output**2)
    target_mag = torch.sqrt(real_target**2 + imag_target**2)
    input_mag = torch.sqrt(real_input**2 + imag_input**2)
    

    recon_loss = F.mse_loss(rd_output, rd_target)
    

    interference_ratio = input_mag / (target_mag + 1e-10)
    interference_mask = interference_ratio > 2.0  
    

    if torch.any(interference_mask):
        suppression_loss = F.mse_loss(output_mag[interference_mask], target_mag[interference_mask])
    else:
        suppression_loss = torch.tensor(0.0, device=rd_output.device)
    

    target_peaks = target_mag > torch.quantile(target_mag, 0.95)  
    
    if torch.any(target_peaks):
        preservation_loss = F.mse_loss(output_mag[target_peaks], target_mag[target_peaks])
    else:
        preservation_loss = torch.tensor(0.0, device=rd_output.device)
    

    total_loss = 0.4 * recon_loss + 0.4 * suppression_loss + 0.2 * preservation_loss
    
    if torch.rand(1) < 0.01: 
        print(f"\nloss composition:")
        print(f"recon: {recon_loss.item():.6f}")
        print(f"mitigation: {suppression_loss.item():.6f}")
        print(f"preserv: {preservation_loss.item():.6f}")
    
    return total_loss