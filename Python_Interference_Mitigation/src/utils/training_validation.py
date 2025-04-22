import os
import time
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from src.data.data_utils import apply_rd_processing_torch



def train_autoencoder(model, train_loader, val_loader, num_epochs=100,
                      learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the autoencoder model with provided data loaders

    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Computing device
    """
    print("\nStarting training...")
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    

    # Obtain absolute path for saving models
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    save_dir = os.path.join(project_root, 'outputs', 'models')
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # model = model.to(device)
    # criterion = nn.MSELoss()     # a class, need to be instantiated
    criterion = radar_signal_loss  # The parentheses would call the function immediately rather than passing the function itself to be used later when calculating loss

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    # Set up gradient clipping to prevent exploding gradients
    clip_value = 1.0          # smaller value -> tighter gradient control
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    # Modified scheduler for better learning rate adjustment
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,
        patience=8,
        # min_lr=1e-6
        min_lr=5e-6
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 15

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch_idx, (interference, clean) in enumerate(train_loader):
            # Move to GPU if available
            if torch.cuda.is_available():
                interference = interference.cuda()
                clean = clean.cuda()

            # Debug info for first batch
            if batch_idx == 0:
                print(f"\nEpoch {epoch + 1} - First batch:")
                print(f"Interference shape: {interference.shape}")
                print(f"Clean shape: {clean.shape}")
                print(f"Value ranges - Interference: [{interference.min():.3f}, {interference.max():.3f}]")
                print(f"Value ranges - Clean: [{clean.min():.3f}, {clean.max():.3f}]")

            """"""""" For dual-path autoencoder usage """""""""
            # Calculate RD maps
            rd_maps_input = apply_rd_processing_torch(interference)
            rd_maps_target = apply_rd_processing_torch(clean)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(interference, rd_maps_input)

            # # Calculate loss
            # loss = custom_radar_loss(output, clean, rd_maps_input, rd_maps_target)


            rd_maps_output = apply_rd_processing_torch(output)
            loss = enhanced_complex_radar_loss(output, clean, rd_maps_output, rd_maps_target)
            """"""""" For dual-path autoencoder usage """""""""

            # # Forward pass
            # optimizer.zero_grad()
            # output = model(interference)
            # loss = criterion(output, clean)

            # # Debug info for first batch
            # if batch_idx == 0:
            #     print(f"Output shape: {output.shape}")
            #     print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            #     print(f"Loss: {loss.item():.6f}")

            # # Features analysis
            # Monitor patterns occasionally during training
            if epoch % 10 == 0 and batch_idx == 0:
                print("\nTraining Pattern Analysis:")
                pattern_stats = analyze_learned_patterns(model, train_loader, device)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_interference, val_clean in val_loader:
                # val_interference = val_interference.to(device)
                # val_clean = val_clean.to(device)

                if torch.cuda.is_available():
                    val_interference = val_interference.cuda()
                    val_clean = val_clean.cuda()

                """"""""" For dual-path autoencoder usage """""""""
                val_interference, val_clean = val_interference.to(device), val_clean.to(device)
                rd_maps_input = apply_rd_processing_torch(val_interference)
                rd_maps_target = apply_rd_processing_torch(val_clean)

                val_output = model(val_interference, rd_maps_input)
                rd_val_output = apply_rd_processing_torch(val_output)
                val_loss = custom_radar_loss(val_output, val_clean, rd_val_output, rd_maps_target)
                val_losses.append(val_loss.item())
                """"""""" For dual-path autoencoder usage """""""""

                # val_output = model(val_interference)
                # val_loss = criterion(val_output, val_clean)
                # val_losses.append(val_loss.item())

            # # Features analysis
            # More comprehensive analysis during validation
            if epoch % 10 == 0:
                print("\nValidation Pattern Analysis:")
                pattern_stats = analyze_learned_patterns(model, val_loader, device)

        # Calculate average losses
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Update learning rate
        scheduler.step(val_loss)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {train_loss:.6f} '
              f'Val Loss: {val_loss:.6f} '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f'Best validation loss updated: {best_val_loss}')
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'outputs/models/best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    training_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_usage = final_memory - initial_memory

    return model, history, memory_usage, training_time


def enhanced_complex_radar_loss(output, target, rd_output, rd_target):

    mse_loss = F.mse_loss(output, target)
    

    output_mag = torch.sqrt(output[:, 0]**2 + output[:, 1]**2)
    target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
    magnitude_loss = F.mse_loss(output_mag, target_mag)

    output_complex = torch.complex(output[:, 0], output[:, 1])
    target_complex = torch.complex(target[:, 0], target[:, 1])

    significant_mask = target_mag > torch.mean(target_mag) * 0.2

    output_phase = torch.angle(output_complex)
    target_phase = torch.angle(target_complex)
    phase_diff = output_phase - target_phase
    phase_loss = torch.mean((1 - torch.cos(phase_diff))[significant_mask])
    

    rd_loss = calculate_detection_loss(rd_output, rd_target)
    

    total_loss = 0.4 * mse_loss + 0.3 * phase_loss + 0.2 * magnitude_loss + 0.1 * rd_loss
    

    if torch.rand(1) < 0.01:
        print(f"\nComplex loss components:")
        print(f"MSE: {mse_loss.item():.6f}")
        print(f"Phase loss: {phase_loss.item():.6f}")
        print(f"Magnitude loss: {magnitude_loss.item():.6f}")
        print(f"RD loss: {rd_loss.item():.6f}")
    
    return total_loss


def custom_radar_loss(output, target, rd_output, rd_target):
    """Combined loss function for both domains"""
    # Time domain MSE
    time_loss = F.mse_loss(output, target)

    # RD domain detection metrics
    rd_loss = calculate_detection_loss(rd_output, rd_target)

    # Weighted combination
    total_loss = 0.7 * time_loss + 0.3 * rd_loss

    # Debug prints
    if torch.rand(1) < 0.01:  # Print occasionally
        print(f"\nLoss Components:")
        print(f"Time Domain Loss: {time_loss.item():.6f}")
        print(f"RD Domain Loss: {rd_loss.item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")

    return total_loss


def custom_radar_loss_modif(output, target, rd_output, rd_target):
    """Enhanced combined loss with pattern monitoring"""
    # Time domain MSE
    time_mse = F.mse_loss(output, target)

    # Additional time domain pattern analysis
    output_mag = torch.sqrt(output[:, 0] ** 2 + output[:, 1] ** 2)
    target_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)

    # Identify quiet regions (should have minimal signal)
    quiet_mask = target_mag < torch.mean(target_mag) * 0.1
    noise_suppression = torch.mean(output_mag[quiet_mask])

    # RD domain detection and suppression
    detection_loss = calculate_detection_loss(rd_output, rd_target)

    # Combined loss with weighting
    total_loss = 0.4 * time_mse + 0.4 * detection_loss + 0.2 * noise_suppression

    # Monitor learned patterns
    if torch.rand(1) < 0.01:  # Print occasionally
        print("\nLearned Pattern Analysis:")
        print(f"Time Domain Error: {time_mse.item():.6f}")
        print(f"Noise Level in Quiet Regions: {noise_suppression.item():.6f}")
        print(
            f"Signal-to-Noise Ratio: {(torch.mean(output_mag[~quiet_mask]) / (torch.mean(output_mag[quiet_mask]) + 1e-10)).item():.2f}")

    return total_loss


def calculate_detection_loss(rd_output, rd_target, threshold_db=-20):

    # Convert to magnitude
    output_mag = torch.sqrt(rd_output[:, 0] ** 2 + rd_output[:, 1] ** 2)
    target_mag = torch.sqrt(rd_target[:, 0] ** 2 + rd_target[:, 1] ** 2)

    # Convert to dB
    output_db = 20 * torch.log10(output_mag / (torch.max(output_mag) + 1e-10))
    target_db = 20 * torch.log10(target_mag / (torch.max(target_mag) + 1e-10))

    # Find peaks (detections)
    output_detections = output_db > threshold_db
    target_detections = target_db > threshold_db

    # Calculate detection metrics
    true_positives = torch.sum(output_detections & target_detections)
    false_positives = torch.sum(output_detections & ~target_detections)
    false_negatives = torch.sum(~output_detections & target_detections)

    # Detection probability and false alarm rate
    detection_prob = true_positives / (true_positives + false_negatives + 1e-10)
    false_alarm_rate = false_positives / (torch.sum(output_detections) + 1e-10)

    # Combined detection loss
    detection_loss = (1 - detection_prob) + 0.5 * false_alarm_rate

    # Add magnitude preservation for detected targets
    magnitude_loss = F.mse_loss(
        output_mag[target_detections],
        target_mag[target_detections]
    ) if torch.any(target_detections) else torch.tensor(0.0, device=rd_output.device)

    # Debug prints
    if torch.rand(1) < 0.01:  # Print occasionally
        print(f"\nDetection Metrics:")
        print(f"Detection Probability: {detection_prob.item():.3f}")
        print(f"False Alarm Rate: {false_alarm_rate.item():.3f}")
        print(f"Magnitude Loss: {magnitude_loss.item():.3f}")

    return detection_loss + 0.3 * magnitude_loss


def calculate_detection_loss_modif(rd_output, rd_target, threshold_db=-20, noise_threshold_db=-40):
    """Enhanced detection loss with interference suppression"""
    # Convert to magnitude and dB
    output_mag = torch.sqrt(rd_output[:, 0] ** 2 + rd_output[:, 1] ** 2)
    target_mag = torch.sqrt(rd_target[:, 0] ** 2 + rd_target[:, 1] ** 2)

    output_db = 20 * torch.log10(output_mag / (torch.max(output_mag) + 1e-10))
    target_db = 20 * torch.log10(target_mag / (torch.max(target_mag) + 1e-10))

    # Target detection
    output_detections = output_db > threshold_db
    target_detections = target_db > threshold_db

    # Interference regions (where there should be no signal)
    interference_mask = (target_db < noise_threshold_db)
    suppression_loss = torch.mean(torch.abs(output_mag[interference_mask]))

    # Calculate detection metrics
    true_positives = torch.sum(output_detections & target_detections)
    false_positives = torch.sum(output_detections & ~target_detections)
    false_negatives = torch.sum(~output_detections & target_detections)

    detection_prob = true_positives / (true_positives + false_negatives + 1e-10)
    false_alarm_rate = false_positives / (torch.sum(output_detections) + 1e-10)

    # Combined loss with stronger suppression
    detection_loss = (1 - detection_prob) + false_alarm_rate + 0.5 * suppression_loss

    # Debug information about learned patterns
    if torch.rand(1) < 0.01:  # Print occasionally
        print("\nPattern Analysis:")
        print(f"Target Detections: {torch.sum(target_detections).item()}")
        print(f"Output Detections: {torch.sum(output_detections).item()}")
        print(f"False Detections: {false_positives.item()}")
        print(f"Interference Level: {suppression_loss.item():.6f}")

    return detection_loss


def radar_signal_loss(output, target):

    mse_loss = F.mse_loss(output, target)

    # Signal amplitude preservation
    output_amp = torch.sqrt(output[:, 0, :, :] ** 2 + output[:, 1, :, :] ** 2)
    target_amp = torch.sqrt(target[:, 0, :, :] ** 2 + target[:, 1, :, :] ** 2)
    amp_loss = F.mse_loss(output_amp, target_amp)

    # Penalize near-zero outputs more heavily
    zero_suppression_loss = torch.mean(1 / (output_amp + 1e-6))

    # Combined loss with weighting
    total_loss = mse_loss + 2.0 * amp_loss + 0.1 * zero_suppression_loss

    # Debug prints
    if torch.rand(1) < 0.01:  # Print occasionally
        print(f"\nLoss Components:")
        print(f"MSE Loss: {mse_loss.item():.6f}")
        print(f"Amplitude Loss: {amp_loss.item():.6f}")
        print(f"Zero Suppression Loss: {zero_suppression_loss.item():.6f}")
        print(f"Total Loss: {total_loss.item():.6f}")

    return total_loss




# # Modified 31.03.2025
def analyze_learned_patterns(model, test_loader, device):
    """Analyze what patterns the model has learned, including phase consistency"""
    model.eval()
    pattern_stats = {
        'false_peaks': [],
        'signal_preservation': [],
        'phase_consistency': []
    }

    with torch.no_grad():
        for interference, clean in test_loader:
            interference, clean = interference.to(device), clean.to(device)
            rd_maps_input = apply_rd_processing_torch(interference)
            output = model(interference, rd_maps_input)

            # Analyze magnitude preservation
            clean_mag = torch.sqrt(clean[:, 0] ** 2 + clean[:, 1] ** 2)
            output_mag = torch.sqrt(output[:, 0] ** 2 + output[:, 1] ** 2)

            clean_complex = torch.complex(clean[:, 0], clean[:, 1])
            output_complex = torch.complex(output[:, 0], output[:, 1])

            # Find peaks in clean signal
            peak_mask = clean_mag > torch.mean(clean_mag) * 1.5

            if torch.any(peak_mask):
                signal_preservation = torch.mean(output_mag[peak_mask] / (clean_mag[peak_mask] + 1e-10)).item()
                phase_consistency = torch.mean(torch.cos(torch.angle(clean_complex) - torch.angle(output_complex))[peak_mask]).item()
            else:
                signal_preservation = 0.0
                phase_consistency = 0.0

            # Analyze false peaks
            false_peak_mask = (~peak_mask) & (output_mag > torch.mean(clean_mag))
            if torch.numel(false_peak_mask) > 0:
                false_peaks = torch.sum(false_peak_mask).item() / torch.numel(false_peak_mask)
            else:
                false_peaks = 0.0

            # Record statistics
            pattern_stats['false_peaks'].append(false_peaks)
            pattern_stats['signal_preservation'].append(signal_preservation)
            pattern_stats['phase_consistency'].append(phase_consistency)

    # Print analysis
    print("\nLearned Pattern Analysis v2:")
    print(f"Average False Peak Rate: {np.mean(pattern_stats['false_peaks']):.3f}")
    print(f"Signal Preservation Ratio: {np.mean(pattern_stats['signal_preservation']):.3f}")
    print(f"Phase Consistency: {np.mean(pattern_stats['phase_consistency']):.3f}")

    return pattern_stats
