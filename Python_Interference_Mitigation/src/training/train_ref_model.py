import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data.data_utils import apply_rd_processing_torch



def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the Reference Convolutional Autoencoder model with RD processing
    
    Args:
        model: The RefConvAutoencoder model
        train_loader: DataLoader for training data (time domain)
        val_loader: DataLoader for validation data (time domain)
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for interference, clean in train_loader:
            interference, clean = interference.to(device), clean.to(device)
            
            # Preprocess: Convert time domain to RD maps
            rd_interference = apply_rd_processing_torch(interference)
            rd_clean = apply_rd_processing_torch(clean)
            
            optimizer.zero_grad()
            output = model(rd_interference)
            
            # Calculate loss
            loss = criterion(output, rd_clean)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for val_interference, val_clean in val_loader:
                val_interference = val_interference.to(device)
                val_clean = val_clean.to(device)
                
                # Preprocess validation data
                rd_val_interference = apply_rd_processing_torch(val_interference)
                rd_val_clean = apply_rd_processing_torch(val_clean)
                
                val_output = model(rd_val_interference)
                val_loss = criterion(val_output, rd_val_clean)
                
                val_losses.append(val_loss.item())
        
        # Calculate average losses
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, '
              f'LR: {current_lr:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_ref_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return model, history



def predict(model, test_data, device='cuda'):
    """
    Generate predictions using the trained model
    
    Args:
        model: Trained RefConvAutoencoder model
        test_data: Input test data with interference [batch, 2, 256, 128]
        device: Device to run inference on
    
    Returns:
        Reconstructed data without interference [batch, 2, 256, 128]
    """
    model = model.to(device)
    model.eval()
    
    test_data = test_data.to(device)
    
    with torch.no_grad():
        output = model(test_data)
    
    return output

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

