import torch.nn as nn
from src.models.hybrid_autoencoder import RadarHybridAutoencoder
from src.models.linear_autoencoder import RadarAutoencoder


def get_model_info(model):
    """Extract model architecture information for both linear and hybrid models"""
    info = {
        'total_params': int(sum(p.numel() for p in model.parameters())),
        'trainable_params': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'layer_info': []
    }

    if isinstance(model, RadarAutoencoder):
        info['model_type'] = 'Linear Autoencoder'

        # Encoder info
        for idx, layer in enumerate(model.encoder):
            layer_info = {
                'layer_idx': int(idx),
                'layer_type': str(layer.__class__.__name__),
                'output_size': int(layer.out_features) if hasattr(layer, 'out_features') else 0
            }
            info['layer_info'].append(layer_info)

        # Decoder info
        for idx, layer in enumerate(model.decoder):
            layer_info = {
                'layer_idx': int(idx + len(model.encoder)),
                'layer_type': str(layer.__class__.__name__),
                'output_size': int(layer.out_features) if hasattr(layer, 'out_features') else 0
            }
            info['layer_info'].append(layer_info)

    elif isinstance(model, RadarHybridAutoencoder):
        info['model_type'] = 'Hybrid Autoencoder'

        # Conv encoder info
        for idx, layer in enumerate(model.conv_encoder):
            layer_info = {
                'layer_idx': int(idx),
                'layer_type': str(layer.__class__.__name__)
            }
            if isinstance(layer, nn.Conv2d):
                layer_info.update({
                    'in_channels': int(layer.in_channels),
                    'out_channels': int(layer.out_channels),
                    'kernel_size': int(layer.kernel_size[0])
                })
            info['layer_info'].append(layer_info)

        # Linear encoder info
        for idx, layer in enumerate(model.linear_encoder):
            if isinstance(layer, nn.Linear):
                layer_info = {
                    'layer_idx': int(idx + len(model.conv_encoder)),
                    'layer_type': 'Linear',
                    'in_features': int(layer.in_features),
                    'out_features': int(layer.out_features)
                }
                info['layer_info'].append(layer_info)

    # Convert layer_info to dictionary for MATLAB compatibility
    info['layer_info'] = {str(i): layer for i, layer in enumerate(info['layer_info'])}
    return info


def count_model_parameters(model):
    """Count and estimate model memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage
    memory_bits = total_params * 32  # 32 bits per float parameter
    memory_mb = memory_bits / (8 * 1024 * 1024)  # Convert to MB

    print("\nModel Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model memory: {memory_mb:.2f} MB")

    return total_params, memory_mb


