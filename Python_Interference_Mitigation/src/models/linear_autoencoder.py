import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RadarAutoencoder(nn.Module):
    def __init__(self, num_samples=1024, num_chirps=128, dropout_rate=0.2):
        super(RadarAutoencoder, self).__init__()

        self.num_samples = int(num_samples)
        self.num_chirps = int(num_chirps)

        self.encoder = nn.Sequential(
            # First block
            nn.Linear(self.num_samples * self.num_chirps, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),

            # Second block
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),

            # Third block
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),

            # Fourth block
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # Symmetric decoder
        self.decoder = nn.Sequential(
            # First block
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),

            # Second block
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),

            # Third block
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),

            # Final block
            nn.Linear(1024, self.num_samples * self.num_chirps)
        )

        # Additional processing layers
        self.range_attention = nn.Sequential(
            nn.Linear(self.num_samples, self.num_samples),
            nn.Sigmoid()
        )

        self.doppler_attention = nn.Sequential(
            nn.Linear(self.num_chirps, self.num_chirps),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Split real and imaginary
        x_real = x[:, 0, :, :].reshape(batch_size, -1)
        x_imag = x[:, 1, :, :].reshape(batch_size, -1)

        # Process through encoder
        latent_real = self.encoder(x_real)
        latent_imag = self.encoder(x_imag)

        # Decode
        decoded_real = self.decoder(latent_real)
        decoded_imag = self.decoder(latent_imag)

        # Reshape for attention
        decoded_real = decoded_real.view(batch_size, self.num_samples, self.num_chirps)
        decoded_imag = decoded_imag.view(batch_size, self.num_samples, self.num_chirps)

        # Apply attention mechanisms
        range_weights = self.range_attention(decoded_real.transpose(1, 2))
        doppler_weights = self.doppler_attention(decoded_real)

        decoded_real = decoded_real * range_weights.transpose(1, 2) * doppler_weights
        decoded_imag = decoded_imag * range_weights.transpose(1, 2) * doppler_weights

        # Final reshape
        decoded_real = decoded_real.unsqueeze(1)
        decoded_imag = decoded_imag.unsqueeze(1)

        return torch.cat([decoded_real, decoded_imag], dim=1)

