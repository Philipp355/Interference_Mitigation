import torch
import torch.nn as nn


class RadarHybridAutoencoder(nn.Module):
    def __init__(self, num_samples=1024, num_chirps=128):
        super().__init__()

        # Calculate feature map size after conv layers
        self.feature_h = num_samples // 8  # After 3 MaxPool2d
        self.feature_w = num_chirps // 8
        self.flat_size = 32 * self.feature_h * self.feature_w

        # # Input normalization for 2 channels
        # self.input_norm = nn.BatchNorm2d(2)

        # # Replace BatchNorm2D with InstanceNorm2D
        # self.input_norm = nn.InstanceNorm2d(2, affine=True)

        # Convolutional encoder (2 channels for real/imag)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),  # 2 input channels
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 128x64

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 64x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )

        self.linear_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128)
        )

        self.linear_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.flat_size)
        )

        # Convolutional decoder (output 2 channels for real/imag)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(8, 2, kernel_size=3, padding=1),  # 2 output channels
            nn.Upsample(scale_factor=2),
            nn.Identity()   # Linear activation to preserve signal characteristic
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Process both channels together, conv
        # x = self.input_norm(x)
        x = self.conv_encoder(x)

        # Linear
        x = self.linear_encoder(x.flatten(1))
        x = self.linear_decoder(x)

        # Reshape and decode using calculated dimensions
        x = x.view(batch_size, 32, self.feature_h, self.feature_w)
        x = self.conv_decoder(x)

        return x


