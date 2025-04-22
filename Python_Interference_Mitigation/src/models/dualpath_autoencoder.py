import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathRadarAutoencoder(nn.Module):
    def __init__(self, num_samples=1024, num_chirps=128):
        super().__init__()

        self.shape_printed = False  # printing flag

        # Calculate RD map size
        self.rd_height = num_samples
        self.rd_width = num_chirps

        # Calculate feature dimensions after conv layers
        self.feature_h = num_samples // 8  # 256/8 = 32
        self.feature_w = num_chirps // 8   # 128/8 = 16

        # RD Domain Path (CNN)
        self.rd_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 2 channels: magnitude and phase
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 512x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 256x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x16
        )

        # # Time Domain Path (FC) - with reduced dimensions
        # self.time_encoder = nn.Sequential(
        #     nn.Linear(num_samples * num_chirps, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2)
        # )
        #
        # # Combined Features
        # self.combined_decoder = nn.Sequential(
        #     nn.Linear(512 + 64 * 128 * 16, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, num_samples * num_chirps)
        # )

        # Time Path - adjusted for 256x128 input
        time_input_size = num_samples * num_chirps  # 256*128 = 32,768
        self.time_encoder = nn.Sequential(
            nn.Linear(time_input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        # Calculate combined feature size
        rd_feature_size = 64 * self.feature_h * self.feature_w  # 64*32*16
        combined_size = 256 + rd_feature_size  # time features + rd features

        # Combined decoder - adjusted accordingly
        self.combined_decoder = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, time_input_size)
        )

        print(f"\nModel Architecture Info:")
        print(f"Input dimensions: {num_samples}x{num_chirps}")
        self.print_parameter_count()

    def print_parameter_count(self):
        """Print parameter count for each component"""
        rd_params = sum(p.numel() for p in self.rd_encoder.parameters())
        time_params = sum(p.numel() for p in self.time_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.combined_decoder.parameters())
        total_params = rd_params + time_params + decoder_params

        print(f"\nParameter Count:")
        print(f"RD Path: {rd_params:,}")
        print(f"Time Path: {time_params:,}")
        print(f"Decoder: {decoder_params:,}")
        print(f"Total: {total_params:,}")

    def forward(self, x, rd_maps):
        batch_size = x.size(0)

        # Print shapes only once
        if not self.shape_printed:
            print(f"\nInput shapes:")
            print(f"Raw signal: {x.shape}")
            print(f"RD maps: {rd_maps.shape}")
            print(f"RD output shape: {torch.Size([batch_size, 2, self.rd_height, self.rd_width])}")

        # Process RD domain
        rd_features = self.rd_encoder(rd_maps)
        rd_features_flat = rd_features.flatten(1)

        # Process time domain (real and imag)
        x_real = x[:, 0, :, :].reshape(batch_size, -1)
        x_imag = x[:, 1, :, :].reshape(batch_size, -1)

        time_features_real = self.time_encoder(x_real)
        time_features_imag = self.time_encoder(x_imag)

        # Combine features
        combined_real = torch.cat([time_features_real, rd_features_flat], dim=1)
        combined_imag = torch.cat([time_features_imag, rd_features_flat], dim=1)

        # Decode
        output_real = self.combined_decoder(combined_real)
        output_imag = self.combined_decoder(combined_imag)

        # Reshape outputs
        output_real = output_real.view(batch_size, 1, self.rd_height, self.rd_width)
        output_imag = output_imag.view(batch_size, 1, self.rd_height, self.rd_width)

        # Debug prints for output shapes
        output = torch.cat([output_real, output_imag], dim=1)

        # Print shapes only once
        if not self.shape_printed:
            print(f"Combined output shape: {output.shape}")
            self.shape_printed = True

        return output


class DualPathRadarAutoencoder_V2(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):  # 更新默认输入维度
        super().__init__()
        self.shape_printed = False
        
        self.feature_h = num_samples // 8  # 32
        self.feature_w = num_chirps // 8   # 16
        
        self.rd_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )

        # Time Domain Path - CNN
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )

        time_feature_size = 64 * self.feature_h * self.feature_w  # 64 * 32 * 16
        
        # Time Domain Path 
        self.time_encoder = nn.Sequential(
            nn.Linear(time_feature_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        rd_feature_size = 64 * self.feature_h * self.feature_w
        combined_size = 256 + rd_feature_size

        self.fc_decoder = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 32 * self.feature_h * self.feature_w) 
        )
        
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x, rd_maps):
        batch_size = x.size(0)
        
        if not self.shape_printed:
            print(f"\nInput shapes:")
            print(f"Raw signal: {x.shape}")
            print(f"RD maps: {rd_maps.shape}")
            print(f"Feature dimensions: {self.feature_h}x{self.feature_w}")

        # Process RD domain
        rd_features = self.rd_encoder(rd_maps)
        rd_features_flat = rd_features.flatten(1)

        # Process time domain
        x_real = x[:, 0:1, :, :]
        x_imag = x[:, 1:2, :, :]

        time_conv_real = self.time_conv(x_real)
        time_conv_imag = self.time_conv(x_imag)

        time_features_real = self.time_encoder(time_conv_real.flatten(1))
        time_features_imag = self.time_encoder(time_conv_imag.flatten(1))

        # Combine features
        combined_real = torch.cat([time_features_real, rd_features_flat], dim=1)
        combined_imag = torch.cat([time_features_imag, rd_features_flat], dim=1)

        # Decode
        decoded_real = self.fc_decoder(combined_real)
        decoded_imag = self.fc_decoder(combined_imag)

        # 重塑为4D张量 [batch_size, channels, height, width]
        decoded_real = decoded_real.view(batch_size, 32, self.feature_h, self.feature_w)
        decoded_imag = decoded_imag.view(batch_size, 32, self.feature_h, self.feature_w)

        # 上采样恢复
        output_real = self.conv_decoder(decoded_real)
        output_imag = self.conv_decoder(decoded_imag)

        # 合并实部和虚部
        output = torch.cat([output_real, output_imag], dim=1)

        if not self.shape_printed:
            print(f"Output shape: {output.shape}")
            self.shape_printed = True

        return output
        

class DualPathRadarAutoencoder_v3(nn.Module):
   def __init__(self, num_samples=256, num_chirps=128):
       super().__init__()
       
       # Input normalization
       self.input_norm = nn.BatchNorm2d(2)
       
       # RD Domain Path (CNN)
       self.rd_encoder = nn.Sequential(
           nn.Conv2d(2, 8, kernel_size=3, padding=1),      # params: 2*8*3*3 + 8 = 152
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 128x64
           
           nn.Conv2d(8, 16, kernel_size=3, padding=1),     # params: 8*16*3*3 + 16 = 1,168
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 64x32
           
           nn.Conv2d(16, 32, kernel_size=3, padding=1),    # params: 16*32*3*3 + 32 = 4,640
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2)   # 32x16
       )
       
       # Time Domain Path with initial CNN layers
       # Shared CNN layers for real and imag parts
       self.time_cnn = nn.Sequential(
           nn.Conv2d(1, 8, kernel_size=3, padding=1),      # params: 1*8*3*3 + 8 = 80
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # Reduce to 128x64
           
           nn.Conv2d(8, 16, kernel_size=3, padding=1),     # params: 8*16*3*3 + 16 = 1,168
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2)   # Further reduce to 64x32
       )
       
       # Calculate sizes after CNN
       self.rd_feature_size = 32 * (num_samples//8) * (num_chirps//8)  # After 3 MaxPool
       # 修正 time_feature_size 计算
       self.time_feature_size = 16 * (num_samples//4) * (num_chirps//4) * 2  #  After 2 MaxPool *2 
        
       
       # Time domain FC layers (reduced size due to CNN)
       self.time_fc = nn.Sequential(
           nn.Linear(self.time_feature_size, 512),         # params: (16*64*32)*512 + 512 ≈ 16.8M
           nn.LeakyReLU(0.2),
           nn.Linear(512, 256)                            # params: 512*256 + 256 = 131,328
       )
       
       # Combined decoder (smaller due to reduced feature sizes)
       combined_size = 256 + self.rd_feature_size  # Time features + RD features
       self.decoder = nn.Sequential(
           nn.Linear(combined_size, 512),                  # params: ~0.5M
           nn.LeakyReLU(0.2),
           nn.Linear(512, 2 * num_samples * num_chirps)    # params: ~0.3M
       )
       
       self.num_samples = num_samples
       self.num_chirps = num_chirps
       
       # Print model info
       self.print_model_info()

   def forward(self, x, rd_maps):
    """
    Forward pass with both time domain and RD domain inputs
    
    Args:
        x: Time domain signal [batch, 2, samples, chirps]
        rd_maps: RD domain maps [batch, 2, samples, chirps]
    """
    batch_size = x.size(0)
    
    # Normalize inputs
    x = self.input_norm(x)
    rd_maps = self.input_norm(rd_maps)
    
    # RD domain path
    rd_features = self.rd_encoder(rd_maps)
    rd_features_flat = rd_features.flatten(1)
    
    # Time domain path - process real and imag with shared CNN
    x_real = x[:, 0:1, :, :]
    x_imag = x[:, 1:2, :, :]
    
    # Apply shared CNN to both parts
    real_features = self.time_cnn(x_real)
    imag_features = self.time_cnn(x_imag)
    
    # Combine and flatten CNN features
    time_features = torch.cat([real_features.flatten(1), 
                             imag_features.flatten(1)], dim=1)
    
    # Apply FC layers
    time_features = self.time_fc(time_features)
    
    # Combine features from both paths
    combined = torch.cat([time_features, rd_features_flat], dim=1)
    
    # Decode
    output = self.decoder(combined)
    output = output.view(batch_size, 2, self.num_samples, self.num_chirps)
    
    return output

   def print_model_info(self):
       """Print model parameter counts"""
       def count_parameters(module):
           return sum(p.numel() for p in module.parameters())
       
       print("\nModel Parameter Count:")
       print(f"RD Path (CNN): {count_parameters(self.rd_encoder):,}")
       print(f"Time Path (CNN): {count_parameters(self.time_cnn):,}")
       print(f"Time Path (FC): {count_parameters(self.time_fc):,}")
       print(f"Decoder: {count_parameters(self.decoder):,}")
       total = count_parameters(self)
       print(f"Total Parameters: {total:,}")
       
       if total > 3_000_000:
           print("\nWarning: Total parameters exceed 3 million!")
           

class DualPathRadarAutoencoder_v3_2(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm2d(2)
        
        # RD Domain Path - 
        self.rd_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4),  # 64x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4),  # 16x8
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)   # 8x4
        )
        
        # Time Domain Path 
        self.time_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4),  # 64x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4)   # 16x8
        )
        

        self.time_feature_size = 32 * 16 * 8 * 2  # 32 channels * 16x8 spatial * 2 (real/imag)
        self.rd_feature_size = 64 * 8 * 4  # 64 channels * 8x4 spatial
        
        self.time_fc = nn.Sequential(
            nn.Linear(self.time_feature_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128)
        )
        

        combined_size = 128 + self.rd_feature_size
        self.decoder_fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 32 * 16 * 8)  
        )
        
    
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 32x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 64x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 128x64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),   # -> 256x128
        )
        
        self.num_samples = num_samples
        self.num_chirps = num_chirps

    def forward(self, x, rd_maps):
        batch_size = x.size(0)
        
        # Normalize inputs
        x = self.input_norm(x)
        rd_maps = self.input_norm(rd_maps)
        
        # RD domain path
        rd_features = self.rd_encoder(rd_maps)
        rd_features_flat = rd_features.view(batch_size, -1)
        
        # Time domain path
        x_real = x[:, 0:1, :, :]
        x_imag = x[:, 1:2, :, :]
        
        real_features = self.time_cnn(x_real)
        imag_features = self.time_cnn(x_imag)
        
        time_features = torch.cat([real_features.flatten(1), 
                                 imag_features.flatten(1)], dim=1)
        
        time_features = self.time_fc(time_features)
        
        # Combine features
        combined = torch.cat([time_features, rd_features_flat], dim=1)
        
        # Decode
        features = self.decoder_fc(combined)
        features = features.view(batch_size, 32, 16, 8) 
        output = self.decoder_conv(features)  # [batch_size, 2, 256, 128]
        
        return output
    

    def print_model_info(self):
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        print("\nModel Parameter Count:")
        print(f"RD Path (CNN): {count_parameters(self.rd_encoder):,}")
        print(f"Time Path (CNN): {count_parameters(self.time_cnn):,}")
        print(f"Time Path (FC): {count_parameters(self.time_fc):,}")
        print(f"Decoder FC: {count_parameters(self.decoder_fc):,}")
        print(f"Decoder Conv: {count_parameters(self.decoder_conv):,}")
        total = count_parameters(self)
        print(f"Total Parameters: {total:,}")


class DualPathRadarAutoencoder_v3_3(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()
        
 
        self.rd_encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        
        self.rd_encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        
        self.rd_encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        
        self.time_encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        
        self.time_encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        
        self.time_encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        
        self.time_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),  # 1x1
            nn.LeakyReLU(0.2)
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256x128
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )
        
        self.num_samples = num_samples
        self.num_chirps = num_chirps

    def forward(self, x, rd_maps):
        batch_size = x.size(0)
        
        rd1 = self.rd_encoder1(rd_maps)
        rd2 = self.rd_encoder2(rd1)
        rd3 = self.rd_encoder3(rd2)
        
        t1 = self.time_encoder1(x)
        t2 = self.time_encoder2(t1)
        t3 = self.time_encoder3(t2)

        t_bottleneck = self.time_bottleneck(t3)
        t_bottleneck = t_bottleneck.view(batch_size, 64, 1, 1)
        t_bottleneck = F.interpolate(t_bottleneck, size=(32, 16), mode='bilinear', align_corners=False)

        combined = torch.cat([rd3, t_bottleneck], dim=1)  # [B, 128, 32, 16]
        fused = self.fusion(combined)  # [B, 64, 32, 16]

        att_weights = self.channel_attention(fused)
        fused = fused * att_weights

        d1 = self.decoder1(fused)  # [B, 32, 64, 32]
        d1_skip = torch.cat([d1, rd2], dim=1)  
        
        d2 = self.decoder2(d1_skip)  # [B, 16, 128, 64]
        d2_skip = torch.cat([d2, rd1], dim=1)  
        
        output = self.decoder3(d2_skip)  # [B, 2, 256, 128]

        return output + x  
        
    def print_model_info(self):

        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        rd_encoder_params = (count_parameters(self.rd_encoder1) + 
                             count_parameters(self.rd_encoder2) + 
                             count_parameters(self.rd_encoder3))
        
        time_encoder_params = (count_parameters(self.time_encoder1) + 
                               count_parameters(self.time_encoder2) + 
                               count_parameters(self.time_encoder3))
        
        bottleneck_params = count_parameters(self.time_bottleneck)
        
        fusion_params = (count_parameters(self.fusion) + 
                         count_parameters(self.channel_attention))
        
        decoder_params = (count_parameters(self.decoder1) + 
                          count_parameters(self.decoder2) + 
                          count_parameters(self.decoder3))
        
        total_params = count_parameters(self)
        
        print(f"RD Encoder Parameters: {rd_encoder_params:,}")
        print(f"Time Encoder Parameters: {time_encoder_params:,}")
        print(f"Bottleneck Parameters: {bottleneck_params:,}")
        print(f"Fusion Parameters: {fusion_params:,}")
        print(f"Decoder Parameters: {decoder_params:,}")
        print(f"Total Parameters: {total_params:,}")


class DualPathRadarAutoencoder_V4_1(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm2d(2)
        
        # RD Domain Path (CNN) - keep as is
        self.rd_encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),     
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 128x64
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),    
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 64x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)   # 32x16 -> 32 channels
        )
        
        # Time Domain Path
        self.time_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 128x64
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 64x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)   # 32x16 -> 32 channels
        )
        
        # Feature fusion using 1x1 convolutions instead of FC
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),  # 96 = 32(RD) + 32*2(Time real/imag)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=1)
        )
        
        self.decoder = nn.Sequential(
            # 32x16 -> 64x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 64x32 -> 128x64
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 128x64 -> 256x128
            nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
        )
        
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        
        self.print_model_info()

    def forward(self, x, rd_maps):
        batch_size = x.size(0)
        
        # Normalize inputs
        x = self.input_norm(x)
        rd_maps = self.input_norm(rd_maps)
        
        # RD domain path
        rd_features = self.rd_encoder(rd_maps)  # [B, 32, 32, 16]
        
        # Time domain path
        x_real = x[:, 0:1, :, :]
        x_imag = x[:, 1:2, :, :]
        
        real_features = self.time_cnn(x_real)  # [B, 32, 32, 16]
        imag_features = self.time_cnn(x_imag)  # [B, 32, 32, 16]
        
        # Concatenate features along channel dimension
        combined = torch.cat([rd_features, real_features, imag_features], dim=1)  # [B, 96, 32, 16]
        
        # Fuse features
        fused = self.fusion_conv(combined)  # [B, 32, 32, 16]
        
        # Decode
        output = self.decoder(fused)  # [B, 2, 256, 128]
        
        return output

    # [print_model_info function remains the same]
    def print_model_info(self):
        """Print detailed model parameter counts"""
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        print("\nModel Parameter Count:")
        
        # RD path
        rd_params = count_parameters(self.rd_encoder)
        print(f"RD Path (CNN): {rd_params:,}")
        
        # Time path
        time_cnn_params = count_parameters(self.time_cnn)
        print(f"Time Path (CNN): {time_cnn_params:,}")
        
        # Feature fusion
        fusion_params = count_parameters(self.fusion_conv)
        print(f"Feature Fusion: {fusion_params:,}")
        
        # Decoder
        decoder_params = count_parameters(self.decoder)
        print(f"Decoder: {decoder_params:,}")
        
        # Total
        total = count_parameters(self)
        print(f"Total Parameters: {total:,}")
        
        if total > 3_000_000:
            print("\nWarning: Total parameters exceed 3 million!")
            
        return {
            'rd_path': rd_params,
            'time_cnn': time_cnn_params,
            'fusion': fusion_params,
            'decoder': decoder_params,
            'total': total
        }

    

class DualPathRadarAutoencoder_v4_2(nn.Module):
   def __init__(self, num_samples=256, num_chirps=128):
       super().__init__()
       
       # Input normalization
       self.input_norm = nn.BatchNorm2d(2)
       
       # RD Domain Path CNN
       self.rd_encoder = nn.Sequential(
           nn.Conv2d(2, 8, kernel_size=3, padding=1),     
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 128x64
           
           nn.Conv2d(8, 16, kernel_size=3, padding=1),    
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 64x32
           
           nn.Conv2d(16, 32, kernel_size=3, padding=1),   
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2)   # 32x16 -> 32 channels
       )
       
       # Time Domain Path CNN
       self.time_cnn = nn.Sequential(
           nn.Conv2d(1, 8, kernel_size=3, padding=1),
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 128x64
           
           nn.Conv2d(8, 16, kernel_size=3, padding=1),
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2),  # 64x32
           
           nn.Conv2d(16, 32, kernel_size=3, padding=1),
           nn.LeakyReLU(0.2),
           nn.MaxPool2d(2)   # 32x16 -> 32 channels
       )
       
       # Attention
       self.attention = nn.Sequential(
           nn.Conv2d(96, 96, kernel_size=1),  # 96 = 32(RD) + 32*2(Time)
           nn.Sigmoid()
       )
       
       # Global Average Pooling
       self.gap = nn.AdaptiveAvgPool2d(1)
       
       # Small FC layer for global features
       self.fc = nn.Sequential(
           nn.Linear(96, 256),
           nn.LeakyReLU(0.2),
           nn.Linear(256, 32)
       )
       
       # Feature fusion with 1x1 convolutions
       self.fusion_conv = nn.Sequential(
           nn.Conv2d(128, 64, kernel_size=1),  # 128 = 96 + 32(FC features)
           nn.LeakyReLU(0.2),
           nn.Conv2d(64, 32, kernel_size=1)
       )
       
       # Decoder
       self.decoder = nn.Sequential(
           # 32x16 -> 64x32
           nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
           nn.LeakyReLU(0.2),
           
           # 64x32 -> 128x64
           nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
           nn.LeakyReLU(0.2),
           
           # 128x64 -> 256x128
           nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
       )
       
       self.num_samples = num_samples
       self.num_chirps = num_chirps
       
       self.print_model_info()

   def forward(self, x, rd_maps):
       batch_size = x.size(0)
       
       # Normalize inputs
       x = self.input_norm(x)
       rd_maps = self.input_norm(rd_maps)
       
       # RD domain path
       rd_features = self.rd_encoder(rd_maps)  # [B, 32, 32, 16]
       
       # Time domain path
       x_real = x[:, 0:1, :, :]
       x_imag = x[:, 1:2, :, :]
       real_features = self.time_cnn(x_real)   # [B, 32, 32, 16]
       imag_features = self.time_cnn(x_imag)   # [B, 32, 32, 16]
       
       # Concatenate features
       combined = torch.cat([rd_features, real_features, imag_features], dim=1)  # [B, 96, 32, 16]
       
       # Apply attention
       attention_weights = self.attention(combined)
       attended_features = combined * attention_weights
       
       # Global features through GAP and FC
       global_features = self.gap(attended_features).flatten(1)  # [B, 96]
       global_features = self.fc(global_features)               # [B, 32]
       
       # Expand global features to match feature map size
       global_features = global_features.view(batch_size, 32, 1, 1)
       global_features = global_features.expand(-1, -1, 32, 16)
       
       # Concatenate with attended features
       fusion_input = torch.cat([attended_features, global_features], dim=1)  # [B, 128, 32, 16]
       
       # Fuse features
       fused = self.fusion_conv(fusion_input)  # [B, 32, 32, 16]
       
       # Decode
       output = self.decoder(fused)  # [B, 2, 256, 128]
       
       return output

   def print_model_info(self):
       """Print detailed model parameter counts"""
       def count_parameters(module):
           return sum(p.numel() for p in module.parameters())
       
       print("\nModel Parameter Count:")
       
       rd_params = count_parameters(self.rd_encoder)
       print(f"RD Path (CNN): {rd_params:,}")
       
       time_cnn_params = count_parameters(self.time_cnn)
       print(f"Time Path (CNN): {time_cnn_params:,}")
       
       attention_params = count_parameters(self.attention)
       print(f"Attention: {attention_params:,}")
       
       fc_params = count_parameters(self.fc)
       print(f"FC Layer: {fc_params:,}")
       
       fusion_params = count_parameters(self.fusion_conv)
       print(f"Feature Fusion: {fusion_params:,}")
       
       decoder_params = count_parameters(self.decoder)
       print(f"Decoder: {decoder_params:,}")
       
       total = count_parameters(self)
       print(f"Total Parameters: {total:,}")
       
       if total > 3_000_000:
           print("\nWarning: Total parameters exceed 3 million!")
           
       return {
           'rd_path': rd_params,
           'time_cnn': time_cnn_params,
           'attention': attention_params,
           'fc': fc_params,
           'fusion': fusion_params,
           'decoder': decoder_params,
           'total': total
       }


class DualPathRadarAutoencoder_v4_3(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()
        self.rd_encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        
        self.rd_encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        
        self.rd_encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        
        # Time Domain Path
        self.time_encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        
        self.time_encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        
        self.time_encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
 
        self.fusion = nn.Conv2d(128, 64, kernel_size=1)  
 
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256x128
            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )
        
        self.num_samples = num_samples
        self.num_chirps = num_chirps

    def forward(self, x, rd_maps):
        batch_size = x.size(0)
        

        rd1 = self.rd_encoder1(rd_maps)
        rd2 = self.rd_encoder2(rd1)
        rd3 = self.rd_encoder3(rd2)
        

        t1 = self.time_encoder1(x)
        t2 = self.time_encoder2(t1)
        t3 = self.time_encoder3(t2)
        

        combined = torch.cat([rd3, t3], dim=1)  
        att_weights = self.channel_attention(combined)  
        combined_weighted = combined * att_weights
        fused = self.fusion(combined_weighted)  # [B, 64, 32, 16]
        

        d1 = self.decoder1(fused)  # [B, 32, 64, 32]
        d1_skip = torch.cat([d1, rd2 + t2], dim=1)  
        d2 = self.decoder2(d1_skip)  # [B, 16, 128, 64]
        d2_skip = torch.cat([d2, rd1 + t1], dim=1)  
        
        output = self.decoder3(d2_skip)  # [B, 2, 256, 128]
        
        return output




class DualPathRadarAutoencoder_init(nn.Module):
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()

        self.shape_printed = False  # printing flag

        # Calculate RD map size
        self.rd_height = num_samples
        self.rd_width = num_chirps

        # Calculate feature dimensions after conv layers
        self.feature_h = num_samples // 8  # 256/8 = 32
        self.feature_w = num_chirps // 8   # 128/8 = 16

        # RD Domain Path (CNN)
        self.rd_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 2 channels: magnitude and phase
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 512x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 256x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x16
        )

        
        # Time Path - adjusted for 256x128 input
        time_input_size = num_samples * num_chirps  # 256*128 = 32,768
        self.time_encoder = nn.Sequential(
            nn.Linear(time_input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        # Calculate combined feature size
        rd_feature_size = 64 * self.feature_h * self.feature_w  # 64*32*16
        combined_size = 256 + rd_feature_size  # time features + rd features

        # Combined decoder - adjusted accordingly
        self.combined_decoder = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, time_input_size)
        )

        print(f"\nModel Architecture Info:")
        print(f"Input dimensions: {num_samples}x{num_chirps}")
        self.print_parameter_count()

    def print_parameter_count(self):
        """Print parameter count for each component"""
        rd_params = sum(p.numel() for p in self.rd_encoder.parameters())
        time_params = sum(p.numel() for p in self.time_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.combined_decoder.parameters())
        total_params = rd_params + time_params + decoder_params

        print(f"\nParameter Count:")
        print(f"RD Path: {rd_params:,}")
        print(f"Time Path: {time_params:,}")
        print(f"Decoder: {decoder_params:,}")
        print(f"Total: {total_params:,}")

    def forward(self, x, rd_maps):
        batch_size = x.size(0)

        # Print shapes only once
        if not self.shape_printed:
            print(f"\nInput shapes:")
            print(f"Raw signal: {x.shape}")
            print(f"RD maps: {rd_maps.shape}")
            print(f"RD output shape: {torch.Size([batch_size, 2, self.rd_height, self.rd_width])}")

        # Process RD domain
        rd_features = self.rd_encoder(rd_maps)
        rd_features_flat = rd_features.flatten(1)

        # Process time domain (real and imag)
        x_real = x[:, 0, :, :].reshape(batch_size, -1)
        x_imag = x[:, 1, :, :].reshape(batch_size, -1)

        time_features_real = self.time_encoder(x_real)
        time_features_imag = self.time_encoder(x_imag)

        # Combine features
        combined_real = torch.cat([time_features_real, rd_features_flat], dim=1)
        combined_imag = torch.cat([time_features_imag, rd_features_flat], dim=1)

        # Decode
        output_real = self.combined_decoder(combined_real)
        output_imag = self.combined_decoder(combined_imag)

        # Reshape outputs
        output_real = output_real.view(batch_size, 1, self.rd_height, self.rd_width)
        output_imag = output_imag.view(batch_size, 1, self.rd_height, self.rd_width)

        # Debug prints for output shapes
        output = torch.cat([output_real, output_imag], dim=1)

        # Print shapes only once
        if not self.shape_printed:
            print(f"Combined output shape: {output.shape}")
            self.shape_printed = True

        return output