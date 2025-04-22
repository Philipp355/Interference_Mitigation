import torch
import torch.nn as nn
import torch.nn.functional as F

class RefDualPathAutoencoder(nn.Module):
    """Combined Reference Dual Path Autoencoder for radar data"""
    def __init__(self, num_samples=256, num_chirps=128):
        super().__init__()
        

        self.time_enc1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        

        self.time_enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        

        self.time_enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        

        self.rd_enc1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 128x64
        )
        

        self.rd_enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 64x32
        )
        

        self.rd_enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)  # 32x16
        )
        

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 48, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 192, kernel_size=1),
            nn.Sigmoid()
        )
        
  
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),  
            nn.LeakyReLU(0.2)
        )
        

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 64x32
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 128x64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 256x128
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        

        self.output_conv = nn.Conv2d(32, 2, kernel_size=1)
        
    def forward(self, time_input, rd_input):


        t1 = self.time_enc1(time_input)
        t2 = self.time_enc2(t1)
        time_features = self.time_enc3(t2)  # [B, 64, 32, 16]
        

        r1 = self.rd_enc1(rd_input)
        r2 = self.rd_enc2(r1)
        rd_features = self.rd_enc3(r2)  # [B, 128, 32, 16]
        

        combined = torch.cat([time_features, rd_features], dim=1)  # [B, 192, 32, 16]
        

        att_weights = self.attention(combined)
        attended = combined * att_weights
        

        fused = self.fusion(attended)  # [B, 128, 32, 16]
        

        bottleneck = self.bottleneck(fused)  # [B, 256, 32, 16]
        

        dec1 = self.decoder1(bottleneck)  # [B, 128, 64, 32]
        dec2 = self.decoder2(dec1)        # [B, 64, 128, 64]
        dec3 = self.decoder3(dec2)        # [B, 32, 256, 128]
        

        output = self.output_conv(dec3)   # [B, 2, 256, 128]
        

        
        return output
        
    def count_parameters(self):

        time_params = sum(p.numel() for p in self.time_enc1.parameters()) + \
                      sum(p.numel() for p in self.time_enc2.parameters()) + \
                      sum(p.numel() for p in self.time_enc3.parameters())
                      
        rd_params = sum(p.numel() for p in self.rd_enc1.parameters()) + \
                    sum(p.numel() for p in self.rd_enc2.parameters()) + \
                    sum(p.numel() for p in self.rd_enc3.parameters())
                    
        attention_params = sum(p.numel() for p in self.attention.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters())
        
        decoder_params = sum(p.numel() for p in self.decoder1.parameters()) + \
                         sum(p.numel() for p in self.decoder2.parameters()) + \
                         sum(p.numel() for p in self.decoder3.parameters()) + \
                         sum(p.numel() for p in self.output_conv.parameters())
        
        total = sum(p.numel() for p in self.parameters())
        
        print(f"model complexity:")
        print(f"  time: {time_params:,} ")
        print(f"  RD: {rd_params:,} ")
        print(f"  attention: {attention_params:,} ")
        print(f"  fusion: {fusion_params:,} ")
        print(f"  bottleneck: {bottleneck_params:,} ")
        print(f"  decoder: {decoder_params:,} ")
        print(f"total param: {total:,}")
        
        return {
            'time_path': time_params,
            'rd_path': rd_params,
            'attention': attention_params,
            'fusion': fusion_params,
            'bottleneck': bottleneck_params,
            'decoder': decoder_params,
            'total': total
        }