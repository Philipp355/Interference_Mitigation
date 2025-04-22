import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with two conv layers and ReLU activation"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class RefConvAutoencoder(nn.Module):
    """Reference Convolutional Autoencoder for radar data"""
    def __init__(self):
        super(RefConvAutoencoder, self).__init__()
        
        # Encoder blocks
        self.enc1 = ConvBlock(2, 32)          
        self.enc2 = ConvBlock(32, 64)         
        self.enc3 = ConvBlock(64, 128)        
        self.enc4 = ConvBlock(128, 256)       
        
        # Pool layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512) # Input: [B, 256, 16, 8] -> Output: [B, 512, 16, 8]
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    
        
        # Decoder blocks
        self.dec1 = ConvBlock(512, 256)       
        self.dec2 = ConvBlock(256, 128)       
        self.dec3 = ConvBlock(128, 64)       
        self.dec4 = ConvBlock(64, 32)       
        
        # Output layer
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)  
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 2, 256, 128] - real and imaginary parts
            
        Returns:
            Reconstructed clean tensor [batch_size, 2, 256, 128]
        """
        # Encoder path with skip connections
        enc1_out = self.enc1(x)              # [B, 32, 256, 128]
        p1 = self.pool(enc1_out)             # [B, 32, 128, 64]
        
        enc2_out = self.enc2(p1)             # [B, 64, 128, 64]
        p2 = self.pool(enc2_out)             # [B, 64, 64, 32]
        
        enc3_out = self.enc3(p2)             # [B, 128, 64, 32]
        p3 = self.pool(enc3_out)             # [B, 128, 32, 16]
        
        enc4_out = self.enc4(p3)             # [B, 256, 32, 16]
        p4 = self.pool(enc4_out)             # [B, 256, 16, 8]
        
        # Bottleneck
        bottleneck_out = self.bottleneck(p4) # [B, 512, 16, 8]
        
        # Decoder path with skip connections
        up1 = self.up1(bottleneck_out)                      # [B, 256, 32, 16]
        merged1 = torch.cat([enc4_out, up1], dim=1)         # [B, 512, 32, 16]
        dec1_out = self.dec1(merged1)                       # [B, 256, 32, 16]
        
        up2 = self.up2(dec1_out)                            # [B, 128, 64, 32]
        merged2 = torch.cat([enc3_out, up2], dim=1)         # [B, 256, 64, 32]
        dec2_out = self.dec2(merged2)                       # [B, 128, 64, 32]
        
        up3 = self.up3(dec2_out)                            # [B, 64, 128, 64]
        merged3 = torch.cat([enc2_out, up3], dim=1)         # [B, 128, 128, 64]
        dec3_out = self.dec3(merged3)                       # [B, 64, 128, 64]
        
        up4 = self.up4(dec3_out)                            # [B, 32, 256, 128]
        merged4 = torch.cat([enc1_out, up4], dim=1)         # [B, 64, 256, 128]
        dec4_out = self.dec4(merged4)                       # [B, 32, 256, 128]
        
        # Output layer
        output = self.out_conv(dec4_out)                    # [B, 2, 256, 128]
        
        return output