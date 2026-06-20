import torch
import torch.nn as nn

class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        
        # --- Encoder (Feature Extraction & Downsampling) ---
        # Block 1: Extracts low-level temporal features. Local spatial size remains unchanged.
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # Halves the signal length (e.g., from 512 to 256)
        self.pool1 = nn.MaxPool1d(kernel_size=2) 
        
        # Block 2: Extracts higher-level abstract features.
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Halves the signal length again (e.g., from 256 to 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2) 

        # --- Bottleneck (Latent Space Representation) ---
        # The deepest layer of the network capturing global sequence dependencies.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # --- Decoder (Signal Reconstruction & Upsampling) ---
        # Upsamples features back by a factor of 2 (e.g., from 128 to 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # After concat with enc2 (64 channels), input channels = 128 + 64 = 192
        self.dec2 = nn.Sequential(
            nn.Conv1d(in_channels=128 + 64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Upsamples features back to original scale (e.g., from 256 to 512)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # After concat with enc1 (32 channels), input channels = 64 + 32 = 96
        self.dec1 = nn.Sequential(
            nn.Conv1d(in_channels=64 + 32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Final Layer: Convolves channels back to 1 to match target clean ECG shape
        self.final = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        # --- Encoder Path ---
        e1 = self.enc1(x)       # Shape: [Batch, 32, Length]
        p1 = self.pool1(e1)     # Shape: [Batch, 32, Length // 2]
        
        e2 = self.enc2(p1)      # Shape: [Batch, 64, Length // 2]
        p2 = self.pool2(e2)     # Shape: [Batch, 64, Length // 4]
        
        # --- Bottleneck ---
        b = self.bottleneck(p2) # Shape: [Batch, 128, Length // 4]
        
        # --- Decoder Path with Skip Connections ---
        u2 = self.up2(b)        # Upsample to [Batch, 128, Length // 2]
        cat2 = torch.cat([u2, e2], dim=1) # Concatenate along channel axis -> [Batch, 192, Length // 2]
        d2 = self.dec2(cat2)    # Smooth out features -> [Batch, 64, Length // 2]
        
        u1 = self.up1(d2)       # Upsample to [Batch, 64, Length]
        cat1 = torch.cat([u1, e1], dim=1) # Concatenate along channel axis -> [Batch, 96, Length]
        d1 = self.dec1(cat1)    # Smooth out features -> [Batch, 32, Length]
        
        # Projection layer
        return self.final(d1)   # Target Shape: [Batch, 1, Length]