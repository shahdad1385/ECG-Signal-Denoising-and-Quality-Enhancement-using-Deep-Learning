import torch
import torch.nn as nn

class BiLSTMBridge(nn.Module):
    """
    BiLSTM module placed within the skip connections to capture bidirectional
    temporal context from encoder feature maps before joining the decoder.
    """
    def __init__(self, in_channels):
        super(BiLSTMBridge, self).__init__()
        # Using hidden_size = in_channels // 2 so bidirectional output equals in_channels
        self.hidden_size = in_channels // 2
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        # LSTM expects: [Batch, Sequence_Length, Features]
        b, c, l = x.size()
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        # Bring back to shape: [Batch, Channels, Length]
        x = lstm_out.transpose(1, 2)
        return x

class LUNet1D(nn.Module):
    """
    LU-Net 1D Architecture inspired by the 2023 Heart Sound Denoising Paper.
    Integrates BiLSTMs within skip connections of a 1D U-Net backbone.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(LUNet1D, self).__init__()
        
        # --- Encoder Path ---
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Length // 2
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Length // 4
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2) # Length // 8

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # --- LU-Net BiLSTM Bridges (Skip Connections) ---
        self.bridge1 = BiLSTMBridge(in_channels=32)
        self.bridge2 = BiLSTMBridge(in_channels=64)
        self.bridge3 = BiLSTMBridge(in_channels=128)

        # --- Decoder Path ---
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # Input channels: 256 (from upsampled bottleneck) + 128 (from bridge3)
        self.dec3 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # Input channels: 128 (from upsampled dec3) + 64 (from bridge2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # Input channels: 64 (from upsampled dec2) + 32 (from bridge1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(64 + 32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Output layer mapping channels down to original 1D state
        self.final = nn.Conv1d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder passes
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Latent Space Bottleneck
        b = self.bottleneck(p3)
        
        # Process Skip connections via Recurrent Bridges
        bridge1_out = self.bridge1(e1)
        bridge2_out = self.bridge2(e2)
        bridge3_out = self.bridge3(e3)
        
        # Decoder passes paired with Skip Connections
        u3 = self.up3(b)
        cat3 = torch.cat([u3, bridge3_out], dim=1)
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d2 if 'd2' in locals() else d3)
        cat2 = torch.cat([u2, bridge2_out], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, bridge1_out], dim=1)
        d1 = self.dec1(cat1)
        
        return self.final(d1)