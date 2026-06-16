import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDenoisingDataset(Dataset):
    def __init__(self, num_samples=1000, signal_length=512, snr_db=5):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.snr_db = snr_db
        
        self.clean_signals = []
        self.noisy_signals = []
        
        # Trigger data generation pipeline immediately upon instantiation
        self._generate_data()
        
    def _generate_data(self):
        # Establish a linear temporal baseline for signal generation
        t = np.linspace(0, 10, self.signal_length)
        
        for _ in range(self.num_samples):
            # 1. Synthesize Clean Heartbeat Waveform
            # Standard sine wave establishes the baseline heartbeat frequency
            clean = np.sin(2 * np.pi * 1.2 * t) 
            # Injected higher frequency components simulate a sharp QRS ventricular spike
            clean += 0.5 * np.sin(2 * np.pi * 3.6 * t) * (clean > 0.5) 
            
            # 2. Synthesize Composite Noise Artifacts
            # Gaussian White Noise simulates high-frequency thermal/sensor jitters
            white_noise = np.random.normal(0, 1, self.signal_length)
            # Low-frequency drift simulates physical patient breathing or electrode displacement
            baseline_wander = 0.3 * np.sin(2 * np.pi * 0.1 * t) 
            total_noise = white_noise + baseline_wander
            
            # 3. Calculate Signal-to-Noise Ratio (SNR) Scalers
            # Determine fundamental root mean square power levels
            signal_power = np.mean(clean ** 2)
            noise_power = np.mean(total_noise ** 2)
            
            # Calculate the scaling factor 'k' using the requested decibel (dB) specification
            k = np.sqrt(signal_power / (noise_power * (10 ** (self.snr_db / 10))))
            # Construct the final corrupted input variant
            noisy = clean + k * total_noise
            
            # 4. Standardize and Normalize Signals
            # Maps data arrays tightly within a uniform [-1, 1] amplitude bounds
            clean = (clean - np.min(clean)) / (np.max(clean) - np.min(clean)) * 2 - 1
            noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy)) * 2 - 1
            
            self.clean_signals.append(clean)
            self.noisy_signals.append(noisy)
            
    def __len__(self):
        # Returns total length of dataset to PyTorch internal iterators
        return self.num_samples
    
    def __getitem__(self, idx):
        # Pack data tensors into [Channels, Length] format expected by 1D Convolution layers
        clean = torch.tensor(self.clean_signals[idx], dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(self.noisy_signals[idx], dtype=torch.float32).unsqueeze(0)
        return noisy, clean