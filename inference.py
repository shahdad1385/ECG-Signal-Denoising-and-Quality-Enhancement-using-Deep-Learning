import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from torch.utils.data import DataLoader
from ecg_dataset import ECGDenoisingDataset
from models.unet_1d import UNet1D
from models.lu_net_1d import LUNet1D
from metrics import evaluate_signal_batch

def plot_spectrogram(axis, signal, title, fs=50):
    """Generates an STFT spectrogram profile on a target plot grid."""
    f, t, Zxx = stft(signal, fs=fs, nperseg=64)
    mesh = axis.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.4, shading='gouraud', cmap='inferno')
    axis.set_title(title, fontsize=10)
    axis.set_ylabel("Freq (Hz)")
    return mesh

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() and config['system']['device'] == 'cuda' else 'cpu')
    
    # Load test tracking structures
    d_cfg = config['dataset']
    test_dataset = ECGDenoisingDataset(d_cfg['test_samples'], d_cfg['signal_length'], d_cfg['snr_db'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize networks and pull frozen weight parameters
    unet = UNet1D().to(device)
    lunet = LUNet1D().to(device)
    
    unet.load_state_dict(torch.load(os.path.join(config['outputs']['weights_dir'], "unet1d_best.pth"), map_location=device))
    lunet.load_state_dict(torch.load(os.path.join(config['outputs']['weights_dir'], "lunet1d_best.pth"), map_location=device))
    
    unet.eval()
    lunet.eval()
    
    # Aggregate benchmarking metrics
    unet_metrics = {"RMSE": [], "PRD": [], "SNR_Imp": [], "SI_SDR": []}
    lunet_metrics = {"RMSE": [], "PRD": [], "SNR_Imp": [], "SI_SDR": []}
    
    sample_packet = None
    
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(test_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            out_u = unet(noisy)
            out_lu = lunet(noisy)
            
            m_u = evaluate_signal_batch(clean, noisy, out_u)
            m_lu = evaluate_signal_batch(clean, noisy, out_lu)
            
            for k in unet_metrics.keys():
                unet_metrics[k].append(m_u[k])
                lunet_metrics[k].append(m_lu[k])
                
            if i == 0: # Cache individual waveform arrays for generating visual plots
                sample_packet = {
                    "noisy": noisy.cpu().squeeze().numpy(),
                    "clean": clean.cpu().squeeze().numpy(),
                    "unet": out_u.cpu().squeeze().numpy(),
                    "lunet": out_lu.cpu().squeeze().numpy()
                }

    # Print summary performance table
    print("\n" + "="*50)
    print(f"{'METRIC SUMMARY COMPARISON':^50}")
    print("="*50)
    print(f"{'Metric':<15} | {'Standard U-Net':<15} | {'Advanced LU-Net':<15}")
    print("-"*50)
    for k in unet_metrics.keys():
        print(f"{k:<15} | {np.mean(unet_metrics[k]):<15.4f} | {np.mean(lunet_metrics[k]):<15.4f}")
    print("="*50)

    # Begin Matrix Visualization Plotting Sequence
    fig, axs = plt.subplots(4, 2, figsize=(15, 12))
    signals = [sample_packet['clean'], sample_packet['noisy'], sample_packet['unet'], sample_packet['lunet']]
    titles = ["Clean ECG Target", "Noisy ECG Input", "Standard 1D U-Net Output", "Advanced 1D LU-Net Output"]
    colors = ['green', 'red', 'purple', 'blue']
    
    for i in range(4):
        # Left columns hold traditional Time Domain Waveforms
        axs[i, 0].plot(signals[i], color=colors[i], lw=1.5)
        axs[i, 0].set_title(titles[i], fontsize=11, fontweight='bold')
        axs[i, 0].grid(True, linestyle='--', alpha=0.6)
        axs[i, 0].set_ylabel("Amplitude")
        if i == 3: axs[i, 0].set_xlabel("Time Samples")
        
        # Right columns hold corresponding Short-Time Fourier Transform Spectrograms
        plot_spectrogram(axs[i, 1], signals[i], f"STFT Spectrogram: {titles[i]}")
        if i == 3: axs[i, 1].set_xlabel("Time (sec)")

    plt.tight_layout()
    plt.savefig(config['outputs']['report_img'], dpi=300)
    print(f"\nMatrix visualization successfully generated and saved to: {config['outputs']['report_img']}")
    plt.show()

if __name__ == "__main__":
    main()