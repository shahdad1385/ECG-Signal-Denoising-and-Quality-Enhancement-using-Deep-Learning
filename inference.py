import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from torch.utils.data import DataLoader
from ecg_dataset import ECGDenoisingDataset
from models.unet_1d import UNet1D
from models.lu_net_1d import LUNet1D
from metrics import evaluate_signal_batch

# Set a professional, clean style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def plot_spectrogram(fig, axis, signal, title, fs=50):
    """Generates a high-fidelity STFT spectrogram with a localized colorbar."""
    f, t, Zxx = stft(signal, fs=fs, nperseg=64)
    # Using 'viridis' or 'magma' for higher perceptual uniformity in scientific publications
    mesh = axis.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.4, shading='gouraud', cmap='magma')
    axis.set_title(title, fontsize=11, fontweight='semibold', pad=8)
    axis.set_ylabel("Freq (Hz)", labelpad=4)
    axis.grid(False) # Spectrograms look cleaner without mesh grid lines overlaid
    
    # Add a dedicated, slim colorbar to each individual subplot
    cbar = fig.colorbar(mesh, ax=axis, pad=0.02, aspect=15)
    cbar.ax.tick_params(labelsize=8)
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
    single_sample_metrics = {}
    
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
                
            if i == 0: # Cache individual waveform arrays and their explicit metrics for the visual overlay
                single_sample_metrics = {"unet": m_u, "lunet": m_lu}
                sample_packet = {
                    "noisy": noisy.cpu().squeeze().numpy(),
                    "clean": clean.cpu().squeeze().numpy(),
                    "unet": out_u.cpu().squeeze().numpy(),
                    "lunet": out_lu.cpu().squeeze().numpy()
                }

    # Print summary performance table to the console
    print("\n" + "="*50)
    print(f"{'METRIC SUMMARY COMPARISON':^50}")
    print("="*50)
    print(f"{'Metric':<15} | {'Standard U-Net':<15} | {'Advanced LU-Net':<15}")
    print("-"*50)
    for k in unet_metrics.keys():
        print(f"{k:<15} | {np.mean(unet_metrics[k]):<15.4f} | {np.mean(lunet_metrics[k]):<15.4f}")
    print("="*50)

    # --- Begin Scientific Matrix Plotting Sequence ---
    fig, axs = plt.subplots(4, 2, figsize=(14, 11), sharex='col')
    fig.suptitle("Physiological ECG Signal Denoising: Time-Frequency Comparative Analysis", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    signals = [sample_packet['clean'], sample_packet['noisy'], sample_packet['unet'], sample_packet['lunet']]
    titles = ["(a) Clean ECG Target Reference", "(b) Noisy ECG Input (3dB SNR + Wander)", 
              "(c) Denoised Output: Standard 1D U-Net", "(d) Denoised Output: Advanced 1D LU-Net"]
    colors = ['#2ebd59', '#e03e3e', '#7b2cbf', '#1d4ed8'] # Clean emerald, deep red, crisp purple, royal blue
    
    for i in range(4):
        # Left Column: Time-Domain Waveforms
        axs[i, 0].plot(signals[i], color=colors[i], lw=1.25, alpha=0.9)
        axs[i, 0].set_title(titles[i], fontsize=11, fontweight='bold', loc='left', pad=6)
        axs[i, 0].set_ylabel("Normalized Amp")
        axs[i, 0].set_xlim(0, len(signals[i]))
        axs[i, 0].grid(True, linestyle=':', alpha=0.6)
        
        # Overlay specific sample metrics directly onto the subplots for fast visual verification
        if i == 2: # Standard U-Net metrics text box
            mu = single_sample_metrics['unet']
            metric_text = f"SNR Imp: +{mu['SNR_Imp']:.2f}dB\nPRD: {mu['PRD']:.1f}%\nSI-SDR: {mu['SI_SDR']:.1f}dB"
            axs[i, 0].text(0.02, 0.08, metric_text, transform=axs[i, 0].transAxes, fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#7b2cbf'))
        elif i == 3: # Advanced LU-Net metrics text box
            mlu = single_sample_metrics['lunet']
            metric_text = f"SNR Imp: +{mlu['SNR_Imp']:.2f}dB\nPRD: {mlu['PRD']:.1f}%\nSI-SDR: {mlu['SI_SDR']:.1f}dB"
            axs[i, 0].text(0.02, 0.08, metric_text, transform=axs[i, 0].transAxes, fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#1d4ed8'))

        # Right Column: Frequency-Domain Short-Time Fourier Transform Spectrograms
        plot_spectrogram(fig, axs[i, 1], signals[i], f"STFT Power Spectrum: Baseline Asset {i+1}")

    # Set bottom axis tracking labels safely
    axs[3, 0].set_xlabel("Time Domain Samples (N)", labelpad=6)
    axs[3, 1].set_xlabel("Temporal Windows (t)", labelpad=6)

    # Adjust layout dynamically, compile file buffers, and export at publication standard (300 DPI)
    plt.tight_layout()
    plt.savefig(config['outputs']['report_img'], dpi=300, bbox_inches='tight')
    print(f"\n[Success] Scientific matrix layout exported smoothly to: {config['outputs']['report_img']}\n")
    plt.show()

if __name__ == "__main__":
    main()