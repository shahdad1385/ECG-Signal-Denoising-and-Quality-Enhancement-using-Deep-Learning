import torch
import numpy as np

def calculate_rmse(clean, denoised):
    """Computes Root Mean Squared Error."""
    return np.sqrt(np.mean((clean - denoised) ** 2))

def calculate_prd(clean, denoised):
    """Computes Percent Root-mean-square Difference indicating distortion."""
    numerator = np.sum((clean - denoised) ** 2)
    denominator = np.sum(clean ** 2) + 1e-8
    return np.sqrt(numerator / denominator) * 100

def calculate_snr_improvement(clean, noisy, denoised):
    """Computes the net decibel gain in Signal-to-Noise Ratio."""
    noise_old = noisy - clean
    noise_new = denoised - clean
    
    snr_initial = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise_old ** 2) + 1e-8))
    snr_final = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise_new ** 2) + 1e-8))
    
    return snr_final - snr_initial

def calculate_si_sdr(clean, denoised):
    """Computes Scale-Invariant Signal-to-Distortion Ratio."""
    clean = clean - np.mean(clean)
    denoised = denoised - np.mean(denoised)
    
    target = (np.dot(denoised, clean) / (np.sum(clean ** 2) + 1e-8)) * clean
    artifact = denoised - target
    
    sdr = 10 * np.log10(np.sum(target ** 2) / (np.sum(artifact ** 2) + 1e-8))
    return sdr

def evaluate_signal_batch(clean_tensor, noisy_tensor, denoised_tensor):
    """Extracts tensors and computes aggregate evaluations across arrays."""
    c = clean_tensor.cpu().squeeze().numpy()
    n = noisy_tensor.cpu().squeeze().numpy()
    d = denoised_tensor.cpu().squeeze().numpy()
    
    # Handle single-sample conversions vs stacked array loops
    if c.ndim == 1:
        return {
            "RMSE": calculate_rmse(c, d),
            "PRD": calculate_prd(c, d),
            "SNR_Imp": calculate_snr_improvement(c, n, d),
            "SI_SDR": calculate_si_sdr(c, d)
        }
        
    rmses, prds, snrs, sdrs = [], [], [], []
    for i in range(c.shape[0]):
        rmses.append(calculate_rmse(c[i], d[i]))
        prds.append(calculate_prd(c[i], d[i]))
        snrs.append(calculate_snr_improvement(c[i], n[i], d[i]))
        sdrs.append(calculate_si_sdr(c[i], d[i]))
        
    return {
        "RMSE": np.mean(rmses),
        "PRD": np.mean(prds),
        "SNR_Imp": np.mean(snrs),
        "SI_SDR": np.mean(sdrs)
    }