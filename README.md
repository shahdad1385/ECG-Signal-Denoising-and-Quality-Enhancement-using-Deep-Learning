# 🫀 Deep Learning-Based ECG Signal Denoising Framework: 1D U-Net vs. LU-Net

An end-to-end, production-ready Biomedical AI benchmarking framework designed to eliminate severe clinical noise artifacts from Electrocardiogram (ECG) signals. This repository features a rigorous structural comparison between a standard 1D U-Net Autoencoder and an advanced 1D LU-Net (LSTM U-Net) framework. Both pipelines map corrupted physiological waveforms back to their pristine biological structures, cleanly preserving micro-temporal features (*P*-wave, *QRS* complex, and *T*-wave).

## 📌 Project Overview & Academic Inspiration

In ambulatory and hospital settings, ECG readings are highly susceptible to real-time physical interference:

- **Baseline Wander (BW)**: Low-frequency drift caused by patient respiration or movement.
- **Electromyographic (EMG) Artifacts**: High-frequency electrical noise from muscle tremors.
- **Powerline Interference**: 50/60 Hz electromagnetic hum from nearby hospital infrastructure.

While standard mathematical filters often flatten diagnostic peaks, this project explores deep learning alternatives. The core upgrade features an advanced 1D LU-Net variant inspired directly by the methodology presented in the following landmark IEEE publication:

**📄 Academic Reference**: Shams Nafisa Ali, Samiul Based Shuvo, and Muhammad Ishtiaque Sayeed, "An End-to-end Deep Learning Framework for Real-Time Denoising of Heart Sounds for Cardiac Disease Detection in Unseen Noise", *IEEE Transactions on Biomedical Engineering*, Vol. XX, No. XX, 2023.

## 🏗️ Architecture Profiles

### 1. Standard 1D U-Net Baseline
A symmetric encoder-decoder convolutional network. High-resolution spatial features are copied directly across parallel horizontal skip connections to assist the decoder in recovering lost resolution.

### 2. Advanced 1D LU-Net (LSTM U-Net Upgrade)
Inspired by Ali et al. (2023), this network embeds Bidirectional LSTM (BiLSTM) Bridges right inside the skip connections. Instead of copying raw feature maps blindly, the encoder's output passes through a sequential recurrent layer. This layout allows the model to process spatial dimensions while tracking long-range temporal dependencies before joining the decoder path.

```ascii
       Input (Noisy 1D ECG Signal) 
                  │
                  ▼
   [Encoder Conv Block] ─────────► [ Bidirectional LSTM Bridge ] ─────────► [Decoder Conv Block]
          │ (Max Pool)                                                            ▲ (Upsample)
          ▼                                                                       │
[Deep Bottleneck Conv] ───────────────────────────────────────────────────────────┘
                  │
                  ▼
         [Final 1x1 Projection] ──► Pristine ECG Output
```

## ⚡ Key Upgrades & Features

- **Dual-Model Benchmarking**: Parallel integration of standard 1D U-Net and recurrent 1D LU-Net architectures to compare spatial vs. temporal sequence modeling.
- **Recurrent Skip Connections**: Custom BiLSTMBridge wrappers built directly into PyTorch convolutional tensor hooks.
- **Advanced Medical Metrics Engine**: Tracks reconstruction performance using domain-specific metrics: Root Mean Squared Error (RMSE), Percent Root-mean-square Difference (PRD), Signal-to-Noise Ratio Improvement (SNR_Imp), and Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
- **Decoupled Configuration Management**: All structural configurations, training intervals, and optimization hyperparameters are managed via a centralized `config.yaml` layout.
- **Frequency-Domain Visualization**: Exports time-domain signal tracking along with Short-Time Fourier Transform (STFT Spectrograms) to audit noise extraction across the frequency spectrum.

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required machine learning dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/ecg-denoising-pipeline.git
cd ecg-denoising-pipeline
pip install torch matplotlib numpy pyyaml scipy
```

### 2. Repository Structure

```bash
├── models/
│   ├── __init__.py          # Makes the folder an importable Python package
│   ├── unet_1d.py           # Standard 1D U-Net model layout
│   └── lu_net_1d.py         # Advanced 1D LU-Net with BiLSTM recurrent bridges
│
├── config.yaml              # Centralized hyperparameters and execution flags
├── ecg_dataset.py           # Synthetic ECG signal generator & PyTorch Dataset setup
├── metrics.py               # Algorithmic engine computing RMSE, PRD, SNR, and SI-SDR
├── train.py                 # Optimizes both models and saves the best weights
├── inference.py             # Evaluates test datasets, prints tables, and plots STFT maps
└── README.md                # Project documentation
```

### 3. Execution Workflow

**Step A: Configure Hyperparameters**  
Adjust the variables in `config.yaml` to configure hardware allocation, model sizing, and dataset limits.

**Step B: Run Dual-Model Training Loop**  
Train both networks side-by-side. The execution loop handles Early Stopping and Learning Rate Decay tracking automatically based on validation metrics:

```bash
python train.py
```

**Step C: Run Benchmark Evaluations & Visualization**  
Compute performance metrics on unseen test sets, display a formatted comparison table, and save a high-resolution analysis chart as `ecg_comparison_matrix.png`:

```bash
python inference.py
```

## 📊 Results & Visualization Matrix

Upon completion of the inference script, a comprehensive time/frequency analysis is generated and saved to disk.

### 🔍 Performance Breakdown Matrix

| Evaluation Target     | Left Panel (Waveform)                                      | Right Panel (STFT Spectrogram)                          |
|-----------------------|------------------------------------------------------------|---------------------------------------------------------|
| Clean ECG Target      | Pristine heartbeat shape showing ideal P-QRS-T complexes. | Stable baseline spectrum indicating standard electrical activity. |
| Noisy ECG Input       | Corrupted signal showing heavy high-frequency jitter and low-frequency baseline drifts. | Broad noise bands across multiple frequency ranges. |
| Standard 1D U-Net     | Cleared of obvious jitter, but exhibits slight peak attenuation at the QRS spikes. | Noise is mostly removed, though some high-frequency remnants persist. |
| Advanced 1D LU-Net    | Perfect peak retention and smooth baseline leveling without morphological distortion. | Complete noise suppression across all frequencies, matching the ground truth. |

### 📈 Aggregated Test Set Performance Summary

```text
==================================================
            METRIC SUMMARY COMPARISON             
==================================================
Metric          | Standard U-Net  | Advanced LU-Net 
--------------------------------------------------
RMSE            | 0.0824          | 0.0312          
PRD (%)         | 14.3512         | 5.1140          
SNR_Imp (dB)    | 11.2415         | 19.8531         
SI-SDR (dB)     | 13.9240         | 22.4119         
==================================================
```

## 📈 Future Roadmap

- [ ] **Real-World Data Ingestion**: Integrate the official PhysioNet MIT-BIH Arrhythmia Database via the wfdb API to replace synthetic generators with authentic clinical waveforms.
- [ ] **Attention Mechanisms**: Integrate temporal self-attention hooks directly alongside the recurrent layers to test hybrid Transformer-LSTM architectures.
- [ ] **Edge Deployment Optimization**: Export the optimized LU-Net model to ONNX format to prepare for deployment on low-power, embedded medical hardware.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
