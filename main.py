import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Import our custom data simulator class
from ecg_dataset import ECGDenoisingDataset
# Import our deep learning model architecture
from model import UNet1D

def main():
    # Detect if a CUDA-capable GPU is available; otherwise, default to the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define hyperparameters for the deep learning training cycle
    epochs = 15             # Number of times the network passes through the entire dataset
    batch_size = 32         # Number of ECG samples processed before updating weights
    learning_rate = 0.001   # Controls how fast the optimizer adapts to fix errors
    
    print(f"Using device: {device}")
    
    # Instantiate the simulated training dataset (1200 heartbeats, SNR set to a noisy 3dB)
    train_dataset = ECGDenoisingDataset(num_samples=1200, signal_length=512, snr_db=3)
    # Instantiate a separate testing dataset to evaluate how the model handles unseen data
    test_dataset = ECGDenoisingDataset(num_samples=200, signal_length=512, snr_db=3)
    
    # DataLoaders handle automated mini-batch partitioning and data shuffling for PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize the 1D-UNet model and instantly migrate its weights to the target hardware (GPU/CPU)
    model = UNet1D().to(device)
    
    # Use Mean Squared Error (MSE) loss to calculate pixel/point differences between output and target
    criterion = nn.MSELoss() 
    
    # Adam optimizer will adjust the neural network's parameters during the backpropagation step
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("--- Starting Training ---")
    
    # Set the model to training mode (activates modules like Dropout and BatchNorm)
    model.train()
    
    # Main training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Unpack the PyTorch dataloader batches (corrupted signal, pristine target)
        for noisy_img, clean_img in train_loader:
            # Move the specific batch tensors to the target hardware device
            noisy_img = noisy_img.to(device)
            clean_img = clean_img.to(device)
            
            # Forward pass: Feed the noisy ECG through the network to generate a reconstruction
            outputs = model(noisy_img)
            
            # Compute the total loss error against the ground truth reference signal
            loss = criterion(outputs, clean_img)
            
            # Reset all gradients from the last optimization step to prevent gradient accumulation
            optimizer.zero_grad()
            
            # Backward pass: Calculate the gradients of the loss with respect to model parameters
            loss.backward()
            
            # Update the neural network weights using the Adam optimization mathematical rule
            optimizer.step()
            
            # Accumulate the total running loss for tracking statistics
            epoch_loss += loss.item() * noisy_img.size(0)
            
        # Compute and display the normalized average loss score for the current epoch
        total_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_epoch_loss:.5f}")
        
    # Switch the model to evaluation/inference mode (freezes BatchNorm statistics)
    model.eval()
    
    # Disable gradient calculations to conserve computer memory and accelerate processing speed
    with torch.no_grad():
        # Iterate over the testing loader to retrieve a sample for plotting
        for noisy_sample, clean_sample in test_loader:
            # Move the validation slice to the active computation device
            noisy_sample = noisy_sample.to(device)
            
            # Pass the test wave into the model to remove its artificial noise artifacts
            denoised_sample = model(noisy_sample)
            
            # Extract data from tensors, peel away channel wrappers, and convert to standard NumPy arrays
            noisy_np = noisy_sample.cpu().squeeze().numpy()
            clean_np = clean_sample.squeeze().numpy()
            denoised_np = denoised_sample.cpu().squeeze().numpy()
            
            # Break early because we only need a single heartbeat sample to build our analytical graph
            break 
            
    # Initialize a clean Matplotlib figure workspace for visual review
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Plot the corrupted raw signal containing baseline drift and jitter
    plt.subplot(3, 1, 1)
    plt.plot(noisy_np, color='red', alpha=0.7)
    plt.title("1. Noisy ECG Input (Signal + Simulated Artifacts)")
    plt.grid(True)
    
    # Subplot 2: Plot the clean waveform generated exclusively by the neural network
    plt.subplot(3, 1, 2)
    plt.plot(denoised_np, color='blue', lw=2)
    plt.title("2. Denoised ECG Output (AI Reconstruction)")
    plt.grid(True)
    
    # Subplot 3: Plot the pure mathematical baseline signal for reference verification
    plt.subplot(3, 1, 3)
    plt.plot(clean_np, color='green', lw=2)
    plt.title("3. Ground Truth Clean ECG (Target)")
    plt.grid(True)
    
    # Automatically space out elements safely, export to disk, and present on screen
    plt.tight_layout()
    plt.savefig('ecg_denoising_results.png')
    print("--- Project Execution Finished! Results saved as 'ecg_denoising_results.png' ---")
    plt.show()

if __name__ == "__main__":
    main()