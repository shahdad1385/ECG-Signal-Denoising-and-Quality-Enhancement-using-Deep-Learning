import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ecg_dataset import ECGDenoisingDataset
from models.unet_1d import UNet1D
from models.lu_net_1d import LUNet1D
from metrics import evaluate_signal_batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, name, config, train_loader, val_loader, device):
    print(f"\n--- Initializing Training Pipeline for: {name} ---")
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    model = model.to(device)
    hparams = config['hyperparameters']
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=hparams['lr_decay_patience'], factor=0.5
    )
    
    os.makedirs(config['outputs']['weights_dir'], exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hparams['epochs']):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                val_loss += criterion(outputs, clean).item() * noisy.size(0)
                
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{hparams['epochs']}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # Early Stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['outputs']['weights_dir'], f"{name.lower()}_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= hparams['early_stopping_patience']:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break
                
    print(f"Training for {name} complete. Best Weight saved.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['system']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() and config['system']['device'] == 'cuda' else 'cpu')
    
    # Initialize Datasets
    d_cfg = config['dataset']
    train_dataset = ECGDenoisingDataset(d_cfg['train_samples'], d_cfg['signal_length'], d_cfg['snr_db'])
    val_dataset = ECGDenoisingDataset(d_cfg['test_samples'], d_cfg['signal_length'], d_cfg['snr_db'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False)
    
    # Train Model 1: Baseline UNet
    unet_model = UNet1D()
    train_model(unet_model, "UNet1D", config, train_loader, val_loader, device)
    
    # Train Model 2: Advanced LUNet
    lunet_model = LUNet1D()
    train_model(lunet_model, "LUNet1D", config, train_loader, val_loader, device)