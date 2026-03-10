import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')
from preprocessing.preprocess import normalize_and_log_single, load_real_data

OUTPUT_DIR = 'results'
IMPUTED_DIR = os.path.join(OUTPUT_DIR, 'imputed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPUTED_DIR, exist_ok=True)

SEED = 123
BETA_VALUE = 0.0001
D_DIMS = [1024, 512, 256]  
Z_DIMS = [256, 128, 64]
BATCH_SIZE = 128 
LEARNING_RATE = 7e-4
NUM_EPOCHS = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeterministicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class TopInferenceLayer(nn.Module):
    def __init__(self, d_dim, z_dim):
        super().__init__()
        self.fc = nn.Linear(d_dim, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
    def forward(self, d):
        h = F.relu(self.fc(d))
        return self.mu(h), self.logvar(h)

class InferenceLayer(nn.Module):
    def __init__(self, d_dim, z_upper_dim, z_dim):
        super().__init__()
        self.combine = nn.Linear(d_dim + z_upper_dim, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
    def forward(self, d, z_upper):
        h = F.relu(self.combine(torch.cat([d, z_upper], dim=1)))
        return self.mu(h), self.logvar(h)

class GenerativeLayer(nn.Module):
    def __init__(self, z_upper_dim, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_upper_dim, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
    def forward(self, z_upper):
        h = F.relu(self.fc(z_upper))
        return self.mu(h), self.logvar(h)

class ReconstructionLayerWithSkip(nn.Module):
    def __init__(self, z_dim, d1_dim, d2_dim, output_dim):
        super().__init__()
        self.combine = nn.Linear(z_dim + d1_dim + d2_dim, 512)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, z1, d1, d2):
        combined = torch.cat([z1, d1, d2], dim=1)
        h = F.relu(self.combine(combined))
        output = self.decoder(h)
        output = F.elu(output, alpha=1.0) + 1.0
        return output

class LVAEImputation(nn.Module):
    def __init__(self, input_dim, d_dims=[512, 256, 128], z_dims=[128, 64, 32]):
        super().__init__()
        self.d1_layer = DeterministicLayer(input_dim, d_dims[0])
        self.d2_layer = DeterministicLayer(d_dims[0], d_dims[1])
        self.d3_layer = DeterministicLayer(d_dims[1], d_dims[2])
        self.q_z3 = TopInferenceLayer(d_dims[2], z_dims[2])
        self.q_z2 = InferenceLayer(d_dims[1], z_dims[2], z_dims[1])
        self.q_z1 = InferenceLayer(d_dims[0], z_dims[1], z_dims[0])
        self.p_z2_given_z3 = GenerativeLayer(z_dims[2], z_dims[1])
        self.p_z1_given_z2 = GenerativeLayer(z_dims[1], z_dims[0])
        self.reconstruction = ReconstructionLayerWithSkip(z_dims[0], d_dims[0], d_dims[1], input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def forward(self, x):
        # Encoder (save d1, d2 for skip connections)
        d1 = self.d1_layer(x)
        d2 = self.d2_layer(d1)
        d3 = self.d3_layer(d2)
        
        # Inference
        q_mu3, q_logvar3 = self.q_z3(d3)
        z3 = self.reparameterize(q_mu3, q_logvar3)
        
        q_mu2, q_logvar2 = self.q_z2(d2, z3)
        z2 = self.reparameterize(q_mu2, q_logvar2)
        
        q_mu1, q_logvar1 = self.q_z1(d1, z2)
        z1 = self.reparameterize(q_mu1, q_logvar1)
        
        # Generative
        p_mu2, p_logvar2 = self.p_z2_given_z3(z3)
        p_mu1, p_logvar1 = self.p_z1_given_z2(z2)
        
        q_params = {'z1': (q_mu1, q_logvar1, z1), 'z2': (q_mu2, q_logvar2, z2), 'z3': (q_mu3, q_logvar3, z3)}
        p_params = {'z2': (p_mu2, p_logvar2), 'z1': (p_mu1, p_logvar1)}
        
        # Reconstruction with skip connections
        recon_x = self.reconstruction(z1, d1, d2)
        
        return recon_x, q_params, p_params

def kl_divergence(q_mu, q_logvar, p_mu=None, p_logvar=None):
    if p_mu is None and p_logvar is None:
        return -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - q_logvar.exp())
    return -0.5 * torch.sum(1 + q_logvar - p_logvar - ((q_mu - p_mu).pow(2) + q_logvar.exp()) / p_logvar.exp())

def ladder_vae_loss(recon_x, x, q_params, p_params, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    q_mu1, q_logvar1, z1 = q_params['z1']
    q_mu2, q_logvar2, z2 = q_params['z2']
    q_mu3, q_logvar3, z3 = q_params['z3']
    p_mu2, p_logvar2 = p_params['z2']
    p_mu1, p_logvar1 = p_params['z1']
    kl_loss = kl_divergence(q_mu3, q_logvar3) + kl_divergence(q_mu2, q_logvar2, p_mu2, p_logvar2) + kl_divergence(q_mu1, q_logvar1, p_mu1, p_logvar1)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def run_single_experiment(my_seed, data_file):
    try:
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)
        
        X_missing, _ = load_real_data(data_file)
        data_normalized = normalize_and_log_single(X_missing, do_normalize=True, do_log=True, target_sum=1e4)
        
        X_tensor = torch.FloatTensor(data_normalized).to(device)
        
        train_loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=BATCH_SIZE, shuffle=True)
        
        model = LVAEImputation(data_normalized.shape[1], D_DIMS, Z_DIMS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"  Training...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            for data_batch, _ in train_loader:
                optimizer.zero_grad()
                recon_data, q_params, p_params = model(data_batch)
                loss, recon_loss, kl_loss = ladder_vae_loss(recon_data, data_batch, q_params, p_params, BETA_VALUE)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{NUM_EPOCHS}")
        
        model.eval()
        with torch.no_grad():
            recon_data, _, _ = model(X_tensor)
            X_imputed = recon_data.cpu().numpy()
        
        # Save imputed data
        imputed_filename = os.path.basename(data_file).replace('.csv', f'_imputed_seed{my_seed}.csv')
        imputed_filepath = os.path.join(IMPUTED_DIR, imputed_filename)
        pd.DataFrame(X_imputed).to_csv(imputed_filepath, index=True, float_format='%.6f')
        
        print(f"Training completed. Imputed data saved to: {imputed_filepath}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    data_file = "data/SplatDrop_counts.csv" # Input file path
    print(f"Running experiment with Seed={SEED}")
    print(f"{'='*70}")
    
    success = run_single_experiment(SEED, data_file)
    
    if success:
        print(f"{'='*70}")
        print(f"Experiment completed successfully!")
        print(f"Imputed file saved in: {IMPUTED_DIR}")
        print(f"{'='*70}")
    else:
        print(f"{'='*70}")
        print(f"Experiment failed!")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
