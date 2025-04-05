#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:42:59 2025

@author: sinap
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your real dataset
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Extract conditioning features
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month

# Define continuous features (excluding datetime and categorical flags)
excluded_cols = ["DateTime", "IsWeekend", "IsHoliday", "BusinessHour"]
features = [col for col in df.columns if col not in excluded_cols]

# Features used for conditioning
conditioning_features = ["Hour", "Month"]

# Features to generate
generation_features = [col for col in features if col not in conditioning_features]

# Normalize features
scaler = StandardScaler()
df_scaled = df[conditioning_features + generation_features].dropna()
scaled_data = scaler.fit_transform(df_scaled)

# Split inputs
cond_data = scaled_data[:, :len(conditioning_features)]
gen_data = scaled_data[:, len(conditioning_features):]

# Convert to tensors
cond_tensor = torch.tensor(cond_data, dtype=torch.float32)
real_tensor = torch.tensor(gen_data, dtype=torch.float32)

# Hyperparameters
latent_dim = 16
cond_dim = cond_tensor.shape[1]
gen_dim = real_tensor.shape[1]
epochs = 500
batch_size = 128
lr = 0.0002

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, gen_dim),
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(gen_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

# Initialize models
G = Generator()
D = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    idx = np.random.randint(0, real_tensor.shape[0], batch_size)
    real_samples = real_tensor[idx]
    cond_samples = cond_tensor[idx]

    # Train Discriminator
    z = torch.randn(batch_size, latent_dim)
    fake_samples = G(z, cond_samples)

    real_validity = D(real_samples, cond_samples)
    fake_validity = D(fake_samples.detach(), cond_samples)

    real_loss = loss_fn(real_validity, torch.ones_like(real_validity))
    fake_loss = loss_fn(fake_validity, torch.zeros_like(fake_validity))
    d_loss = (real_loss + fake_loss) / 2

    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    z = torch.randn(batch_size, latent_dim)
    gen_samples = G(z, cond_samples)
    gen_validity = D(gen_samples, cond_samples)

    g_loss = loss_fn(gen_validity, torch.ones_like(gen_validity))

    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# ===============================
# ✅ Generate 50 Synthetic Rows
# ===============================

# Choose random conditioning values (or use real ones)
sample_hours = np.random.randint(0, 24, size=50)
sample_months = np.random.randint(1, 13, size=50)
conditioning_input = np.stack([sample_hours, sample_months], axis=1)
conditioning_input_scaled = scaler.transform(
    pd.DataFrame(conditioning_input, columns=conditioning_features)
    .assign(**{col: 0 for col in generation_features})  # dummy values for rest
)[:, :len(conditioning_features)]

cond_tensor_sample = torch.tensor(conditioning_input_scaled, dtype=torch.float32)
z = torch.randn(50, latent_dim)
generated_scaled = G(z, cond_tensor_sample).detach().numpy()

# Reconstruct full data for inverse transform
combined_scaled = np.hstack([conditioning_input_scaled, generated_scaled])
generated_df = pd.DataFrame(scaler.inverse_transform(combined_scaled),
                            columns=conditioning_features + generation_features)

# Final synthetic output
print(generated_df.head(10))
generated_df.to_csv("synthetic_50_rows_cgan.csv", index=False)
print("✅ Saved synthetic data to 'synthetic_50_rows_cgan.csv'")

