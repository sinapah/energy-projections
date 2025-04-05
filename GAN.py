#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:42:59 2025

@author: sinap
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ===========================
# 📥 Load and Prepare Data
# ===========================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
df = df.dropna()

# Extract time features
df["Month"] = df["DateTime"].dt.month

df["Day"] = df["DateTime"].dt.day
df["DayOfWeek"] = df["DateTime"].dt.weekday
df["Hour"] = df["DateTime"].dt.hour

# Define conditional and continuous features
cond_cols = ["Month", "Day", "DayOfWeek", "Hour"]
cont_cols = [col for col in df.columns if col not in cond_cols + ["DateTime"]]

# Normalize both conditional and continuous features
all_cols = cond_cols + cont_cols
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[all_cols]), columns=all_cols)

# ===========================
# 📦 PyTorch Dataset
# ===========================
class EnergyDataset(Dataset):
    def __init__(self, df):
        self.cond = df[cond_cols].values.astype(np.float32)
        self.cont = df[cont_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return self.cond[idx], self.cont[idx]

# ===========================
# 🧠 Generator and Discriminator
# ===========================
class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, noise, cond):
        x = torch.cat([noise, cond], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, cond_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.net(x)

# ===========================
# ⚙️ Training Setup
# ===========================
BATCH_SIZE = 128
NOISE_DIM = 16
EPOCHS = 200

dataset = EnergyDataset(df_scaled)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, len(cond_cols), len(cont_cols))
disc = Discriminator(len(cond_cols), len(cont_cols))

criterion = nn.BCELoss()
g_optimizer = optim.Adam(gen.parameters(), lr=0.001)
d_optimizer = optim.Adam(disc.parameters(), lr=0.001)

# ===========================
# 🏋️ Train GAN
# ===========================
for epoch in range(EPOCHS):
    for cond_batch, real_batch in dataloader:
        batch_size = cond_batch.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, NOISE_DIM)
        fake_batch = gen(noise, cond_batch)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        d_loss_real = criterion(disc(real_batch, cond_batch), real_labels)
        d_loss_fake = criterion(disc(fake_batch.detach(), cond_batch), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        noise = torch.randn(batch_size, NOISE_DIM)
        fake_batch = gen(noise, cond_batch)
        g_loss = criterion(disc(fake_batch, cond_batch), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# ===========================
# 🧪 Generate Synthetic Samples
# ===========================
def generate_synthetic_samples(n=50):
    with torch.no_grad():
        # Randomly sample conditions from training data
        cond_sample = df_scaled[cond_cols].sample(n).values.astype(np.float32)
        cond_sample_tensor = torch.tensor(cond_sample)
        noise = torch.randn(n, NOISE_DIM)
        synthetic = gen(noise, cond_sample_tensor).numpy()

        # Combine and inverse scale
        combined = np.concatenate([cond_sample, synthetic], axis=1)
        generated_df = pd.DataFrame(scaler.inverse_transform(combined), columns=all_cols)

        # Round time features
        for col in ["Month", "Day", "DayOfWeek", "Hour"]:
            if col in generated_df:
                if col == "Month":
                    generated_df[col] = generated_df[col].round().clip(1, 12).astype(int)
                elif col == "Day":
                    generated_df[col] = generated_df[col].round().clip(1, 31).astype(int)
                elif col == "DayOfWeek":
                    generated_df[col] = generated_df[col].round().clip(0, 6).astype(int)
                elif col == "Hour":
                    generated_df[col] = generated_df[col].round().clip(0, 23).astype(int)

        return generated_df

# Generate and show synthetic samples
synthetic_df = generate_synthetic_samples(50)
print(synthetic_df.head())


