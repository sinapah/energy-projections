#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:55:26 2025

@author: sinap
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

# Select the relevant features for TimeGAN
features = [col for col in df.columns if col.endswith("_temp") or col.endswith("_humidity")] + [
    "Market Demand", "Ontario Demand", "HOEP"
]
# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Convert to sequences of fixed length (e.g., 24-hour windows)
sequence_length = 24  # Adjust based on dataset needs
sequences = []

for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i: i + sequence_length])

sequences = np.array(sequences)

# Define and train the TimeGAN model
synthesizer = TimeGAN(epochs=1, batch_size=128)
synthesizer.fit(sequences)

# Generate synthetic sequences
num_samples = len(sequences)
synthetic_data = synthesizer.sample(num_samples)

# Reshape & Inverse Transform
synthetic_data = synthetic_data.reshape(-1, len(features))
synthetic_data = scaler.inverse_transform(synthetic_data)

# Convert back to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=features)

# Convert back to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=features)

# Save to CSV
synthetic_df.to_csv("synthetic_timegan_data.csv", index=False)
print("✅ Synthetic data saved to synthetic_timegan_data.csv")