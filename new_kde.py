#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 13:26:35 2025

@author: sinap
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import gaussian_kde
import holidays
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

# Load the 7K Dataset
df = pd.read_csv("full_features_7000.csv")
#df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
#df["Hour"] = df["DateTime"].dt.hour
#df["Month"] = df["DateTime"].dt.month
#df["DayOfWeek"] = df["DateTime"].dt.weekday
#df["Day"] = df["DateTime"].dt.day

# Feature Setup
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]
continuous_features += [col for col in df.columns if col.endswith(("temp", "humidity"))]

# Autoencoder Definition
def build_autoencoder(input_dim, latent_dim=5):
    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(latent_dim)
    ])

    decoder = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_dim)
    ])
    
    

    def weighted_mse(weights):
        def loss(y_true, y_pred):
            return K.mean(K.square((y_true - y_pred) * weights), axis=-1)
        return loss
    
    # weights: same shape as input_dim, emphasize humidity
    weights = np.ones(input_dim)
    for i, col in enumerate(continuous_features):
        if "humidity" in col:
            weights[i] = 2.0  # or higher if needed

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss=weighted_mse(weights))
    return autoencoder, encoder, decoder

# Train Autoencoder + KDE per (month, 4-hour window)
models_dict = {}

for month in range(1, 13):
    for start_hour in range(0, 24, 4):
        end_hour = start_hour + 3
        subset = df[(df["Month"] == month) & (df["Hour"] >= start_hour) & (df["Hour"] <= end_hour)]
        if len(subset) > 100:
            X = subset[continuous_features].dropna().values
            scaler = QuantileTransformer(output_distribution='normal')
            X_scaled = scaler.fit_transform(X)

            autoencoder, encoder, decoder = build_autoencoder(X_scaled.shape[1])
            autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

            latent = encoder.predict(X_scaled)
            kde = gaussian_kde(latent.T)

            models_dict[(month, start_hour)] = {
                "scaler": scaler,
                "encoder": encoder,
                "decoder": decoder,
                "kde": kde
            }

print(f"✅ Trained Autoencoder + KDE models for {len(models_dict)} (month, 4-hour) groups.")

# Generate Synthetic Data
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_data(n_samples=7000):
    np.random.seed(42)

    # Generate timestamps with random (Month, Hour)
    month_hour_keys = list(models_dict.keys())
    sampled_keys = np.random.choice(len(month_hour_keys), size=n_samples)
    
    synthetic_rows = []
    base_date = datetime(2025, 1, 1)

    for i in range(n_samples):
        month, start_hour = month_hour_keys[sampled_keys[i]]
        hour = start_hour + np.random.randint(0, 4)  # random hour within the 4-hour window
        day = np.random.randint(1, 28)
        year = 2025
        dt = datetime(year, month, day, hour)

        # Generate latent sample and decode
        model = models_dict[(month, start_hour)]
        latent_sample = model["kde"].resample(1).T
        decoded_sample = model["decoder"].predict(latent_sample)[0]
        scaled_back = model["scaler"].inverse_transform([decoded_sample])[0]

        row = {
            "DateTime": dt,
            "Month": month,
            "Day": day,
            "Hour": hour,
            "DayOfWeek": dt.weekday()
        }
        for i, col in enumerate(continuous_features):
            row[col] = scaled_back[i]
        synthetic_rows.append(row)

    synthetic_data = pd.DataFrame(synthetic_rows)
    
    # Derived columns
    synthetic_data["IsWeekend"] = synthetic_data["DayOfWeek"].isin([5, 6]).astype(int)
    synthetic_data["IsHoliday"] = synthetic_data["DateTime"].dt.date.apply(
        lambda x: 1 if x in ontario_holidays else 0
    )
    synthetic_data["BusinessHour"] = (
        (synthetic_data["Hour"] >= 8) &
        (synthetic_data["Hour"] <= 17) &
        (synthetic_data["IsWeekend"] == 0) &
        (synthetic_data["IsHoliday"] == 0)
    ).astype(int)

    return synthetic_data


# Generate and Save Synthetic Data
synthetic_sample = generate_synthetic_data(n_samples=7000)

real_df = pd.read_csv("full_features_7000.csv")

synthetic_sample = synthetic_sample[real_df.columns]
synthetic_sample.to_csv("new_kde.csv", index=False)
print(synthetic_sample.head())







