#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:20:37 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import holidays
from tensorflow.keras import layers, models

# ============================
# 📊 Load the Historical Dataset
# ============================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["DayOfWeek"] = df["DateTime"].dt.weekday
df["Day"] = df["DateTime"].dt.day

# ============================
# 🔍 Feature Setup
# ============================
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]
continuous_features += [col for col in df.columns if col.endswith(("temp", "humidity"))]

# ============================
# 🤖 Autoencoder Definition
# ============================
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

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder, decoder

# ============================
# 🧠 Train Autoencoder + KDE per (month, hour)
# ============================
models_dict = {}

for month in range(1, 13):
    for hour in range(24):
        subset = df[(df["Month"] == month) & (df["Hour"] == hour)]
        if len(subset) > 100:
            X = subset[continuous_features].dropna().values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            autoencoder, encoder, decoder = build_autoencoder(X_scaled.shape[1])
            autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

            latent = encoder.predict(X_scaled)
            kde = gaussian_kde(latent.T)

            models_dict[(month, hour)] = {
                "scaler": scaler,
                "encoder": encoder,
                "decoder": decoder,
                "kde": kde
            }

print(f"✅ Trained Autoencoder + KDE models for {len(models_dict)} (month, hour) groups.")

# ============================
# 🔮 Generate Synthetic Data
# ============================
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_data(start_date="2025-01-01", years=1):
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=365 * years)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    synthetic_data = pd.DataFrame({"DateTime": timestamps})
    synthetic_data["Hour"] = synthetic_data["DateTime"].dt.hour
    synthetic_data["Month"] = synthetic_data["DateTime"].dt.month
    synthetic_data["DayOfWeek"] = synthetic_data["DateTime"].dt.weekday
    synthetic_data["Day"] = synthetic_data["DateTime"].dt.day

    for feature in continuous_features:
        synthetic_data[feature] = np.nan

    np.random.seed(42)
    for i, row in synthetic_data.iterrows():
        month, hour = row["Month"], row["Hour"]
        if (month, hour) in models_dict:
            model = models_dict[(month, hour)]
            latent_sample = model["kde"].resample(1).T
            decoded_sample = model["decoder"].predict(latent_sample)[0]
            scaled_back = model["scaler"].inverse_transform([decoded_sample])[0]
            synthetic_data.loc[i, continuous_features] = scaled_back
        else:
            # fallback
            for f in continuous_features:
                synthetic_data.loc[i, f] = df[f].dropna().sample(1).values[0]

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

# ============================
# ✅ Generate 50 Samples
# ============================
synthetic_sample = generate_synthetic_data(start_date="2025-01-01", years=2)
print(synthetic_sample)

# 💾 Save
synthetic_sample.to_csv("synthetic_data_autoencoder_kde.csv", index=False)





