#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:20:37 2025

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

# ============================
# 🧠 Train Autoencoder + KDE per (month, 4-hour window)
# ============================
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
        start_hour = (hour // 4) * 4
        key = (month, start_hour)

        if key in models_dict:
            model = models_dict[key]
            latent_sample = model["kde"].resample(1).T
            decoded_sample = model["decoder"].predict(latent_sample)[0]
            scaled_back = model["scaler"].inverse_transform([decoded_sample])[0]
            synthetic_data.loc[i, continuous_features] = scaled_back
        else:
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
# ✅ Generate and Save Synthetic Data
# ============================
synthetic_sample = generate_synthetic_data(start_date="2025-01-01", years=1)

real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

synthetic_sample = synthetic_sample[real_df.columns]
synthetic_sample.to_csv("synthetic_data_autoencoder_kde_window4.csv", index=False)
print(synthetic_sample.head())






