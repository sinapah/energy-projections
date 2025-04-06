#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 19:04:54 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import holidays
import matplotlib.pyplot as plt

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
# 🤖 GMM Definition
# ============================
def build_gmm(input_dim, n_components=5):
    gmm = GaussianMixture(n_components=n_components)
    return gmm

# ============================
# 🧠 Train GMM per (month, 4-hour window)
# ============================
models_dict = {}

for month in range(1, 13):
    for start_hour in range(0, 24, 4):
        end_hour = start_hour + 3
        subset = df[(df["Month"] == month) & (df["Hour"] >= start_hour) & (df["Hour"] <= end_hour)]
        if len(subset) > 100:
            X = subset[continuous_features].dropna().values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            gmm = build_gmm(X_scaled.shape[1], n_components=5)  # Adjust n_components as needed
            gmm.fit(X_scaled)

            models_dict[(month, start_hour)] = {
                "scaler": scaler,
                "gmm": gmm
            }

print(f"✅ Trained GMM models for {len(models_dict)} (month, 4-hour) groups.")

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
            gmm = model["gmm"]

            # Sample from GMM
            latent_sample = gmm.sample(1)[0]
            decoded_sample = latent_sample[0]  # GMM sample for continuous features
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
synthetic_sample.to_csv("synthetic_data_gmm_window4.csv", index=False)
print(synthetic_sample.head())

# ============================
# 📊 Plot Synthetic Data to Validate Seasonality
# ============================
synthetic_2025 = synthetic_sample[(synthetic_sample["DateTime"] >= "2025-01-01") & 
                                  (synthetic_sample["DateTime"] < "2026-01-01")]

plt.figure(figsize=(14, 6))
plt.plot(synthetic_2025["DateTime"], synthetic_2025["Ontario Demand"], label="Ontario Demand", alpha=0.7)
plt.plot(synthetic_2025["DateTime"], synthetic_2025["Market Demand"], label="Market Demand", alpha=0.7)

plt.title("🕒 GMM-Based Synthetic Demand Trends for 2025")
plt.xlabel("Time")
plt.ylabel("Demand Level")
plt.legend()
plt.show()
