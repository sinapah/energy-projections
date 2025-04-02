#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:15:57 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import holidays

# ============================
# 📊 Load the Historical Dataset
# ============================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Extract time features
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["DayOfWeek"] = df["DateTime"].dt.weekday
df["Day"] = df["DateTime"].dt.day

# ============================
# 🔥 Fit Univariate KDE for Each Feature
# ============================

# Select continuous features
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]

# Add all city-specific weather features
for col in df.columns:
    if col.endswith(("temp", "humidity")):
        continuous_features.append(col)

# Fit a KDE model for each feature
kde_models = {}

for feature in continuous_features:
    kde = gaussian_kde(df[feature].dropna(), bw_method="scott")
    kde_models[feature] = kde

print(f"✅ Fitted KDE models for {len(kde_models)} features.")

# ============================
# 🔥 Generate Synthetic Data Using Univariate KDE
# ============================
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_kde(start_date="2025-01-01", years=10):
    """
    Generates synthetic data for the next X years using univariate KDE models.
    - start_date: Initial date for synthetic data.
    - years: Number of years to simulate.
    """
    # Define the time range for simulation
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=365 * years)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    # Prepare empty dataframe
    synthetic_data = pd.DataFrame({"DateTime": timestamps})
    synthetic_data["Hour"] = synthetic_data["DateTime"].dt.hour
    synthetic_data["Month"] = synthetic_data["DateTime"].dt.month
    synthetic_data["DayOfWeek"] = synthetic_data["DateTime"].dt.weekday
    synthetic_data["Day"] = synthetic_data["DateTime"].dt.day

    # Generate synthetic values using KDE models
    np.random.seed(42)  # For reproducibility

    for feature in continuous_features:
        synthetic_data[feature] = kde_models[feature].resample(len(synthetic_data)).flatten()

    # ============================
    # 🛠️ Add Business Flags
    # ============================
    synthetic_data["IsWeekend"] = synthetic_data["DayOfWeek"].isin([5, 6]).astype(int)

    synthetic_data["IsHoliday"] = synthetic_data["DateTime"].dt.date.apply(
        lambda x: 1 if x in ontario_holidays else 0
    )

    # Define business hours (8 AM to 5 PM) only on non-holidays and non-weekends
    synthetic_data["BusinessHour"] = (
        (synthetic_data["Hour"] >= 8) &
        (synthetic_data["Hour"] <= 17) &
        (synthetic_data["IsWeekend"] == 0) &
        (synthetic_data["IsHoliday"] == 0)
    ).astype(int)

    print(f"✅ Synthetic data generated for {len(timestamps)} timestamps.")
    return synthetic_data

# ============================
# ✅ Example Usage
# ============================

# Generate synthetic data for the next 5 years
synthetic_5_years = generate_synthetic_kde()

# Display the first few rows
print(synthetic_5_years.head())

# ============================
# 💾 Save the Synthetic Data
# ============================
output_file = "synthetic_data_kde_univariate.csv"
synthetic_5_years.to_csv(output_file, index=False)

print(f"✅ Synthetic data saved to '{output_file}'")
