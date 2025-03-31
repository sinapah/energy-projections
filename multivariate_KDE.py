#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:20:37 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
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

# ============================
# 🔥 Fit PCA + KDE for Each Month-Hour Group
# ============================

# Select continuous features
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]

# Add all city-specific weather features
for col in df.columns:
    if col.endswith(("temp", "humidity")):
        continuous_features.append(col)

# Dictionary to hold PCA and KDE models
pca_kde_models = {}

# Fit PCA + KDE for each (month, hour) combination
for month in range(1, 13):
    for hour in range(24):
        subset = df[(df["Month"] == month) & (df["Hour"] == hour)]
        
        if len(subset) > 100:  # Only fit KDE if we have enough data points
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=min(len(continuous_features), len(subset) - 1))  
            transformed = pca.fit_transform(subset[continuous_features])
            
            # Fit KDE on PCA-transformed space
            kde = gaussian_kde(transformed.T)

            # Store PCA and KDE model
            pca_kde_models[(month, hour)] = (pca, kde)

print(f"✅ Fitted PCA + KDE models for {len(pca_kde_models)} (month, hour) combinations.")

# ============================
# 🔥 Generate Synthetic Data with Time-Aware PCA + KDE
# ============================
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_time_aware_pca_kde(start_date="2025-01-01", years=5):
    """
    Generates synthetic data for the next 5 years using time-aware PCA + KDE models.
    - start_date: Initial date for synthetic data.
    - years: Number of years to simulate.
    """
    # Define the time range for 5 years (hourly intervals)
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=365 * years)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    # Prepare empty dataframe
    synthetic_data = pd.DataFrame({"DateTime": timestamps})
    synthetic_data["Hour"] = synthetic_data["DateTime"].dt.hour
    synthetic_data["Month"] = synthetic_data["DateTime"].dt.month
    synthetic_data["DayOfWeek"] = synthetic_data["DateTime"].dt.weekday

    # Generate synthetic values using time-aware KDE models
    np.random.seed(42)  # For reproducibility

    # Initialize synthetic features
    for feature in continuous_features:
        synthetic_data[feature] = np.nan

    # Generate samples
    for i, row in synthetic_data.iterrows():
        month, hour = row["Month"], row["Hour"]
        
        if (month, hour) in pca_kde_models:
            pca, kde = pca_kde_models[(month, hour)]
            
            # Sample from the time-aware KDE in PCA space
            sample = kde.resample(1).flatten()

            # Inverse transform back to original space
            synthetic_sample = pca.inverse_transform(sample)

            # Assign synthetic values
            synthetic_data.loc[i, continuous_features] = synthetic_sample
        else:
            # Fallback to random sampling if no KDE is available
            for feature in continuous_features:
                synthetic_data.loc[i, feature] = np.random.choice(df[feature].dropna())

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
synthetic_5_years = generate_synthetic_time_aware_pca_kde()

# Display the first few rows
print(synthetic_5_years.head())

# ============================
# 💾 Save the Synthetic Data
# ============================
output_file = "synthetic_data_pca_kde.csv"
synthetic_5_years.to_csv(output_file, index=False)

print(f"✅ Synthetic data saved to '{output_file}'")




