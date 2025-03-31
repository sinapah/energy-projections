#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:36:34 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate
from datetime import datetime, timedelta
import holidays
import matplotlib.pyplot as plt

# ============================
# 📊 Load Historical Dataset
# ============================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Extract time-based features
df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["DayOfWeek"] = df["DateTime"].dt.weekday

# ============================
# 🔥 Fit Gaussian Copula Model
# ============================
# Select only continuous and meaningful features
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]

# Add all city-specific weather features
for col in df.columns:
    if col.endswith(("temp", "humidity")):
        continuous_features.append(col)

# Add time-based features to capture seasonality
features = continuous_features + ["Hour", "Month", "DayOfWeek"]

# ============================
# 🔥 Generate Future Synthetic Data (5 Years)
# ============================
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_copula_data(start_date="2025-01-01", years=5):
    """
    Generates copula-based synthetic data for the next 5 years at hourly intervals.
    """
    # Define time range
    end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=365 * years)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    # Create empty DataFrame with timestamps
    synthetic_data = pd.DataFrame({"DateTime": timestamps})

    # Add time-based features
    synthetic_data["Hour"] = synthetic_data["DateTime"].dt.hour
    synthetic_data["Month"] = synthetic_data["DateTime"].dt.month
    synthetic_data["DayOfWeek"] = synthetic_data["DateTime"].dt.weekday

    # Generate synthetic continuous features conditioned by time features
    synthetic_features = []
    
    print("Generating synthetic data with Copula...")
    for month in range(1, 13):  # Generate by month to preserve seasonality
        # Filter historical data for the current month
        print(f"Processing month {month}")
        month_data = df[df["Month"] == month][features]

        # Fit the copula for the specific month
        copula_month = GaussianMultivariate()
        copula_month.fit(month_data)

        # Generate synthetic samples
        n_samples = len(synthetic_data[synthetic_data["Month"] == month])
        samples = copula_month.sample(n_samples)
        synthetic_features.append(samples)

    # Combine all monthly samples
    synthetic_features_df = pd.concat(synthetic_features, ignore_index=True)

    # Merge with the timestamp-based data
    for col in continuous_features:
        synthetic_data[col] = synthetic_features_df[col].values

    # ============================
    # 🛠️ Add Business Flags
    # ============================
    synthetic_data["IsWeekend"] = synthetic_data["DayOfWeek"].isin([5, 6]).astype(int)

    synthetic_data["IsHoliday"] = synthetic_data["DateTime"].dt.date.apply(
        lambda x: 1 if x in ontario_holidays else 0
    )

    # Define business hours (8 AM to 5 PM) on non-holidays and non-weekends
    synthetic_data["BusinessHour"] = (
        (synthetic_data["Hour"] >= 8) & 
        (synthetic_data["Hour"] <= 17) & 
        (synthetic_data["IsWeekend"] == 0) & 
        (synthetic_data["IsHoliday"] == 0)
    ).astype(int)

    print(f"✅ Synthetic data generated with {len(timestamps)} records.")
    
    return synthetic_data

# ============================
# 🚀 Generate and Save Synthetic Data
# ============================

# Generate synthetic data for the next 5 years
synthetic_5_years = generate_synthetic_copula_data()

# Display sample output
print(synthetic_5_years.head())

# Save the synthetic data
output_file = "synthetic_copula_data_5_years.csv"
synthetic_5_years.to_csv(output_file, index=False)

print(f"✅ Synthetic data saved to {output_file}")

# ============================
# 📊 Plot Synthetic Data to Validate Seasonality
# ============================

synthetic_2025 = synthetic_5_years[(synthetic_5_years["DateTime"] >= "2025-01-01") & 
                                   (synthetic_5_years["DateTime"] < "2026-01-01")]

plt.figure(figsize=(14, 6))
plt.plot(synthetic_2025["DateTime"], synthetic_2025["Ontario Demand"], label="Ontario Demand", alpha=0.7)
plt.plot(synthetic_2025["DateTime"], synthetic_2025["Market Demand"], label="Market Demand", alpha=0.7)

plt.title("🕒 Copula-Based Synthetic Demand Trends for 2025")
plt.xlabel("Time")
plt.ylabel("Demand Level")
plt.legend()
plt.show()

