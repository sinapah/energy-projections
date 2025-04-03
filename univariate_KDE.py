#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:15:57 2025

@author: sinap
"""

#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:31:08 2025

@author: sinap
"""

# ============================
# 📊 KDE for Demand Simulation
# ============================

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
# 🔥 KDE for All Continuous Features
# ============================

# Select continuous features
continuous_features = ["Ontario Demand", "Market Demand", "HOEP"]

# Add all city-specific weather features
for col in df.columns:
    if col.endswith(("temp", "humidity")):
        continuous_features.append(col)

# Fit KDE models for each continuous feature
kde_models = {}
for feature in continuous_features:
    kde_models[feature] = gaussian_kde(df[feature].dropna())

print(f"✅ Fitted KDE models for {len(continuous_features)} features.")

# ============================
# 🔥 Generate Future Synthetic Data (5 Years)
# ============================
ontario_holidays = holidays.Canada(subdiv="ON")

def generate_synthetic_data_5_years(start_date="2025-01-01", years=10):
    """
    Generates synthetic data for the next 5 years at hourly intervals.
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

    # Generate synthetic values for all features
    np.random.seed(42)  # For reproducibility
    for feature, kde_model in kde_models.items():
        print(f"Generating synthetic values for {feature}...")
        
        # Generate synthetic feature values by sampling from KDE
        n_samples = len(synthetic_data)
        synthetic_data[feature] = kde_model.resample(n_samples).flatten()
    
    synthetic_data["Hour"] = synthetic_data["DateTime"].dt.hour
    synthetic_data["Month"] = synthetic_data["DateTime"].dt.month
    synthetic_data["Day"] = synthetic_data["DateTime"].dt.day
    synthetic_data["DayOfWeek"] = synthetic_data["DateTime"].dt.dayofweek

    # (Optional) Create a binary feature for weekends
    synthetic_data["IsWeekend"] = synthetic_data["DayOfWeek"].isin([5, 6]).astype(int)

    synthetic_data["IsHoliday"] = synthetic_data["DateTime"].dt.date.apply(lambda x: 1 if x in ontario_holidays else 0)

    # Define business hours (8 AM to 5 PM) only on non-holidays and non-weekends
    synthetic_data["BusinessHour"] = ((synthetic_data["Hour"] >= 8) & 
                                   (synthetic_data["Hour"] <= 17) & 
                                   (synthetic_data["IsWeekend"] == 0) & 
                                   (synthetic_data["IsHoliday"] == 0)).astype(int)
    
    print(f"✅ Synthetic data generated for {len(timestamps)} timestamps.")

    return synthetic_data


# ============================
# 🔥 Example Usage
# ============================

# Generate synthetic data for the next 5 years
synthetic_5_years = generate_synthetic_data_5_years()

# Display the first few rows
print(synthetic_5_years.head())

# ============================
# 💾 Save the Synthetic Data
# ============================
synthetic_5_years.to_csv("synthetic_data_kde_univariate.csv", index=False)

print("✅ Synthetic data saved to 'synthetic_data_5_years.csv'")

synthetic_2025 = synthetic_5_years[(synthetic_5_years["DateTime"] >= "2025-01-01") & 
                                   (synthetic_5_years["DateTime"] < "2026-01-01")]

print(f"✅ Synthetic data for 2025 contains {len(synthetic_2025)} records.")

plt.figure(figsize=(14, 6))

plt.plot(synthetic_2025["DateTime"], synthetic_2025["Ontario Demand"], label="Ontario Demand", alpha=0.7)
plt.plot(synthetic_2025["DateTime"], synthetic_2025["Market Demand"], label="Market Demand", alpha=0.7)

plt.title("🕒 Synthetic Demand Trends for 2025")
plt.xlabel("Time")
plt.ylabel("Demand Level")
plt.legend()
plt.show()
