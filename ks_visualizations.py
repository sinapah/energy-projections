#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:35:18 2025

@author: sinap
"""
from scipy.stats import ks_2samp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
#synthetic_df = pd.read_csv("synthetic_data_pca_kde.csv", parse_dates=["DateTime"])
synthetic_df = pd.read_csv("synthetic_data_kde_univariate.csv", parse_dates=["DateTime"])

# Drop non-numeric and categorical columns
excluded_cols = ["DateTime", "IsWeekend", "IsHoliday", "BusinessHour"]
continuous_features = [col for col in df.columns if col not in excluded_cols]

# Compute KS scores
ks_scores = {}
for col in continuous_features:
    ks_stat, _ = ks_2samp(df[col], synthetic_df[col])
    ks_scores[col] = ks_stat

# Convert to DataFrame
ks_df = pd.DataFrame.from_dict(ks_scores, orient='index', columns=['KS Score'])
ks_df = ks_df.sort_values(by="KS Score", ascending=False)  # Sort for better visualization

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(ks_df.T, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Heatmap of KS Scores for Original vs. Univariate KDE Based Synthetic Data")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha="right")
plt.show()
