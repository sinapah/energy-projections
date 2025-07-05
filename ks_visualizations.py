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
synthetic_df = pd.read_csv("gen_data_rescaled_7000x54.csv")
# Drop non-numeric and categorical columns
excluded_cols = ["DateTime", "IsWeekend", "IsHoliday", "BusinessHour"]
continuous_features = [col for col in df.columns if col not in excluded_cols]

# Compute KS scores
ks_scores = {}
for col in continuous_features:
    ks_stat, _ = ks_2samp(df[col], synthetic_df[col])
    ks_scores[col] = ks_stat
print(ks_scores)

# Convert to DataFrame
ks_df = pd.DataFrame.from_dict(ks_scores, orient='index', columns=['KS Score'])
ks_df = ks_df.sort_values(by="KS Score", ascending=False)  # Sort for better visualization

# Plot heatmap
plt.figure(figsize=(18, 3))
sns.heatmap(ks_df.T, cmap="coolwarm", linewidths=0.5, fmt=".2f")
#plt.title("Heatmap of KS Scores for Original vs. GAN Based Synthetic Data")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha="right")
plt.show()

'''
def plot_correlation_matrix(corr_matrix, title):
    plt.figure(figsize=(12, 10))  # Larger figure for readability
    sns.heatmap(
        corr_matrix, 
        
        fmt=".2f",  # Show only 2 decimal places
        cmap="coolwarm",  # Diverging colormap
        center=0, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.8}  # Shrink colorbar for better spacing
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.show()

# Compute correlation matrices
corr_original = df.corr(numeric_only=True)
corr_synthetic = synthetic_df.corr(numeric_only=True)

# Plot original data correlation matrix
plot_correlation_matrix(corr_original, "Original Data Correlation Matrix")

# Plot synthetic data correlation matrix
plot_correlation_matrix(corr_synthetic, "Multivariate KDE Based Synthetic Data Correlation Matrix")
'''
