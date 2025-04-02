#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:21:58 2025

@author: sinap
"""

from scipy.stats import ks_2samp, entropy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
synthetic_df = pd.read_csv("synthetic_data_pca_kde.csv", parse_dates=["DateTime"])

# Compare each continuous feature's distribution
for col in df.columns:
    if col not in ["DateTime", "IsWeekend", "IsHoliday", "BusinessHour"]:
        print(f"\n🔍 Feature: {col}")
        
        # KS Test
        ks_stat, ks_p = ks_2samp(df[col], synthetic_df[col])
        print(f"KS Statistic: {ks_stat:.4f}")
        
        # KL Divergence (avoid log(0) by adding small constant)
        original_dist = np.histogram(df[col], bins=50, density=True)[0] + 1e-10
        synthetic_dist = np.histogram(synthetic_df[col], bins=50, density=True)[0] + 1e-10

        kl_divergence = entropy(original_dist, synthetic_dist)
        print(f"KL Divergence: {kl_divergence:.4f}")

corr_original = df.corr(numeric_only=True)
corr_synthetic = synthetic_df.corr(numeric_only=True)

# Plot the first correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_original, annot=True, cmap="coolwarm", center=0)
plt.title("Original Correlation Matrix")
plt.show()

# Plot the second correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_synthetic, annot=True, cmap="coolwarm", center=0)
plt.title("Synthetic Correlation Matrix")
plt.show()

# Compare basic statistics
stats_comparison = pd.DataFrame({
    "Original Mean": df.mean(numeric_only=True),
    "Synthetic Mean": synthetic_df.mean(numeric_only=True),
    "Original Std": df.std(numeric_only=True),
    "Synthetic Std": synthetic_df.std(numeric_only=True)
})

print(stats_comparison)
stats_comparison.to_csv("basic_stats_comparison_synthetic_data.csv")

# ============================
# 🔥 Prepare the Data for Box Plot Comparison
# ============================
# Combine both datasets into a single DataFrame for easy plotting
def prepare_comparison_df(original_df, synthetic_df, features):
    """Combine original and synthetic data into a single DataFrame with labels."""
    combined_df = pd.DataFrame()
    
    for feature in features:
        original_temp = pd.DataFrame({
            "Feature": feature,
            "Value": original_df[feature],
            "Dataset": "Original"
        })

        synthetic_temp = pd.DataFrame({
            "Feature": feature,
            "Value": synthetic_df[feature],
            "Dataset": "Synthetic"
        })

        combined_df = pd.concat([combined_df, original_temp, synthetic_temp], axis=0)

    return combined_df.reset_index(drop=True)

# Select continuous features only
continuous_features = [col for col in df.columns if col not in ["DateTime", "Hour", "Month", "Day", "DayOfWeek"]]

# Prepare the combined DataFrame
comparison_df = prepare_comparison_df(df, synthetic_df, continuous_features)

# ============================
# 🔥 Visualization: Box and Whisker Plots
# ============================
# Set plot style
sns.set(style="whitegrid")

# Plot boxplots for comparison
fig, axes = plt.subplots(nrows=len(continuous_features) // 3 + 1, ncols=3, figsize=(18, len(continuous_features) * 1.5))
axes = axes.flatten()

for i, feature in enumerate(continuous_features):
    ax = axes[i]
    
    sns.boxplot(x="Dataset", y="Value", data=comparison_df[comparison_df["Feature"] == feature], ax=ax)
    ax.set_title(f"{feature}")
    ax.set_xlabel("")
    ax.set_ylabel("Value")

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Box and Whisker Plots: Original vs Synthetic Data", fontsize=18, y=1.02)
plt.show()

'''
### TSNE

# Load data
original_data = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
synthetic_data = pd.read_csv("synthetic_data_pca_kde.csv", parse_dates=["DateTime"])

original_data = original_data.drop(columns=("DateTime"))
synthetic_data = synthetic_data.drop(columns=("DateTime"))

original_data = original_data[:87601]
synthetic_data = synthetic_data

print(original_data.shape)
print(synthetic_data.shape)

print(original_data.columns)
print(synthetic_data.columns)

# Ensure both datasets have the same columns
assert list(original_data.columns.sort_values()) == list(synthetic_data.columns.sort_values()), "Columns do not match!"

# Add a label column to differentiate data
original_data["label"] = "Original"
synthetic_data["label"] = "Synthetic"

# Concatenate both datasets
combined_data = pd.concat([original_data, synthetic_data])

# Extract feature values (excluding labels)
X = combined_data.drop(columns=["label"]).values
y = combined_data["label"].values  # Labels for coloring

print(combined_data.tail)

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_embedded = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
colors = {'Original': 'red', 'Synthetic': 'blue'}
markers = {'Original': 'o', 'Synthetic': 'x'}

for label in np.unique(y):
    mask = (y == label)  # Boolean mask
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                c=colors[label], label=label, alpha=0.2, 
                marker=markers[label], s=7, edgecolors="black")  # Outlined markers

plt.xlabel("t-SNE X (all features ex. demand)")
plt.ylabel("t-SNE Y (hourly demand)")
plt.title("t-SNE Visualization of Original vs. Multivarite KDE-based Synthetic Data")
plt.legend()
plt.show()
'''
