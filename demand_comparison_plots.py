#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots comparing Ontario electricity demand in the original (real) data against
the KDE and TimeGAN synthetic datasets: average demand by hour, seasonal
boxplots, and daily demand by month-day.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# =============================
# Load Original Data
# =============================
df_original = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df_original["DateTime"] = pd.to_datetime(df_original["DateTime"], utc=True)
df_original["Hour"] = df_original["DateTime"].dt.hour
df_original["Shifted_Hour"] = (df_original["Hour"] - 3) % 24
hourly_demand_original = df_original.groupby("Shifted_Hour")["Ontario Demand"].mean()

# =============================
# Load KDE Synthetic Data
# =============================
df_kde = pd.read_csv("synthetic_data_autoencoder_kde_window4.csv")
df_kde["Shifted_Hour"] = (df_kde["Hour"] - 3) % 24
hourly_demand_kde = df_kde.groupby("Shifted_Hour")["Ontario Demand"].mean()

# =============================
# Load GAN Synthetic Data
# =============================
df_gan = pd.read_csv("gen_data_rescaled_7000x54.csv")
df_gan["Hour"] = df_gan["Hour"].round().astype(int)
df_gan["Day"] = df_gan["Day"].round().astype(int)
df_gan["Month"] = df_gan["Month"].round().astype(int)

# Optional: Save the corrected version (overwrite or new file)
df_gan.to_csv("gen_data_rescaled_7000x54_hour_fixed.csv", index=False)
df_gan["Shifted_Hour"] = (df_gan["Hour"] - 3) % 24
hourly_demand_gan = df_gan.groupby("Shifted_Hour")["Ontario Demand"].mean()

# =============================
# Plot all three datasets
# =============================
plt.figure(figsize=(12, 6))

sns.lineplot(x=hourly_demand_original.index, y=hourly_demand_original.values, label="Original Data", marker='o', color="green")
sns.lineplot(x=hourly_demand_kde.index, y=hourly_demand_kde.values, label="KDE", marker='s', color="blue")
sns.lineplot(x=hourly_demand_gan.index, y=hourly_demand_gan.values, label="TimeGAN", marker='^', color="orange")

# Customize the plot
plt.xlabel("Hour of Day", fontsize=14)
plt.ylabel("Average Ontario Demand (MW)", fontsize=14)
plt.xticks(range(0, 24))
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# ============================
# 🔁 Helper: Map Month to Season
# ============================
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

season_order = ["Winter", "Spring", "Summer", "Fall"]

# ============================
# 📂 Load Datasets
# ============================
real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
kde_df = pd.read_csv("synthetic_data_autoencoder_kde_window4.csv")
gan_df = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")

# Ensure Month column exists
for df in [real_df, kde_df, gan_df]:
    if "Month" not in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors='coerce')
        df["Month"] = df["DateTime"].dt.month
    df["Season"] = df["Month"].apply(get_season)

# ============================
# 📊 Seasonal Boxplots
# ============================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

boxplot_inputs = [
    (real_df, "Blues", "Real Data"),
    (kde_df, "Greens", "KDE Synthetic Data"),
    (gan_df, "Oranges", "TimeGAN Synthetic Data"),
]
for ax, (df, palette, title) in zip(axes, boxplot_inputs):
    sns.boxplot(x="Season", y="Ontario Demand", data=df, order=season_order, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Season")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylabel("Ontario Demand")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# =============================
# 📆 Compute Average Daily Demand using Month and Day
# =============================

def compute_average_daily_demand_from_md(df, label):
    # Round and ensure types
    df["Month"] = df["Month"].round().astype(int)
    df["Day"] = df["Day"].round().astype(int)

    # Filter invalid days (e.g. Feb 30)
    valid_df = df[(df["Month"] >= 1) & (df["Month"] <= 12) & (df["Day"] >= 1) & (df["Day"] <= 31)]
    
    # Group by Month and Day tuple
    daily_totals = valid_df.groupby(["Month", "Day"])["Ontario Demand"].sum()
    print(f"{label}: Average Daily Ontario Demand = {daily_totals.mean():.2f} MW")
    return daily_totals

# Reformat (Month, Day) index to "MM-DD" for plotting; groupby already sorts by month and day
daily_datasets = {
    "Real Data": (real_df, "green"),
    "KDE Synthetic": (kde_df, "blue"),
    "GAN Synthetic": (gan_df, "orange"),
}

plt.figure(figsize=(16, 6))
for label, (df, color) in daily_datasets.items():
    daily = compute_average_daily_demand_from_md(df, label)
    daily.index = [f"{m:02d}-{d:02d}" for m, d in daily.index]
    sns.lineplot(data=daily, label=label, color=color)

plt.title("Daily Ontario Demand from Jan 1 to Dec 31 (Based on Month-Day)", fontsize=14)
plt.xlabel("Month-Day")
plt.ylabel("Total Daily Ontario Demand (MW)")
plt.xticks(rotation=45, fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

