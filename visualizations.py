#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:31:19 2025

@author: sinap
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
sns.lineplot(x=hourly_demand_kde.index, y=hourly_demand_kde.values, label="KDE Synthetic", marker='s', color="blue")
sns.lineplot(x=hourly_demand_gan.index, y=hourly_demand_gan.values, label="GAN Synthetic", marker='^', color="orange")

# Customize the plot
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Average Ontario Demand (MW)", fontsize=12)
plt.title("Average Hourly Ontario Demand: Original vs. KDE vs. GAN", fontsize=14)
plt.xticks(range(0, 24))
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

#=================
#Show sample demand for a day
#=================

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

sns.boxplot(x="Season", y="Ontario Demand", data=real_df, order=season_order, palette="Blues", ax=axes[0])
axes[0].set_title("Real Data")
axes[0].set_xlabel("Season")
axes[0].set_ylabel("Ontario Demand")

sns.boxplot(x="Season", y="Ontario Demand", data=kde_df, order=season_order, palette="Greens", ax=axes[1])
axes[1].set_title("KDE Synthetic Data")
axes[1].set_xlabel("Season")

sns.boxplot(x="Season", y="Ontario Demand", data=gan_df, order=season_order, palette="Oranges", ax=axes[2])
axes[2].set_title("GAN Synthetic Data")
axes[2].set_xlabel("Season")

for ax in axes:
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Ontario Energy Demand Across Seasons (Real vs Synthetic)", fontsize=16)
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
    print(daily_totals)
    # Compute average
    average_daily = daily_totals.mean()
    print(f"{label}: Average Daily Ontario Demand = {average_daily:.2f} MW")
    return daily_totals

# Apply to each dataset
real_daily_md = compute_average_daily_demand_from_md(real_df, "Real Data")
kde_daily_md = compute_average_daily_demand_from_md(kde_df, "KDE Synthetic")
gan_daily_md = compute_average_daily_demand_from_md(gan_df, "GAN Synthetic")

# Reformat index to show "MM-DD" for plotting
real_plot = real_daily_md.copy()
real_plot.index = [f"{m:02d}-{d:02d}" for m, d in real_plot.index]

kde_plot = kde_daily_md.copy()
kde_plot.index = [f"{m:02d}-{d:02d}" for m, d in kde_plot.index]

gan_plot = gan_daily_md.copy()
gan_plot.index = [f"{m:02d}-{d:02d}" for m, d in gan_plot.index]

# Sort by month and day
real_plot = real_plot.sort_index()
kde_plot = kde_plot.sort_index()
gan_plot = gan_plot.sort_index()

# Plot
plt.figure(figsize=(16, 6))
sns.lineplot(data=real_plot, label="Real Data", color="green")
sns.lineplot(data=kde_plot, label="KDE Synthetic", color="blue")
sns.lineplot(data=gan_plot, label="GAN Synthetic", color="orange")

plt.title("Daily Ontario Demand from Jan 1 to Dec 31 (Based on Month-Day)", fontsize=14)
plt.xlabel("Month-Day")
plt.ylabel("Total Daily Ontario Demand (MW)")
plt.xticks(rotation=45, fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


'''

# Extract the hour
df["Shifted_Hour"] = (df["Hour"] - 3) % 24

# Compute the average demand per hour
hourly_demand = df.groupby("Shifted_Hour")["Ontario Demand"].mean()

# Plot the results
plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_demand.index, y=hourly_demand.values, marker="o", color="blue")

plt.xlabel("Hour of the Day")
plt.ylabel("Average Demand (MW)")
plt.title("Average Hourly Energy Demand vs Hour of the Day")
plt.xticks(range(0, 24))  # Ensure all hours are labeled
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()
'''
'''
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)


# Extract the month and map it to a season
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

df["Season"] = df["DateTime"].dt.month.apply(get_season)

# Sort seasons properly
season_order = ["Winter", "Spring", "Summer", "Fall"]

# Plot demand levels by season
plt.figure(figsize=(10, 6))
sns.boxplot(x="Season", y="Ontario Demand", data=df, order=season_order, palette="coolwarm")

plt.xlabel("Season")
plt.ylabel("Demand Level")
plt.title("Energy Demand Levels Across Seasons")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


# Extract the hour
df["Hour"] = df["DateTime"].dt.hour

df["Shifted_Hour"] = (df["Hour"] - 3) % 24

# Compute the average demand per hour
hourly_demand = df.groupby("Shifted_Hour")["Ontario Demand"].mean()

# Plot the results
plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_demand.index, y=hourly_demand.values, marker="o", color="blue")

plt.xlabel("Hour of the Day")
plt.ylabel("Average Demand (MW)")
plt.title("Average Hourly Energy Demand vs Hour of the Day")
plt.xticks(range(0, 24))  # Ensure all hours are labeled
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()
'''
'''

dates = {"2016-02-22": "Monday February 22, 2016", "2016-02-21": "Sunday February 21, 2016", "2016-08-11":"Wednesday August 11, 2016", "2016-08-13": "Saturday August 13, 2016"}
for date in dates:
    df_day = df[df["DateTime"].dt.date == pd.to_datetime(date).date()]
    
    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(df_day["Hour"], df_day["Ontario Demand"], color="blue", alpha=0.7)
    
    # Labels and title
    plt.xlabel("Hour of the Day")
    plt.ylabel("Energy Demand (MW)")
    plt.title(f"Hourly Energy Demand on {dates[date]}")
    plt.xticks(range(24))  # Ensure all hours are labeled
    
    # Set y-axis range and ticks
    plt.ylim(0, 24000)  # Y-axis range from 0 to 24,000
    plt.yticks(range(0, 24001, 2000))  # Ticks every 2,000
    
    # Grid and show plot
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# Show hourly price average
# Calculate the average HOEP price per hour
hourly_price = df.groupby("Hour")["HOEP"].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(hourly_price["Hour"], hourly_price["HOEP"], color='skyblue')

# Labels and title
plt.xlabel("Hour of the Day")
plt.ylabel("Average HOEP Price (CAD/MWh)")
plt.title("Average Hourly HOEP Price")
plt.xticks(range(24))
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show the plot
plt.show()

#=================
#Draw Bar Graph For Comparisons
#=================

# Load the dataset
df = pd.read_csv("prediction_results_comparison.csv")

# Select the first 10 rows
df_samples = [df.head(50), df.tail(100)]

for df_sample in df_samples:
    # Extract data
    times = df_sample["DateTime"].str[:-6] # Assuming there is a "Time" column
    actual = df_sample["Actual_Ontario_Demand"]
    rt = df_sample["Predicted_DT"]
    ann = df_sample["Predicted_ANN"]
    svm_rbf = df_sample["Predicted SVM - Non Linear"]
    svm_l = df_sample["Prediced SVM - Linear"]
    
    # Define bar width and positions
    bar_width = 0.15
    x = np.arange(len(times))  # Position of each group
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(x - 2*bar_width, actual, width=bar_width, label="Actual", color="black")
    plt.bar(x - bar_width, rt, width=bar_width, label="Regression Tree", color="blue")
    plt.bar(x, ann, width=bar_width, label="Ann", color="red")
    plt.bar(x + bar_width, svm_rbf, width=bar_width, label="SVM - Non Linear", color="green")
    plt.bar(x + 2*bar_width, svm_l, width=bar_width, label="SVM - Linear", color="orange")
    
    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Energy Demand (MW)")
    plt.title("Actual vs. Predicted Demand (samples from test collection)")
    plt.xticks(x, times, rotation=45)  # Rotate time labels for clarity
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Show plot
    plt.tight_layout()
    plt.show()
'''