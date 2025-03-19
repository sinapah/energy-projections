#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:45:34 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("prediction_results_comparison.csv", parse_dates=["DateTime"])

# Load merged weather data for time-based features
weather_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df = pd.merge(df, weather_df[["DateTime", "Hour", "DayOfWeek"]], on="DateTime", how="left")

# 📊 1. Calculate Variance and Std Dev by Hour
hourly_stats = df.groupby("Hour").agg(
    mean_demand=("Actual_Ontario_Demand", "mean"),
    std_demand=("Actual_Ontario_Demand", "std"),
    var_demand=("Actual_Ontario_Demand", "var")
).reset_index()

# 📉 2. Plot variance by hour
plt.figure(figsize=(12, 6))
plt.plot(hourly_stats["Hour"], hourly_stats["std_demand"], marker='o', color='orange', label="Variance")
plt.xlabel("Hour of the Day")
plt.ylabel("SD of Actual Demand (MW²)")
plt.title("Standard Deviation of Actual Demand by Hour")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(range(24))
plt.legend()
plt.show()

#======
# Analyze Months
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Extract month names for better readability
df["Month"] = df["DateTime"].dt.month
df["Month_Name"] = df["DateTime"].dt.strftime("%B")

# Calculate the absolute error between ANN and actual values
df["ANN_Error"] = abs(df["Predicted_ANN"] - df["Actual_Ontario_Demand"])

# 📊 1. Aggregate error by month
monthly_error = df.groupby("Month_Name").agg(
    avg_error=("ANN_Error", "mean"),
    std_error=("ANN_Error", "std"),
    max_error=("ANN_Error", "max"),
    min_error=("ANN_Error", "min")
).reset_index()

# Sort by calendar month order
month_order = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]
monthly_error["Month_Name"] = pd.Categorical(monthly_error["Month_Name"], categories=month_order, ordered=True)
monthly_error = monthly_error.sort_values("Month_Name")

# 📊 2. Plot the monthly average error
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(monthly_error["Month_Name"], monthly_error["avg_error"], color="skyblue", label="Average Error")

# Labels and title
ax.set_xlabel("Month")
ax.set_ylabel("Average Error (MW)")
ax.set_title("Average Monthly ANN Model Error")
plt.xticks(rotation=45)

# Add grid and legend
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

plt.show()

# 📊 3. Display the statistics
print("\nMonthly Error Statistics:")
print(monthly_error)