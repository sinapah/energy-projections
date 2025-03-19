#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:24:38 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("prediction_results_comparison.csv", parse_dates=["DateTime"])

# Load merged weather data
weather_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

# Calculate the absolute difference between ANN and actual demand
df["Difference"] = abs(df["Predicted_ANN"] - df["Actual_Ontario_Demand"])

# Merge with overall data to add time-based and weather features
df = pd.merge(df, weather_df[["DateTime", "Hour", "DayOfWeek", "IsWeekend"]], on="DateTime", how="left")

# 🟢 1. Differences by Day of the Week
day_diff = df.groupby("DayOfWeek")["Difference"].mean()

plt.figure(figsize=(12, 6))
plt.bar(day_diff.index, day_diff.values, color="royalblue", alpha=0.8)
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Difference (MW)")
plt.title("Average ANN Prediction Error by Day of the Week")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(ticks=range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.show()

# 🟡 2. Differences by Hour of the Day
hour_diff = df.groupby("Hour")["Difference"].mean()

plt.figure(figsize=(12, 6))
plt.plot(hour_diff.index, hour_diff.values, marker="o", color="orange", label="Average Difference")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Difference (MW)")
plt.title("Average ANN Prediction Error by Hour of the Day")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(range(24))
plt.legend()
plt.show()
