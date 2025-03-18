#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:31:19 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#=================
#Show sample demand for a day
#=================

# Load the dataset and parse DateTime column
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)

# Filter for Feb 1, 2016
df_day = df[df["DateTime"].dt.date == pd.to_datetime("2016-08-10").date()]

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.bar(df_day["Hour"], df_day["Ontario Demand"], color="blue", alpha=0.7)

# Labels and title
plt.xlabel("Hour of the Day")
plt.ylabel("Energy Demand (MW)")
plt.title("Hourly Energy Demand on Wednesday, August 10, 2016")
plt.xticks(range(24))  # Ensure all hours are labeled

# Show the plot
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

#=================
#Draw Bar Graph For Comparisons
#=================

# Load the dataset
df = pd.read_csv("prediction_results_comparison.csv")

# Select the first 10 rows
df_sample = df.head(10)

# Extract data
times = df_sample["DateTime"]  # Assuming there is a "Time" column
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
plt.title("Actual vs. Predicted Demand (First 10 Rows)")
plt.xticks(x, times, rotation=45)  # Rotate time labels for clarity
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()
