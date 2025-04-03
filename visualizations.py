#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:31:19 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#=================
#Show sample demand for a day
#=================

# Load the dataset and parse DateTime column
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