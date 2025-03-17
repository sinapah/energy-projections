#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:31:19 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt

#=================
#Show sample demand for a day
#=================

# Load the dataset and parse DateTime column
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

# Filter for Feb 1, 2016
df_day = df[df["DateTime"].dt.date == pd.to_datetime("2016-02-10").date()]

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.bar(df_day["Hour"], df_day["Ontario Demand"], color="blue", alpha=0.7)

# Labels and title
plt.xlabel("Hour of the Day")
plt.ylabel("Energy Demand (MW)")
plt.title("Hourly Energy Demand on February 1, 2016")
plt.xticks(range(24))  # Ensure all hours are labeled

# Show the plot
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
