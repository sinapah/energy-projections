#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:09:44 2025

@author: sinap
"""

import pandas as pd
import glob
import os

# Define file paths
energy_files = glob.glob("demand stats/*.csv")  # Adjust path
weather_files = glob.glob("weather stats/*.csv")  # Adjust path

# Step 1: Load and concatenate energy data
energy_dfs = []
for file in energy_files:
    df = pd.read_csv(file, skiprows=3)
    
    df["Hour"] = df["Hour"].replace(24, 0)  # Convert 24 to 0
    df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime first
    df["Date"] = df.apply(lambda row: row["Date"] + pd.Timedelta(days=1) if row["Hour"] == 0 else row["Date"], axis=1)
    df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Hour"].astype(str) + ":00:00")
    df["DateTime"] = df["DateTime"].dt.tz_localize("America/Toronto", nonexistent='shift_forward', ambiguous='NaT') # need to localize this otherwise there will be an error when merging
    
    df.drop(columns=["Date", "Hour"], inplace=True)  # Clean up
    df = df.dropna(subset=["DateTime"]) # dropping na for datetime. Created on line 26.
    energy_dfs.append(df)

energy_data = pd.concat(energy_dfs, ignore_index=True)

# Step 2: Load weather data and merge
for weather_file in weather_files:
    print(weather_file)
    city_name = os.path.basename(weather_file).replace("weatherstats_", "").replace("_hourly.csv", "")
    
    weather_df = pd.read_csv(weather_file)
    weather_df["DateTime"] = pd.to_datetime(weather_df["date_time_local"], errors="coerce")  # Convert to datetime
    
    if weather_df["DateTime"].dt.tz is None:
        # If naive, first localize to CST/CDT (America/Chicago)
        weather_df["DateTime"] = weather_df["DateTime"].dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward")

    # Convert to America/Toronto (EST/EDT)
    weather_df["DateTime"] = weather_df["DateTime"].dt.tz_convert("America/Toronto")
    
    weather_df = weather_df[["DateTime", "temperature", "windchill"]]  # Keep necessary columns
    weather_df.rename(columns={"temperature": f"{city_name}_temp", "windchill": f"{city_name}_windchill"}, inplace=True)
    weather_df = weather_df.dropna(subset=["DateTime"])
    # Merge with energy data using inner join to ensure both datasets have matching DateTime
    energy_data = pd.merge(energy_data, weather_df, on="DateTime", how="inner")

# Step 3: Sort by DateTime
energy_data = energy_data.sort_values(by="DateTime")

# Step 3: Save final dataset
energy_data.to_csv("merged_energy_weather.csv", index=False)
