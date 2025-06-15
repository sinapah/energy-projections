#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:40:02 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
df_gan = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")
df_gan["Hour"] = df_gan["Hour"].round().astype(int)
df_gan["Day"] = df_gan["Day"].round().astype(int)
df_gan["Month"] = df_gan["Month"].round().astype(int)
