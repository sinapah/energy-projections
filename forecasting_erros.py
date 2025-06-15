#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:24:03 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your predictions
df = pd.read_csv("predictions_with_features.csv")

# Compute absolute error
df["Absolute_Error"] = abs(df["True_Demand"] - df["Predicted_Demand"])

# --- Error by Hour of Day ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="Hour", y="Absolute_Error", data=df)
plt.title("Forecasting Error by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()

# --- Error During High Variability ---
# Compute rolling std to identify high-variability periods
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")
df['Rolling_Std'] = df["True_Demand"].rolling(window=24).std()  # 24-hour window

# Flag high variability periods (e.g., top 25% std)
threshold = df['Rolling_Std'].quantile(0.75)
df["High_Variability"] = df["Rolling_Std"] > threshold

# Plot error during high vs. low variability
plt.figure(figsize=(8, 6))
sns.boxplot(x="High_Variability", y="Absolute_Error", data=df)
plt.xticks([0, 1], ["Low Variability", "High Variability"])
plt.title("Forecasting Error During High vs. Low Variability")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()
