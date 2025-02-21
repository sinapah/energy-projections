#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:43:31 2025

@author: sinap
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the merged dataset
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

# Get count of NaNs
nan_count = df.isna().sum()

print(nan_count)

# Extract time-based features
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)  # Ensure it's datetime
df["DateTime"] = df["DateTime"].dt.tz_localize(None)  # Remove timezone

df["Hour"] = df["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

# (Optional) Create a binary feature for weekends
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

df = df.dropna()
nan_count = df.isna().sum()
print(nan_count)

# Drop non-numeric columns
df = df.drop(columns=["DateTime"])

# Define features (X) and target variable (y)
X = df.drop(columns=["Ontario Demand"])  # Features (weather, time)
y = df["Ontario Demand"]  # Target (energy demand)

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
