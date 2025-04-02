#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:58:49 2025

@author: sinap
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
print(df.shape)
# Handle missing values
df = df.dropna()
print(df.shape)
# Store datetime separately for evaluation
datetime_col = df["DateTime"]

# Extract features and target variable
X = df.drop(columns=["DateTime", "Ontario Demand", "Market Demand"])  # Features
y = df["Ontario Demand"]  # Target variable


# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test, datetime_train, datetime_test = train_test_split(
    X, y, datetime_col, test_size=0.2, random_state=42
)

# Standardize features for ANN (Decision Tree doesn't need this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


X_test_scaled = scaler.transform(X_test)
# List of max_depth values to try
depth_range = list(range(1, 21))  # Trying depths from 1 to 20

# Lists to store the evaluation metrics for each depth
r2_scores = []
rmse_scores = []
mae_scores = []

# Train and evaluate models for each max_depth
for depth in depth_range:
    # Initialize and train the model
    dt_model = DecisionTreeRegressor(max_depth=depth, min_samples_split=10)
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_dt = dt_model.predict(X_test)
    
    # Compute metrics
    r2 = r2_score(y_test, y_pred_dt)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    mae = mean_absolute_error(y_test, y_pred_dt)
    
    # Store the results
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot R² Score
plt.subplot(3, 1, 1)
plt.plot(depth_range, r2_scores, marker='o', label="R² Score", color='blue')
plt.xlabel('Max Depth')
plt.ylabel('R² Score')
plt.title('R² Score vs Max Depth')
plt.grid(True)

# Plot RMSE
plt.subplot(3, 1, 2)
plt.plot(depth_range, rmse_scores, marker='o', label="RMSE", color='red')
plt.xlabel('Max Depth')
plt.ylabel('RMSE')
plt.title('RMSE vs Max Depth')
plt.grid(True)

# Plot MAE
plt.subplot(3, 1, 3)
plt.plot(depth_range, mae_scores, marker='o', label="MAE", color='green')
plt.xlabel('Max Depth')
plt.ylabel('MAE')
plt.title('MAE vs Max Depth')
plt.grid(True)

plt.tight_layout()
plt.show()
