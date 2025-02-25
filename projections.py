#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:43:31 2025

@author: sinap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])

# Handle missing values
df = df.dropna()

# Store datetime separately for evaluation
datetime_col = df["DateTime"]

# Extract features and target variable
X = df.drop(columns=["DateTime", "Ontario Demand"])  # Features
y = df["Ontario Demand"]  # Target variable


# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test, datetime_train, datetime_test = train_test_split(
    X, y, datetime_col, test_size=0.2, random_state=42
)

# Standardize features for ANN (Decision Tree doesn't need this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 📌 Decision Tree Model
# ============================
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Compute evaluation metrics for Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print("\n📊 Decision Tree Performance:")
print(f"MAE: {mae_dt:.2f}")
print(f"RMSE: {rmse_dt:.2f}")
print(f"R² Score: {r2_dt:.4f}")

# ============================
# ANN Model (Neural Network)
# ============================

# Define ANN architecture
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer
])

# Compile the model
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Make predictions using ANN
y_pred_ann = ann_model.predict(X_test_scaled).flatten()

# Compute evaluation metrics for ANN
mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
r2_ann = r2_score(y_test, y_pred_ann)

print("\n📊 ANN Model Performance:")
print(f"MAE: {mae_ann:.2f}")
print(f"RMSE: {rmse_ann:.2f}")
print(f"R² Score: {r2_ann:.4f}")

# ============================
# Save Results to CSV
# ============================
results_df = pd.DataFrame({
    "DateTime": datetime_test.values,
    "Actual_Ontario_Demand": y_test.values,
    "Predicted_DT": y_pred_dt,
    "Predicted_ANN": y_pred_ann
})

results_df = results_df.sort_values(by="DateTime")
results_df.to_csv("prediction_results_comparison.csv", index=False)

# ============================
# 📊 Plot Actual vs Predicted Demand (ANN & DT)
# ============================
plt.figure(figsize=(12, 6))

# Sort values for proper time-series plotting
sorted_indices = np.argsort(datetime_test)
sorted_dates = np.array(datetime_test)[sorted_indices]
sorted_actual = np.array(y_test)[sorted_indices]
sorted_predicted_dt = np.array(y_pred_dt)[sorted_indices]
sorted_predicted_ann = np.array(y_pred_ann)[sorted_indices]

# Plot actual demand
plt.plot(sorted_dates, sorted_actual, label="Actual Demand", color="blue", linewidth=2)

# Plot Decision Tree predictions
plt.plot(sorted_dates, sorted_predicted_dt, label="Decision Tree Prediction", color="red", linestyle="dashed", linewidth=2)

# Plot ANN predictions
plt.plot(sorted_dates, sorted_predicted_ann, label="ANN Prediction", color="green", linestyle="dashed", linewidth=2)

plt.xlabel("DateTime")
plt.ylabel("Ontario Energy Demand")
plt.title("Actual vs Predicted Energy Demand (ANN vs Decision Tree)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
