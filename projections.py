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

df = df.dropna()
nan_count = df.isna().sum()
print(nan_count)

datetime_col = df["DateTime"]

# Drop non-numeric columns
df = df.drop(columns=["DateTime"])

# Define features (X) and target variable (y)
X = df.drop(columns=["Ontario Demand"])  # Features (weather, time)
y = df["Ontario Demand"]  # Target (energy demand)

for col in df:
    print(f"Column {col}: {df[col].unique()}")
    
# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test, datetime_train, datetime_test = train_test_split(X, y, datetime_col, test_size=0.2)

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

# Create a DataFrame with DateTime, actual Ontario demand, and predicted Ontario demand
results_df = pd.DataFrame({
    "DateTime": datetime_test.values,  
    "Actual_Ontario_Demand": y_test.values,  
    "Predicted_Ontario_Demand": y_pred  
})

# Save to CSV
results_df.to_csv("prediction_results.csv", index=False)

# Plot actual vs predicted values over time
plt.figure(figsize=(12, 6))

# Sort by DateTime
sorted_indices = np.argsort(datetime_test)
sorted_dates = np.array(datetime_test)[sorted_indices]
sorted_actual = np.array(y_test)[sorted_indices]
sorted_predicted = np.array(y_pred)[sorted_indices]

plt.plot(sorted_dates, sorted_actual, label="Actual Demand", color="blue", linewidth=2)
plt.plot(sorted_dates, sorted_predicted, label="Predicted Demand", color="red", linestyle="dashed", linewidth=2)

plt.xlabel("DateTime")
plt.ylabel("Ontario Energy Demand")
plt.title("Actual vs Predicted Energy Demand")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.show()

