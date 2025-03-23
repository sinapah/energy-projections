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
from keras.models import Sequential
from keras.layers import Dense
import joblib
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

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

# Save the trained scaler to a file
joblib.dump(scaler, "scaler.pkl")

X_test_scaled = scaler.transform(X_test)


# ============================
# 📌 Decision Tree Model
# ============================
dt_model = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
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

# Get most important features
feature_importance_dt = dt_model.feature_importances_

# Convert to DataFrame for better readability
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance_dt
}).sort_values(by="Importance", ascending=False)

# Display top features
print("\n🌳 Decision Tree Feature Importance:")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="blue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Decision Tree Feature Importance")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(20, 12))
'''
# Plot the tree
plot_tree(
    dt_model,                # Your Decision Tree model
    feature_names=X.columns, # Feature names
    filled=True,             # Color the nodes based on the decision
    rounded=True,            # Rounded corners for readability
    fontsize=10               # Font size
)

plt.savefig("decision_tree_visualization.pdf", format="pdf", bbox_inches="tight")
'''
# ============================
# ANN Model (Neural Network)
# ============================

# Define ANN architecture
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

ann_model.save("ann_energy_model.h5")

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

'''
# Compute feature importance via permutation
perm_importance = permutation_importance(ann_model, X_test_scaled, y_test, scoring='neg_mean_squared_error')

# Convert to DataFrame
feature_importance_ann = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\n🧠 ANN Feature Importance:")
print(feature_importance_ann)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_ann["Feature"], feature_importance_ann["Importance"], color="green")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("ANN Feature Importance (Permutation)")
plt.gca().invert_yaxis()
plt.show()
'''
# ============================
# 📌 Support Vector Machine (SVM) Model - Use RBF (non-linear) as the Kernel
# ============================
svm_model_nl = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model_nl.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm_nl = svm_model_nl.predict(X_test_scaled)

# Compute evaluation metrics
mae_svm = mean_absolute_error(y_test, y_pred_svm_nl)
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm_nl))
r2_svm = r2_score(y_test, y_pred_svm_nl)

print("\n📊 SVM Non Linear Model Performance:")
print(f"MAE: {mae_svm:.2f}")
print(f"RMSE: {rmse_svm:.2f}")
print(f"R² Score: {r2_svm:.4f}")

joblib.dump(svm_model_nl, "svm_model_nl.pkl")

# ============================
# 📌 Support Vector Machine (SVM) Model - Use Linear Kernel
# ============================
svm_model_linear = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svm_model_linear.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm_linear = svm_model_linear.predict(X_test_scaled)

# Compute evaluation metrics
mae_svm = mean_absolute_error(y_test, y_pred_svm_linear)
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm_linear))
r2_svm = r2_score(y_test, y_pred_svm_linear)

print("\n📊 SVM Linear Model Performance:")
print(f"MAE: {mae_svm:.2f}")
print(f"RMSE: {rmse_svm:.2f}")
print(f"R² Score: {r2_svm:.4f}")

joblib.dump(svm_model_nl, "svm_model_linear.pkl")

# ============================
# 📊 Plot Actual vs Predicted Demand (SVM)
# ============================
plt.figure(figsize=(12, 6))

# Sort values for proper time-series plotting
sorted_indices = np.argsort(datetime_test)
sorted_dates = np.array(datetime_test)[sorted_indices]
sorted_actual = np.array(y_test)[sorted_indices]
sorted_predicted_svm = np.array(y_pred_svm_nl)[sorted_indices]

# Plot actual demand
plt.plot(sorted_dates, sorted_actual, label="Actual Demand", color="blue", linewidth=2)

# Plot SVM predictions
plt.plot(sorted_dates, sorted_predicted_svm, label="SVM Prediction", color="purple", linestyle="dashed", linewidth=2)

plt.xlabel("DateTime")
plt.ylabel("Ontario Energy Demand")
plt.title("Actual vs Predicted Energy Demand (SVM)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# ============================
# Save Results to CSV
# ============================
results_df = pd.DataFrame({
    "DateTime": datetime_test.values,
    "Actual_Ontario_Demand": y_test.values,
    "Predicted_DT": y_pred_dt,
    "Predicted_ANN": y_pred_ann,
    "Predicted SVM - Non Linear": y_pred_svm_nl,
    "Prediced SVM - Linear": y_pred_svm_linear
})

results_df = results_df.sort_values(by="DateTime")
results_df.to_csv("prediction_results_comparison.csv", index=False)
