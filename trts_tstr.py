#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:07:36 2025

@author: sinap
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Load datasets
real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
synthetic_df = pd.read_csv("gen_data_rescaled_7000x54.csv")

#synthetic_df = pd.read_csv("synthetic_data_autoencoder_kde_window4.csv")
synthetic_df = synthetic_df.round(1)
print(synthetic_df.head)
real_df = real_df.drop('DateTime', axis=1)
print(real_df.shape)
print(synthetic_df.shape)
#synthetic_df = synthetic_df.drop("DateTime", axis=1)
synthetic_df = synthetic_df[real_df.columns]

# Handle missing values
real_df = real_df.dropna()
synthetic_df = synthetic_df.dropna()

# Extract features and target variable
excluded_cols = [ "Ontario Demand", "Market Demand"]
X_real = real_df.drop(columns=excluded_cols)
y_real = real_df["Ontario Demand"]
X_syn = synthetic_df.drop(columns=excluded_cols)
y_syn = synthetic_df["Ontario Demand"]

# Standardize features for ANN & SVM
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)
X_syn_scaled = scaler.transform(X_syn)
joblib.dump(scaler, "scaler.pkl")

# Correct order: split raw, then scale
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2)

scaler = StandardScaler()
X_train_real_scaled = scaler.fit_transform(X_train_real)
X_test_real_scaled = scaler.transform(X_test_real)

# Do the same for synthetic
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(X_syn, y_syn, test_size=0.2)
X_train_syn_scaled = scaler.transform(X_train_syn)  # 🔁 Use the same scaler as real data!
X_test_syn_scaled = scaler.transform(X_test_syn)

# ============================
# 📌 Decision Tree Model (TRTS & TSTR)
# ============================
dt_model_real = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt_model_real.fit(X_train_real, y_train_real)
dt_model_syn = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt_model_syn.fit(X_train_syn, y_train_syn)

# Evaluate TRTS & TSTR
y_pred_trts_dt = dt_model_real.predict(X_test_syn) ## change tot X_test_syn
y_pred_tstr_dt = dt_model_syn.predict(X_test_real)

mae_trts_dt = mean_absolute_error(y_test_syn, y_pred_trts_dt)
rmse_trts_dt = np.sqrt(mean_squared_error(y_test_syn, y_pred_trts_dt))
r2_trts_dt = r2_score(y_test_syn, y_pred_trts_dt)  ## changet to y_test_syn

mae_tstr_dt = mean_absolute_error(y_test_real, y_pred_tstr_dt)
rmse_tstr_dt = np.sqrt(mean_squared_error(y_test_real, y_pred_tstr_dt))
r2_tstr_dt = r2_score(y_test_real, y_pred_tstr_dt)

print(f"📊 Decision Tree Results:")
print(f"DT TRTS - MAE: {mae_trts_dt:.2f}, RMSE: {rmse_trts_dt:.2f}, R²: {r2_trts_dt:.4f}")
print(f"DT TSTR - MAE: {mae_tstr_dt:.2f}, RMSE: {rmse_tstr_dt:.2f}, R²: {r2_tstr_dt:.4f}")

# ============================
# 📌 ANN Model (TRTS & TSTR)
# ============================
ann_model_real = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_real_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model_real.compile(optimizer='adam', loss='mean_squared_error')
ann_model_real.fit(X_train_real_scaled, y_train_real, epochs=100, batch_size=32, verbose=1)  # SHOW ITERATIONS

ann_model_syn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_syn_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model_syn.compile(optimizer='adam', loss='mean_squared_error')
ann_model_syn.fit(X_train_syn_scaled, y_train_syn, epochs=100, batch_size=32, verbose=1)  # SHOW ITERATIONS

# Evaluate TRTS & TSTR
y_pred_trts_ann = ann_model_real.predict(X_test_syn_scaled).flatten()
y_pred_tstr_ann = ann_model_syn.predict(X_test_real_scaled).flatten()

mae_trts_ann = mean_absolute_error(y_test_syn, y_pred_trts_ann)
rmse_trts_ann = np.sqrt(mean_squared_error(y_test_syn, y_pred_trts_ann))
r2_trts_ann = r2_score(y_test_syn, y_pred_trts_ann)

mae_tstr_ann = mean_absolute_error(y_test_real, y_pred_tstr_ann)
rmse_tstr_ann = np.sqrt(mean_squared_error(y_test_real, y_pred_tstr_ann))
r2_tstr_ann = r2_score(y_test_real, y_pred_tstr_ann)

print(f"📊 ANN Results:")
print(f"ANN TRTS - MAE: {mae_trts_ann:.2f}, RMSE: {rmse_trts_ann:.2f}, R²: {r2_trts_ann:.4f}")
print(f"ANN TSTR - MAE: {mae_tstr_ann:.2f}, RMSE: {rmse_tstr_ann:.2f}, R²: {r2_tstr_ann:.4f}")

# ============================
# 📌 SVM RBF Model (TRTS & TSTR)
# ============================
svm_rbf_real = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_rbf_real.fit(X_train_real_scaled, y_train_real)
svm_rbf_syn = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_rbf_syn.fit(X_train_syn_scaled, y_train_syn)

# Evaluate TRTS & TSTR
y_pred_trts_svm_rbf = svm_rbf_real.predict(X_test_syn_scaled)
y_pred_tstr_svm_rbf = svm_rbf_syn.predict(X_test_real_scaled)

print(f"📊 SVM RBF Results:")
print(f"SVM RBF TRTS - MAE: {mean_absolute_error(y_test_syn, y_pred_trts_svm_rbf):.2f}")
print(f"SVM RBF TSTR - MAE: {mean_absolute_error(y_test_real, y_pred_tstr_svm_rbf):.2f}")
r2_trts_svm_rbf = r2_score(y_test_syn, y_pred_trts_svm_rbf)
r2_tstr_svm_rbf = r2_score(y_test_real, y_pred_tstr_svm_rbf)

print(f"SVM RBF TRTS - R²: {r2_trts_svm_rbf:.4f}")
print(f"SVM RBF TSTR - R²: {r2_tstr_svm_rbf:.4f}")
# ============================
# 📌 SVM Linear Model (TRTS & TSTR)
# ============================
svm_linear_real = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svm_linear_real.fit(X_train_real_scaled, y_train_real)
svm_linear_syn = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svm_linear_syn.fit(X_train_syn_scaled, y_train_syn)

# Evaluate TRTS & TSTR
y_pred_trts_svm_linear = svm_linear_real.predict(X_test_syn_scaled)
y_pred_tstr_svm_linear = svm_linear_syn.predict(X_test_real_scaled)

print(f"📊 SVM Linear Results:")
print(f"SVM Linear TRTS - MAE: {mean_absolute_error(y_test_syn, y_pred_trts_svm_linear):.2f}")
print(f"SVM Linear TSTR - MAE: {mean_absolute_error(y_test_real, y_pred_tstr_svm_linear):.2f}")

r2_trts_svm_linear = r2_score(y_test_syn, y_pred_trts_svm_linear)
r2_tstr_svm_linear = r2_score(y_test_real, y_pred_tstr_svm_linear)

print(f"SVM Linear TRTS - R²: {r2_trts_svm_linear:.4f}")
print(f"SVM Linear TSTR - R²: {r2_tstr_svm_linear:.4f}")