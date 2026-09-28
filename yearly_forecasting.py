#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 14:27:16 2025

@author: sinap
"""

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def save_predictions_with_timeinfo_yearly(y_true, y_pred, time_df, dataset_name, model_name, mode, output_folder="FourYearPredictions"):
    os.makedirs(output_folder, exist_ok=True)
    result_df = pd.DataFrame({
        "month": time_df["Month"].values,
        "day": time_df["Day"].values,
        "hour": time_df["Hour"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "error": np.abs(np.array(y_true) - np.array(y_pred))
    })
    result_df.to_csv(f"{output_folder}/{mode}_{dataset_name}_{model_name}_predictions.csv", index=False)

# Load datasets
real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
kde_df = pd.read_csv("new_kde.csv")
gan_df = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")
real_test_df = pd.read_csv("full_features_7000.csv")

# Define target and features
target_col = "Ontario Demand"
drop_cols = ["DateTime", "Ontario Demand", "Market Demand"]
real_df["DateTime"] = pd.to_datetime(real_df["DateTime"], errors="coerce")

def prepare_data(df):
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y = df[target_col]
    time_info = df[["Month", "Day", "Hour"]].reset_index(drop=True)
    return X, y, time_info

# Prepare training set (Real 2016–2019)
X_train_real = real_df[(real_df["DateTime"].dt.year >= 2016) & 
                       (real_df["DateTime"].dt.year <= 2019)].dropna(subset=[target_col])
X_train_real, y_train_real, _ = prepare_data(X_train_real)

# Prepare test sets
X_test_real, y_test_real, time_real = prepare_data(real_test_df)
X_test_kde, y_test_kde, time_kde = prepare_data(kde_df)
X_test_gan, y_test_gan, time_gan = prepare_data(gan_df)

all_test_sets = {
    "Real": (X_test_real, y_test_real, time_real),
    "KDE": (X_test_kde, y_test_kde, time_kde),
    "GAN": (X_test_gan, y_test_gan, time_gan)
}

# Train and evaluate on Real → [Real, KDE, GAN]
mode = "TRTS"
print(f"\nTraining models on Real data...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_real)

# Models
dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt.fit(X_train_real, y_train_real)

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train_scaled, y_train_real)

svr_linear = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svr_linear.fit(X_train_scaled, y_train_real)

ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
ann_model.fit(X_train_scaled, y_train_real, epochs=100, batch_size=32, verbose=0)

for name, (X_test, y_test, time_info) in all_test_sets.items():
    X_test_scaled = scaler.transform(X_test)

    preds_dt = dt.predict(X_test)
    save_predictions_with_timeinfo_yearly(y_test, preds_dt, time_info, name, "DecisionTree", mode)
    print(f"{mode} | {name} | DecisionTree - MAE: {mean_absolute_error(y_test, preds_dt):.2f}, R²: {r2_score(y_test, preds_dt):.4f}")

    preds_rbf = svr_rbf.predict(X_test_scaled)
    save_predictions_with_timeinfo_yearly(y_test, preds_rbf, time_info, name, "SVM_RBF", mode)
    print(f"{mode} | {name} | SVM RBF - MAE: {mean_absolute_error(y_test, preds_rbf):.2f}, R²: {r2_score(y_test, preds_rbf):.4f}")

    preds_linear = svr_linear.predict(X_test_scaled)
    save_predictions_with_timeinfo_yearly(y_test, preds_linear, time_info, name, "SVM_Linear", mode)
    print(f"{mode} | {name} | SVM Linear - MAE: {mean_absolute_error(y_test, preds_linear):.2f}, R²: {r2_score(y_test, preds_linear):.4f}")

    preds_ann = ann_model.predict(X_test_scaled).flatten()
    save_predictions_with_timeinfo_yearly(y_test, preds_ann, time_info, name, "ANN", mode)
    print(f"{mode} | {name} | ANN - MAE: {mean_absolute_error(y_test, preds_ann):.2f}, R²: {r2_score(y_test, preds_ann):.4f}")

