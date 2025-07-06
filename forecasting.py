#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 19:25:49 2025

@author: sinap
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load datasets
real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
kde_df = pd.read_csv("synthetic_data_pca_kde.csv", parse_dates=["DateTime"])
gan_df = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")  # No parse_dates here

# Define target and columns to drop
target_col = "Ontario Demand"
drop_cols = ["DateTime", "Ontario Demand", "Market Demand"]

# Prepare training data (months 1-10) from real data
def get_real_train(df):
    df = df.dropna(subset=[target_col])
    train_df = df[df["Month"].between(1, 10)].sample(n=7000, random_state=42)
    X_train = train_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_train = train_df[target_col]
    return X_train, y_train

# Prepare test data (months 11-12) from any dataset
def get_test_data(df):
    df = df.dropna(subset=[target_col])
    test_df = df[df["Month"] >= 11]
    X_test = test_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_test = test_df[target_col]
    return X_test, y_test

# Get training data from real
X_train_real, y_train_real = get_real_train(real_df)

# Prepare test sets
X_test_real, y_test_real = get_test_data(real_df)
X_test_kde, y_test_kde = get_test_data(kde_df)
X_test_gan, y_test_gan = get_test_data(gan_df)

# Align column order
feature_columns = X_train_real.columns
X_test_real = X_test_real[feature_columns]
X_test_kde = X_test_kde[feature_columns]
X_test_gan = X_test_gan[feature_columns]

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_real)
X_test_scaled = {
    "Real": scaler.transform(X_test_real),
    "KDE": scaler.transform(X_test_kde),
    "GAN": scaler.transform(X_test_gan)
}

X_test_dict = {
    "Real": X_test_real,
    "KDE": X_test_kde,
    "GAN": X_test_gan
}
y_test_dict = {
    "Real": y_test_real,
    "KDE": y_test_kde,
    "GAN": y_test_gan
}

# Train models on real data
print("\nTraining models on real data (Months 1–10)...")

# Decision Tree
dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt.fit(X_train_real, y_train_real)

# SVM RBF
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train_scaled, y_train_real)

# SVM Linear
svr_linear = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svr_linear.fit(X_train_scaled, y_train_real)

# ANN
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
ann_model.fit(X_train_scaled, y_train_real, epochs=100, batch_size=32, verbose=0)

# Evaluate models on each test set
for name in ["Real", "KDE", "GAN"]:
    print(f"\nTesting on {name} months 11–12:")

    # Decision Tree
    preds_dt = dt.predict(X_test_dict[name])
    print(f"Decision Tree - MAE: {mean_absolute_error(y_test_dict[name], preds_dt):.2f}, R²: {r2_score(y_test_dict[name], preds_dt):.4f}")

    # SVM RBF
    preds_rbf = svr_rbf.predict(X_test_scaled[name])
    print(f"SVM RBF - MAE: {mean_absolute_error(y_test_dict[name], preds_rbf):.2f}, R²: {r2_score(y_test_dict[name], preds_rbf):.4f}")

    # SVM Linear
    preds_linear = svr_linear.predict(X_test_scaled[name])
    print(f"SVM Linear - MAE: {mean_absolute_error(y_test_dict[name], preds_linear):.2f}, R²: {r2_score(y_test_dict[name], preds_linear):.4f}")

    # ANN
    preds_ann = ann_model.predict(X_test_scaled[name]).flatten()
    print(f"ANN - MAE: {mean_absolute_error(y_test_dict[name], preds_ann):.2f}, R²: {r2_score(y_test_dict[name], preds_ann):.4f}")
