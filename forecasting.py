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
gan_df = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")

# Define target and features
target_col = "Ontario Demand"
drop_cols = ["DateTime", "Ontario Demand", "Market Demand"]

def preprocess_by_month(df, name):
    df = df.dropna(subset=[target_col])
    train_df = df[df["Month"].between(1, 10)]
    test_df = df[df["Month"] >= 11]
    train_df = train_df.sample(n=7000, random_state=42) if len(train_df) > 7000 else train_df

    X_train = train_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test

datasets = {
    "Real": preprocess_by_month(real_df, "Real"),
    "KDE": preprocess_by_month(kde_df, "KDE"),
    "GAN": preprocess_by_month(gan_df, "GAN")
}

def evaluate(name, X_train, X_test, y_train, y_test):
    print(f"\n {name} Data Results")

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    print(f" Decision Tree - MAE: {mean_absolute_error(y_test, preds_dt):.2f}, R²: {r2_score(y_test, preds_dt):.4f}")

    # SVM RBF
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_rbf.fit(X_train_scaled, y_train)
    preds_rbf = svr_rbf.predict(X_test_scaled)
    print(f" SVM RBF - MAE: {mean_absolute_error(y_test, preds_rbf):.2f}, R²: {r2_score(y_test, preds_rbf):.4f}")

    # SVM Linear
    svr_linear = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
    svr_linear.fit(X_train_scaled, y_train)
    preds_linear = svr_linear.predict(X_test_scaled)
    print(f" SVM Linear - MAE: {mean_absolute_error(y_test, preds_linear):.2f}, R²: {r2_score(y_test, preds_linear):.4f}")

    # ANN
    ann_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

    preds_ann = ann_model.predict(X_test_scaled).flatten()
    mae_ann = mean_absolute_error(y_test, preds_ann)
    r2_ann = r2_score(y_test, preds_ann)
    print(f" ANN - MAE: {mae_ann:.2f}, R²: {r2_ann:.4f}")

# Run evaluation
for name, (X_train, X_test, y_train, y_test) in datasets.items():
    evaluate(name, X_train, X_test, y_train, y_test)

