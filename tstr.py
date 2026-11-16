#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSTR (Train Synthetic, Test Real).

Trains Decision Tree, ANN, SVM-RBF and SVM-Linear models on synthetic data and
evaluates them on held-out real data. Runs once for each synthetic source.
"""
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.utils import set_random_seed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

REAL_PATH = "data/merged_energy_weather.csv"
SYNTHETIC_SOURCES = {
    "KDE": "data/synthetic_data_autoencoder_kde_window4.csv",
    "GAN": "data/timegan_synthetic_hour_fixed.csv",
}
PREDICTIONS_ROOT = "data/predictions"
TARGET = "Ontario Demand"
EXCLUDED_COLS = [TARGET, "Market Demand"]
RANDOM_STATE = 42


def load_real():
    df = pd.read_csv(REAL_PATH, parse_dates=["DateTime"])
    df = df.drop(columns="DateTime").dropna()
    return df


def load_synthetic(path, columns):
    df = pd.read_csv(path).round(1)
    df = df[columns].dropna()
    return df


def build_ann(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_all(X_train, X_train_scaled, y_train, X_test, X_test_scaled):
    """Fit each model on the training data and return its test predictions."""
    preds = {}

    dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
    dt.fit(X_train, y_train)
    preds["DecisionTree"] = dt.predict(X_test)

    ann = build_ann(X_train_scaled.shape[1])
    ann.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)
    preds["ANN"] = ann.predict(X_test_scaled).flatten()

    svm_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svm_rbf.fit(X_train_scaled, y_train)
    preds["SVM - RBF"] = svm_rbf.predict(X_test_scaled)

    svm_linear = SVR(kernel="linear", C=100, gamma=0.1, epsilon=0.1)
    svm_linear.fit(X_train_scaled, y_train)
    preds["SVM - Linear"] = svm_linear.predict(X_test_scaled)

    return preds


def save_predictions(y_true, y_pred, X_test, synthetic_name, model_name, root):
    folder = os.path.join(root, f"{synthetic_name} Predictions")
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame({
        "month": X_test["Month"].values,
        "day": X_test["Day"].values,
        "hour": X_test["Hour"].values,
        "y_true": np.asarray(y_true),
        "y_pred": y_pred,
        "error": np.abs(np.asarray(y_true) - y_pred),
    })
    df.to_csv(os.path.join(folder, f"TSTR_{model_name}_predictions.csv"), index=False)


def print_metrics(y_true, preds):
    short = {"DecisionTree": "DT", "ANN": "ANN", "SVM - RBF": "SVM RBF", "SVM - Linear": "SVM Linear"}
    results = {}
    for name, y_pred in preds.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results[name] = (mae, rmse, r2)
        print(f"{short[name]} TSTR - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return results


def run_tstr(real_df, syn_df, synthetic_name, root=PREDICTIONS_ROOT):
    X_real = real_df.drop(columns=EXCLUDED_COLS)
    y_real = real_df[TARGET]
    X_syn = syn_df.drop(columns=EXCLUDED_COLS)
    y_syn = syn_df[TARGET]

    # Train on synthetic data only; the real holdout is used for testing.
    X_train, _, y_train, _ = train_test_split(X_syn, y_syn, test_size=0.1, random_state=RANDOM_STATE)
    _, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.1, random_state=RANDOM_STATE)

    # Scaler is fit on the synthetic training data only, so no real-data statistics are used.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    preds = predict_all(X_train, X_train_scaled, y_train, X_test, X_test_scaled)
    print_metrics(y_test, preds)

    for model_name, y_pred in preds.items():
        save_predictions(y_test, y_pred, X_test, synthetic_name, model_name, root)


def main():
    set_random_seed(RANDOM_STATE)
    real_df = load_real()
    for synthetic_name, path in SYNTHETIC_SOURCES.items():
        print(f"\n=== TSTR with {synthetic_name} synthetic data ===")
        syn_df = load_synthetic(path, real_df.columns)
        run_tstr(real_df, syn_df, synthetic_name)


if __name__ == "__main__":
    main()
