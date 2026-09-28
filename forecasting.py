#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 19:25:49 2025

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

# =============================
# Save predictions to folder
# =============================
def save_predictions_with_timeinfo_4yr(y_true, y_pred, time_df, train_source, test_source, model_name, output_folder="YearlyPredictions"):
    os.makedirs(output_folder, exist_ok=True)
    result_df = pd.DataFrame({
        "month": time_df["Month"].values,
        "day": time_df["Day"].values,
        "hour": time_df["Hour"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "error": np.abs(np.array(y_true) - np.array(y_pred))
    })
    
    prefix = "TRTS" if train_source == "Real" else "TSTS"
    file_path = f"{output_folder}/{prefix}_{test_source}_{model_name}_predictions.csv"
    result_df.to_csv(file_path, index=False)

# =============================
# Data Preparation
# =============================
real_df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
#kde_df = pd.read_csv("synthetic_data_pca_kde.csv", parse_dates=["DateTime"])
kde_df = pd.read_csv("new_kde.csv")
gan_df = pd.read_csv("gen_data_rescaled_7000x54_hour_fixed.csv")

target_col = "Ontario Demand"
drop_cols = ["DateTime", "Ontario Demand", "Market Demand"]

def get_train_data(df, max_samples=7000):
    df = df.dropna(subset=[target_col])
    train_df = df[df["Month"].between(1, 10)]
    if len(train_df) < max_samples:
        print(f"⚠️ Only {len(train_df)} rows available for training (requested {max_samples}). Using all available.")
        sampled_df = train_df
    else:
        sampled_df = train_df.sample(n=max_samples, random_state=42)
    X_train = sampled_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_train = sampled_df[target_col]
    return X_train, y_train

def get_test_data(df):
    df = df.dropna(subset=[target_col])
    test_df = df[df["Month"] >= 11]
    X_test = test_df.drop(columns=drop_cols + ["Month"], errors="ignore")
    y_test = test_df[target_col]
    time_info = test_df[["Month", "Day", "Hour"]]
    return X_test, y_test, time_info

# Real training
X_train_real, y_train_real = get_train_data(real_df)

# Test sets
X_test_real, y_test_real, time_real = get_test_data(real_df)
X_test_kde, y_test_kde, time_kde = get_test_data(kde_df)
X_test_gan, y_test_gan, time_gan = get_test_data(gan_df)

# Align feature columns
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
time_dict = {
    "Real": time_real,
    "KDE": time_kde,
    "GAN": time_gan
}

# =============================
# Train models on Real
# =============================
print("\n🔵 Training on Real data (Months 1–10)...")

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

# Evaluate Real → [Real, KDE, GAN]
for test_name in ["Real", "KDE", "GAN"]:
    print(f"\n🔍 Real → {test_name}:")
    preds_dt = dt.predict(X_test_dict[test_name])
    preds_rbf = svr_rbf.predict(X_test_scaled[test_name])
    preds_linear = svr_linear.predict(X_test_scaled[test_name])
    preds_ann = ann_model.predict(X_test_scaled[test_name]).flatten()

    save_predictions_with_timeinfo_4yr(y_test_dict[test_name], preds_dt, time_dict[test_name], "Real", test_name, "DecisionTree")
    save_predictions_with_timeinfo_4yr(y_test_dict[test_name], preds_rbf, time_dict[test_name], "Real", test_name, "SVM_RBF")
    save_predictions_with_timeinfo_4yr(y_test_dict[test_name], preds_linear, time_dict[test_name], "Real", test_name, "SVM_Linear")
    save_predictions_with_timeinfo_4yr(y_test_dict[test_name], preds_ann, time_dict[test_name], "Real", test_name, "ANN")

    print(f"  Decision Tree - MAE: {mean_absolute_error(y_test_dict[test_name], preds_dt):.2f}, R²: {r2_score(y_test_dict[test_name], preds_dt):.4f}")
    print(f"  SVM RBF       - MAE: {mean_absolute_error(y_test_dict[test_name], preds_rbf):.2f}, R²: {r2_score(y_test_dict[test_name], preds_rbf):.4f}")
    print(f"  SVM Linear    - MAE: {mean_absolute_error(y_test_dict[test_name], preds_linear):.2f}, R²: {r2_score(y_test_dict[test_name], preds_linear):.4f}")
    print(f"  ANN           - MAE: {mean_absolute_error(y_test_dict[test_name], preds_ann):.2f}, R²: {r2_score(y_test_dict[test_name], preds_ann):.4f}")

# =============================
# KDE → KDE
# =============================
print("\n🟣 Training and testing on KDE data...")

X_train_kde, y_train_kde = get_train_data(kde_df)
X_test_kde2, y_test_kde2, time_kde2 = get_test_data(kde_df)
X_train_kde_scaled = scaler.fit_transform(X_train_kde)
X_test_kde_scaled2 = scaler.transform(X_test_kde2)

dt_kde = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt_kde.fit(X_train_kde, y_train_kde)

svr_rbf_kde = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf_kde.fit(X_train_kde_scaled, y_train_kde)

svr_linear_kde = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svr_linear_kde.fit(X_train_kde_scaled, y_train_kde)

ann_kde = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_kde_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_kde.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
ann_kde.fit(X_train_kde_scaled, y_train_kde, epochs=100, batch_size=32, verbose=0)

print(f"\n🔍 KDE → KDE:")
preds_dt_kde = dt_kde.predict(X_test_kde2)
preds_rbf_kde = svr_rbf_kde.predict(X_test_kde_scaled2)
preds_linear_kde = svr_linear_kde.predict(X_test_kde_scaled2)
preds_ann_kde = ann_kde.predict(X_test_kde_scaled2).flatten()

save_predictions_with_timeinfo_4yr(y_test_kde2, preds_dt_kde, time_kde2, "KDE", "KDE", "DecisionTree")
save_predictions_with_timeinfo_4yr(y_test_kde2, preds_rbf_kde, time_kde2, "KDE", "KDE", "SVM_RBF")
save_predictions_with_timeinfo_4yr(y_test_kde2, preds_linear_kde, time_kde2, "KDE", "KDE", "SVM_Linear")
save_predictions_with_timeinfo_4yr(y_test_kde2, preds_ann_kde, time_kde2, "KDE", "KDE", "ANN")

print(f"  Decision Tree - MAE: {mean_absolute_error(y_test_kde2, preds_dt_kde):.2f}, R²: {r2_score(y_test_kde2, preds_dt_kde):.4f}")
print(f"  SVM RBF       - MAE: {mean_absolute_error(y_test_kde2, preds_rbf_kde):.2f}, R²: {r2_score(y_test_kde2, preds_rbf_kde):.4f}")
print(f"  SVM Linear    - MAE: {mean_absolute_error(y_test_kde2, preds_linear_kde):.2f}, R²: {r2_score(y_test_kde2, preds_linear_kde):.4f}")
print(f"  ANN           - MAE: {mean_absolute_error(y_test_kde2, preds_ann_kde):.2f}, R²: {r2_score(y_test_kde2, preds_ann_kde):.4f}")


# =============================
# GAN → GAN
# =============================
print("\n🟢 Training and testing on GAN data...")

X_train_gan, y_train_gan = get_train_data(gan_df)
X_test_gan2, y_test_gan2, time_gan2 = get_test_data(gan_df)
X_train_gan_scaled = scaler.fit_transform(X_train_gan)
X_test_gan_scaled2 = scaler.transform(X_test_gan2)

dt_gan = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
dt_gan.fit(X_train_gan, y_train_gan)

svr_rbf_gan = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf_gan.fit(X_train_gan_scaled, y_train_gan)

svr_linear_gan = SVR(kernel='linear', C=100, gamma=0.1, epsilon=0.1)
svr_linear_gan.fit(X_train_gan_scaled, y_train_gan)

ann_gan = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_gan_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_gan.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
ann_gan.fit(X_train_gan_scaled, y_train_gan, epochs=100, batch_size=32, verbose=0)

print(f"\n🔍 GAN → GAN:")
preds_dt_gan = dt_gan.predict(X_test_gan2)
preds_rbf_gan = svr_rbf_gan.predict(X_test_gan_scaled2)
preds_linear_gan = svr_linear_gan.predict(X_test_gan_scaled2)
preds_ann_gan = ann_gan.predict(X_test_gan_scaled2).flatten()

save_predictions_with_timeinfo_4yr(y_test_gan2, preds_dt_gan, time_gan2, "GAN", "GAN", "DecisionTree")
save_predictions_with_timeinfo_4yr(y_test_gan2, preds_rbf_gan, time_gan2, "GAN", "GAN", "SVM_RBF")
save_predictions_with_timeinfo_4yr(y_test_gan2, preds_linear_gan, time_gan2, "GAN", "GAN", "SVM_Linear")
save_predictions_with_timeinfo_4yr(y_test_gan2, preds_ann_gan, time_gan2, "GAN", "GAN", "ANN")

print(f"  Decision Tree - MAE: {mean_absolute_error(y_test_gan2, preds_dt_gan):.2f}, R²: {r2_score(y_test_gan2, preds_dt_gan):.4f}")
print(f"  SVM RBF       - MAE: {mean_absolute_error(y_test_gan2, preds_rbf_gan):.2f}, R²: {r2_score(y_test_gan2, preds_rbf_gan):.4f}")
print(f"  SVM Linear    - MAE: {mean_absolute_error(y_test_gan2, preds_linear_gan):.2f}, R²: {r2_score(y_test_gan2, preds_linear_gan):.4f}")
print(f"  ANN           - MAE: {mean_absolute_error(y_test_gan2, preds_ann_gan):.2f}, R²: {r2_score(y_test_gan2, preds_ann_gan):.4f}")

