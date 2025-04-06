#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:55:13 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from keras.models import load_model
from datetime import datetime
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================
# 📊 Load Pre-trained Models
# ============================
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_energy_model.h5")
df = pd.read_csv("synthetic_data_autoencoder_kde_window4.csv", parse_dates=["DateTime"])
print(df.shape)
df = df.drop(columns=["DateTime", "Market Demand"])
print(df.shape)
# Fill missing values
df = df.dropna()

# ============================
# 🧪 Prepare Features & Target
# ============================
target = "Ontario Demand"
exclude = ["DateTime", target]
features = [col for col in df.columns if col not in exclude]

X = df[features].values
y = df[target].values

# Apply scaler
X_scaled = scaler.transform(X)

# ============================
# 🤖 Make Predictions
# ============================
y_pred = ann_model.predict(X_scaled).flatten()

# ============================
# 📈 Evaluation
# ============================
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("📊 Evaluation on Real Test Data:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")
