#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:31:18 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from tkinter import *
from tkinter import messagebox, ttk
from tkcalendar import Calendar
import joblib
from keras.models import load_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from datetime import datetime

# ============================
# 📊 Load Pre-trained Models
# ============================
# Load scaler and models
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_energy_model.h5")
lstm_model = load_model("lstm_energy_model.h5")
dt_model = joblib.load("decision_tree_model.pkl")
#svm_model = joblib.load("svm_model.pkl")

# ============================
# 📌 Load Dataset
# ============================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df = df.dropna()

# Extract datetime and features
datetime_col = df["DateTime"]
X = df.drop(columns=["DateTime", "Ontario Demand", "Market Demand"])
y = df["Ontario Demand"]

# ============================
# 🔥 GUI Functions
# ============================

def get_prediction():
    """ Get predictions using user-modified feature values """
    selected_date = cal.get_date()

    # Convert date to match dataset format
    selected_datetime = datetime.strptime(selected_date, "%m/%d/%y")

    # Check if the selected date is valid
    if selected_datetime not in datetime_col.values:
        messagebox.showerror("Error", "Selected date is not in the dataset.")
        return

    # Get the row corresponding to the selected date
    idx = datetime_col[datetime_col == selected_datetime].index[0]
    
    # Create input feature array based on user modifications
    user_input = []
    for entry in feature_entries:
        value = float(entry.get())
        user_input.append(value)

    # Scale the user input
    X_scaled = scaler.transform([user_input])

    # ANN Prediction
    ann_pred = ann_model.predict(X_scaled)[0][0]

    # LSTM Prediction (reshape for time series)
    lstm_input = np.array(X_scaled).reshape(1, 1, -1)  # Reshape for LSTM input shape
    lstm_pred = lstm_model.predict(lstm_input)[0][0]

    # Decision Tree Prediction
    dt_pred = dt_model.predict(X_scaled)[0]

    # SVM Prediction
    #svm_pred = svm_model.predict(X_scaled)[0]

    # Display results
    actual = y.iloc[idx]

    result_text.set(f"""
    ✅ Date: {selected_date}

    📊 Actual Demand: {actual:.2f}

    🔥 Predictions:
    - ANN: {ann_pred:.2f}
    - LSTM: {lstm_pred:.2f}
    - Decision Tree: {dt_pred:.2f}
    
    """)


def load_features():
    """ Load and display the features for the selected date """
    selected_date = cal.get_date()

    # Convert date to match dataset format
    selected_datetime = datetime.strptime(selected_date, "%m/%d/%y")

    # Check if the selected date exists
    if selected_datetime not in datetime_col.values:
        messagebox.showerror("Error", "Selected date is not in the dataset.")
        return

    # Get the row corresponding to the selected date
    idx = datetime_col[datetime_col == selected_datetime].index[0]

    # Display feature values as placeholders
    for i, col in enumerate(X.columns):
        value = X.iloc[idx, i]
        feature_entries[i].delete(0, END)
        feature_entries[i].insert(0, str(value))


# ============================
# 🛠️ GUI Setup
# ============================
root = Tk()
root.title("Energy Demand Prediction Simulator")
root.geometry("1200x700")

# Title Label
Label(root, text="Energy Demand Prediction Simulator", font=("Helvetica", 18, "bold")).pack(pady=10)

# Date Picker
Label(root, text="Select Date:", font=("Helvetica", 12)).pack()
cal = Calendar(root, selectmode='day', date_pattern='mm/dd/yy')
cal.pack(pady=10)

# Load Features Button
btn_load = Button(root, text="Load Features", command=load_features, font=("Helvetica", 12, "bold"), bg="lightblue")
btn_load.pack(pady=5)

# Frame for Features
frame = Frame(root)
frame.pack(pady=20)

# Feature labels and entry fields
feature_entries = []
for i, feature in enumerate(X.columns):
    row = i // 4
    col = i % 4
    lbl = Label(frame, text=feature, font=("Helvetica", 10, "bold"))
    lbl.grid(row=row, column=col * 2, padx=5, pady=5)

    entry = Entry(frame, width=10)
    entry.grid(row=row, column=(col * 2) + 1, padx=5, pady=5)
    feature_entries.append(entry)

# Predict Button
btn_predict = Button(root, text="Predict", command=get_prediction, font=("Helvetica", 12, "bold"), bg="green", fg="white")
btn_predict.pack(pady=20)

# Results Display
result_text = StringVar()
result_label = Label(root, textvariable=result_text, font=("Helvetica", 12), justify=LEFT)
result_label.pack(pady=10)

# ============================
# 🛠️ GUI Execution
# ============================
root.mainloop()
