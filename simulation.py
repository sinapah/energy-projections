#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 16:31:18 2025

@author: sinap
"""

import pandas as pd
import numpy as np
import threading
import logging
from tkinter import *
from tkinter import messagebox, Canvas, Scrollbar, Frame, ttk
from tkcalendar import Calendar
from keras.models import load_model
from datetime import datetime
import joblib

# ============================
# 📊 Logging Setup
# ============================
logging.basicConfig(
    filename="simulator_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================
# 📊 Load Pre-trained Models
# ============================
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_energy_model.h5")
dt_model = joblib.load("decision_tree_model.pkl")
svm_model_nl = joblib.load("svm_model_nl.pkl")
svm_model_linear = joblib.load("svm_model_linear.pkl")

# ============================
# 📌 Load and Normalize Dataset
# ============================
df = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
df = df.dropna()

# ✅ Ensure consistent datetime format (strip timezone info)
df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:00:00")

datetime_col = df["DateTime"]
X = df.drop(columns=["DateTime", "Ontario Demand", "Market Demand"])
y = df["Ontario Demand"]

# ============================
# 🔥 GUI Functions with Multithreading
# ============================

def get_prediction():
    """ Runs the prediction in a separate thread to prevent GUI freezing """
    threading.Thread(target=run_prediction).start()


def run_prediction():
    """ Perform model predictions """
    selected_date = cal.get_date()
    selected_hour = hour_picker.get()

    try:
        # Convert selected date and hour into the proper format
        selected_datetime = datetime.strptime(f"{selected_date} {selected_hour}:00:00", "%m/%d/%y %H:%M:%S")
        formatted_datetime = selected_datetime.strftime("%Y-%m-%d %H:00:00")

        logging.info(f"Selected date and time: {formatted_datetime}")

        if formatted_datetime not in datetime_col.values:
            messagebox.showerror("Error", "Selected date and time is not in the dataset.")
            return

        idx = datetime_col[datetime_col == formatted_datetime].index[0]

        # Prepare input features
        user_input = []
        for entry in feature_entries:
            value = float(entry.get())
            user_input.append(value)
        
        logging.info(f"User Input: {user_input}")

        # Scale the input
        X_scaled = scaler.transform([user_input])

        # Model predictions
        ann_pred = ann_model.predict(X_scaled)[0][0]
        dt_pred = dt_model.predict(X_scaled)[0]
        svm_nl_pred = svm_model_nl.predict(X_scaled)[0]
        svm_linear_pred = svm_model_linear.predict(X_scaled)[0]

        # Actual value
        actual = y.iloc[idx]

        # Update the GUI with the results
        result_text.set(f"""
        ✅ Date: {selected_date} {selected_hour}:00

        📊 Actual Demand: {actual:.2f}

        🔥 Predictions:
        - ANN: {ann_pred:.2f}
        - Decision Tree: {dt_pred:.2f}
        - SVM (RBF): {svm_nl_pred:.2f}
        - SVM (Linear): {svm_linear_pred:.2f}
        """)

        logging.info(f"Prediction results: ANN={ann_pred}, DT={dt_pred}, SVM_RBF={svm_nl_pred}, SVM_Linear={svm_linear_pred}")

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")


def load_features():
    """ Load and display the features for the selected date and hour """
    selected_date = cal.get_date()
    selected_hour = hour_picker.get()

    try:
        # Format selected date and hour to match dataset
        selected_datetime = datetime.strptime(f"{selected_date} {selected_hour}:00:00", "%m/%d/%y %H:%M:%S")
        formatted_datetime = selected_datetime.strftime("%Y-%m-%d %H:00:00")

        if formatted_datetime not in datetime_col.values:
            messagebox.showerror("Error", "Selected date and time is not in the dataset.")
            return

        idx = datetime_col[datetime_col == formatted_datetime].index[0]

        # Display feature values
        for i, col in enumerate(X.columns):
            value = X.iloc[idx, i]
            feature_entries[i].delete(0, END)
            feature_entries[i].insert(0, str(value))

        logging.info(f"Loaded features for {formatted_datetime}")

    except Exception as e:
        logging.error(f"Error loading features: {str(e)}")
        messagebox.showerror("Error", f"Failed to load features: {str(e)}")


# ============================
# 🛠️ GUI Setup with Scrollable Frame and Hour Picker
# ============================

# Main Window
root = Tk()
root.title("Energy Demand Prediction Simulator")
root.geometry("1200x700")

# Create Canvas and Scrollbar
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient=VERTICAL, command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

# Pack canvas and scrollbar
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# Title Label
Label(scrollable_frame, text="Energy Demand Prediction Simulator", font=("Helvetica", 18, "bold")).pack(pady=10)

# Date Picker
Label(scrollable_frame, text="Select Date:", font=("Helvetica", 12)).pack()
cal = Calendar(scrollable_frame, selectmode='day', date_pattern='mm/dd/yy')
cal.pack(pady=10)

# Hour Picker
Label(scrollable_frame, text="Select Hour:", font=("Helvetica", 12)).pack()
hour_picker = ttk.Combobox(scrollable_frame, values=[f"{i:02d}" for i in range(1, 24)], width=5)
hour_picker.current(0)
hour_picker.pack(pady=10)

# Load Features Button
btn_load = Button(scrollable_frame, text="Load Features", command=load_features, font=("Helvetica", 12, "bold"), bg="lightblue")
btn_load.pack(pady=5)

# Frame for Features
frame = Frame(scrollable_frame)
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
btn_predict = Button(scrollable_frame, text="Predict", command=get_prediction, font=("Helvetica", 12, "bold"), bg="green", fg="black")
btn_predict.pack(pady=20)

# Results Display
result_text = StringVar()
result_label = Label(scrollable_frame, textvariable=result_text, font=("Helvetica", 12), justify=LEFT)
result_label.pack(pady=10)

# ============================
# 🛠️ GUI Execution
# ============================
root.mainloop()


