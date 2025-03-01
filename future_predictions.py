#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:35:23 2025

@author: sinap
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import holidays

# Load Ontario holidays
ontario_holidays = holidays.Canada(subdiv="ON")

# Load the trained ANN model
model = load_model("ann_energy_model.h5")

def predict_energy_demand(future_datetime, historical_data, model):
    historical_data["DateTime"] = pd.to_datetime(historical_data["DateTime"])

    # Extract time-based features
    future_features = {
        "Hour": future_datetime.hour,
        "Month": future_datetime.month,
        "Day": future_datetime.day,
        "DayOfWeek": future_datetime.weekday(),
        "IsWeekend": 1 if future_datetime.weekday() in [5, 6] else 0,
        "IsHoliday": 1 if future_datetime in ontario_holidays else 0,
        "BusinessHour": 1 if ((future_datetime.hour >= 8) & 
                                       (future_datetime.hour <= 17) & 
                                       (future_datetime.weekday() not in [5, 6]) & 
                                       (future_datetime in ontario_holidays)) else 0
    }

    # Identify temperature and humidity columns dynamically
    temp_columns = [col for col in historical_data.columns if col.endswith("_temp")]
    humidity_columns = [col for col in historical_data.columns if col.endswith("_humidity")]

    # Filter past data for the same hour & month
    relevant_data = historical_data[
        (historical_data["DateTime"].dt.hour == future_features["Hour"]) &
        (historical_data["DateTime"].dt.day == future_features["Day"]) &
        (historical_data["DateTime"].dt.month == future_features["Month"])
    ]

    # Compute averages for relevant past records
    avg_price = relevant_data["HOEP"].mean()

    avg_temps = relevant_data[temp_columns].mean().to_dict()

    avg_humidity = relevant_data[humidity_columns].mean().to_dict()
        
    interleaved_weather = {}
    for temp_col, hum_col in zip(temp_columns, humidity_columns):
        interleaved_weather[temp_col] = avg_temps[temp_col]  # Add temp first
        interleaved_weather[hum_col] = avg_humidity[hum_col]
    
    # Create a feature dictionary
    input_features = {
        "HOEP": avg_price,
        **interleaved_weather,    # Include all humidity averages
        **future_features  # Include time-based features
    }
    print(input_features)
    print(input_features.keys())
    # Columns to exclude from the input features
    excluded_columns = ["DateTime", "Market Demand", "Ontario Demand"]

    # Select only the relevant feature columns
    feature_columns = [col for col in historical_data.columns if col not in excluded_columns]
    
    # Ensure only relevant columns are used in input_features
    input_array = np.array([[input_features[col] for col in feature_columns]])
    
    # Make prediction
    predicted_demand = model.predict(input_array)[0, 0]
    return predicted_demand

historical_data = pd.read_csv("merged_energy_weather.csv")  # Load dataset

future_dt = datetime.datetime(2025, 3, 15, 14)  # Predict for March 15, 2025, at 2 PM


prediction = predict_energy_demand(future_dt, historical_data, model)
print(f"Predicted energy demand for {future_dt}: {prediction}")

future_dt = datetime.datetime(2025, 2, 27, 15)  # Predict for Feb 27, 2025, at 3 PM

prediction = predict_energy_demand(future_dt, historical_data, model)
print(f"Predicted energy demand for {future_dt}: {prediction}")

future_dt = datetime.datetime(2022, 2, 27, 15)  # Predict for Feb 27, 2025, at 3 PM

prediction = predict_energy_demand(future_dt, historical_data, model)
print(f"Predicted energy demand for {future_dt}: {prediction}")