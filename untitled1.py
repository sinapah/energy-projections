#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:20:37 2025

@author: sinap
"""

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('merged_energy_weather.csv')

# Step 2: Randomly select 5000 rows
  # random_state is optional, it's for reproducibility

# Extract features and target variable
 # Features

removable_cols = []
for col in df.columns:
    if col.startswith(("orillia","oshawa", "peterborough", "brockville", "cornwall", "pickering", "sarnia", "guelph", "brantford", "niagarafalls", "barrie", "markham", "vaughan", "ottawa")):
        removable_cols.append(col)

df = df.drop(columns=removable_cols)

sampled_df = df.sample(n=5000, random_state=42)
X = sampled_df.drop(columns=["DateTime", "Ontario Demand", "Market Demand"]) 
y = sampled_df["Ontario Demand"]  # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Fit a Lasso model
lasso = Lasso(alpha=0.1)  # adjust alpha for regularization strength
lasso.fit(X_train, y_train)

# Step 4: Get the coefficients of the model
coefficients = lasso.coef_

# Step 5: Identify which features are dropped (coefficients equal to 0)
dropped_features = X.columns[coefficients == 0]
print(dropped_features)

# Remove dropped features from the original DataFrame
X_selected = X.drop(columns=dropped_features)

# Now create a new DataFrame with the dropped features removed
sampled_df_selected = sampled_df.drop(columns=dropped_features)

# Reset the index of the DataFrame (optional)
sampled_df_selected = sampled_df_selected.reset_index(drop=True)

sampled_df_selected = sampled_df_selected.drop(columns=["DateTime", "Ontario Demand", "Market Demand"])
# Save the DataFrame without the dropped features to a new CSV
sampled_df_selected.to_csv("reduced_sample.csv", index=False)

# Print the shape and preview of the updated DataFrame
print(sampled_df_selected.shape)

