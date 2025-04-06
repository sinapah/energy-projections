#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:07:33 2025

@author: sinap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.manifold import TSNE

real = pd.read_csv("merged_energy_weather.csv", parse_dates=["DateTime"])
synthetic = pd.read_csv("synthetic_data_autoencoder_kde_window4.csv", parse_dates=["DateTime"])
#synthetic = pd.read_csv("synthetic_data_gmm_window4.csv", parse_dates=["DateTime"])

print(synthetic.shape)
real = real.drop(columns=["DateTime"])
synthetic = synthetic.drop(columns=["DateTime"])

synthetic = synthetic[real.columns]

common_features = list(set(real.columns) & set(synthetic.columns))
print(common_features)
for col in common_features:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(real[col], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic[col], label="Synthetic", fill=True, alpha=0.5)
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
real_corr = real.corr()
synthetic_corr = synthetic.corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", center=0)
axes[0].set_title("Real Data Correlation")

sns.heatmap(synthetic_corr, ax=axes[1], cmap="coolwarm", center=0)
axes[1].set_title("Synthetic Data Correlation")

plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier

real["source"] = 0
synthetic["source"] = 1

combined = pd.concat([real, synthetic])

combined.to_csv('combined_synthetic_real.csv', index=False)
combined = combined.dropna()  # just in case

X = combined.drop(columns=["Ontario Demand", "source"])
y = combined["source"]

clf = RandomForestClassifier()
clf.fit(X, y)

print(f"Classifier accuracy at distinguishing real vs synthetic: {clf.score(X, y):.2f}")

plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], feature_names=X.columns, filled=True)
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_proj = tsne.fit_transform(X)

plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("t-SNE Projection: Real (0) vs Synthetic (1)")
plt.show()