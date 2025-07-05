import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ===========================
# Map Month to Season
# ===========================
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

season_order = ["Winter", "Spring", "Summer", "Fall"]

# ===========================
# Load Prediction Files
# ===========================
def load_prediction_files(folder, source_label):
    dataframes = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)

            # Extract model and mode
            parts = filename.split("_")
            if source_label == "Real":
                model = parts[1]
                mode = "Real"
            else:
                mode = parts[0]
                model = parts[1]

            df["Model"] = model
            df["Mode"] = mode
            df["Source"] = source_label

            # Add Season
            if "month" in df.columns and "hour" in df.columns:
                df["Season"] = df["month"].apply(get_season)
            else:
                raise ValueError(f"'month' or 'hour' column not found in {filename}")

            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# ===========================
# Load Data from All Sources
# ===========================
df_gan = load_prediction_files("GAN Predictions", "TimeGAN")
df_kde = load_prediction_files("KDE Predictions", "KDE")
df_real = load_prediction_files("Real Predictions", "Real")

df_all = pd.concat([df_gan, df_kde, df_real], ignore_index=True)

# ===========================
# Plot 1: Hourly Forecast Error
# ===========================
hourly_error = df_all.groupby(["hour", "Source"])["error"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_error, x="hour", y="error", hue="Source", marker='o')
plt.xlabel("Hour of Day", fontsize=14)
plt.ylabel("Average Absolute Error", fontsize=14)
plt.xticks(range(0, 24, 2), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ===========================
# Plot 2: Seasonal Forecast Error
# ===========================
seasonal_error = df_all.groupby(["Season", "Source"])["error"].mean().reset_index()
seasonal_error["Season"] = pd.Categorical(seasonal_error["Season"], categories=season_order, ordered=True)
seasonal_error = seasonal_error.sort_values("Season")

plt.figure(figsize=(10, 6))
sns.barplot(data=seasonal_error, x="Season", y="error", hue="Source")
plt.xlabel("Season")
plt.ylabel("Average Absolute Error")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
