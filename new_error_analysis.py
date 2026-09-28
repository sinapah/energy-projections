import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# ===========================
# Season Mapping Helper
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
# General File Loader (term-aware)
# ===========================
def load_prediction_files(folder, term_label, assume_real_train=False):
    dataframes = []
    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)

        # Ensure required columns
        if "month" not in df.columns or "hour" not in df.columns:
            raise ValueError(f"'month' or 'hour' column not found in {filename}")
        df["Season"] = df["month"].apply(get_season)

        # Parse file naming
        parts = filename.replace(".csv", "").split("_")

        if assume_real_train:
            # Format: Real_KDE_predictions.csv or Real_GAN_predictions.csv etc.
            train_src = "Real"
            test_src = parts[1]
        else:
            # Format: TRTS_KDE_predictions.csv, TSTS_GAN_predictions.csv, KDE_predictions.csv
            if parts[0] == "TRTS":
                train_src = "Real"
                test_src = parts[1]
            elif parts[0] == "TSTS":
                train_src = parts[1]
                test_src = parts[1]
            else:
                train_src = parts[0]
                test_src = parts[0]
        
        if train_src == "GAN":
            train_src = "TimeGAN"
        
        if test_src == "GAN":
            test_src = "TimeGAN"
            
        df["TrainSource"] = train_src
        df["TestSource"] = test_src
        df["Term"] = term_label
        df["Label"] = f"{term_label} - {train_src} → {test_src}"

        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# ===========================
# Load Data
# ===========================
df_short = load_prediction_files("YearlyPredictions", "Short Term", assume_real_train=False)
df_long = load_prediction_files("FourYearPredictions", "Long Term", assume_real_train=True)

# Combine all for line plot
df_all = pd.concat([df_short, df_long], ignore_index=True)

# ===========================
# Line Plot: Hourly Forecast Error
# ===========================
hourly_error = df_all.groupby(["hour", "Label"])["error"].mean().reset_index()

plt.figure(figsize=(12, 6))
palette = sns.color_palette("tab10", n_colors=hourly_error["Label"].nunique())

sns.lineplot(
    data=hourly_error,
    x="hour",
    y="error",
    hue="Label",
    linewidth=2.2,
    marker=None,
    palette=palette
)

plt.xlabel("Hour of Day", fontsize=14)
plt.ylabel("Average Absolute Error", fontsize=14)
plt.xticks(range(0, 24, 2), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Scenario", fontsize=11, title_fontsize=12, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===========================
# Bar Plot: Seasonal Error (Long Term Only)
# ===========================
seasonal_df = df_long.copy()
seasonal_df["Season"] = pd.Categorical(seasonal_df["Season"], categories=season_order, ordered=True)

seasonal_error = (
    seasonal_df
    .groupby(["Season", "TrainSource", "TestSource"])
    .agg(mean_error=("error", "mean"))
    .reset_index()
)
seasonal_error["Label"] = "Long Term - " + seasonal_error["TrainSource"] + " → " + seasonal_error["TestSource"]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=seasonal_error,
    x="Season",
    y="mean_error",
    hue="Label"
)
plt.xlabel("Season", fontsize=13)
plt.ylabel("Average Absolute Error", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Scenario", fontsize=11, title_fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()

