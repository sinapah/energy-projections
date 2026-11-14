#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared dataset helpers. Run this file directly to regenerate the
hour-corrected TimeGAN dataset used by the other scripts.
"""

import pandas as pd

GAN_RAW_PATH = "data/timegan_synthetic_raw.csv"
GAN_HOUR_FIXED_PATH = "data/timegan_synthetic_hour_fixed.csv"


def fix_gan_hours(raw_path=GAN_RAW_PATH, fixed_path=GAN_HOUR_FIXED_PATH):
    """Round the TimeGAN Hour, Day, and Month columns to integers and save the corrected copy."""
    df = pd.read_csv(raw_path)
    for col in ["Hour", "Day", "Month"]:
        df[col] = df[col].round().astype(int)
    df.to_csv(fixed_path, index=False)
    return df


if __name__ == "__main__":
    fix_gan_hours()
