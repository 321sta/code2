# -*- coding: utf-8 -*-
"""加载并合并 2023.10-2024.06 与 2024.10-2025.06 两期冬小麦日尺度数据"""
import pandas as pd
import numpy as np
from config import (
    CLEANED_23_24, CLEANED_24_25,
    WHEAT_ETC_23_24, WHEAT_ETC_24_25,
    FEATURE_COLS, TARGET_COL,
)

def load_season_23_24():
    """2023.10-2024.06 生育期：气象 + ET0/ETC/kc"""
    weather = pd.read_csv(CLEANED_23_24)
    weather["date"] = pd.to_datetime(weather["date"])
    etc = pd.read_csv(WHEAT_ETC_23_24)
    etc["date"] = pd.to_datetime(etc["date"])
    df = weather.merge(etc[["date", "et0", "etc", "kc"]], on="date", how="inner")
    return df

def load_season_24_25():
    """2024.10-2025.06 生育期"""
    weather = pd.read_csv(CLEANED_24_25)
    weather["date"] = pd.to_datetime(weather["date"])
    etc = pd.read_csv(WHEAT_ETC_24_25)
    etc["date"] = pd.to_datetime(etc["date"])
    df = weather.merge(etc[["date", "et0", "etc", "kc"]], on="date", how="inner")
    return df

def get_merged_data():
    """合并两期数据，用于建模"""
    d1 = load_season_23_24()
    d2 = load_season_24_25()
    d1["season"] = "2023-2024"
    d2["season"] = "2024-2025"
    merged = pd.concat([d1, d2], ignore_index=True)
    return merged


def get_merged_data_with_stage():
    """合并两期数据并保留生育期 stage，用于 5.3.2 不同生育期预测误差分析"""
    etc1 = pd.read_csv(WHEAT_ETC_23_24)
    etc1["date"] = pd.to_datetime(etc1["date"])
    etc2 = pd.read_csv(WHEAT_ETC_24_25)
    etc2["date"] = pd.to_datetime(etc2["date"])
    w1 = pd.read_csv(CLEANED_23_24)
    w1["date"] = pd.to_datetime(w1["date"])
    w2 = pd.read_csv(CLEANED_24_25)
    w2["date"] = pd.to_datetime(w2["date"])
    d1 = w1.merge(etc1[["date", "et0", "etc", "kc", "stage"]], on="date", how="inner")
    d2 = w2.merge(etc2[["date", "et0", "etc", "kc", "stage"]], on="date", how="inner")
    d1["season"] = "2023-2024"
    d2["season"] = "2024-2025"
    return pd.concat([d1, d2], ignore_index=True)


def get_X_y_with_stage(df=None, target_col=None):
    """提取特征、目标及生育期 stage；返回 X, y, scaler, feature_names, stages"""
    if df is None:
        df = get_merged_data_with_stage()
    if target_col is None:
        target_col = TARGET_COL
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].astype(float).values
    y = df[target_col].astype(float).values.reshape(-1, 1)
    stages = df["stage"].astype(str).values if "stage" in df.columns else None
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1e-8
    X = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std == 0:
        y_std = 1e-8
    y = (y - y_mean) / y_std
    return X, y, (X_mean, X_std, y_mean, y_std), avail, stages

def get_X_y(df=None, target_col=None):
    """提取特征 X 与目标 y，并做简单标准化（按列零均值单位方差）。"""
    if df is None:
        df = get_merged_data()
    if target_col is None:
        target_col = TARGET_COL
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].astype(float).values
    y = df[target_col].astype(float).values.reshape(-1, 1)
    # 标准化
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1e-8
    X = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std == 0:
        y_std = 1e-8
    y = (y - y_mean) / y_std
    return X, y, (X_mean, X_std, y_mean, y_std), avail

if __name__ == "__main__":
    df = get_merged_data()
    print("Merged shape:", df.shape)
    print("Columns:", list(df.columns))
    X, y, scaler, cols = get_X_y()
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Features:", cols)
