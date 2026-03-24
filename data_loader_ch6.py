# -*- coding: utf-8 -*-
"""
第六章 气象数据缺失情景：情景 I/II/III 特征设计
情景 I：完整气象要素（温度、湿度、风速、辐射 + DOY）
情景 II：无辐射（温度、湿度、风速 + DOY）
情景 III：仅温度（T_max, T_min + DOY）
"""
import pandas as pd
import numpy as np
from data_loader import get_merged_data
from config import TARGET_COL

# 情景 I：完整气象 + DOY（含 kc 便于 ETc 预测）
SCENARIO_I_COLS = ["T_mean", "RH_mean", "u_2_mean", "Rs_mean", "DOY", "kc"]
# 情景 II：无辐射
SCENARIO_II_COLS = ["T_mean", "RH_mean", "u_2_mean", "DOY", "kc"]
# 情景 III：仅温度 + DOY
SCENARIO_III_COLS = ["T_max", "T_min", "DOY"]


def get_merged_data_with_doy():
    """合并两期数据并添加 DOY（日序，1–366）"""
    df = get_merged_data()
    df["DOY"] = pd.to_datetime(df["date"]).dt.dayofyear
    return df


def get_X_y_scenario(scenario, df=None, target_col=None):
    """
    按情景返回特征 X 与目标 y（标准化）。
    scenario: "I" | "II" | "III"
    返回: X, y, scaler_dict, feature_names
    """
    if df is None:
        df = get_merged_data_with_doy()
    if target_col is None:
        target_col = TARGET_COL
    if scenario.upper() == "I":
        cols = [c for c in SCENARIO_I_COLS if c in df.columns]
    elif scenario.upper() == "II":
        cols = [c for c in SCENARIO_II_COLS if c in df.columns]
    elif scenario.upper() == "III":
        cols = [c for c in SCENARIO_III_COLS if c in df.columns]
    else:
        raise ValueError("scenario must be I, II, or III")
    X = df[cols].astype(float).values
    y = df[target_col].astype(float).values.reshape(-1, 1)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1e-8
    X = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    if y_std == 0:
        y_std = 1e-8
    y = (y - y_mean) / y_std
    scaler = (X_mean, X_std, y_mean, y_std)
    return X, y, scaler, cols
