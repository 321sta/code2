# -*- coding: utf-8 -*-
"""
Hargreaves-Samani 经验公式：仅用温度与日序估算 ET0，用于 6.3 与 IBA-ELM（仅温度情景）对比。
ET0 = 0.0023 * Ra * (T_mean + 17.8) * sqrt(T_max - T_min)
Ra: 地外辐射 MJ/(m2·d)，由纬度和日序计算。
"""
import numpy as np


def deg2rad(deg):
    return np.deg2rad(deg)


def sol_dec(day_of_year):
    """太阳赤纬 (rad)"""
    return 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)


def inv_rel_dist_earth_sun(day_of_year):
    """日地相对距离倒数"""
    return 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)


def sunset_hour_angle(lat_rad, sol_dec_rad):
    """日落时角 (rad)"""
    cos_omega = -np.tan(lat_rad) * np.tan(sol_dec_rad)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    return np.arccos(cos_omega)


def et_rad(lat_rad, sol_dec_rad, sha_rad, ird):
    """地外辐射 Ra, MJ/(m2·d)"""
    Gsc = 0.0820  # MJ/(m2·min)
    tmp = sha_rad * np.sin(lat_rad) * np.sin(sol_dec_rad) + np.cos(lat_rad) * np.cos(sol_dec_rad) * np.sin(sha_rad)
    return (24 * 60 / np.pi) * Gsc * ird * np.maximum(tmp, 0)


def hargreaves_et0(T_min, T_max, T_mean, Ra):
    """
    Hargreaves-Samani ET0 (mm/d).
    T_min, T_max, T_mean in °C; Ra in MJ/(m2·d).
    """
    delta_T = np.maximum(T_max - T_min, 0.1)  # 避免 sqrt(0)
    return 0.0023 * Ra * (T_mean + 17.8) * np.sqrt(delta_T)


def et0_series_from_df(df, latitude_deg=34.0):
    """
    由 DataFrame 的 date, T_min, T_max, T_mean 计算每日 ET0 (mm/d)。
    latitude_deg: 纬度（度），默认 34°N（黄淮冬麦区典型纬度）。
    """
    import pandas as pd
    date = pd.to_datetime(df["date"])
    doy = date.dt.dayofyear.values
    T_min = df["T_min"].astype(float).values
    T_max = df["T_max"].astype(float).values
    T_mean = df["T_mean"].astype(float).values
    lat_rad = deg2rad(latitude_deg)
    Ra = np.zeros_like(doy, dtype=float)
    for i, d in enumerate(doy):
        sd = sol_dec(d)
        ird = inv_rel_dist_earth_sun(d)
        sha = sunset_hour_angle(lat_rad, sd)
        Ra[i] = et_rad(lat_rad, sd, sha, ird)
    et0 = hargreaves_et0(T_min, T_max, T_mean, Ra)
    return et0


def etc_hargreaves(df, latitude_deg=34.0):
    """
    ETc = kc * ET0_HS。返回与 df 同长的 ETc 数组 (mm/d)。
    """
    import pandas as pd
    et0 = et0_series_from_df(df, latitude_deg)
    kc = df["kc"].astype(float).values
    return kc * et0
