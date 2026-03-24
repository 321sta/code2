# -*- coding: utf-8 -*-
"""
5.1.2 评价指标：决定系数 R2、均方根误差 RMSE、平均绝对误差 MAE
"""
import numpy as np


def rmse(y_true, y_pred):
    """均方根误差 RMSE"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """平均绝对误差 MAE"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """决定系数 R2"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_metrics(y_true, y_pred):
    """返回 R2, RMSE, MAE 字典"""
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }
