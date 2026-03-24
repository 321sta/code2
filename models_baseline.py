# -*- coding: utf-8 -*-
"""
5.3.1 常见对比模型：SVM (SVR)、BP (MLP)，用于与 IBA-ELM / BA-ELM / 标准 ELM 对比
"""
import numpy as np

try:
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def fit_svr(X, y, **kwargs):
    """SVR 回归，默认 RBF 核。返回 fitted 模型，用于 predict(X)."""
    if not HAS_SKLEARN:
        return None
    y_flat = np.asarray(y).ravel()
    model = SVR(kernel="rbf", C=10.0, gamma="scale", **kwargs)
    model.fit(X, y_flat)
    return model


def fit_mlp(X, y, hidden_layer_sizes=(15,), max_iter=500, **kwargs):
    """BP 神经网络 (MLPRegressor)。返回 fitted 模型。"""
    if not HAS_SKLEARN:
        return None
    y_flat = np.asarray(y).ravel()
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
        **kwargs,
    )
    model.fit(X, y_flat)
    return model


def predict_svr(model, X):
    if model is None:
        return np.full(len(X), np.nan)
    return model.predict(X).reshape(-1, 1)


def predict_mlp(model, X):
    if model is None:
        return np.full(len(X), np.nan)
    return model.predict(X).reshape(-1, 1)
