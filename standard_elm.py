# -*- coding: utf-8 -*-
"""
标准 ELM：随机初始化输入权重与偏置，无优化，用于 5.2.1/5.3.1 对比
"""
import numpy as np
from elm import ELM


def standard_elm_fit(X, y, n_hidden, lb=-1.0, ub=1.0, seed=42):
    """
    标准 ELM：W_in 和 b 在 [lb, ub] 内随机生成，仅求 beta。
    返回: elm, history=None（无优化过程，可返回单点 K 折 RMSE 作为“曲线”的常值）
    """
    n_in = X.shape[1]
    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out == 1:
        y = y.reshape(-1, 1)
    np.random.seed(seed)
    W_in = np.random.uniform(lb, ub, (n_in, n_hidden))
    b = np.random.uniform(lb, ub, n_hidden)
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X, y)
    return {"elm": elm, "best_rmse": None, "history": None}
