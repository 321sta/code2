# -*- coding: utf-8 -*-
"""
极限学习机 (ELM)：单隐层前馈网络，输入权重与偏置给定后，输出权重由 Moore-Penrose 伪逆解析得到。
用于 4.3：IBA 优化 ELM 的输入权重和隐层偏置。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ELM:
    """ELM：固定输入权重 W_in (n_in x n_hidden) 与偏置 b (n_hidden,)，求输出权重 beta (n_hidden x n_out)。"""

    def __init__(self, n_in, n_hidden, n_out=1):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.W_in = None   # (n_in, n_hidden)
        self.b = None      # (n_hidden,)
        self.beta = None   # (n_hidden, n_out)

    def set_weights(self, W_in, b):
        """设置输入权重与偏置（由 IBA 优化得到）。"""
        self.W_in = np.asarray(W_in).reshape(self.n_in, self.n_hidden)
        self.b = np.asarray(b).reshape(self.n_hidden)

    def fit(self, X, y, lam=0.0):
        """给定 W_in, b 后，求输出权重 beta。
        lam=0: 标准伪逆（Moore-Penrose）；lam>0: L2 正则化岭回归解 (H'H+λI)^{-1}H'y。
        X: (n_sample, n_in), y: (n_sample, n_out)
        """
        if self.W_in is None or self.b is None:
            raise ValueError("set_weights(W_in, b) must be called first")
        H = sigmoid(X @ self.W_in + self.b)  # (n_sample, n_hidden)
        if lam > 0:
            I = np.eye(H.shape[1])
            self.beta = np.linalg.solve(H.T @ H + lam * I, H.T @ y)
        else:
            self.beta = np.linalg.pinv(H) @ y
        return self

    def predict(self, X):
        """X: (n_sample, n_in) -> (n_sample, n_out)"""
        H = sigmoid(X @ self.W_in + self.b)
        return H @ self.beta


def elm_fitness_from_vector(
    vec,
    n_in,
    n_hidden,
    n_out,
    X_train,
    y_train,
    X_val,
    y_val,
    lam=0.0,
):
    """
    将高维向量 vec 解码为 W_in 和 b，构建 ELM 并在 (X_val, y_val) 上计算 RMSE 作为适应度。
    lam>0 时使用 L2 正则化（IBA-RELM）。
    """
    W_in, b = vector_to_elm_weights(vec, n_in, n_hidden, n_out)
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X_train, y_train, lam=lam)
    pred = elm.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - pred) ** 2))
    return rmse


def elm_weights_to_vector(W_in, b):
    """4.3.2 将 ELM 的输入权重和偏置高维向量化，便于蝙蝠位置编码。"""
    return np.concatenate([W_in.ravel(), b.ravel()])


def vector_to_elm_weights(vec, n_in, n_hidden, n_out=1):
    """将蝙蝠位置（高维向量）解码为 W_in 和 b。"""
    size_w = n_in * n_hidden
    W_in = vec[:size_w].reshape(n_in, n_hidden)
    b = vec[size_w : size_w + n_hidden]
    return W_in, b


def kfold_rmse_fitness(
    vec,
    n_in,
    n_hidden,
    n_out,
    X,
    y,
    k=5,
    rng=None,
    folds=None,
    lam=0.0,
):
    """
    适应度函数：K 折交叉验证的 RMSE 的均值，作为蝙蝠的适应度。越小越好。
    若提供 folds（list of array），则使用固定划分，保证优化过程中适应度可比。
    lam>0 时使用 L2 正则化（IBA-RELM）。
    """
    n = len(X)
    if folds is None:
        if rng is None:
            rng = np.random.default_rng()
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = [np.asarray(f) for f in np.array_split(indices, k)]
    rmses = []
    all_idx = np.arange(n)
    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        rmse = elm_fitness_from_vector(vec, n_in, n_hidden, n_out, X_tr, y_tr, X_val, y_val, lam=lam)
        rmses.append(rmse)
    return np.mean(rmses)
