# -*- coding: utf-8 -*-
"""
BA-ELM：标准蝙蝠算法优化 ELM，用于 5.3.1 预测精度对比（IBA-ELM vs BA-ELM vs 标准 ELM vs SVM/BP）
"""
import numpy as np
from elm import ELM, vector_to_elm_weights, kfold_rmse_fitness
from ba_standard import standard_ba
from config import N_POP, N_GEN, N_HIDDEN, K_FOLD, DIM_SCALE


def ba_elm_fit(
    X,
    y,
    n_hidden=None,
    k_fold=None,
    n_pop=None,
    n_gen=None,
    lb=-1.0,
    ub=1.0,
    seed=42,
):
    """
    标准 BA 优化 ELM，适应度 = K 折 RMSE。
    返回: elm, best_rmse, history
    """
    n_in = X.shape[1]
    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out == 1:
        y = y.reshape(-1, 1)
    n_hidden = n_hidden or N_HIDDEN
    k_fold = k_fold or K_FOLD
    n_pop = n_pop or N_POP
    n_gen = n_gen or N_GEN
    dim = n_in * n_hidden + n_hidden
    lb_vec = lb * DIM_SCALE
    ub_vec = ub * DIM_SCALE
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    folds = [np.asarray(f) for f in np.array_split(indices, k_fold)]

    def fitness(vec):
        return kfold_rmse_fitness(vec, n_in, n_hidden, n_out, X, y, k=k_fold, folds=folds)

    best_pos, best_fit, history = standard_ba(
        fitness, dim, n_pop=n_pop, n_gen=n_gen, lb=lb_vec, ub=ub_vec, seed=seed
    )
    W_in, b = vector_to_elm_weights(best_pos, n_in, n_hidden, n_out)
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X, y)
    return {
        "elm": elm,
        "best_rmse": best_fit,
        "history": history,
    }
