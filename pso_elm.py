# -*- coding: utf-8 -*-
"""
PSO-ELM：粒子群优化 ELM 的输入权重与偏置，用于 5.2.1 收敛曲线对比（IBA-ELM vs ELM vs PSO-ELM）
"""
import numpy as np
from elm import ELM, vector_to_elm_weights, kfold_rmse_fitness


def pso_elm_fit(
    X,
    y,
    n_hidden,
    k_fold=5,
    n_pop=30,
    n_gen=50,
    lb=-1.0,
    ub=1.0,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    seed=42,
):
    """
    PSO 优化 ELM：位置 = ELM 权重向量，适应度 = K 折 RMSE。
    返回: elm, best_rmse, history (收敛曲线)
    """
    n_in = X.shape[1]
    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out == 1:
        y = y.reshape(-1, 1)
    dim = n_in * n_hidden + n_hidden
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    folds = [np.asarray(f) for f in np.array_split(indices, k_fold)]

    def fitness(vec):
        return kfold_rmse_fitness(vec, n_in, n_hidden, n_out, X, y, k=k_fold, folds=folds)

    # PSO 初始化
    np.random.seed(seed)
    pos = np.random.uniform(lb, ub, (n_pop, dim))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_fit = np.array([fitness(pos[i]) for i in range(n_pop)])
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    history = [gbest_fit]

    for g in range(n_gen - 1):
        w = w_max - (w_max - w_min) * (g / max(n_gen - 1, 1))
        for i in range(n_pop):
            r1, r2 = np.random.rand(2)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], lb, ub)
            f = fitness(pos[i])
            if f < pbest_fit[i]:
                pbest[i] = pos[i].copy()
                pbest_fit[i] = f
            if f < gbest_fit:
                gbest = pos[i].copy()
                gbest_fit = f
        history.append(gbest_fit)

    W_in, b = vector_to_elm_weights(gbest, n_in, n_hidden, n_out)
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X, y)
    return {
        "elm": elm,
        "best_rmse": gbest_fit,
        "history": np.array(history),
    }
