# -*- coding: utf-8 -*-
"""
4.3 IBA 优化 ELM 的融合设计
4.3.1 适应度：K 折交叉验证 RMSE
4.3.2 蝙蝠位置 = ELM 输入权重与偏置的高维向量
4.3.3 整体流程与调用入口
"""
import numpy as np
from elm import (
    ELM,
    vector_to_elm_weights,
    kfold_rmse_fitness,
)
from iba import iba
from config import (
    N_POP, N_GEN, N_HIDDEN, K_FOLD,
    W_MAX, W_MIN, N_ELITE, CHAOS_TYPE,
    DIM_SCALE,
)


def get_elm_dim(n_in, n_hidden, n_out=1):
    """蝙蝠位置维度 = n_in*n_hidden + n_hidden"""
    return n_in * n_hidden + n_hidden


def iba_elm_fit(
    X,
    y,
    n_hidden=None,
    k_fold=None,
    n_pop=None,
    n_gen=None,
    lb=-1.0,
    ub=1.0,
    chaos_type=None,
    w_max=None,
    w_min=None,
    n_elite=None,
    use_chaos=True,
    seed=42,
):
    """
    4.3.3 IBA-ELM 整体流程：
    1) 确定解空间维度 = n_in*n_hidden + n_hidden
    2) 适应度 = K 折交叉验证 RMSE
    3) 用 IBA 搜索最优向量，再解码为 W_in, b
    4) 用全量数据重新拟合 ELM 得到 beta，返回模型与最优 RMSE
    """
    n_in = X.shape[1]
    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out == 1:
        y = y.reshape(-1, 1)
    n_hidden = n_hidden or N_HIDDEN
    k_fold = k_fold or K_FOLD
    n_pop = n_pop or N_POP
    n_gen = n_gen or N_GEN
    chaos_type = chaos_type or CHAOS_TYPE
    w_max = w_max or W_MAX
    w_min = w_min or W_MIN
    n_elite = n_elite or N_ELITE

    dim = get_elm_dim(n_in, n_hidden, n_out)
    lb_vec = lb * DIM_SCALE
    ub_vec = ub * DIM_SCALE
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    folds = [np.asarray(f) for f in np.array_split(indices, k_fold)]

    def fitness(vec):
        return kfold_rmse_fitness(vec, n_in, n_hidden, n_out, X, y, k=k_fold, folds=folds)

    best_pos, best_fit, history = iba(
        fitness,
        dim,
        n_pop=n_pop,
        n_gen=n_gen,
        lb=lb_vec,
        ub=ub_vec,
        chaos_type=chaos_type,
        w_max=w_max,
        w_min=w_min,
        n_elite=n_elite,
        use_chaos=use_chaos,
        seed=seed,
    )
    W_in, b = vector_to_elm_weights(best_pos, n_in, n_hidden, n_out)
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X, y)
    return {
        "elm": elm,
        "best_rmse": best_fit,
        "history": history,
        "best_pos": best_pos,
        "W_in": W_in,
        "b": b,
    }
