# -*- coding: utf-8 -*-
"""
IBA-RELM：将正则化系数 λ 纳入 IBA 搜索空间，与 ELM 输入权重联合优化。

相较于 IBA-ELM（iba_elm.py），本模块新增：
  - 搜索向量末尾追加 1 维，存储 log10(λ)，搜索范围 [-4, 2]
    即 λ ∈ [1e-4, 100]，由 IBA 自适应确定，无需人工调参
  - ELM 求解 beta 时改用 L2 正则化岭回归：(H'H + λI)^{-1} H'y
  - 模型名称升级为 IBA-RELM（Regularized ELM）

搜索向量结构（总维度 = n_in*n_hidden + n_hidden + 1）：
  [ W_in(拉平), b, log10(λ) ]
    ←─── 原 IBA-ELM 空间 ───→  ←新增→
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

# λ 的 log10 搜索范围：对应 λ ∈ [1e-4, 100]
LAM_LOG_LB = -4.0
LAM_LOG_UB = 2.0


def get_relm_dim(n_in, n_hidden, n_out=1):
    """IBA-RELM 搜索向量维度 = n_in*n_hidden + n_hidden + 1（+1 为 log10(λ)）"""
    return n_in * n_hidden + n_hidden + 1


def iba_relm_fit(
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
    IBA-RELM 整体流程：
    1) 搜索向量末位追加 log10(λ)，维度 +1
    2) λ 的搜索范围独立设为 [LAM_LOG_LB, LAM_LOG_UB]，其余维度不变
    3) 适应度 = K 折交叉验证 RMSE（使用 lam = 10^vec[-1] 的正则化 ELM）
    4) IBA 寻优后解码 W_in, b, λ，用全量数据岭回归拟合最终 ELM
    返回与 iba_elm_fit 格式一致的字典，额外包含 best_lambda。
    """
    n_in = X.shape[1]
    n_out = y.shape[1] if y.ndim > 1 else 1
    if n_out == 1:
        y = y.reshape(-1, 1)

    n_hidden = n_hidden or N_HIDDEN
    k_fold   = k_fold   or K_FOLD
    n_pop    = n_pop    or N_POP
    n_gen    = n_gen    or N_GEN
    chaos_type = chaos_type or CHAOS_TYPE
    w_max    = w_max    or W_MAX
    w_min    = w_min    or W_MIN
    n_elite  = n_elite  or N_ELITE

    dim = get_relm_dim(n_in, n_hidden, n_out)

    # 构造逐维度搜索边界：前 dim-1 维同 IBA-ELM，末位为 log10(λ)
    lb_arr = np.full(dim, lb * DIM_SCALE)
    ub_arr = np.full(dim, ub * DIM_SCALE)
    lb_arr[-1] = LAM_LOG_LB
    ub_arr[-1] = LAM_LOG_UB

    # 固定 K 折划分，保证每代适应度可比
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    folds = [np.asarray(f) for f in np.array_split(indices, k_fold)]

    def fitness(vec):
        lam = float(10 ** vec[-1])           # 解码正则化系数
        weights_vec = vec[:-1]               # 前段为 W_in 和 b
        return kfold_rmse_fitness(
            weights_vec, n_in, n_hidden, n_out, X, y,
            k=k_fold, folds=folds, lam=lam,
        )

    best_pos, best_fit, history = iba(
        fitness,
        dim,
        n_pop=n_pop,
        n_gen=n_gen,
        lb=lb_arr,
        ub=ub_arr,
        chaos_type=chaos_type,
        w_max=w_max,
        w_min=w_min,
        n_elite=n_elite,
        use_chaos=use_chaos,
        seed=seed,
    )

    # 解码最优解
    best_lam = float(10 ** best_pos[-1])
    W_in, b = vector_to_elm_weights(best_pos[:-1], n_in, n_hidden, n_out)

    # 用全量数据 + 最优 λ 拟合最终模型
    elm = ELM(n_in, n_hidden, n_out)
    elm.set_weights(W_in, b)
    elm.fit(X, y, lam=best_lam)

    return {
        "elm": elm,
        "best_rmse": best_fit,
        "history": history,
        "best_pos": best_pos,
        "W_in": W_in,
        "b": b,
        "best_lambda": best_lam,
    }
