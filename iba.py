# -*- coding: utf-8 -*-
"""
4.2 改进蝙蝠算法 (IBA)
4.2.1 混沌映射（Tent/Logistic）优化初始种群
4.2.2 自适应惯性权重平衡全局与局部搜索
4.2.3 精英保留策略防止种群退化
"""
import numpy as np
from chaos import chaos_init_population


def adaptive_weight(gen, max_gen, w_max=0.9, w_min=0.4):
    """4.2.2 自适应惯性权重：随迭代从 w_max 线性降至 w_min，前期全局、后期局部"""
    return w_max - (w_max - w_min) * (gen / max_gen)


def iba(
    fitness_func,
    dim,
    n_pop=30,
    n_gen=50,
    f_min=0.0,
    f_max=2.0,
    A0=0.9,
    r0=0.5,
    alpha=0.97,
    gamma=0.1,
    lb=-1.0,
    ub=1.0,
    chaos_type="tent",
    w_max=0.9,
    w_min=0.4,
    n_elite=2,
    use_chaos=True,
    seed=None,
):
    """
    IBA：混沌初始化(可选) + 自适应惯性权重 + 精英保留(可选)。
    use_chaos=False 用于消融实验（无混沌）；n_elite=0 用于消融（无精英）。
    返回: best_pos, best_fit, history_best_fit
    """
    if seed is not None:
        np.random.seed(seed)
    # 4.2.1 混沌初始化 或 随机初始化（消融用）
    if use_chaos:
        pop = chaos_init_population(n_pop, dim, lb, ub, map_type=chaos_type, seed=seed)
    else:
        pop = np.random.uniform(lb, ub, (n_pop, dim))
    vel = np.zeros_like(pop)
    fit = np.array([fitness_func(pop[i]) for i in range(n_pop)])
    # 4.2.3 精英：按适应度排序，保留前 n_elite 个
    order = np.argsort(fit)
    elite_indices = order[:n_elite]
    best_idx = order[0]
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    A = np.full(n_pop, A0)
    r = np.full(n_pop, r0)
    history_best = [best_fit]

    for g in range(n_gen - 1):
        w = adaptive_weight(g, n_gen, w_max, w_min)  # 4.2.2 自适应惯性权重
        for i in range(n_pop):
            fi = f_min + (f_max - f_min) * np.random.rand()
            # 速度更新：加入惯性权重 w
            vel[i] = w * vel[i] + (pop[i] - best_pos) * fi
            new_pos = pop[i] + vel[i]
            new_pos = np.clip(new_pos, lb, ub)
            if np.random.rand() > r[i]:
                new_pos = best_pos + 0.001 * np.random.randn(dim)
                new_pos = np.clip(new_pos, lb, ub)
            new_fit = fitness_func(new_pos)
            if new_fit < fit[i] and np.random.rand() < A[i]:
                pop[i] = new_pos
                fit[i] = new_fit
                A[i] = alpha * A[i]
                r[i] = r0 * (1 - np.exp(-gamma * (g + 1)))
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = new_pos.copy()
        # 4.2.3 精英保留：用精英个体替换最差的 n_elite 个（n_elite=0 则跳过，消融用）
        if n_elite > 0:
            order = np.argsort(fit)
            elite_indices = order[:n_elite]
            worst_indices = order[-n_elite:]
            for k, idx in enumerate(worst_indices):
                if idx not in elite_indices:
                    pop[idx] = pop[elite_indices[k % n_elite]].copy()
                    fit[idx] = fit[elite_indices[k % n_elite]]
        # 重新更新全局最优
        best_idx = np.argmin(fit)
        if fit[best_idx] < best_fit:
            best_fit = fit[best_idx]
            best_pos = pop[best_idx].copy()
        history_best.append(best_fit)
    return best_pos, best_fit, np.array(history_best)
