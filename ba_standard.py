# -*- coding: utf-8 -*-
"""
标准蝙蝠算法 (Standard Bat Algorithm, BA)
用于 4.1 节：分析易陷入局部最优、后期收敛慢等缺陷
"""
import numpy as np


def standard_ba(
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
    seed=None,
):
    """
    标准 BA：通过频率、速度、响度与脉冲发射率更新位置。
    返回: best_pos, best_fit, history_best_fit (用于画收敛曲线，体现后期收敛慢)
    """
    if seed is not None:
        np.random.seed(seed)
    # 随机初始化种群（无混沌，多样性依赖随机）
    pop = np.random.uniform(lb, ub, (n_pop, dim))
    vel = np.zeros_like(pop)
    fit = np.array([fitness_func(pop[i]) for i in range(n_pop)])
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    A = np.full(n_pop, A0)
    r = np.full(n_pop, r0)
    history_best = [best_fit]

    for g in range(n_gen - 1):
        for i in range(n_pop):
            # 频率
            fi = f_min + (f_max - f_min) * np.random.rand()
            # 速度更新（无惯性权重，易导致后期步长单一、收敛慢）
            vel[i] = vel[i] + (pop[i] - best_pos) * fi
            new_pos = pop[i] + vel[i]
            # 边界
            new_pos = np.clip(new_pos, lb, ub)
            # 局部搜索：若随机数 > r，则围绕当前最优做局部扰动（易陷入局部最优）
            if np.random.rand() > r[i]:
                new_pos = best_pos + 0.001 * np.random.randn(dim)
                new_pos = np.clip(new_pos, lb, ub)
            new_fit = fitness_func(new_pos)
            # 接受准则：更优且响度大于随机数
            if new_fit < fit[i] and np.random.rand() < A[i]:
                pop[i] = new_pos
                fit[i] = new_fit
                A[i] = alpha * A[i]
                r[i] = r0 * (1 - np.exp(-gamma * (g + 1)))
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = new_pos.copy()
        history_best.append(best_fit)
    return best_pos, best_fit, np.array(history_best)
