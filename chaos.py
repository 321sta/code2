# -*- coding: utf-8 -*-
"""
4.2.1 混沌映射：Tent 与 Logistic，用于优化初始种群多样性
"""
import numpy as np


def tent_map(x, mu=2.0):
    """Tent 混沌映射: x_{n+1} = mu * min(x_n, 1-x_n), 通常 mu=2"""
    return mu * np.minimum(x, 1.0 - x)


def logistic_map(x, mu=4.0):
    """Logistic 混沌映射: x_{n+1} = mu * x_n * (1 - x_n), 通常 mu=4"""
    return mu * x * (1.0 - x)


def chaos_sequence(map_type, length, x0=0.1, mu=None):
    """生成一维混沌序列。map_type: 'tent' 或 'logistic'"""
    if mu is None:
        mu = 2.0 if map_type == "tent" else 4.0
    seq = np.zeros(length)
    seq[0] = x0
    f = tent_map if map_type == "tent" else logistic_map
    for i in range(1, length):
        seq[i] = f(seq[i - 1], mu)
        # 避免不动点
        if abs(seq[i] - seq[i - 1]) < 1e-10:
            seq[i] = (seq[i] + 0.1) % 1.0
    return seq


def chaos_init_population(n_pop, dim, lb, ub, map_type="tent", seed=None):
    """
    用混沌序列生成初始种群，使个体在解空间分布更均匀、多样性更好。
    对应 4.2.1 引入混沌映射优化初始种群多样性。
    """
    if seed is not None:
        np.random.seed(seed)
    total = n_pop * dim
    seq = chaos_sequence(map_type, total)
    pop = lb + (ub - lb) * seq.reshape(n_pop, dim)
    return pop
