# -*- coding: utf-8 -*-
"""
第四章 图表绘制脚本
4.1 标准 BA 缺陷（收敛曲线：易陷入局部最优、后期收敛慢）
4.2.1 混沌映射序列（Tent/Logistic）
4.2.2 自适应惯性权重曲线
4.3 IBA-ELM 收敛曲线与预测效果
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import FIG_DIR, W_MAX, W_MIN
from chaos import chaos_sequence, chaos_init_population
from ba_standard import standard_ba
from iba import iba, adaptive_weight

# 使用可显示中文的字体（若系统无则回退默认）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def sphere(x):
    """简单球函数，用于对比 BA 与 IBA 收敛"""
    return np.sum(x ** 2)


def rastrigin(x):
    """Rastrigin 多峰函数，易陷入局部最优"""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def fig_4_1_ba_defects():
    """4.1 标准蝙蝠算法缺陷：在同一测试函数上对比 BA 与 IBA 收敛曲线（易陷入局部最优、后期收敛慢）"""
    dim, n_pop, n_gen = 10, 30, 80
    lb, ub = -5.0, 5.0
    runs = 5
    ba_curves = []
    iba_curves = []
    for seed in range(runs):
        _, _, h_ba = standard_ba(sphere, dim, n_pop=n_pop, n_gen=n_gen, lb=lb, ub=ub, seed=seed)
        _, _, h_iba = iba(sphere, dim, n_pop=n_pop, n_gen=n_gen, lb=lb, ub=ub, seed=seed)
        ba_curves.append(h_ba)
        iba_curves.append(h_iba)
    ba_mean = np.mean(ba_curves, axis=0)
    ba_std = np.std(ba_curves, axis=0)
    iba_mean = np.mean(iba_curves, axis=0)
    iba_std = np.std(iba_curves, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ba_mean, label="BA (standard)", color="C0")
    ax.fill_between(np.arange(len(ba_mean)), ba_mean - ba_std, ba_mean + ba_std, alpha=0.3, color="C0")
    ax.plot(iba_mean, label="IBA (improved)", color="C1")
    ax.fill_between(np.arange(len(iba_mean)), iba_mean - iba_std, iba_mean + iba_std, alpha=0.3, color="C1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    #ax.set_title("4.1 BA vs IBA convergence (sphere function)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_4_1_ba_defects.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_4_1_ba_defects")


def fig_4_2_1_chaos():
    """4.2.1 混沌映射：Tent 与 Logistic 序列对比，体现初始种群多样性优化"""
    n = 200
    tent_seq = chaos_sequence("tent", n)
    log_seq = chaos_sequence("logistic", n)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))
    axes[0].plot(tent_seq, color="C0", label="Tent")
    axes[0].set_ylabel("Tent map")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(log_seq, color="C1", label="Logistic")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Logistic map")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    #fig.suptitle("4.2.1 Chaos sequences for initial population diversity")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_4_2_1_chaos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_4_2_1_chaos")


def fig_4_2_2_adaptive_weight():
    """4.2.2 自适应惯性权重：随迭代从 w_max 线性降至 w_min"""
    max_gen = 50
    gens = np.arange(max_gen + 1)
    w = [adaptive_weight(g, max_gen, W_MAX, W_MIN) for g in gens]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(gens, w, color="C0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia weight w")
    #ax.set_title("4.2.2 Adaptive inertia weight (global → local)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_4_2_2_adaptive_weight.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_4_2_2_adaptive_weight")


def fig_4_3_convergence_and_fit():
    """4.3 IBA-ELM 收敛曲线与预测值 vs 实测值（需先运行 run_iba_elm.py）"""
    conv_path = os.path.join(FIG_DIR, "iba_elm_convergence.csv")
    pred_path = os.path.join(FIG_DIR, "iba_elm_predictions.csv")
    if not os.path.isfile(conv_path) or not os.path.isfile(pred_path):
        print("Run run_iba_elm.py first to generate iba_elm_convergence.csv and iba_elm_predictions.csv")
        return
    import pandas as pd
    conv = pd.read_csv(conv_path)
    pred = pd.read_csv(pred_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(conv["gen"], conv["best_rmse"], color="C0")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("K-fold CV RMSE")
    #axes[0].set_title("4.3 IBA-ELM convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(pred["y_true"], pred["y_pred"], alpha=0.6, s=20)
    lims = [min(pred["y_true"].min(), pred["y_pred"].min()), max(pred["y_true"].max(), pred["y_pred"].max())]
    axes[1].plot(lims, lims, "k--", label="y=x")
    axes[1].set_xlabel("Observed (standardized)")
    axes[1].set_ylabel("Predicted (standardized)")
    #axes[1].set_title("4.3 IBA-ELM fit (ETc)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_4_3_iba_elm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_4_3_iba_elm")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_4_1_ba_defects()
    fig_4_2_1_chaos()
    fig_4_2_2_adaptive_weight()
    fig_4_3_convergence_and_fit()
