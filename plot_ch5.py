# -*- coding: utf-8 -*-
"""
第五章 图表绘制
5.2.1 适应度收敛曲线对比（IBA-ELM vs PSO-ELM vs ELM）
5.2.2 消融实验（IBA full vs no chaos vs no elite）
5.3.1 预测精度对比（R2、RMSE、MAE 柱状图）
5.3.2 冬小麦不同生育期预测误差特征
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import FIG_DIR

# 中文字体：优先 macOS 常见字体，再 Windows SimHei，保证生育期等中文标签正常显示
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def fig_5_2_1_convergence():
    """5.2.1 适应度收敛曲线对比：IBA-ELM vs PSO-ELM vs ELM"""
    path = os.path.join(FIG_DIR, "ch5_convergence_compare.csv")
    if not os.path.isfile(path):
        print("Run run_ch5.py first to generate ch5_convergence_compare.csv")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for name in ["IBA-ELM", "PSO-ELM", "ELM"]:
        d = df[df["model"] == name]
        if name == "ELM":
            ax.axhline(y=d["fitness"].iloc[0], color="C2", linestyle="--", label=name)
        else:
            ax.plot(d["gen"], d["fitness"], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("K-fold CV RMSE (fitness)")
    #ax.set_title("5.2.1 Convergence: IBA-ELM vs PSO-ELM vs ELM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_5_2_1_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_5_2_1_convergence")


def fig_5_2_2_ablation():
    """5.2.2 消融实验：混沌映射与精英策略的有效性"""
    path = os.path.join(FIG_DIR, "ch5_ablation.csv")
    if not os.path.isfile(path):
        print("Run run_ch5.py first to generate ch5_ablation.csv")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for name in ["IBA(full)", "IBA(no chaos)", "IBA(no elite)"]:
        d = df[df["model"] == name]
        ax.plot(d["gen"], d["fitness"], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("K-fold CV RMSE (fitness)")
    #ax.set_title("5.2.2 Ablation: chaos & elite strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_5_2_2_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_5_2_2_ablation")


def fig_5_3_1_metrics():
    """5.3.1 预测精度对比：IBA-ELM vs BA-ELM vs ELM vs MLP（R2、RMSE、MAE）"""
    path = os.path.join(FIG_DIR, "ch5_metrics_compare.csv")
    if not os.path.isfile(path):
        print("Run run_ch5.py first to generate ch5_metrics_compare.csv")
        return
    df = pd.read_csv(path)
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    # R2 越大越好
    axes[0].bar(x - width, df["R2"], width, label="R2", color="C0")
    axes[0].set_ylabel("R2")
    #axes[0].set_title("R2 (higher better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")
    # RMSE 越小越好
    axes[1].bar(x, df["RMSE"], width, label="RMSE", color="C1")
    axes[1].set_ylabel("RMSE")
    #axes[1].set_title("RMSE (lower better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    # MAE 越小越好
    axes[2].bar(x + width, df["MAE"], width, label="MAE", color="C2")
    axes[2].set_ylabel("MAE")
    #axes[2].set_title("MAE (lower better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15, ha="right")
    axes[2].grid(True, alpha=0.3, axis="y")
    #fig.suptitle("5.3.1 Prediction accuracy comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_5_3_1_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_5_3_1_metrics")


def fig_5_3_2_stage_errors():
    """5.3.2 冬小麦不同生育期的预测误差特征（各模型在各生育期的 RMSE）"""
    path = os.path.join(FIG_DIR, "ch5_stage_errors.csv")
    if not os.path.isfile(path):
        print("Run run_ch5.py first to generate ch5_stage_errors.csv")
        return
    df = pd.read_csv(path)
    stages = df["stage"].unique().tolist()
    models = df["model"].unique().tolist()
    # 按生育期做分组柱状图：每个 stage 一组，组内各 model 的 RMSE
    x = np.arange(len(stages))
    width = 0.15
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(stages) * 1.2), 4))
    for i, model in enumerate(models):
        vals = [df[(df["model"] == model) & (df["stage"] == s)]["RMSE"].values[0] if len(df[(df["model"] == model) & (df["stage"] == s)]) else 0 for s in stages]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Growth stage")
    #ax.set_title("5.3.2 Prediction error by growth stage")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # 底部留足空间，避免中文下标被裁切
    plt.savefig(os.path.join(FIG_DIR, "fig_5_3_2_stage_errors.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_5_3_2_stage_errors")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_5_2_1_convergence()
    fig_5_2_2_ablation()
    fig_5_3_1_metrics()
    fig_5_3_2_stage_errors()
