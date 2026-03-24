# -*- coding: utf-8 -*-
"""
数据部分图表（用于论文/报告“数据”章节）
1. 气象数据的统计特征图：温度、湿度、风速等随时间变化的折线图，2023-2024 与 2024-2025 两期并列/对比展示
2. 目标基准值（ETc）的变化趋势图：结合冬小麦生育期在图上标注（返青期、拔节期等）
3. 数据相关性热力图：输入（气象因子）与输出（ETc）的相关系数热力图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from config import FIG_DIR, FEATURE_COLS, TARGET_COL
from data_loader import get_merged_data, get_merged_data_with_stage

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 气象因子中文名（用于折线图图例与热力图）
FEATURE_LABELS = {
    "T_mean": "日均温(℃)",
    "RH_mean": "相对湿度(%)",
    "P_mean": "气压(hPa)",
    "u_2_mean": "风速(m/s)",
    "Rs_mean": "辐射(MJ/m²)",
    "kc": "作物系数",
    "etc": "ETc(mm)",
}


def fig_data_1_weather_timeseries():
    """
    1. 气象数据的统计特征图：温度、湿度、风速等随时间变化的折线图。
    2023-2024 与 2024-2025 两期并列展示（左一列 2023-2024，右一列 2024-2025）。
    """
    df = get_merged_data()
    df["date"] = pd.to_datetime(df["date"])
    # 选主要气象因子：温度、湿度、风速、气压、辐射（不含 kc，kc 更偏农学）
    weather_cols = ["T_mean", "RH_mean", "u_2_mean", "P_mean", "Rs_mean"]
    avail = [c for c in weather_cols if c in df.columns]
    if not avail:
        avail = [c for c in FEATURE_COLS if c in df.columns and c != "kc"]
    seasons = ["2023-2024", "2024-2025"]
    n_var = len(avail)
    fig, axes = plt.subplots(n_var, 2, figsize=(11, 2.2 * n_var), sharex="col")
    if n_var == 1:
        axes = axes.reshape(1, -1)
    for col_idx, season in enumerate(seasons):
        d = df[df["season"] == season].sort_values("date").copy()
        d = d.reset_index(drop=True)
        x = np.arange(len(d))
        for row_idx, var in enumerate(avail):
            ax = axes[row_idx, col_idx]
            ax.plot(x, d[var].values, color="C0", lw=1, alpha=0.9)
            ax.set_ylabel(FEATURE_LABELS.get(var, var))
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(season)
            if row_idx == n_var - 1:
                ax.set_xlabel("日期")
    # 左列用日期作为 x 更直观（可选：用日期标签）
    for col_idx, season in enumerate(seasons):
        d = df[df["season"] == season].sort_values("date")
        axes[0, col_idx].set_title(season)
        if len(d) > 0:
            ticks = np.linspace(0, len(d) - 1, min(6, len(d)), dtype=int)
            axes[n_var - 1, col_idx].set_xticks(ticks)
            axes[n_var - 1, col_idx].set_xticklabels(
                d["date"].iloc[ticks].dt.strftime("%m-%d").tolist(), rotation=25
            )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_data_1_weather_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_data_1_weather_timeseries")


def fig_data_2_etc_with_stages():
    """
    2. 目标基准值（ETc）的变化趋势图，结合冬小麦生育期在图上标注；
    每年一张图：2023-2024、2024-2025 分别输出。
    """
    df = get_merged_data_with_stage()
    df["date"] = pd.to_datetime(df["date"])
    # 统一生育期颜色映射（两期一致）
    all_stages = df["stage"].unique().tolist()
    stage_to_color = {s: plt.cm.Set3(i / max(1, len(all_stages))) for i, s in enumerate(all_stages)}

    for season in ["2023-2024", "2024-2025"]:
        d = df[df["season"] == season].sort_values("date").reset_index(drop=True)
        if d.empty:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(d["date"].values, d["etc"].values, color="navy", lw=1.2, alpha=0.9, label="ETc")
        y_max = d["etc"].max()
        i = 0
        while i < len(d):
            s = d["stage"].iloc[i]
            j = i
            while j < len(d) and d["stage"].iloc[j] == s:
                j += 1
            ax.axvspan(d["date"].iloc[i], d["date"].iloc[j - 1], alpha=0.25, color=stage_to_color[s])
            mid_idx = (i + j - 1) // 2
            ax.text(d["date"].iloc[mid_idx], y_max * 0.98, s, ha="center", va="top", fontsize=8, rotation=25)
            i = j
        ax.set_xlabel("日期")
        ax.set_ylabel("ETc (mm)")
        ax.set_title(season)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        fname = "fig_data_2_etc_with_stages_{}.png".format(season.replace("-", "_"))
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved", fname)


def fig_data_3_correlation_heatmap():
    """
    3. 数据相关性热力图：输入（气象因子）与输出（ETc）的相关系数矩阵热力图。
    """
    df = get_merged_data()
    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].astype(float).corr()
    labels = [FEATURE_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    vmin, vmax = -1, 1
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(corr.values, cmap="RdBu_r", aspect="auto", norm=norm)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax, label="相关系数")
    ax.set_title("气象因子与 ETc 相关系数热力图")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_data_3_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_data_3_correlation_heatmap")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_data_1_weather_timeseries()
    fig_data_2_etc_with_stages()
    fig_data_3_correlation_heatmap()
    print("Data figures done.")
