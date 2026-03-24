# -*- coding: utf-8 -*-
"""
第六章 图表绘制
6.2 不同情景下 IBA-ELM 预测表现（R2/RMSE/MAE）
6.3 IBA-ELM vs Hargreaves-Samani 对比
6.4 气象因子与 ETc 相关性及敏感性
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import FIG_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def fig_6_2_scenario_metrics():
    """6.2 情景 I/II/III 下 IBA-ELM 的 R2、RMSE、MAE 对比"""
    path = os.path.join(FIG_DIR, "ch6_scenario_metrics.csv")
    if not os.path.isfile(path):
        print("Run run_ch6.py first to generate ch6_scenario_metrics.csv")
        return
    df = pd.read_csv(path)
    labels = ["Scenario I\n(full)", "Scenario II\n(no radiation)", "Scenario III\n(temp only)"]
    x = np.arange(3)
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    axes[0].bar(x - width, df["R2"], width, color="C0", label="R2")
    axes[0].set_ylabel("R2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    #axes[0].set_title("R2 (higher better)")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(x, df["RMSE"], width, color="C1", label="RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    #axes[1].set_title("RMSE (lower better)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[2].bar(x + width, df["MAE"], width, color="C2", label="MAE")
    axes[2].set_ylabel("MAE")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    #axes[2].set_title("MAE (lower better)")
    axes[2].grid(True, alpha=0.3, axis="y")
    #fig.suptitle("6.2 IBA-ELM performance under different input scenarios")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_6_2_scenario_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_6_2_scenario_metrics")


def _plot_scenario_metrics_for_model(csv_name, fig_name):
    """通用：读取 ch6_scenario_metrics_xxx.csv 并绘制 R2/RMSE/MAE 三子图。"""
    path = os.path.join(FIG_DIR, csv_name)
    if not os.path.isfile(path):
        print(f"Run run_ch6.py first to generate {csv_name}")
        return
    df = pd.read_csv(path)
    labels = ["Scenario I\n(full)", "Scenario II\n(no radiation)", "Scenario III\n(temp only)"]
    x = np.arange(3)
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    axes[0].bar(x - width, df["R2"], width, color="C0")
    axes[0].set_ylabel("R2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(x, df["RMSE"], width, color="C1")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[2].bar(x + width, df["MAE"], width, color="C2")
    axes[2].set_ylabel("MAE")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_name}")


def fig_6_2_ba_elm_scenario_metrics():
    """6.2 情景 I/II/III 下 BA-ELM 的 R2、RMSE、MAE 对比"""
    _plot_scenario_metrics_for_model("ch6_scenario_metrics_ba_elm.csv",
                                     "fig_6_2_ba_elm_scenario_metrics.png")


def fig_6_2_elm_scenario_metrics():
    """6.2 情景 I/II/III 下 ELM 的 R2、RMSE、MAE 对比"""
    _plot_scenario_metrics_for_model("ch6_scenario_metrics_elm.csv",
                                     "fig_6_2_elm_scenario_metrics.png")


def fig_6_2_mlp_scenario_metrics():
    """6.2 情景 I/II/III 下 MLP 的 R2、RMSE、MAE 对比"""
    _plot_scenario_metrics_for_model("ch6_scenario_metrics_mlp.csv",
                                     "fig_6_2_mlp_scenario_metrics.png")


def fig_6_3_iba_vs_hs():
    """6.3 仅温度情景下 IBA-ELM vs Hargreaves-Samani：实测 vs 预测散点/柱状对比"""
    path_metrics = os.path.join(FIG_DIR, "ch6_iba_vs_hargreaves.csv")
    path_daily = os.path.join(FIG_DIR, "ch6_iba_vs_hs_daily.csv")
    if not os.path.isfile(path_metrics) or not os.path.isfile(path_daily):
        print("Run run_ch6.py first to generate ch6_iba_vs_hargreaves.csv and ch6_iba_vs_hs_daily.csv")
        return
    df_m = pd.read_csv(path_metrics)
    df_d = pd.read_csv(path_daily)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 左：R2/RMSE/MAE 柱状对比
    x = np.arange(2)
    width = 0.25
    axes[0].bar(x - width, df_m["R2"], width, label="R2", color="C0")
    axes[0].bar(x, df_m["RMSE"], width, label="RMSE", color="C1")
    axes[0].bar(x + width, df_m["MAE"], width, label="MAE", color="C2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_m["model"].tolist())
    axes[0].set_ylabel("Value")
    #axes[0].set_title("6.3 IBA-ELM vs Hargreaves-Samani (temp-only)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    # 右：实测 vs IBA-ELM vs H-S 散点（取前 200 点避免过密）
    n_show = min(200, len(df_d))
    axes[1].scatter(df_d["y_true"].iloc[:n_show], df_d["IBA_ELM"].iloc[:n_show], alpha=0.6, s=15, label="IBA-ELM")
    axes[1].scatter(df_d["y_true"].iloc[:n_show], df_d["Hargreaves_Samani"].iloc[:n_show], alpha=0.6, s=15, label="H-S")
    lims = [df_d["y_true"].min(), df_d["y_true"].max()]
    axes[1].plot(lims, lims, "k--", lw=1, label="y=x")
    axes[1].set_xlabel("Observed ETc")
    axes[1].set_ylabel("Predicted ETc")
    #axes[1].set_title("Predicted vs observed (temp-only)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_6_3_iba_vs_hs.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_6_3_iba_vs_hs")


def fig_6_4_correlation_sensitivity():
    """6.4 气象因子与 ETc 的相关性；敏感性（+1 std 输出变化）"""
    path_corr = os.path.join(FIG_DIR, "ch6_correlation.csv")
    path_sens = os.path.join(FIG_DIR, "ch6_sensitivity.csv")
    if not os.path.isfile(path_corr):
        print("Run run_ch6.py first to generate ch6_correlation.csv")
        return
    df_c = pd.read_csv(path_corr)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 相关性
    axes[0].barh(df_c["factor"], df_c["correlation_with_ETc"], color="C0", alpha=0.8)
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Correlation with ETc")
    #axes[0].set_title("6.4(a) Correlation of factors with ETc")
    axes[0].grid(True, alpha=0.3, axis="x")
    if os.path.isfile(path_sens):
        df_s = pd.read_csv(path_sens)
        axes[1].barh(df_s["factor"], df_s["sensitivity_mean_abs_delta"], color="C1", alpha=0.8)
        axes[1].set_xlabel("Mean |delta output| (+1 std input)")
        #axes[1].set_title("6.4(b) Sensitivity (IBA-ELM, Scenario I)")
        axes[1].grid(True, alpha=0.3, axis="x")
    else:
        axes[1].set_visible(False)
    #fig.suptitle("6.4 Meteorological factor: correlation & sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_6_4_correlation_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_6_4_correlation_sensitivity")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_6_2_scenario_metrics()
    fig_6_2_ba_elm_scenario_metrics()
    fig_6_2_elm_scenario_metrics()
    fig_6_2_mlp_scenario_metrics()
    fig_6_3_iba_vs_hs()
    fig_6_4_correlation_sensitivity()
