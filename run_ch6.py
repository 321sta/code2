# -*- coding: utf-8 -*-
"""
第六章 气象数据缺失情景下的冬小麦 ETc 预测
6.1 情景设计（data_loader_ch6）
6.2 不同情景下 IBA-ELM 预测表现
6.3 IBA-ELM vs Hargreaves-Samani（仅温度情景）
6.4 气象因子对 ETc 的敏感性/相关性分析
"""
import os
import numpy as np
import pandas as pd
from config import FIG_DIR, N_HIDDEN, K_FOLD, N_POP
from data_loader_ch6 import get_merged_data_with_doy, get_X_y_scenario
from metrics import compute_metrics, rmse, mae, r2_score
from iba_relm import iba_relm_fit as iba_elm_fit
from ba_elm import ba_elm_fit
from standard_elm import standard_elm_fit
from models_baseline import fit_mlp
from hargreaves_samani import etc_hargreaves

N_GEN_DEMO = min(30, 50)


def run_scenario_metrics():
    """6.2 情景 I/II/III 下 IBA-ELM 的 R2、RMSE、MAE"""
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    models = {}
    for scenario in ["I", "II", "III"]:
        X, y, scaler, cols = get_X_y_scenario(scenario)
        y_flat = y.ravel()
        res = iba_elm_fit(
            X, y,
            n_hidden=N_HIDDEN,
            k_fold=K_FOLD,
            n_pop=N_POP,
            n_gen=N_GEN_DEMO,
            seed=42,
        )
        models[scenario] = res["elm"]
        pred = res["elm"].predict(X).ravel()
        m = compute_metrics(y_flat, pred)
        rows.append({
            "scenario": scenario,
            "features": ",".join(cols),
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(FIG_DIR, "ch6_scenario_metrics.csv"), index=False)
    print("Saved ch6_scenario_metrics.csv")
    return models


def run_scenario_metrics_baseline():
    """6.2 情景 I/II/III 下 BA-ELM / ELM / MLP 的 R2、RMSE、MAE"""
    os.makedirs(FIG_DIR, exist_ok=True)
    model_names = ["ba_elm", "elm", "mlp"]
    results = {name: [] for name in model_names}

    for scenario in ["I", "II", "III"]:
        X, y, scaler, cols = get_X_y_scenario(scenario)
        y_flat = y.ravel()

        # BA-ELM
        res = ba_elm_fit(X, y, n_hidden=N_HIDDEN, k_fold=K_FOLD, n_pop=N_POP, n_gen=N_GEN_DEMO, seed=42)
        pred = res["elm"].predict(X).ravel()
        m = compute_metrics(y_flat, pred)
        results["ba_elm"].append({"scenario": scenario, "features": ",".join(cols),
                                   "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]})

        # Standard ELM
        res = standard_elm_fit(X, y, n_hidden=N_HIDDEN, seed=42)
        pred = res["elm"].predict(X).ravel()
        m = compute_metrics(y_flat, pred)
        results["elm"].append({"scenario": scenario, "features": ",".join(cols),
                                "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]})

        # MLP
        mlp = fit_mlp(X, y, hidden_layer_sizes=(N_HIDDEN,), max_iter=500)
        pred = mlp.predict(X)
        m = compute_metrics(y_flat, pred)
        results["mlp"].append({"scenario": scenario, "features": ",".join(cols),
                                "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"]})

    for name, rows in results.items():
        csv_path = os.path.join(FIG_DIR, f"ch6_scenario_metrics_{name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved ch6_scenario_metrics_{name}.csv")


def run_hargreaves_compare():
    """6.3 仅温度情景下 IBA-ELM vs Hargreaves-Samani (ETc = kc * ET0_HS)"""
    df = get_merged_data_with_doy()
    X, y, scaler, cols = get_X_y_scenario("III")
    y_flat = y.ravel()
    # IBA-ELM 预测（标准化空间）
    res = iba_elm_fit(X, y, n_hidden=N_HIDDEN, k_fold=K_FOLD, n_pop=N_POP, n_gen=N_GEN_DEMO, seed=42)
    pred_elm = res["elm"].predict(X).ravel()
    # 反标准化到原始尺度，便于与 H-S 对比
    _, _, y_mean, y_std = scaler
    pred_elm_orig = pred_elm * y_std + y_mean
    y_orig = y_flat * y_std + y_mean
    # H-S ETc
    etc_hs = etc_hargreaves(df)
    m_elm = compute_metrics(y_orig, pred_elm_orig)
    m_hs = compute_metrics(y_orig, etc_hs)
    compare = pd.DataFrame([
        {"model": "IBA-ELM", "R2": m_elm["R2"], "RMSE": m_elm["RMSE"], "MAE": m_elm["MAE"]},
        {"model": "Hargreaves-Samani", "R2": m_hs["R2"], "RMSE": m_hs["RMSE"], "MAE": m_hs["MAE"]},
    ])
    compare.to_csv(os.path.join(FIG_DIR, "ch6_iba_vs_hargreaves.csv"), index=False)
    # 保存逐日预测值供作图
    pd.DataFrame({
        "y_true": y_orig,
        "IBA_ELM": pred_elm_orig,
        "Hargreaves_Samani": etc_hs,
    }).to_csv(os.path.join(FIG_DIR, "ch6_iba_vs_hs_daily.csv"), index=False)
    print("Saved ch6_iba_vs_hargreaves.csv, ch6_iba_vs_hs_daily.csv")
    return compare


def run_sensitivity_correlation():
    """6.4 气象因子与 ETc 的相关性；基于情景 I 的 IBA-ELM 敏感性（扰动 +1 std）"""
    df = get_merged_data_with_doy()
    # 相关性：原始尺度下各气象因子与 ETc 的 Pearson 相关
    feat_cols = ["T_mean", "RH_mean", "P_mean", "u_2_mean", "Rs_mean", "kc", "DOY", "T_max", "T_min"]
    avail = [c for c in feat_cols if c in df.columns]
    corr_rows = []
    y_etc = df["etc"].astype(float).values
    for c in avail:
        x = df[c].astype(float).values
        r = np.corrcoef(x, y_etc)[0, 1]
        if np.isnan(r):
            r = 0
        corr_rows.append({"factor": c, "correlation_with_ETc": r})
    pd.DataFrame(corr_rows).to_csv(os.path.join(FIG_DIR, "ch6_correlation.csv"), index=False)

    # 敏感性：情景 I 训练 IBA-ELM，对每个输入 +1 标准差扰动，看输出变化
    X, y, scaler, cols = get_X_y_scenario("I")
    X_mean, X_std = scaler[0], scaler[1]
    res = iba_elm_fit(X, y, n_hidden=N_HIDDEN, k_fold=K_FOLD, n_pop=N_POP, n_gen=N_GEN_DEMO, seed=42)
    elm = res["elm"]
    pred_base = elm.predict(X).ravel()
    sens_rows = []
    for j in range(X.shape[1]):
        X_pert = X.copy()
        X_pert[:, j] = X_pert[:, j] + 1.0  # +1 std in standardized space
        pred_pert = elm.predict(X_pert).ravel()
        delta = np.mean(np.abs(pred_pert - pred_base))
        sens_rows.append({"factor": cols[j], "sensitivity_mean_abs_delta": delta})
    pd.DataFrame(sens_rows).to_csv(os.path.join(FIG_DIR, "ch6_sensitivity.csv"), index=False)
    print("Saved ch6_correlation.csv, ch6_sensitivity.csv")


def main():
    run_scenario_metrics()
    run_scenario_metrics_baseline()
    run_hargreaves_compare()
    run_sensitivity_correlation()
    print("Chapter 6 experiments done. Run plot_ch6.py to generate figures.")


if __name__ == "__main__":
    main()
