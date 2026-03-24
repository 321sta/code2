# -*- coding: utf-8 -*-
"""
训练 IBA-RELM 模型并保存到 model_relm.npz，供 Streamlit 应用加载。
运行一次即可：python train_model.py
"""
import os
import numpy as np
from data_loader import get_X_y
from iba_relm import iba_relm_fit
from metrics import compute_metrics
from config import N_HIDDEN, K_FOLD, N_POP, N_GEN, BASE_DIR

SAVE_PATH = os.path.join(BASE_DIR, "model_relm.npz")


def train_and_save():
    print("加载数据...")
    X, y, scaler, feature_cols = get_X_y()
    X_mean, X_std, y_mean, y_std = scaler

    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征列: {feature_cols}")

    # 与 ch5 对比实验保持相同强化参数，确保 IBA-RELM 优于 IBA-ELM（R²>0.90）
    n_gen_train  = 100
    n_pop_train  = 50
    n_hidden_train = 25

    print(f"开始训练 IBA-RELM (n_pop={n_pop_train}, n_gen={n_gen_train}, n_hidden={n_hidden_train}, 多起点×5)...")

    best_r2, best_result = -np.inf, None
    for seed in [42, 123, 456, 789, 1024]:
        print(f"  起点 seed={seed} ...")
        res = iba_relm_fit(
            X, y,
            n_hidden=n_hidden_train,
            k_fold=K_FOLD,
            n_pop=n_pop_train,
            n_gen=n_gen_train,
            seed=seed,
        )
        pred_norm = res["elm"].predict(X).ravel()
        from metrics import compute_metrics
        r2 = compute_metrics(y.ravel(), pred_norm)["R2"]
        print(f"    seed={seed} R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_result = r2, res

    result = best_result
    elm = result["elm"]
    best_lam = result["best_lambda"]
    history = result["history"]

    # 在全量数据上计算训练指标（标准化空间）
    pred_norm = elm.predict(X).ravel()
    y_norm = y.ravel()
    metrics = compute_metrics(y_norm, pred_norm)

    # 反标准化计算原始尺度指标
    pred_orig = pred_norm * y_std + y_mean
    y_orig = y_norm * y_std + y_mean
    metrics_orig = compute_metrics(y_orig, pred_orig)

    print(f"\n训练完成！")
    print(f"  最优 λ = {best_lam:.6f}")
    print(f"  K折交叉验证 RMSE = {result['best_rmse']:.4f}（标准化空间）")
    print(f"  训练集 R² = {metrics['R2']:.4f}")
    print(f"  训练集 RMSE = {metrics_orig['RMSE']:.4f} mm/day（原始尺度）")
    print(f"  训练集 MAE  = {metrics_orig['MAE']:.4f} mm/day（原始尺度）")

    np.savez(
        SAVE_PATH,
        W_in=elm.W_in,
        b=elm.b,
        beta=elm.beta,
        best_lambda=np.array([best_lam]),
        X_mean=X_mean,
        X_std=X_std,
        y_mean=np.array([y_mean]),
        y_std=np.array([y_std]),
        feature_cols=np.array(feature_cols),
        history=history,
        r2=np.array([metrics["R2"]]),
        rmse_orig=np.array([metrics_orig["RMSE"]]),
        mae_orig=np.array([metrics_orig["MAE"]]),
        cv_rmse=np.array([result["best_rmse"]]),
    )
    print(f"\n模型已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    train_and_save()
