# -*- coding: utf-8 -*-
"""
运行 IBA-ELM：加载两期冬小麦数据，训练 IBA-ELM 模型，保存收敛曲线与预测结果供作图使用。
"""
import os
import numpy as np
import pandas as pd
from data_loader import get_X_y
from iba_relm import iba_relm_fit as iba_elm_fit
from metrics import compute_metrics
from config import FIG_DIR, N_POP, N_GEN, N_HIDDEN, K_FOLD

def main():
    X, y, scaler, feat_names = get_X_y()
    n_in = X.shape[1]
    print("Features:", feat_names, "n_in:", n_in, "samples:", len(X))

    # IBA-ELM 训练（与互动系统保持一致：n_pop=50, n_gen=100, n_hidden=25, 多起点×5）
    best_r2, best_res = -np.inf, None
    for seed in [42, 123, 456, 789, 1024]:
        res = iba_elm_fit(
            X, y,
            n_hidden=25,
            k_fold=K_FOLD,
            n_pop=50,
            n_gen=100,
            seed=seed,
        )
        pred = res["elm"].predict(X).ravel()
        r2 = compute_metrics(y.ravel(), pred)["R2"]
        if r2 > best_r2:
            best_r2, best_res = r2, res
    res = best_res
    elm = res["elm"]
    history = res["history"]
    best_rmse = res["best_rmse"]

    # 全量预测
    y_pred = elm.predict(X)

    # 保存收敛曲线与预测结果
    os.makedirs(FIG_DIR, exist_ok=True)
    pd.DataFrame({"gen": np.arange(len(history)), "best_rmse": history}).to_csv(
        os.path.join(FIG_DIR, "iba_elm_convergence.csv"), index=False
    )
    pd.DataFrame({
        "y_true": y.ravel(),
        "y_pred": y_pred.ravel(),
    }).to_csv(os.path.join(FIG_DIR, "iba_elm_predictions.csv"), index=False)
    print("Best K-fold CV RMSE:", best_rmse)
    print("Convergence and predictions saved to", FIG_DIR)

if __name__ == "__main__":
    main()
