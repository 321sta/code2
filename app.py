# -*- coding: utf-8 -*-
"""
冬小麦智能灌溉需水预测系统
基于 IBA-ELM（改进蝙蝠算法优化正则化极限学习机）

运行方式：
    1. 先训练模型（只需一次）：python train_model.py
    2. 启动应用：              streamlit run app.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))

from elm import ELM
from config import BASE_DIR, FEATURE_COLS

MODEL_PATH = os.path.join(BASE_DIR, "model_relm.npz")

# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    """加载预训练 IBA-ELM 模型（需先运行 train_model.py）"""
    if not os.path.exists(MODEL_PATH):
        return None
    data = np.load(MODEL_PATH, allow_pickle=True)
    n_in  = data["W_in"].shape[0]
    n_hidden = data["W_in"].shape[1]
    elm = ELM(n_in, n_hidden, 1)
    elm.W_in = data["W_in"]
    elm.b    = data["b"]
    elm.beta = data["beta"]
    meta = {
        "X_mean":      data["X_mean"],
        "X_std":       data["X_std"],
        "y_mean":      float(data["y_mean"]),
        "y_std":       float(data["y_std"]),
        "feature_cols": list(data["feature_cols"]),
        "best_lambda": float(data["best_lambda"]),
        "r2":          float(data["r2"]),
        "rmse_orig":   float(data["rmse_orig"]),
        "mae_orig":    float(data["mae_orig"]),
        "cv_rmse":     float(data["cv_rmse"]),
        "history":     data["history"],
    }
    return elm, meta


def predict_etc(elm, meta, feature_values: dict) -> float:
    """输入原始气象特征字典，返回反标准化后的 ETc（mm/day）"""
    cols = meta["feature_cols"]
    x_raw = np.array([feature_values[c] for c in cols], dtype=float)
    x_norm = (x_raw - meta["X_mean"]) / meta["X_std"]
    y_norm = elm.predict(x_norm.reshape(1, -1)).ravel()[0]
    etc = y_norm * meta["y_std"] + meta["y_mean"]
    return max(float(etc), 0.0)


def irrigation_advice(etc_value: float) -> tuple:
    """根据 ETc 值给出灌溉建议，返回 (等级, 建议文字, 颜色)"""
    if etc_value < 1.5:
        return "无需灌溉", f"今日 ETc = {etc_value:.2f} mm，蒸散量较低，暂无需灌溉。", "#4CAF50"
    elif etc_value < 3.0:
        vol = round(etc_value * 3, 1)
        return "轻度灌溉", f"今日 ETc = {etc_value:.2f} mm，建议补水 **{vol} mm**（约合 {vol*10} m³/亩），可适量微喷。", "#FF9800"
    elif etc_value < 5.0:
        vol = round(etc_value * 4, 1)
        return "中度灌溉", f"今日 ETc = {etc_value:.2f} mm，建议补水 **{vol} mm**（约合 {vol*10} m³/亩），建议沟灌或滴灌。", "#FF5722"
    else:
        vol = round(etc_value * 5, 1)
        return "大量灌溉", f"今日 ETc = {etc_value:.2f} mm，需水量较高！建议补水 **{vol} mm**（约合 {vol*10} m³/亩），及时灌溉防旱。", "#F44336"


# ──────────────────────────────────────────────
# 特征参数范围与说明
# ──────────────────────────────────────────────
FEATURE_CONFIG = {
    "T_mean":   {"label": "日均气温 (°C)",          "min": -10.0, "max": 35.0,  "default": 15.0,  "step": 0.5},
    "RH_mean":  {"label": "日均相对湿度 (%)",        "min": 10.0,  "max": 100.0, "default": 65.0,  "step": 1.0},
    "P_mean":   {"label": "日均大气压 (hPa)",          "min": 980.0, "max": 1040.0,"default": 1010.0,"step": 1.0},
    "u_2_mean": {"label": "2m 高风速 (m/s)",         "min": 0.0,   "max": 10.0,  "default": 2.0,   "step": 0.1},
    "Rs_mean":  {"label": "太阳辐射 (MJ/m²/day)",   "min": 0.0,   "max": 30.0,  "default": 12.0,  "step": 0.5},
    "kc":       {"label": "作物系数 Kc",             "min": 0.3,   "max": 1.3,   "default": 0.7,   "step": 0.05},
}

KC_STAGE_HINT = """
| 生育期 | Kc 参考值 |
|--------|----------|
| 出苗—分蘖 | 0.40–0.60 |
| 越冬—返青 | 0.50–0.70 |
| 拔节—孕穗 | 0.80–1.10 |
| 抽穗—灌浆 | 1.05–1.20 |
| 乳熟—成熟 | 0.70–0.90 |
"""


# ──────────────────────────────────────────────
# 图表函数
# ──────────────────────────────────────────────

def plot_convergence(history):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history, color="#1976D2", linewidth=2)
    ax.set_xlabel("迭代次数", fontsize=11)
    ax.set_ylabel("K折RMSE（标准化空间）", fontsize=11)
    ax.set_title("IBA-ELM 收敛曲线", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_batch_result(dates, y_pred, y_true=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(y_pred)), y_pred, color="#1976D2", linewidth=2,
            marker="o", markersize=4, label="IBA-ELM 预测 ETc")
    if y_true is not None:
        ax.plot(range(len(y_true)), y_true, color="#E53935", linewidth=1.5,
                linestyle="--", marker="s", markersize=3, label="实测 ETc")
    ax.axhline(y=3.0, color="#FF9800", linestyle=":", linewidth=1, alpha=0.7, label="中度灌溉阈值 3mm")
    ax.axhline(y=5.0, color="#F44336", linestyle=":", linewidth=1, alpha=0.7, label="大量灌溉阈值 5mm")
    if dates is not None and len(dates) == len(y_pred):
        step = max(1, len(dates) // 10)
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([str(d)[:10] for d in dates[::step]], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ETc (mm/day)", fontsize=11)
    ax.set_title("批量预测结果", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_gauge(etc_value):
    """仿仪表盘：显示当日 ETc 相对位置"""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    thresholds = [0, 1.5, 3.0, 5.0, 7.5]
    colors = ["#4CAF50", "#FF9800", "#FF5722", "#F44336"]
    labels = ["无需灌溉", "轻度", "中度", "大量"]
    for i in range(4):
        ax.barh(0, thresholds[i+1] - thresholds[i], left=thresholds[i],
                height=0.5, color=colors[i], alpha=0.7)
    clamped = min(etc_value, 7.4)
    ax.axvline(x=clamped, color="black", linewidth=3, ymin=0.1, ymax=0.9)
    ax.scatter([clamped], [0], color="black", s=80, zorder=5)
    for i, (t, lbl) in enumerate(zip(thresholds[:-1], labels)):
        ax.text(t + (thresholds[i+1]-t)/2, -0.35, lbl,
                ha="center", va="top", fontsize=8, color="white" if i >= 2 else "black",
                fontweight="bold")
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.6, 0.5)
    ax.set_xlabel("ETc (mm/day)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("需水强度指示", fontsize=11, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Streamlit 主程序
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="冬小麦智能灌溉预测系统",
        page_icon="🌾",
        layout="wide",
    )

    # ── 侧边栏 ──────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌾 系统信息")
        st.markdown("**冬小麦智能灌溉需水预测**")
        st.markdown("核心模型：**IBA-ELM**")
        st.markdown("（改进蝙蝠算法优化正则化极限学习机）")
        st.divider()

        result = load_model()
        if result is None:
            st.error("⚠️ 未找到模型文件！\n\n请先运行：\n```\npython train_model.py\n```")
            st.stop()

        elm, meta = result
        st.success("✅ 模型已加载")
        st.markdown(f"- **最优 λ**：`{meta['best_lambda']:.5f}`")
        st.markdown(f"- **R²**：`{meta['r2']:.4f}`")
        st.markdown(f"- **RMSE**：`{meta['rmse_orig']:.4f}` mm/day")
        st.markdown(f"- **MAE**：`{meta['mae_orig']:.4f}` mm/day")
        st.divider()
        st.markdown("**使用说明**")
        st.markdown(
            "1. 输入当日气象数据\n"
            "2. 点击【开始预测】\n"
            "3. 查看 ETc 预测值与灌溉建议\n"
            "4. 也可上传 CSV 进行批量预测"
        )
        st.divider()
        st.caption("作物：冬小麦 | 数据：黄淮海地区\n适用生育期：10月—次年6月")

    # ── 页面标题 ────────────────────────────────
    st.title("🌾 冬小麦智能灌溉需水预测系统")
    st.markdown(
        "基于 **IBA-ELM**（改进蝙蝠算法优化正则化极限学习机）"
        "，输入气象数据，自动预测日蒸散量（ETc）并给出灌溉建议。"
    )
    st.divider()

    # ── 主选项卡 ────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 单次预测", "📂 批量预测（CSV）", "📈 模型信息"])

    # ══ Tab 1：单次预测 ══════════════════════════
    with tab1:
        st.subheader("输入今日气象数据")

        col1, col2, col3 = st.columns(3)
        inputs = {}

        with col1:
            cfg = FEATURE_CONFIG["T_mean"]
            inputs["T_mean"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="T_mean"
            )
            cfg = FEATURE_CONFIG["RH_mean"]
            inputs["RH_mean"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="RH_mean"
            )

        with col2:
            cfg = FEATURE_CONFIG["P_mean"]
            inputs["P_mean"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="P_mean"
            )
            cfg = FEATURE_CONFIG["u_2_mean"]
            inputs["u_2_mean"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="u_2_mean"
            )

        with col3:
            cfg = FEATURE_CONFIG["Rs_mean"]
            inputs["Rs_mean"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="Rs_mean"
            )
            cfg = FEATURE_CONFIG["kc"]
            inputs["kc"] = st.number_input(
                cfg["label"], min_value=cfg["min"], max_value=cfg["max"],
                value=cfg["default"], step=cfg["step"], key="kc"
            )
            with st.expander("Kc 参考表"):
                st.markdown(KC_STAGE_HINT)

        st.markdown("")
        predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=False)

        if predict_btn:
            etc_val = predict_etc(elm, meta, inputs)
            level, advice, color = irrigation_advice(etc_val)

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="预测日蒸散量 ETc",
                    value=f"{etc_val:.2f} mm/day",
                )
                st.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:8px;"
                    f"text-align:center;color:white;font-size:16px;font-weight:bold'>"
                    f"💧 {level}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.pyplot(plot_gauge(etc_val))

            with res_col2:
                st.markdown("#### 灌溉建议")
                st.info(advice)
                st.markdown("#### 输入数据汇总")
                summary = {FEATURE_CONFIG[k]["label"]: [v] for k, v in inputs.items()}
                st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # ══ Tab 2：批量预测 ══════════════════════════
    with tab2:
        st.subheader("批量预测（上传 CSV 文件）")
        st.markdown(
            "上传包含气象数据的 CSV 文件，系统将对每行数据进行 ETc 预测。\n\n"
            "**必须包含的列**：`T_mean, RH_mean, P_mean, u_2_mean, Rs_mean, kc`\n\n"
            "可选列：`date`（日期，用于图表横轴）、`etc`（实测值，用于对比）"
        )

        # 示例 CSV 下载
        sample_data = pd.DataFrame({
            "date": ["2024-03-01", "2024-03-02", "2024-03-03"],
            "T_mean": [12.5, 14.0, 10.2],
            "RH_mean": [68.0, 72.0, 80.0],
            "P_mean": [0.0, 2.5, 0.0],
            "u_2_mean": [2.1, 1.8, 3.0],
            "Rs_mean": [14.5, 12.0, 8.0],
            "kc": [0.85, 0.85, 0.85],
        })
        st.download_button(
            "⬇ 下载示例 CSV 模板",
            sample_data.to_csv(index=False).encode("utf-8-sig"),
            "sample_input.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

        if uploaded is not None:
            try:
                df_input = pd.read_csv(uploaded)
                st.markdown(f"已读取 **{len(df_input)} 行**数据，列：{list(df_input.columns)}")

                required = ["T_mean", "RH_mean", "P_mean", "u_2_mean", "Rs_mean", "kc"]
                missing = [c for c in required if c not in df_input.columns]
                if missing:
                    st.error(f"缺少必要列：{missing}")
                else:
                    with st.spinner("正在预测..."):
                        preds = []
                        for _, row in df_input.iterrows():
                            feat = {c: float(row[c]) for c in required}
                            preds.append(predict_etc(elm, meta, feat))
                        df_input["ETc_pred (mm/day)"] = preds
                        df_input["灌溉建议"] = [irrigation_advice(v)[0] for v in preds]

                    st.success("预测完成！")

                    # 图表
                    dates = df_input["date"].values if "date" in df_input.columns else None
                    y_true = df_input["etc"].values if "etc" in df_input.columns else None
                    st.pyplot(plot_batch_result(dates, preds, y_true))

                    # 统计
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    stat_col1.metric("平均 ETc", f"{np.mean(preds):.2f} mm/day")
                    stat_col2.metric("最大 ETc", f"{np.max(preds):.2f} mm/day")
                    stat_col3.metric("最小 ETc", f"{np.min(preds):.2f} mm/day")
                    n_irr = sum(1 for v in preds if v >= 1.5)
                    stat_col4.metric("需灌溉天数", f"{n_irr} 天")

                    # 结果表格
                    st.dataframe(df_input, use_container_width=True)

                    # 下载
                    st.download_button(
                        "⬇ 下载预测结果",
                        df_input.to_csv(index=False).encode("utf-8-sig"),
                        "prediction_result.csv",
                        "text/csv",
                    )
            except Exception as e:
                st.error(f"读取文件失败：{e}")

    # ══ Tab 3：模型信息 ══════════════════════════
    with tab3:
        st.subheader("IBA-ELM 模型信息")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("#### 模型参数")
            from config import N_POP, N_GEN, N_HIDDEN, K_FOLD, CHAOS_TYPE, W_MAX, W_MIN, N_ELITE
            params = {
                "模型名称": "IBA-ELM",
                "隐层神经元数": N_HIDDEN,
                "种群规模 N_pop": N_POP,
                "迭代代数 N_gen": N_GEN,
                "K 折交叉验证": K_FOLD,
                "混沌映射类型": CHAOS_TYPE,
                "惯性权重范围": f"[{W_MIN}, {W_MAX}]",
                "精英保留数量": N_ELITE,
                "最优正则化系数 λ": f"{meta['best_lambda']:.6f}",
                "λ 搜索空间": "10^[-4, 2]（log10 尺度）",
            }
            st.table(pd.DataFrame.from_dict(params, orient="index", columns=["值"]))

        with info_col2:
            st.markdown("#### 训练性能指标")
            perf = {
                "R²（决定系数）": f"{meta['r2']:.4f}",
                "RMSE（均方根误差）": f"{meta['rmse_orig']:.4f} mm/day",
                "MAE（平均绝对误差）": f"{meta['mae_orig']:.4f} mm/day",
                "K折 CV RMSE（标准化空间）": f"{meta['cv_rmse']:.4f}",
            }
            st.table(pd.DataFrame.from_dict(perf, orient="index", columns=["值"]))

            st.markdown("#### 系统架构（三层）")
            st.markdown(
                "**数据层** — 气象数据输入（手动 / CSV 上传）\n\n"
                "**业务层** — IBA-ELM 模型预测引擎\n\n"
                "**展示层** — 预测结果 + 灌溉建议可视化"
            )

        st.divider()
        st.markdown("#### IBA-ELM 收敛曲线")
        st.pyplot(plot_convergence(meta["history"]))



if __name__ == "__main__":
    main()
