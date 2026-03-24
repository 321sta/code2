# -*- coding: utf-8 -*-
"""数据路径与全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

# 两个冬小麦生育期数据文件
CLEANED_23_24 = os.path.join(DATA_DIR, "cleaned23_24.csv")
CLEANED_24_25 = os.path.join(DATA_DIR, "cleaned24_25.csv")
WHEAT_ETC_23_24 = os.path.join(DATA_DIR, "wheat_daily_etc_2023_2024.csv")
WHEAT_ETC_24_25 = os.path.join(DATA_DIR, "wheat_daily_etc_2024_2025.csv")
ET0_23_24 = os.path.join(DATA_DIR, "2324_ET_0.csv")
ET0_24_25 = os.path.join(DATA_DIR, "2425_ET_0.csv")

# 特征列（用于ELM输入）
FEATURE_COLS = ["T_mean", "RH_mean", "P_mean", "u_2_mean", "Rs_mean", "kc"]
# 目标列
TARGET_COL = "etc"  # 或 "et0"

# IBA-ELM 超参数
N_POP = 30          # 蝙蝠种群规模
N_GEN = 50          # 迭代代数
N_HIDDEN = 15       # ELM 隐层神经元数
K_FOLD = 5          # K折交叉验证
DIM_SCALE = 1.0     # 搜索空间缩放

# 混沌映射类型: 'tent' 或 'logistic'
CHAOS_TYPE = "tent"
# 惯性权重范围
W_MAX, W_MIN = 0.9, 0.4
# 精英保留数量
N_ELITE = 2

# 图表输出目录
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
