## 数据与配置
config.py：数据路径、特征列（T_mean, RH_mean, P_mean, u_2_mean, Rs_mean, kc）、目标（ETc）、IBA-ELM 超参数、图表输出目录。
data_loader.py：加载并合并两期数据（2023.10–2024.06、2024.10–2025.06），提供 get_X_y() 做标准化并返回特征 X 与目标 y。
## 标准 BA 与 IBA（4.1、4.2）
ba_standard.py：标准蝙蝠算法，用于 4.1 节说明“易陷入局部最优、后期收敛慢”。
chaos.py：4.2.1 混沌映射（Tent / Logistic）及基于混沌的初始种群生成。
iba.py：4.2 改进蝙蝠算法——混沌初始化、4.2.2 自适应惯性权重、4.2.3 精英保留。
## ELM 与 IBA-ELM（4.3）
elm.py：ELM 类；vector_to_elm_weights / vector_to_elm_weights 实现 4.3.2 的高维向量映射；kfold_rmse_fitness 实现 4.3.1 的 K 折交叉验证 RMSE 适应度。
iba_elm.py：4.3.3 整体流程——适应度=K 折 RMSE、蝙蝠位置=ELM 输入权重与偏置向量、IBA 寻优后解码并拟合 ELM。
## 运行与作图
run_iba_elm.py：用两期数据训练 IBA-ELM，将收敛曲线和预测结果写入 figures/。
plot_figures.py：
4.1：BA vs IBA 收敛对比 → figures/fig_4_1_ba_defects.png
4.2.1：Tent/Logistic 混沌序列 → figures/fig_4_2_1_chaos.png
4.2.2：自适应惯性权重曲线 → figures/fig_4_2_2_adaptive_weight.png
4.3：IBA-ELM 收敛曲线 + 预测 vs 实测 → figures/fig_4_3_iba_elm.png