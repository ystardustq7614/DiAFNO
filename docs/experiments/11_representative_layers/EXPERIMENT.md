# 实验 11：代表层（middle=14 / bottom=0）确定性基线

> 状态：**原实验已完成（2026-09-03）；勘误后正式 Ep4 test 待补**
> 制定日期：2026-09-03
> 科学问题：surface 的确定性预测能力能否代表垂向？
> （[历史实施计划 §6 工作包 5](../../project/archive/MULTISTEP_PLAN_20260901.md)）

## 设计

- 两层独立单层 probe：**middle（sigma index 14）**、**bottom（sigma index 0）**，
  与 surface（实验 07）唯一的差异是 `depth_index`（`middle_smoke`/`bottom_smoke`
  preset，架构/patch/embed/预算/协议完全一致——单变量）。
- 先单步 `persistence_residual`（随机初始化，lr 1e-3，10 epochs，单卡 4090，
  batch 4，与实验 07 同预算）；checkpoint 选型 = validation **day-1 native RMSE**
  （h1，stride 7，156 窗口）。
- 过单步门槛的层进入 **MS5**（K=5，schedule `1,2,1,3,1,4,1,5`，从该层单步最优
  weights-only 初始化，MS 默认 lr 1e-4 / 5 epochs，同实验 10 协议）。
- sigma index 随水深变化，**不得把 index 换算成固定米深**。
- 中/底层的 per-layer stats 缓存（`stats_d14/d0_clipnone.npz`）由 trainer 首跑
  现算（train-only，mmap 单层流式）。

## 预注册门槛（开跑前冻结）

难度参照（[实验 06 全层数据画像](../06_full3d/RESULTS.md)，
`diag_uv_predictability_20260901/summary.csv`，val
persistence RMSE，rho 网格物理单位）：

| 层 | u d1 / d15 | v d1 / d15 |
|---|---|---|
| bottom (0) | 0.0496 / 0.0943 | 0.0285 / 0.0485 |
| middle (14) | 0.1048 / 0.2018 | 0.0492 / 0.0777 |
| surface (29)（参照） | 0.1504 / 0.3116 | 0.0972 / 0.1685 |

1. **单步 probe Go**：val day-1 native RMSE ratio（model/该层 persistence，
   同 156 窗口协议）**< 1.0**；coastal/offshore 与 u/v 分开报告；不用 pooled
   overall 掩盖分项。
2. **层 MS5 Go**（镜像实验 10 surface 预注册）：
   - day-1 native RMSE ≤ 该层单步最优 × 1.02；
   - validation 15-day overall ratio < 0.941；
   - u、v 各自 overall ratio < 1.0；
   - day 10–15 每日 ratio < 1.0；
   - 结构诊断：crossover 消失、corr 全 lead 占优（bias/var_ratio 如实记录）。
3. test 在配置与 checkpoint 冻结后**各只运行一次**（h15）。
4. 代表层只判垂向难度与 full3d 投资价值，**不充当 full3d 结论**
   （[历史实施计划 §6 工作包 5](../../project/archive/MULTISTEP_PLAN_20260901.md)）。

## 执行编排

- GPU 0 = middle，GPU 3 = bottom（并行）；逐 epoch 选型评估与训练共享同卡。
- 监控：tmux + 值守 subagent（用户要求的标准模式）。

## 状态

- [x] middle 单步 probe（day-1 ratio 0.582 ✅；2026-09-04 勘误：原记录 0.770 为誊写
  错误，当前仓库可用的归档 NPZ/日志均为 0.582，见 `RESULTS.md` 勘误节）
- [x] bottom 单步 probe（day-1 ratio 0.568 ✅）
- [x] 单步门槛判断 → 两层 MS5（均从 probe Ep10 weights-only 初始化）
- [x] 原选型 + test + 结构诊断（bottom Ep5 全门槛 Go；middle 原选型 Ep2 不合规——
  day-1 门槛基于无法复现的 probe 数值误算，按预注册规则正式改选 Ep4，test 待补，
  见 `RESULTS.md` 勘误节）
- 结果见 `RESULTS.md`。
