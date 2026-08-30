# 实验 04 结果：surface SD2 15 天 rollout

> 结论：SD2 相对旧实验有改善，但所有 lead day 仍败于 persistence，full3d 门槛未通过。

## 汇总结果

原生 C-grid、native mask、RMSE 单位 m/s：

| lead day | model | persistence | model/persistence | zero |
|---:|---:|---:|---:|---:|
| 1 | 0.2568 | 0.1167 | 2.201 | 0.2640 |
| 5 | 0.3409 | 0.2076 | 1.642 | 0.2571 |
| 10 | 0.3444 | 0.2232 | 1.543 | 0.2434 |
| 15 | 0.3543 | 0.2152 | 1.646 | 0.2648 |
| **15-day overall** | **0.3442** | **0.2098** | **1.640** | **0.2568** |

day-1 分变量：

- u：0.2649 vs persistence 0.1387 m/s，比例 1.91。
- v：0.2485 vs persistence 0.0895 m/s，比例 2.78。

15 天内没有任何 lead day、任何变量稳定胜过 persistence。

## 与旧 SD1 的对照

- day-1 pooled RMSE 相对旧实验降低 22.5%。
- 15-day overall RMSE 相对旧实验降低 18.6%。
- 这是“SD2 训练 + churn=0”组合协议的改善，不能把全部收益单独归因于 sigma 修正。
- 15-day overall 的 v 几乎没有改善：约 0.2392 → 0.2382 m/s。

## 空间结果

预测保留海岸轮廓，但丢失大尺度真实流场结构，并叠加细碎纹理和近岸极值。
原评估图对 truth、prediction、error 分别自动取色阶，因此不能跨列直接用颜色强弱比较。

- [day-1 u](../../../checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/figures_h15_ch0_e1_s123_ckptEp3/d01_s00_u.png)
- [day-1 v](../../../checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/figures_h15_ch0_e1_s123_ckptEp3/d01_s00_v.png)
- [day-15 u](../../../checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/figures_h15_ch0_e1_s123_ckptEp3/d15_s00_u.png)
- [day-15 v](../../../checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/figures_h15_ch0_e1_s123_ckptEp3/d15_s00_v.png)

## 分析与决策

模型 day-1 仅比 zero field 略好，长期 overall 反而比 zero 高 34.1%。这不是单纯的
autoregressive 误差累积；单步条件预测本身已经不合格。

- 本实验标记为失败，但所有产物保留。
- 暂停 full3d。
- 进入 [条件通路与可预测性诊断](../05_condition_diagnostics/EXPERIMENT.md)。
