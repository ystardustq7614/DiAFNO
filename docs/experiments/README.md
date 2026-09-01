# PRE 实验索引

> 更新日期：2026-09-01

| 编号 | 实验 | 状态 | 实验设计 | 结果与分析 |
|---:|---|---|---|---|
| 01 | surface SD1 旧尺度基线 | 已完成，失败 | [EXPERIMENT](./01_surface_sd1_baseline/EXPERIMENT.md) | [RESULTS](./01_surface_sd1_baseline/RESULTS.md) |
| 02 | surface SD2 修复后重训 | 已完成，未过门槛 | [EXPERIMENT](./02_surface_sd2_retrain/EXPERIMENT.md) | [RESULTS](./02_surface_sd2_retrain/RESULTS.md) |
| 03 | day-1 采样与 checkpoint 消融 | 已完成，未找到合格配置 | [EXPERIMENT](./03_sampler_ablation/EXPERIMENT.md) | [RESULTS](./03_sampler_ablation/RESULTS.md) |
| 04 | surface SD2 15 天 rollout | 已完成，失败 | [EXPERIMENT](./04_surface_sd2_rollout/EXPERIMENT.md) | [RESULTS](./04_surface_sd2_rollout/RESULTS.md) |
| 05 | 条件通路与可预测性诊断 | 部分完成，已定位主要方向 | [EXPERIMENT](./05_condition_diagnostics/EXPERIMENT.md) | [RESULTS](./05_condition_diagnostics/RESULTS.md) |
| 06 | full3d 30 层实验 | 未执行，建议待 surface 长 lead 解决后启动 | [EXPERIMENT](./06_full3d/EXPERIMENT.md) | [RESULTS](./06_full3d/RESULTS.md) |
| 07 | surface persistence-residual 确定性基线 | **已完成，Phase 3 Go**；含长时效/区域分解诊断与 Phase 5 两项 A/B（均判不保留） | [EXPERIMENT](./07_residual_baseline/EXPERIMENT.md) | [RESULTS](./07_residual_baseline/RESULTS.md) |

## 实验之间的决策关系

```text
SD1 失败
  └─ 修复 sigma_data 与训练卫生
       └─ SD2 重训仍未过 validation 门槛
            ├─ sampler 消融：只能小幅改善，不能救回模型
            ├─ 15 天 test：所有 lead day 均败于 persistence
            └─ 条件诊断：任务有信号、condition 已接入，但条件预测能力不足
                 ├─ 暂停 full3d，先建立 condition-only 确定性基线
                 └─ 07 persistence-residual：Phase 3 Go（val day-1 0.781×，test 0.833×）
                      ├─ 长时效诊断：overall 仅持平的根因 = 方差塌缩+相关衰减+偏差漂移
                      ├─ Phase 5① 静态 mask 输入 A/B → 不保留（9/10 epoch 落后）
                      ├─ Phase 5② remask 回灌 A/B → 维持 rf0（分段效应：中段好长段差）
                      └─ Phase 6：对残差做生成式建模（residual diffusion），新计划待立
                           └─ full3d：待 surface 长 lead 解决后再启动
```

共用数据、网格和指标定义见 [数据说明](../data/PRE_ocean_data.md) 与
[运行手册](../operations/PRE_runbook.md)。
