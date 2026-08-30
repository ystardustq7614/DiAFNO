# 实验 03 结果：day-1 采样与 checkpoint 消融

> 结论：`churn=0` 是当前较优设置，但任何配置都没有接近 persistence。

## 消融结果

同一 validation 集 156 个窗口，pooled native masked RMSE：

| 设置 | RMSE (m/s) | / persistence | 判断 |
|---|---:|---:|---|
| Ep2, churn=0, E=1 | 0.2991 | 2.312 | 差 |
| **Ep3, churn=0, E=1** | **0.2584** | **1.998** | 单轨迹最佳 checkpoint |
| Ep4, churn=0, E=1 | 0.3305 | 2.555 | 继续训练恶化 |
| Ep3, churn=80, E=1 | 0.3234 | 2.500 | churn 明显有害 |
| Ep3, churn=0, E=4 | 0.2471 | 1.911 | 改善 4.4%，仍失败 |
| Ep3, sigma_max=3 | 0.2851 | 2.204 | 无效 |
| persistence | 0.1294 | 1.000 | 必须战胜的基线 |
| zero field | 0.2620 | 2.025 | Ep3 单轨迹仅略好于零场 |

![SD2 训练、消融和 rollout 总览](../../../plots/07_sd2_result_overview.png)

## 结果分析

- `churn=0` 相对 `churn=80` 明显更好，应保留确定性 Heun。
- E=4 只改善 4.4%，仍是 persistence 的 1.91 倍；现阶段不值得支付约 4 倍采样成本。
- Ep3 是三组 checkpoint 中最好的一组；Ep4 验证恶化与训练曲线一致。
- `sigma_max=3` 没有改善，说明问题不是单一高噪声上限参数。
- 超参数会改变结果，但不能把不合格模型调成合格模型。

## 决策

- 15 天 rollout 使用 `Ep3 + churn=0 + E=1`，用于确认长期失败形态和保留正式 test 证据。
- 不继续扩张 sampler 搜索。
- 后续把主要精力转向条件预测目标与 backbone/condition 诊断。
