# 实验 09：surface rollout remask feedback A/B

> 状态：**已完成**（2026-09-01）
> 科学问题：每步把预测帧的陆地点重新置零再回灌，能否稳定改善 15 天自回归？

## 目标与假设

使用实验 07 的同一个 Ep10 checkpoint，在 validation 上比较：

- rf0：历史行为，模型预测整帧直接回灌；
- rf1：每步预测先乘双变量 rho mask，再进入下一步 condition。

假设是 rf1 能阻止陆地填值污染后续 condition，并在不改变 day-1 的前提下改善长 lead。

## 任务与执行状态

| 任务 | 状态 | 结果入口 |
|---|---|---|
| rf0 validation 15-day rollout | 已完成 | [RESULTS](./RESULTS.md) |
| rf1 validation 15-day rollout | 已完成 | [RESULTS](./RESULTS.md) |
| 逐 lead 与 overall 对比 | 已完成 | [RESULTS](./RESULTS.md) |
| 远端“day-2 改善约 7.7%”复核 | 已完成 | [RESULTS](./RESULTS.md) |

## 设计与控制变量

- checkpoint、split、窗口、seed、batch、model 和所有 sampler metadata 相同；
- 唯一变量是 `remask_feedback=False/True`；
- 使用 validation 而非 test 做 A/B 决策；
- day-1 应完全相同，因为尚未发生反馈。

## 指标与判定

- 逐 lead pooled native RMSE；
- rf1/rf0 与各自 model/persistence ratio；
- 15-day overall；
- 只有 rf1 在中后段和 overall 上稳定改善才保留。

实际配置、产物、数值与分析见 [RESULTS.md](./RESULTS.md)。
