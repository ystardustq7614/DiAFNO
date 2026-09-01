# 实验 08：surface 静态 mask 输入 A/B

> 状态：**已完成**（2026-08-31）
> 科学问题：把双变量 rho mask 作为额外静态条件，能否稳定改善 day-1，尤其是近岸预测？

## 目标与假设

实验比较：

- A：实验 07 的 14 通道动态 `u/v` condition；
- B：相同动态 condition，再增加 `mask_u_rho`、`mask_v_rho` 两个静态通道。

假设是显式海陆边界能帮助 IAFNO 区分近岸动力结构，使 overall 与 coastal 指标稳定优于 A。

## 任务与执行状态

| 任务 | 状态 | 结果入口 |
|---|---|---|
| B 臂训练入口检查 | 已完成 | 工程验证见项目 Changelog |
| B 臂 surface 10 epoch 训练 | 已完成 | [RESULTS](./RESULTS.md) |
| A/B 逐 epoch validation day-1 对比 | 已完成 | [RESULTS](./RESULTS.md) |
| coastal/offshore × u/v 分解 | 已完成 | [RESULTS](./RESULTS.md) |

## 设计与控制变量

| 项目 | A | B |
|---|---|---|
| 动态 condition | 14 通道 `u/v` | 同 A |
| 静态 condition | 无 | 2 通道双变量 rho mask |
| persistence base | condition 最后一天 | 同 A |
| 模型/训练预算 | 实验 07 配置 | 同 A |
| validation 协议 | day-1，156 窗口 | 同 A |

除静态 mask 输入外，数据、归一化、backbone、loss、batch、LR、epoch 和随机种子保持一致。

## 指标与判定

- 每个 epoch 的 validation day-1 native pooled RMSE；
- coastal（距陆地 ≤5 格）/offshore × u/v 的 model/persistence ratio；
- 只有 B 在 overall 和区域分项上表现出稳定改善才保留，否则回到 A。

实际配置、产物、数值与分析见 [RESULTS.md](./RESULTS.md)。
