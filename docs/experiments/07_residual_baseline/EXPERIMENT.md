# 实验 07：surface persistence-residual 确定性基线

> 状态：**已完成**（2026-08-31）
> 科学问题：IAFNO backbone 能否把过去 7 天 `u/v` 直接映射成优于 persistence 的
> 次日流场，并在 15 天自回归中保持优势？

## 目标与假设

建立确定性、condition-only 的 `PersistenceResidualIAFNO` 基线：

```text
prediction = condition 最后一天 + learned residual
```

核心假设是：7 天 condition 中存在足够的次日预测信号；如果该模型优于 persistence，
则此前 diffusion 路径失败不能简单归因于 backbone 或 condition 无效。

## 任务与执行状态

| 任务 | 状态 | 结果入口 |
|---|---|---|
| 真实数据训练入口检查 | 已完成 | 工程验证见项目 Changelog |
| surface 10 epoch 短训练 | 已完成 | [RESULTS](./RESULTS.md) |
| 逐 checkpoint validation day-1 选型 | 已完成 | [RESULTS](./RESULTS.md) |
| 冻结 checkpoint 后 test day-1/15-day 报告 | 已完成 | [RESULTS](./RESULTS.md) |
| 长时效 bias/variance/correlation 诊断 | 已完成 | [RESULTS](./RESULTS.md) |

静态 mask 输入和 rollout remask 不属于本实验，分别记录在实验 08、09。

## 实验设置

| 项目 | 设置 |
|---|---|
| preset/objective | `surface_smoke` / `persistence_residual` |
| 模型 | 与 diffusion 相同的 `IAFNODiff` backbone；残差输出头零初始化 |
| 输入/输出 | 过去 7 天 `u/v`（14 通道）→ 次日 `u/v`（2 通道） |
| 残差基准 | `base = cond[:, -2:]` |
| 损失 | 双变量 rho mask 下的 normalized `masked_mse_loss` |
| 数据 | 连续 split、train-only min-max、`clip_pct=None` |
| 训练 | 单卡 batch 4、lr `1e-3`、cosine、最多 10 epoch |
| rollout | 15 天确定性自回归，`remask_feedback=False` |

## 对照与控制变量

- 主对照：persistence；附加参考：zero、rho-oracle、ridge/linear probe；
- 与 surface SD2 diffusion 保持相同数据划分、归一化、mask、backbone 和训练预算；
- 核心变化只有目标函数：扩散采样改为确定性 persistence-residual 回归；
- checkpoint 只由 validation 选择，test 不参与训练、选型或超参数决策。

## 指标与判定

- 训练记录 normalized masked MSE、验证 rel-L2、耗时和吞吐；
- checkpoint 按 validation day-1 native C-grid RMSE 选择，不使用 `best.pth` 排名；
- 科学报告使用物理单位 m/s 的 day-1/15-day RMSE、MAE、model/persistence ratio；
- 长时效附加报告 u/v 分项、bias、variance ratio、spatial correlation 和 crossover day；
- day-1 严格优于 persistence 才允许进入 test 15-day 报告。

实际运行配置、产物、数值与分析见 [RESULTS.md](./RESULTS.md)。
