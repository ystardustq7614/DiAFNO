# 实验 06：full3d 30 层训练与评估

> 状态：未执行；被 surface 科学门槛阻塞

## 实验目的

在全部 30 个 sigma 层上验证 7 天 u/v → 次日 u/v 的训练和 15 天 rollout，
评估 surface 结论能否扩展到完整三维海流。

## 计划设置

| 项目 | 设置 |
|---|---|
| 空间 shape | `400×441×30` |
| patch | `4×3×2` |
| token grid | `100×147×15 = 220,500` |
| 模型 | embed 128，implicit 2，explicit 4 |
| batch | 1 |
| rollout | 15 天 |
| sampler | 先沿用通过 surface 消融的配置 |

## 对照与控制变量

- persistence、zero field、rho-oracle。
- 与 surface 使用相同连续 split、变量语义、双 mask 和 train-only 归一化原则。
- 只改变垂向层数和为显存调整的模型容量；任何容量调整都必须记录。

## 记录指标

- 资源：统计缓存耗时、batch 耗时、epoch 耗时、峰值显存、GPU 利用率和 I/O 波动。
- 训练：train loss、validation 指标、AMP skipped update、checkpoint。
- test：每个 lead day × u/v × 30 层的 RMSE/MAE、overall RMSE。
- 分析：按层、近岸/开阔海、lead day 和 persistence skill 分层。

## 执行方法

前置门槛通过后，将训练和评估的 `PRESET` 统一设为 `"full3d"`：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u pre_trainer.py
CUDA_VISIBLE_DEVICES=<gpu> python -u pre_evaluate.py
```

正式训练前先运行统计缓存、单 batch 显存和 I/O 探针，再决定 epoch 数和评估窗口数。

## 预期结果与门槛

- 前置门槛：surface day-1 和 15 天必须稳定优于 persistence。
- 工程门槛：无 OOM，至少完成一个 epoch 和最小 rollout。
- 科学门槛：正式 test 的 `model/persistence < 1`，并检查各层是否存在系统性退化。
- OOM 时依次将 embed 128→96、implicit 2→1；不改变能精确整除 441 的 patch。

## 当前为什么不执行

surface SD2 的 day-1 ratio 为 2.201，15-day overall ratio 为 1.640，明确未过门槛。
此时投入 full3d 只会把未解决的单步条件预测问题放大，并增加显存和 I/O 成本。

实际状态见 [RESULTS.md](./RESULTS.md)。
