# 实验 06：full3d 30 层训练与评估

> 状态：**部分执行**（画像/资源 probe/K1 smoke/single-step pilot 完成；K3 按预注册条件阻塞，正式长训未准入）
> 科学问题：surface 上的确定性预测能力能否扩展到全部 30 个 sigma 层？

## 目标与假设

验证过去 7 天全层 `u/v` → 未来 `u/v` 的训练与自回归预测，并确定不同 sigma 层的
可预测性、误差结构和资源成本。surface 结果不能直接外推到 full3d，因此先以逐级 probe
建立证据，再决定是否投入正式训练。

## 任务与执行状态

| 任务 | 状态 | 结果入口 |
|---|---|---|
| 全 30 层尺度、增量和 persistence 画像 | 已完成（2026-09-01，门禁 PASS） | [RESULTS](./RESULTS.md) |
| stats cache、单样本 I/O 和峰值显存 probe | 已完成（2026-09-03） | [RESULTS](./RESULTS.md) |
| deterministic K1 real-data smoke | 已完成（SMOKE PASS） | [RESULTS](./RESULTS.md) |
| deterministic single-step 1 epoch pilot | 已完成（训练健康，逐层信号未出现） | [RESULTS](./RESULTS.md) |
| detached K3 pilot | 阻塞（依赖 pilot 逐层信号，未满足） | [RESULTS](./RESULTS.md) |
| 正式训练与 15-day test | 未执行；尚未准入 | [RESULTS](./RESULTS.md) |

## 计划设置

| 项目 | 设置 |
|---|---|
| 空间 shape | `400×441×30` |
| patch | `4×3×2` |
| token grid | `100×147×15 = 220,500` |
| 初始模型 | embed 128，implicit 2，explicit 4 |
| batch | 1 |
| 首个 objective | `persistence_residual` |
| rollout | probe 从 K1 开始；K3 和 15 天按门槛逐级开放 |

## 对照与控制变量

- persistence、zero field、rho-oracle；
- 与 surface 使用相同连续 split、变量语义、双 mask 和 train-only 归一化原则；
- full3d 结果必须报告逐层及 upper/middle/bottom band，不用 pooled overall 掩盖坏层；
- 容量、patch、horizon 每次只改变一项并记录。

## 指标与判定

- 数据：逐层 valid count、尺度、增量、persistence RMSE/MAE；
- 资源：stats 缓存耗时、batch/epoch 耗时、峰值显存、吞吐、GPU 利用率、I/O 波动；
- 训练：loss、validation 指标、AMP skipped update、checkpoint 完整性；
- 科学指标：lead × u/v × layer 的 native RMSE/MAE 和 model/persistence ratio。

准入顺序：数据画像通过 → 资源 probe 无异常 → K1 smoke → single-step pilot 有信号 →
K3 pilot → 另行冻结正式预算。OOM 时可依次尝试 embed 128→96、implicit 2→1；任何容量
调整都成为新的受控配置。

实际状态和后续结果见 [RESULTS.md](./RESULTS.md)。
