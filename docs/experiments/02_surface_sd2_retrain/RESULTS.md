# 实验 02 结果：surface SD2 修复后重训

> 结论：尺度修复生效，但训练仍未通过 validation 门槛；第 5 epoch 早停。

## 实际运行记录

| epoch | train loss | val masked relative L2 | 备注 |
|---:|---:|---:|---|
| 1 | 0.15017 | 1.65138 | 2 次 AMP skipped update |
| 2 | 0.04517 | 1.68291 | 1 次 AMP skipped update |
| 3 | 0.03753 | **1.52958** | 最佳 checkpoint |
| 4 | 0.02832 | 2.18132 | train 下降、validation 明显恶化 |
| 5 | 0.02331 | 1.58135 | 未刷新最佳，触发 early stop |

train loss 从 epoch 1 到 5 下降约 84.5%，但 validation 指标始终大于 1 且波动明显。

## Checkpoint 核验

- `config.stats_sigma = 0.0856042`
- `config.sigma_data_scale = 2.0`
- `config.sigma_data = 0.1712084`
- `best.pth` 的 62 个模型张量与 `Ep3.pth` 逐元素一致。

因此可以排除“尺度修复未生效”和“评估错 checkpoint”。

本次产物对应“最多 10 epoch + early stop”的执行配置。当前工作树已消除原来的
4/10 epoch 漂移：`pre_config.py` 保存 `surface_smoke.num_epochs=10`，
`pre_trainer.py` 的 `EPOCH_OVERRIDES={}` 默认不覆盖它。

## 学习率附属对照

服务器曾在仓库外副本 `/data2/user/zyq/projects/DiAFNO_lr3e4/` 运行 `lr=3e-4` 对照，
产物写入独立的 `PRE_lr3e4` 根目录，后来归档在兄弟分支
`origin/adapt-weather-ocean-lr3e4`，不在当前工作树中。本次整理没有复制其中的
checkpoint、NPZ 或图片。

| 设置 | val masked relative L2 | day-1 native RMSE | / persistence |
|---|---:|---:|---:|
| `lr=3e-4`, Ep1（early-stop best） | **2.14188** | **0.3259** | **2.520** |
| `lr=3e-4`, Ep10（手动续训） | 2.56845 | 0.3779 | 2.922 |
| `lr=1e-3`, Ep3（主实验） | **1.52958** | **0.2584** | **1.998** |

这些历史日志足以说明把学习率单独降至 `3e-4` 没有挽救当前方案，不能说明已经完成
系统的学习率搜索。续训阶段还放宽了 early-stop patience，因此 Ep10 只用于观察继续训练，
不应表述为与主实验完全等价的单变量对照。

## 与预期对照

| 判定项 | 预期 | 实际 | 结论 |
|---|---:|---:|---|
| epoch 2 val relative L2 | < 0.8 且下降 | 1.68291，反而上升 | 未通过 |
| best val relative L2 | < 1 | 1.52958 | 未通过 |
| 训练稳定泛化 | train/val 同向改善 | train 降、val 波动恶化 | 未通过 |

## 分析

修正 `sigma_data` 后，模型并没有形成可靠的条件预测能力。继续增加 epoch 更可能强化
训练目标而非改善真实条件预测。训练去噪 loss 与最终“从条件生成下一天”的目标存在明显
失配信号。

## 结论与后续决策

- 采用 `Ep3.pth` 进入受控的 day-1 采样消融，但不把它视为合格模型。
- 暂停增加训练轮数。
- sampler 若只能小幅改善，应转向 condition-only 基线和条件通路诊断。

后续实验见 [采样消融](../03_sampler_ablation/EXPERIMENT.md)。
