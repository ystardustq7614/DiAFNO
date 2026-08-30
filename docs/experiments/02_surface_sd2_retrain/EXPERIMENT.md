# 实验 02：surface SD2 修复后重训

> 状态：已执行
> 执行日期：2026-08-28～2026-08-29
> 结果目录：`checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/`

## 实验目的

修正旧实验的 `sigma_data` 尺度和训练卫生问题后重新训练 surface 模型，判断旧失败
是否主要由尺度 bug 导致，并为采样消融及正式 rollout 选择 checkpoint。

## 核心变量与对照

| 项目 | SD1 旧实验 | SD2 重训 |
|---|---:|---:|
| `stats_sigma` | 0.0856042 | 0.0856042 |
| `sigma_data_scale` | 等效 1.0 | **2.0** |
| `sigma_data` | 0.0856042 | **0.1712084** |
| checkpoint 目录 | 无后缀 | `_SD2` 后缀 |
| 训练控制 | 旧 AMP/scheduler 逻辑 | 修复 AMP、scheduler、非有限 loss、early stop 和完整 resume |

其余数据、surface preset、连续 split、mask、归一化和模型主体保持一致。

## 执行设置

- 输入/输出：7 天表层 u/v → 次日表层 u/v。
- batch 4，embed 180，implicit 4，explicit 4，32 sampling steps。
- 原计划最多 10 epoch，连续 2 epoch validation 不改善则 early stop。
- 每个 checkpoint 单独保存；`best.pth` 必须与最佳 epoch 权重一致。

实际归档产物使用“最多 10 epoch + early stop”，并在第 5 epoch 停止。当前代码已让
`pre_config.py` 的 `surface_smoke.num_epochs=10` 成为唯一正常默认，
`EPOCH_OVERRIDES={}` 不再改变该上限，因此默认命令与本实验的轮数协议一致。

主实验结束后还在服务器仓库外副本 `DiAFNO_lr3e4` 做过 `lr=3e-4` 的附属对照；它不属于
当前分支的结果目录，设计与结果证据见 [RESULTS.md](./RESULTS.md) 的“学习率附属对照”。

## 记录指标

- 每 epoch train loss、validation masked relative L2、AMP skipped updates、耗时。
- checkpoint 中的 `stats_sigma`、`sigma_data_scale`、`sigma_data` 和 epoch。
- `best.pth` 与对应 `EpN.pth` 的张量一致性。
- 后续 day-1 native RMSE；validation relative L2 只作筛选信号。

## 执行方法

确认训练上限、early stop 和 `SIGMA_DATA_SCALE=2.0` 后运行：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u pre_trainer.py
```

启动后先核对日志中的 `sigma_data`、run tag 和 epoch 上限，再允许无人值守训练。

## 预期结果与 Go/No-Go

- 启动检查：日志必须出现 `sigma_data=0.17120 (scale 2.000x)` 和 `_SD2` 目录。
- 早期门槛：epoch 2 的 validation relative L2 应小于 0.8 且较 epoch 1 下降。
- 科学门槛：选出的 checkpoint 在独立评估中必须有 `model/persistence < 1`。
- 若 validation 仍约 1.5 或更高，停止延长训练，转向条件预测目标和采样链诊断。

## 主要产物

- `train_surface_sd2.log`
- `Ep1.pth`～实际停止 epoch、`best.pth`、`loss.dat`
- checkpoint 配置元数据

实际训练曲线与判断见 [RESULTS.md](./RESULTS.md)。
