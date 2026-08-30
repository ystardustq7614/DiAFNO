# 实验 01：surface SD1 旧尺度基线

> 状态：已执行
> 执行日期：2026-08-27
> 结果目录：`checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7/`

## 实验目的

验证 PRE 的完整工程链路是否能从表层 7 天 u/v 条件训练单步 DiAFNO，并在 test
集自回归预测 15 天；同时建立后续修复实验必须对照的旧基线。

## 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | surface 层，`H×W×Z = 400×441×1` |
| 输入/输出 | 7 天 u/v（14 通道）→ 次日 u/v（2 通道） |
| 模型 | embed 180，implicit 4，explicit 4，batch 4 |
| 训练 | 10 epoch，32-step sampling，seed 123 |
| EDM 尺度 | `sigma_data≈0.08560`；事后确认这是错误的 SD1 尺度 |
| 长期预测 | test 集 15 天 autoregressive rollout |

## 对照与控制变量

- 模型对照：persistence、zero field、rho-oracle。
- 时间切分、mask、归一化、rho→native 映射和 test 窗口固定。
- 结果保留为反面基线，不与 SD2 checkpoint 混目录。

## 记录指标

- 每 epoch：train loss、validation masked relative L2、耗时。
- test：每个 lead day × u/v × sigma layer 的 native masked RMSE/MAE。
- 汇总：有效点数加权 overall RMSE、model/persistence ratio。
- 现象：空间偏置、近岸极值、rollout 是否随 lead day 失稳。

## 执行方法

以下命令只记录当时使用的入口，不是当前 HEAD 的直接复现命令。当前
`pre_trainer.py` 强制使用 SD2 尺度和 `_SD2` 目录；若要重放 SD1，必须恢复当时的
历史代码/配置，并使用新的归档目录，不能用当前默认配置覆盖现有产物。

```bash
python pre_trainer.py
python pre_evaluate.py
```

历史产物已存在，不应在同一目录覆盖重跑。

## 预期结果与判定

- 工程预期：训练、checkpoint、15 天 rollout 和指标文件全部落盘。
- 科学最低门槛：每个 lead day 的 `model/persistence < 1`。
- 若 validation relative L2 持续大于 1 或 test 明显差于 persistence，标记失败并进入尺度/训练/采样诊断。

## 主要产物

- `loss.dat`、`Ep1.pth`～`Ep10.pth`、`best.pth`
- `eval_test.npz`
- `figures/`

实际数值和失败分析见 [RESULTS.md](./RESULTS.md)。
