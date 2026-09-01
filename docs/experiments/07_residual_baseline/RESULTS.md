# 实验 07 结果：surface persistence-residual 确定性基线

> 状态：**已完成**
> 结论：day-1 明确优于 persistence，但 15-day overall 持平略差；backbone 与 condition
> 已证明可用，主要困难转为长时效自回归退化。

## 实际运行与产物

- 环境：conda env `diafno`，Python 3.10.20，torch 2.4.1+cu124，RTX 4090 24G 单卡；
- run 目录：`/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/`；
- checkpoint：`Ep1.pth`～`Ep10.pth`、`best.pth`、`loss.dat`；
- 训练日志：`~/checkpoints/PRE/train_residual_full_surface.log`；
- 选型日志：`eval_val_day1_selection_RES.log`；
- test：`eval_test_h15_ch0_e1_s123_rf0_ckptEp10.npz` 及对应 figures；
- 长时效诊断：`leadtime_diag_ckptEp10.npz/.png`。

## 训练结果

10/10 epochs 跑满，未触发 early stop，总耗时约 3 h 35 min，吞吐稳定在
1.60～1.63 step/s。

| epoch | train_loss | val_masked_relL2 |
|---:|---:|---:|
| 1 | 0.00116 | 0.58275 |
| 2 | 0.00096 | 0.53001 |
| 3 | 0.00092 | 0.52922 |
| 4 | 0.00086 | 0.55743 |
| 5 | 0.00083 | 0.50534 |
| 6 | 0.00077 | 0.48424 |
| 7 | 0.00070 | 0.44761 |
| 8 | 0.00064 | 0.42285 |
| 9 | 0.00059 | 0.41326 |
| 10 | 0.00055 | **0.40325** |

## Validation checkpoint 选型

协议：每个 `Ep{n}.pth` 使用 156 个 validation 窗口、day-1 deterministic 评估；
指标为原生 C-grid masked pooled RMSE。

| epoch | model RMSE (m/s) | persistence | ratio |
|---:|---:|---:|---:|
| 1 | 0.1380 | 0.1294 | 1.067 |
| 2 | 0.1260 | 0.1294 | 0.974 |
| 3 | 0.1209 | 0.1294 | 0.934 |
| 4 | 0.1251 | 0.1294 | 0.967 |
| 5 | 0.1173 | 0.1294 | 0.907 |
| 6 | 0.1125 | 0.1294 | 0.869 |
| 7 | 0.1074 | 0.1294 | 0.830 |
| 8 | 0.1031 | 0.1294 | 0.797 |
| 9 | 0.1022 | 0.1294 | 0.790 |
| 10 | **0.1011** | 0.1294 | **0.781** |

Ep10 由 validation 选定并冻结，test 未参与选择。

## Test 结果

协议：154 个 test 窗口，`ROLLOUT_DAYS=15`、`EVAL_STRIDE=7`、
`REMASK_FEEDBACK=False`。

| 指标 | model | persistence | ratio |
|---|---:|---:|---:|
| day-1 pooled RMSE | **0.0973** | 0.1167 | **0.833** |
| 15-day overall pooled RMSE | 0.2136 | **0.2098** | 1.018 |
| 15-day u overall RMSE | 0.2651 | **0.2615** | 1.014 |
| 15-day v overall RMSE | 0.1449 | **0.1405** | 1.031 |

确定性基线相较 SD2 diffusion 的 test day-1 `0.2568` 和 overall `0.3442` 有大幅改善；
rho-oracle RMSE `0.0031`，说明 rho→native 映射误差不是主要瓶颈。

## 长时效诊断

在 77 个 stride-14 test 窗口上重放 Ep10 的 15 天 rollout，统计 native-grid bias、
variance ratio 和逐窗口 spatial correlation。

| lead | u ratio | u bias | u variance ratio | u corr model | u corr persistence |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.879 | -0.005 | 0.868 | **0.915** | 0.878 |
| 3 | 0.937 | -0.074 | 0.690 | **0.710** | 0.689 |
| 7 | 1.014 | -0.100 | 0.587 | 0.478 | **0.573** |
| 15 | 1.117 | +0.065 | 0.536 | 0.388 | **0.605** |

证据表明：

1. u 的方差从接近真值快速塌缩到约 0.53～0.60；
2. 空间相关在中段开始低于 persistence；
3. bias 随 rollout 漂移并变号；
4. v 的 late lead 退化更多来自相关损失和正偏差，不完全由方差不足解释；
5. pooled crossover 在全量评估约为 day 5。

## 执行中发现的问题

- test 进程没有显式设置 `CUDA_VISIBLE_DEVICES`，实际与其他任务共享 GPU 0；该确定性
  评估显存约 1.3 GB，数值有效，但后续实验必须明确记录 GPU。
- `scripts/diag_leadtime_residual.py` 保存 NPZ 时 model 与 persistence 使用同名 key，
  persistence 会覆盖 model。PNG、终端输出和本文已记录数字有效；复用 NPZ 前必须修复。

## 结论与后续影响

- 7 天 condition 包含可利用信号，IAFNO backbone 能给出优于 persistence 的次日预测；
- 单步 teacher-forcing 训练不能保持 15 天 rollout 优势，长时效退化是当前主问题；
- 后续应先检验 detached autoregressive multi-step，而不是同时加入扩散、新输入或新 loss；
- 当前困难、验收门槛和执行顺序见
  [当前困难与下一步](../../project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md)。
