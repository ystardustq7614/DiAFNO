# DiAFNO / PRE 项目交接概要

> 更新日期：2026-09-01
> 用途：让新成员或 agent 在几分钟内了解项目目标、当前证据、主要困难和接手入口。
> 当前困难与执行顺序以
> [《当前困难与下一步》](./CURRENT_CHALLENGES_AND_NEXT_STEPS.md)为唯一方向文档；
> 实验数字以各实验目录的 `RESULTS.md` 为准。

## 当前结论

项目的首要科学目标是：用过去 7 天的海流场，确定性预测未来 1–15 天的 `u/v`。
当前 surface 模型已经证明 IAFNO backbone 能利用条件信息预测次日流场，但尚未解决
连续 15 天自回归时的误差累积：

- 最佳确定性模型 test day-1 RMSE 为 `0.0973 m/s`，优于 persistence 的
  `0.1167 m/s`（ratio `0.833`）；
- test 15-day overall 为 `0.2136`，略差于 persistence 的 `0.2098`
  （ratio `1.018`）；
- 优势约在 day 4–5 后消失，并出现方差塌缩、空间相关衰减和偏差漂移；
- 静态 mask 输入与逐步 remask 两项消融均没有稳定改善 overall，因此当前保留
  14 个动态 condition 通道、无静态 mask、`remask_feedback=False`；
- full3d 30 层正式训练尚未执行。

下一步不是直接加入扩散或完整 BPTT，而是先建立 detached-autoregressive multi-step
训练：训练时真实回灌模型预测，但只对选定 lead 的最后一步反传，以较低显存成本检验
exposure bias 是否是长时效退化的关键原因。

## 任务与数据

| 项目 | 当前定义 |
|---|---|
| 数据 | 1994–2022，共 10,591 个连续日平均 ROMS/COAWST 场 |
| 空间 | `400×441×30`，30 个地形追随 sigma 层；`k=29` 为 surface，`k=0` 为 bottom |
| 输入 | 过去 7 天共定位后的曲线网格方向 `u/v`，day-major 14 通道 |
| 单步目标 | 下一天 `u/v`，2 通道 |
| 正式预测 | 连续 15 天自回归 rollout |
| 时间切分 | train `[0,8401)`；val `[8401,9496)`；test `[9496,10591)` |
| 正式指标 | 映射回原生 C-grid 后，在 native mask 上计算物理单位 m/s 的 RMSE/MAE |

这里预测的 `u/v` 是原始曲线网格 ξ/η 方向分量，不是 east/north 分量。完整变量、网格、
mask、异常值和归一化说明见 [PRE 数据说明](../data/PRE_ocean_data.md)。

## 已完成到什么程度

### 数据与管线

- 已完成原始数据审计，以及 staggered C-grid `u/v` 到 rho-grid 的共定位；
- 已建立连续时间 split、train-only 归一化、双变量 mask 和 surface/full3d preset；
- 已建立单步训练、15 天自回归评估、persistence/zero/rho-oracle 基线和逐 lead 指标；
- 已有 CPU 合成回归测试、真实数据 smoke、checkpoint 恢复和单卡/DDP 保护；
- 训练与评估都能输出适合服务器长任务监控的进度行。

### 模型与实验结论

| 实验 | 结论 |
|---|---|
| [01 surface SD1 diffusion](../experiments/01_surface_sd1_baseline/RESULTS.md) | 失败，归一化尺度有误 |
| [02 surface SD2 diffusion](../experiments/02_surface_sd2_retrain/RESULTS.md) | 修正尺度后仍未过 persistence |
| [03 sampler/checkpoint 消融](../experiments/03_sampler_ablation/RESULTS.md) | 采样参数只能小幅改善，不能救回模型 |
| [04 SD2 15-day rollout](../experiments/04_surface_sd2_rollout/RESULTS.md) | 所有 lead 均败于 persistence |
| [05 条件与可预测性诊断](../experiments/05_condition_diagnostics/RESULTS.md) | condition 中有可利用信号，模型也确实读取 condition |
| [06 full3d](../experiments/06_full3d/RESULTS.md) | 尚未执行 |
| [07 persistence-residual](../experiments/07_residual_baseline/RESULTS.md) | day-1 明确优于 persistence；15-day overall 仍持平略差 |
| [08 静态 mask 输入 A/B](../experiments/08_static_mask_ablation/RESULTS.md) | 不保留静态 mask；原 14 通道模型更好 |
| [09 remask feedback A/B](../experiments/09_remask_feedback_ablation/RESULTS.md) | 保持 `rf0`；中段改善但长段转差，overall 无增益 |

每个实验的目标、任务与执行状态在 `EXPERIMENT.md`，实际数字、分析和科学结论在
`RESULTS.md`。入口见 [实验索引](../experiments/README.md)。

## 当前主要困难

1. **训练与使用方式不一致**：当前训练只优化真实 condition 下的 day-1，正式使用却把
   自己的预测连续回灌 15 次。
2. **长时效结构退化**：问题不只是 RMSE 变大，还包括方差、空间相关和 bias 同时恶化。
3. **u/v 机制不完全相同**：u 更明显方差不足；v 的长 lead 更受相关损失和正偏差影响。
4. **垂向证据不足**：当前可靠实验集中在 surface，不能直接推断 30 层 full3d 表现。
5. **资源证据不足**：full3d 尚无实测峰值显存、吞吐和 I/O 基线。
6. **诊断产物缺陷**：`scripts/diag_leadtime_residual.py` 保存 NPZ 时 model/persistence
   key 重名；已归档 PNG 和终端统计有效，但复用 NPZ 前必须修复。

## 下一步怎么做

详细门槛、文件改动和执行后回顾表见
[《当前困难与下一步》](./CURRENT_CHALLENGES_AND_NEXT_STEPS.md)。当前顺序是：

1. 审计全 30 层 `u/v` 的尺度、增量、persistence 和归一化压缩；
2. 实现并测试 detached multi-step 训练，同时修复诊断 NPZ key；
3. 从实验 07 Ep10 权重启动 surface MS5，使用 fresh optimizer/scheduler；
4. 只有 MS5 保住 day-1 并改善 15-day validation，才继续 MS10；
5. 再做 surface/middle/bottom 代表层，以及 full3d 资源 probe、K1 smoke、K3 pilot；
6. TBPTT、新输入、loss weighting、direct multi-horizon 和 residual diffusion 都是
   证据触发的后续分支，不与 MS5 同时修改。

## 接手入口

- 当前方向：[当前困难与下一步](./CURRENT_CHALLENGES_AND_NEXT_STEPS.md)
- 实验事实：[实验索引](../experiments/README.md)
- 数据语义：[PRE 数据说明](../data/PRE_ocean_data.md)
- 运行方法：[PRE 运行手册](../operations/PRE_runbook.md)
- 代码/文档变更：[项目 Changelog](./CHANGELOG.md)
- 训练入口：`pre_trainer.py`
- 评估入口：`pre_evaluate.py`
- 数据集：`pre_dataset.py`
- 确定性模型：`pre_models.py`
- 自回归 rollout：`pre_rollout.py`

服务器目标环境是 conda env `diafno`。运行前需实测 `nvidia-smi`、PyTorch/CUDA 状态；
本地仓库不包含服务器上的 4.1 TB 原始数据。`pre_trainer.py` 和 `pre_evaluate.py` 都是
脚本，不应被其他模块 import。

## 文档事实源

- **这份概要**只回答“项目是什么、做到哪里、难点是什么、从哪里接手”，不保存完整
  数据字典、实验表格或分步计划。
- **当前困难与下一步**负责方向、准入门槛、待办顺序和事后回顾，不复制实验结果表。
- **实验文档**一目录一问题：`EXPERIMENT.md` 记录目标、任务、设计和执行状态；
  `RESULTS.md` 记录真实配置、产物、结果、问题与分析。
- **Changelog**记录代码和文档的实现变化及验证，不把代码修改结果写进实验结果。
- **Runbook**只描述当前已经可执行的操作；计划中的环境变量在实现前不得写入 runbook。
