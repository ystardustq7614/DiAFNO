# DiAFNO / PRE 项目交接概要

> 更新日期：2026-09-04
> 用途：让新成员或 agent 在几分钟内了解项目目标、当前证据、主要困难和接手入口。
> 当前困难与执行顺序以
> [《当前困难与下一步》](./CURRENT_CHALLENGES_AND_NEXT_STEPS.md)为唯一方向文档；
> 实验数字以各实验目录的 `RESULTS.md` 为准。

## 当前结论

项目的首要科学目标是：用过去 7 天的海流场，确定性预测未来 1–15 天的 `u/v`。
当前 surface 模型已经证明 IAFNO backbone 能利用条件信息预测次日流场，且
detached-autoregressive multi-step 训练（实验 10）已把长时效误差累积压到
persistence 之下：

- 当前最优模型为 surface **MS10 Ep2**（实验 10）：test day-1 RMSE `0.0972 m/s`
  vs persistence `0.1167`（ratio `0.833`，不退化）；test 15-day overall
  `0.1759` vs `0.2098`（**ratio `0.838`**），day 4–5 crossover 消除；
  ⚠️ 正式 checkpoint 一律用 `..._RES_MS10/Ep2.pth`：run 目录的 `best.pth` 按训练期
  `val_masked_relL2` 对应 **Ep3**，不是选型产物（MS5 正式对应 `Ep4.pth`，同理）；
- 演进链：单步基线（实验 07）overall ratio `1.018`（day 4–5 后失去优势）→
  MS5 Ep4 `0.871` → MS10 Ep2 `0.838`；
- 修复可垂向泛化（实验 11）：bottom 单层 MS5 过全部预注册门槛，test overall
  `0.813`（单步 `0.930`）；middle 按勘误后的预注册规则正式选中 Ep4，test overall
  `0.851`（单步 `1.183`），test 门槛全过；validation gate 5 corr 仅在 d15 边缘
  未过，已裁定接受并如实保留，原 Ep2 test `0.830` 仅作探索性记录；
- 遗留缺陷：方差塌缩（var_ratio ~0.3@d15）、d15 附近 ratio 回升（test 0.894）
  与轻微 bias 漂移仍在；现有证据不足以准入任何后续分支，surface u d15 ratio
  `0.906` 仅作为未来重新评估 loss weighting 的触发观察项；
- 静态 mask 输入与逐步 remask 两项消融均没有稳定改善 overall，因此当前保留
  14 个动态 condition 通道、无静态 mask、`remask_feedback=False`；
- full3d 30 层：画像/资源 probe/K1 smoke/1-epoch pilot 已完成（实测 ≈2.3 h/epoch，
  50 epoch ≈ 5 天；单步峰值 22.6 GB），pilot 无逐层信号；已选 Path B，冻结等待
  独立正式预算，K3 继续按预注册条件阻塞（实验 06）。

当前无待执行实验：等待 full3d 独立预算，或由新证据触发后续分支重新预注册。

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
- 已有 CPU 合成回归测试（pre_smoke_test 59 项）、真实数据 smoke、checkpoint 恢复
  和单卡/DDP 保护；
- 已实现 detached multi-step（MS5/MS10）训练与单卡/DDP2 smoke；
- 训练与评估都能输出适合服务器长任务监控的进度行。

### 模型与实验结论

| 实验 | 结论 |
|---|---|
| [01 surface SD1 diffusion](../experiments/01_surface_sd1_baseline/RESULTS.md) | 失败，归一化尺度有误 |
| [02 surface SD2 diffusion](../experiments/02_surface_sd2_retrain/RESULTS.md) | 修正尺度后仍未过 persistence |
| [03 sampler/checkpoint 消融](../experiments/03_sampler_ablation/RESULTS.md) | 采样参数只能小幅改善，不能救回模型 |
| [04 SD2 15-day rollout](../experiments/04_surface_sd2_rollout/RESULTS.md) | 所有 lead 均败于 persistence |
| [05 条件与可预测性诊断](../experiments/05_condition_diagnostics/RESULTS.md) | condition 中有可利用信号，模型也确实读取 condition |
| [06 full3d](../experiments/06_full3d/RESULTS.md) | 画像/资源 probe/K1/pilot 完成；pilot 无逐层信号，已选 Path B 冻结待独立正式预算，K3 继续阻塞 |
| [07 persistence-residual](../experiments/07_residual_baseline/RESULTS.md) | day-1 明确优于 persistence；15-day overall 仍持平略差 |
| [08 静态 mask 输入 A/B](../experiments/08_static_mask_ablation/RESULTS.md) | 不保留静态 mask；原 14 通道模型更好 |
| [09 remask feedback A/B](../experiments/09_remask_feedback_ablation/RESULTS.md) | 保持 `rf0`；中段改善但长段转差，overall 无增益 |
| [10 multi-step MS5/MS10](../experiments/10_multistep_deterministic/RESULTS.md) | detached multi-step 成立：test overall 1.018→0.871→0.838，crossover 消除 |
| [11 代表层 middle/bottom](../experiments/11_representative_layers/RESULTS.md) | bottom 全门槛 Go（0.813）；middle 正式 Ep4 test 0.851，test 门槛全过，gate 5 d15 边缘缺陷已裁定接受；Ep2 的 0.830 为探索性结果 |

每个实验的目标、任务与执行状态在 `EXPERIMENT.md`，实际数字、分析和科学结论在
`RESULTS.md`。入口见 [实验索引](../experiments/README.md)。

## 当前主要困难

1. **长时效残差缺陷**：crossover 已由 multi-step 消除，但方差塌缩
   （var_ratio ~0.3@d15）、d15 附近 ratio 回升（test 0.894）与轻微 bias 漂移仍在。
2. **full3d pilot 无信号**：1-epoch single-step pilot 训练健康但 60 个逐层 day-1
   ratio 全部 ≈1.000，K3 按预注册条件阻塞；已选 Path B，冻结等待独立正式预算。
3. **full3d 资源门槛高**：实测 ≈2.3 h/epoch（50 epoch ≈ 5 天）、单步峰值 22.6 GB
   （24 GB 卡无同卡推理余量），正式投入需另行冻结预算。
4. **残余长时效缺陷**：MS 后 u/v 不对称已大幅消解，但两变量仍共同存在方差塌缩；
   surface u d15 ratio `0.906` 是未来重新评估 loss weighting 的触发观察项。
5. **DDP+AMP 陷阱**：detached 反馈 forward 必须包在 `autocast(enabled=False)` 内
   （autocast 权重缓存会使 fp16 副本 detached、DDP 梯度规约失败；2026-09-03 已修复，
   新增多步相关代码时需保持警惕）。

## 下一步怎么做

详细门槛和裁定见 [《当前困难与下一步》](./CURRENT_CHALLENGES_AND_NEXT_STEPS.md)：

1. full3d 保持 Path B 冻结；预算落实后，先设计显存/评估成本压缩方案并复核
   per-band 归一化，再重新立项；
2. 六个后续模型分支当前均不立项；只有出现方向文档 §6 定义的新证据时，才按新的
   预注册重新评估。

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
