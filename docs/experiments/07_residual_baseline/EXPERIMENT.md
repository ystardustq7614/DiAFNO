# 实验 07：surface persistence-residual 确定性基线

> 状态：未执行（代码已实现并通过 CPU 回归测试；真实数据训练与评估待服务器执行）

## 实验目的

建立一个**确定性、condition-only、以条件第 7 天为 persistence 基准预测残差**的 IAFNO
基线（`PersistenceResidualIAFNO`），回答当前 diffusion 路径未能回答的问题：backbone
能否把 7 天条件直接映射成足够准确的次日流场。扩散路径保留为对照，本轮不修改
EDM、噪声日程或采样器。

背景依据（见 `docs/project/PROJECT_HANDOFF_SUMMARY.md`）：

- condition-only linear/ridge probe RMSE `0.1177 m/s` 优于 persistence `0.1293 m/s`，
  说明 7 天条件中存在可利用信号；
- SD2 diffusion 真实条件 day-1 RMSE `0.2584 m/s`（persistence 的 2.201 倍），
  空间相关性也低于 persistence；
- 条件扰动实验证明模型会使用条件，但没有转化为准确的空间预测。

## 实验设置

| 项目 | 设置 |
|---|---|
| objective | `persistence_residual`（`DIAFNO_OBJECTIVE`） |
| 模型 | 与 diffusion 相同的 `IAFNODiff` backbone（`surface_smoke`：400×441×1、patch 4×3×1、embed 180、implicit 4、explicit 4）；残差输出头零初始化 |
| 残差基准 | 条件最后一天 `base = cond[:, -2:]`；未训练时输出严格等于 persistence（trainer 启动自检） |
| 损失 | 双变量 rho mask 下的 `masked_mse_loss`（逐样本有效格点均值，batch 均值） |
| 时间嵌入 | 常量 `c_noise = 0.25·ln(0.002)`（`time_sigma` 记录在 checkpoint） |
| 数据/归一化 | 与 SD2 完全一致：连续 split、train-only min-max、不裁剪、`_SD2` stats 缓存 |
| batch/lr | 每卡 4、lr 1e-3、cosine（同 `surface_smoke` 预设）；多卡不缩放 lr |
| run 目录 | `surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES`（smoke/DDP 追加 `_SMOKE`/`_DDP<n>`） |

## 对照与控制变量

- persistence、zero、rho-oracle 基线内建于 `pre_evaluate.py`；ridge/linear probe
  `0.1177 m/s` 作为参考线。
- 与 SD2 diffusion 保持相同数据划分、归一化、mask、训练预算与 GPU 数；
  单一变量是 objective（扩散采样 vs 确定性残差回归）。
- rollout `remask_feedback` 先固定 `False`（历史行为）；单变量 A/B 属 Phase 5，不与本实验捆绑。

## 记录指标

- 训练：masked MSE loss、`val_masked_relL2`（early stop 用）、updates/skips、耗时吞吐。
- 选型：对每个 `Ep{n}.pth` 以 `SPLIT="val"`、`ROLLOUT_DAYS=1` 跑 `pre_evaluate.py`，
  按 **validation day-1 native RMSE** 选 checkpoint（训练日志的 rel-L2 不是选型指标）。
- 报告：day-1 与 15-day 的 native masked RMSE/MAE（pooled）、model/persistence 比值、
  u/v 分项、coastal/open-ocean 分项、空间相关性。

## 执行方法

```bash
# 1) 单卡真实数据 smoke（结构与正式完全一致，仅 4 batch/1 epoch）
CUDA_VISIBLE_DEVICES=<gpu> DIAFNO_OBJECTIVE=persistence_residual python pre_trainer.py

# 2) 目标 world size 的 DDP smoke（与单卡 smoke 产物隔离）
DIAFNO_OBJECTIVE=persistence_residual torchrun --standalone --nproc_per_node=4 pre_trainer.py

# 3) surface 短训练（Phase 3；Go 后才进入全量预算）
CUDA_VISIBLE_DEVICES=<gpu> DIAFNO_OBJECTIVE=persistence_residual \
  DIAFNO_TRAIN_MODE=full python pre_trainer.py

# 4) validation 选型（逐 checkpoint，day-1）
#    pre_evaluate.py: CHECKPOINT='<...>/_RES/EpN.pth'、SPLIT='val'、ROLLOUT_DAYS=1
# 5) 冻结配置后 test 报告（SPLIT='test'、ROLLOUT_DAYS=15）
```

## 预期结果与门槛

- 工程门槛：单卡与 DDP smoke 打印 `SMOKE PASS`；零初始化 identity 自检通过；
  loss/指标 finite；checkpoint 在 `*_RES` 目录完整落盘。
- Phase 3 Go 条件：最佳 **validation day-1 native RMSE 严格优于 persistence**（< `0.1293 m/s`
  的对应 validation 数值）；否则停止扩大预算，先诊断优化/输入/目标。
- Phase 4 第一目标：test day-1 稳定优于 persistence `0.1293 m/s`；进一步目标：达到或优于
  ridge probe `0.1177 m/s`。
- Go 之后的后续决策（不属本实验）：residual diffusion、双静态 mask 输入 A/B、
  remask A/B，再评估 full3d。

## 当前为什么不执行

代码与 CPU 回归测试已完成；真实数据训练需要服务器 GPU 与 `/data2` 数据，
单卡/DDP smoke 尚未运行，validation 选型与 test 报告更无从谈起。

实际状态见 [RESULTS.md](./RESULTS.md)。
