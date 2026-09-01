# 实验 07：surface persistence-residual 确定性基线 — 结果

> 状态：smoke、Phase 3 短训练、validation day-1 选型、Phase 4 test 报告、
> 长时效诊断、Phase 5①（mask 输入 A/B → 保留 A）与 Phase 5②（remask A/B →
> 维持 rf0）均已执行（2026-08-31 / 09-01）。Phase 3 **Go**；test day-1 优于
> persistence（0.833）、15-day overall 持平（1.018）；Phase 5 两项均判
> "不保留"。剩余：Phase 6 residual diffusion / full3d 决策。

## 执行环境与产物

- 服务器 conda env `diafno`（Python 3.10.20, torch 2.4.1+cu124），GPU 1（RTX 4090 24G）单卡。
- run 目录：`/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/`
  （`Ep1.pth`~`Ep10.pth`、`best.pth`、`loss.dat`，共 741M）。
- 训练日志：`~/checkpoints/PRE/train_residual_full_surface.log`；
  smoke 日志：`train_residual_smoke_single.log`、`train_residual_smoke_ddp2.log`；
  选型日志：`eval_val_day1_selection_RES.log`。

## smoke（Phase 2，均已通过）

- 单卡（`DIAFNO_OBJECTIVE=persistence_residual`，默认 smoke 模式）：`SMOKE PASS`，
  4 updates/rank、skipped 0、finite train/val、零初始化 identity 自检通过；
  `train_loss 0.03270`，`val_masked_relL2 1.75240`；产物目录
  `surface_smoke_BS4_EMD180_I4_E4_S4_C7_SD2_RES_SMOKE/`。
- DDP world size 2（GPU 1+2，`torchrun --standalone --nproc_per_node=2`）：`SMOKE PASS`，
  每 rank 4 updates、skipped 0，进度行 `scope=rank0_shard_of_2`（rank 0 独占输出），
  仅 rank 0 写 checkpoint；`train_loss 0.02694`，`val_masked_relL2 1.66825`；产物目录
  `surface_smoke_BS4_EMD180_I4_E4_S4_C7_SD2_RES_SMOKE_DDP2/`。
- 备注：smoke 末行 `lr=0.00e+00` 为 cosine `T_max=1×4 步` 退火到底的设计行为，非异常。

## 短训练（Phase 3，已完成）

单卡 GPU 1，`DIAFNO_TRAIN_MODE=full`，10/10 epochs 跑满（未触发 early stop），
总耗时 12894.3 s ≈ 3 h 35 min，吞吐稳定 1.60~1.63 step/s（2098 步/epoch）。

| epoch | train_loss | val_masked_relL2 | 备注 |
|---|---|---|---|
| 1 | 0.00116 | 0.58275 | best |
| 2 | 0.00096 | 0.53001 | best |
| 3 | 0.00092 | 0.52922 | best |
| 4 | 0.00086 | 0.55743 | 恶化（early-stop 计数 1/2） |
| 5 | 0.00083 | 0.50534 | best（计数清零） |
| 6 | 0.00077 | 0.48424 | best |
| 7 | 0.00070 | 0.44761 | best |
| 8 | 0.00064 | 0.42285 | best |
| 9 | 0.00059 | 0.41326 | best |
| 10 | 0.00055 | **0.40325** | best（最终模型） |

## validation day-1 选型（Go/No-Go，已完成）

协议：逐个 `Ep{n}.pth` 用 `pre_evaluate.py`（`SPLIT="val"`、`ROLLOUT_DAYS=1`、
确定性模型强制 `ENSEMBLE_SIZE=1`、`REMASK_FEEDBACK=False`）评估；156 个 val 窗口
（stride 7）；指标为原生交错网格 masked pooled RMSE。10 轮全部 exit=0，产物
`eval_val_h1_ch0_e1_s123_rf0_ckptEp{n}.npz` + 对应 figures 目录。

| epoch | model d1 RMSE (m/s) | pers d1 RMSE | ratio |
|---|---|---|---|
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

## test 报告（Phase 4，已完成）

配置冻结：`CHECKPOINT=Ep10.pth`（validation day-1 选出）、`SPLIT="test"`、
`ROLLOUT_DAYS=15`、`EVAL_STRIDE=7`（154 个窗口，horizon 15 每窗需 22 天）、
确定性模型强制 `ENSEMBLE_SIZE=1`、`REMASK_FEEDBACK=False`。单次运行
`status=completed`（~6.7 min），零异常。产物：
`eval_test_h15_ch0_e1_s123_rf0_ckptEp10.npz` + `figures_h15_..._ckptEp10/`。

```
mode | d1 RMSE | pers d1 | ratio | overall RMSE | pers overall | ratio
 model | 0.0973 | 0.1167 | 0.833 | 0.2136 | 0.2098 | 1.018
  zero | 0.2640 | 0.1167 | 2.263 | 0.2568 | 0.2098 | 1.224
oracle | 0.0031 | 0.1167 | 0.027 | 0.0031 | 0.2098 | 0.015
```

- **test day-1 优于 persistence**（Phase 4 第一目标达成）：`0.0973 m/s` vs
  `0.1167 m/s`（ratio 0.833，改善约 17%）。
- **15-day overall 与 persistence 基本持平**：`0.2136` vs `0.2098`（ratio 1.018，
  略差 1.8%）——day-1 优势随 lead time 递减，长时效自回归误差累积仍未解决。
- 与扩散路径对照：SD2 diffusion test d1 `0.2568` / overall `0.3442`；本基线
  d1 改善约 2.6 倍、overall 改善约 1.6 倍。
- rho-oracle `0.0031` 再次确认 rho→native 映射误差可忽略，差距来自模型本身。

## 长时效误差诊断（2026-08-31，追加）

目的：解释 test day-1 优势（ratio 0.833）为何在长时效衰减为 overall 持平（1.018）。
方法：`scripts/diag_leadtime_residual.py` 复用官方 rollout 协议（Ep10、test、
确定性、`REMASK_FEEDBACK=False`）在 77 个 stride-14 窗口上重放 15 天 rollout，
逐 lead day 统计 native 网格上的 signed bias、方差比（pred/truth，<1 即模糊化）
与逐窗口空间相关（评估 NPZ 不存这三类量）。产物：
`leadtime_diag_ckptEp10.npz` / `.png`（run 目录内）。

核心数字（u 分量）：

| lead | ratio | bias_m | var_ratio_m | corr_m | corr_p |
|---|---|---|---|---|---|
| 1 | 0.879 | -0.005 | 0.868 | **0.915** | 0.878 |
| 3 | 0.937 | -0.074 | 0.690 | 0.710 | 0.689 |
| 7 | 1.014 | -0.100 | 0.587 | 0.478 | 0.573 |
| 15 | 1.117 | **+0.065** | 0.536 | 0.388 | 0.605 |

结论（按证据强度排序）：

1. **方差塌缩（模糊化）是主导问题**：模型方差比从 d1 的 0.87 跌至 d7 起的
   ~0.53-0.60（u）——MSE 确定性回归的典型均值回归；d1-3 模糊尚轻（0.74-0.87），
   与 ratio<1 的优势期吻合。
2. **空间相关中段即低于 persistence**：u 的逐窗口空间相关 d7 起模型（0.48）
   低于 persistence（0.57），d15 仅 0.39 vs 0.61——模型失去大尺度形态，而
   persistence 保有缓变背景场；这比 RMSE 比值更早、更清晰地揭示劣化。
3. **偏差漂移并变号**：u 的 bias 从 d1 的 -0.005 漂移到 d4-7 的 ~-0.11，再变号为
   d15 的 +0.065；v 同步正漂（d14 +0.047）——模糊/含误差的预测回灌条件窗造成的
   系统性漂移，与自回归反馈污染一致。
4. **u/v 不对称**：v 的方差比 d15 回升到 1.03（非模糊主导），其长时效劣化更多
   来自相关损失（0.40 vs 0.62）与正偏差；u 则始终方差不足。
5. 交叉点：pooled ratio 首次 >1 在 d4（本 77 窗子集）/ d5（154 窗全量），d5-12
   在 1.0 附近徘徊，d13-15 明确恶化（1.10-1.15）。

对后续方向的含义（记录，不擅自决策）：模糊化 + 相关塌缩正是「确定性回归 vs
生成式采样」的分界证据，直接支撑修改计划 Phase 6 预留的 residual diffusion 讨论；
`remask_feedback` A/B 与双静态 mask 输入（Phase 5）对偏差漂移可能有边际改善，
但不恢复方差。

## Phase 5① 双静态 mask 输入 A/B（2026-08-31，追加）

设计（依 CODE_MODIFICATION_PLAN §3.4）：A = 14 通道（本实验已有 run）；
B = 14 动态通道 + 2 个静态 mask 通道（`mask_u_rho`/`mask_v_rho`，非交集）。
实现：`DIAFNO_STATIC_MASK=1` 启用；静态通道经 `pre_rollout` 的 `static_cond`
参数单独前传（滑窗保持纯动态 14 通道，persistence base 语义不变）；run 目录
追加 `_MSK`；checkpoint 记录 `static_mask_input`/`model_cond_chans`，评估端按
元数据自动重建（`pre_smoke_test.py` 新增 2 项回归测试，共 47 项 PASS）。

**B 臂执行**：单卡 smoke `SMOKE PASS`（零初始化 identity 含静态通道校验通过）；
10/10 epochs 训练完成（3 h 36 min，best `val_masked_relL2 0.40038`@ep10，
全程单调改善无 early-stop；与 A 臂逐 epoch 交替领先）。

**validation day-1 选型对比**（156 窗口，确定性评估，persistence 0.1294）：

| Ep | A d1 RMSE | A ratio | B d1 RMSE | B ratio |
|---|---|---|---|---|
| 1 | 0.1380 | 1.067 | 0.1492 | 1.153 |
| 2 | 0.1260 | 0.974 | 0.1315 | 1.017 |
| 3 | 0.1209 | 0.934 | 0.1287 | 0.995 |
| 4 | 0.1251 | 0.967 | **0.1248** | **0.965** |
| 5 | 0.1173 | 0.907 | 0.1221 | 0.944 |
| 6 | 0.1125 | 0.869 | 0.1197 | 0.925 |
| 7 | 0.1074 | 0.830 | 0.1107 | 0.856 |
| 8 | 0.1031 | 0.797 | 0.1059 | 0.818 |
| 9 | 0.1022 | 0.790 | 0.1047 | 0.810 |
| 10 | **0.1011** | **0.781** | 0.1024 | 0.792 |

**区域分解**（val day-1，156 窗口，coastal = 距陆地 ≤5 格；产物
`region_diag_ckptEp10.npz`，脚本 `scripts/diag_region_breakdown.py`；
model/persistence ratio）：

| arm | coastal u | coastal v | offshore u | offshore v |
|---|---|---|---|---|
| A | **0.854** | **0.891** | **0.770** | **0.795** |
| B | 0.864 | 0.900 | 0.782 | 0.801 |

**决策：B 臂不保留（A/B → A）**。判据"相同验证协议下稳定改善"未满足：A 最优
0.1011 < B 最优 0.1024，且 10 个 epoch 中 9 个 A 领先、区域分解 4 项全部 A 优。
静态 mask 通道未带来可测改善——IAFNO 的 FFT 全局混合下，归一化后陆地填 0 的
动态通道可能已隐式携带了大部分掩膜信息。近岸改善幅度整体小于离岸
（A coastal 0.867 vs offshore 0.777），近岸误差是后续改进的明确靶点，但
静态 mask 输入不是其解法。B 臂产物保留在 `..._RES_MSK/` 供复核，不进入后续流程。

## Phase 5② rollout 陆地回灌修正 A/B（2026-09-01，追加）

设计：同一 checkpoint（A 臂 Ep10）在 validation 跑 15 天确定性 rollout，
rf0 = 历史行为（预测整帧回灌，含陆地填值）；rf1 = 每步预测先重应用双变量
rho mask（陆地置 0）再回灌。`pre_evaluate.py` 原生支持（`REMASK_FEEDBACK` +
`rf{0|1}` tag + npz 元数据），无需改代码；统一 `OUTPUT_TAG="rfab"` 避免与
test 报告的图目录同名冲突。每臂 154 窗口 ~2.6 分钟，两臂 exit=0，产物
`eval_val_h15_ch0_e1_s123_rf{0,1}_ckptEp10_rfab.npz`（被中断的首轮半成品
npz/空图目录已清理）。

逐 lead day pooled RMSE（rf1/rf0 与对 persistence 比值）：

| lead | rf0 | rf1 | rf1/rf0 | rf0/pers | rf1/pers |
|---|---|---|---|---|---|
| 1 | 0.1430 | 0.1430 | 1.000 | 0.780 | 0.780 |
| 2 | 0.2076 | 0.2066 | 0.995 | 0.807 | 0.803 |
| 3 | 0.2516 | 0.2433 | 0.967 | 0.831 | 0.804 |
| 4 | 0.2877 | 0.2707 | 0.941 | 0.897 | 0.844 |
| 5 | 0.3091 | 0.2885 | 0.933 | 0.929 | 0.867 |
| 6 | 0.3218 | 0.2973 | 0.924 | 0.943 | 0.871 |
| 7 | 0.3162 | 0.2912 | **0.921** | 0.943 | 0.869 |
| 8 | 0.3321 | 0.3157 | 0.951 | 0.946 | 0.899 |
| 9 | 0.3267 | 0.3273 | 1.002 | 0.949 | 0.951 |
| 10 | 0.3207 | 0.3356 | 1.046 | 0.957 | 1.002 |
| 11 | 0.3221 | 0.3472 | 1.078 | 0.947 | 1.021 |
| 12 | 0.3338 | 0.3572 | 1.070 | 0.962 | 1.029 |
| 13 | 0.3451 | 0.3612 | 1.046 | 0.974 | 1.019 |
| 14 | 0.3556 | 0.3609 | 1.015 | 1.013 | 1.028 |
| 15 | 0.3702 | 0.3874 | 1.046 | 1.002 | 1.049 |

overall（脚本 pooled）：rf0 0.2180（ratio 0.941）vs rf1 0.2183（0.943）。

**结论：不保留 rf1（默认维持 rf0 历史行为）**。rf1 呈现清晰的分段效应：
- day 1 完全一致（无反馈，预期）；
- **day 2-8 中期稳定改善**（-0.5%~-7.9%，day 7 最大 -7.9%）；
- **day 9-15 长期转差**（+1.5%~+7.8%，day 10-15 rf1 对 persistence 比值越过 1.0）；
- 15 天 overall 持平略差（+0.14%）。
不满足"相同验证协议下稳定改善"判据。判读：海洋边界置零在中期抑制了陆地
填值回灌的污染，但长期造成条件窗的人为不连续；**逐 lead 分段 remask**
（仅中段启用）理论上可兼得，但属新变体，不在本轮单变量范围内。

**远端"day-2 改善约 7.7%"声明复核（HANDOFF 未完成项 5）**：未复现——
本协议下 day-2 仅 -0.49%（0.2076→0.2066）；与 7.7% 同量级的改善出现在
**day 4-7**（-5.9%~-7.9%）。远端数字的 lead 定义/配置不可考，其方向
（中期改善）与本复现一致，数值归属更正为 day 4-7。

## 结论

- **Phase 3 Go**：最佳 checkpoint Ep10 的 validation day-1 native RMSE `0.1011 m/s`
  严格优于 persistence `0.1294 m/s`（ratio 0.781，改善约 22%），也优于 ridge probe
  参考线 `0.1177 m/s`。ratio 自 Ep2 起持续 < 1 并随 epoch 单调下降。
- **Phase 4 test**：day-1 `0.0973 m/s` 优于 persistence（ratio 0.833）；15-day
  overall `0.2136 m/s` 与 persistence（0.2098）基本持平（ratio 1.018）。
- 对比扩散路径：SD2 diffusion 的 day-1 RMSE 为 persistence 的 2.201 倍；确定性
  persistence-residual 基线在同一数据/预算下反超 persistence，说明 backbone 具备
  把 7 天条件映射为次日流场的能力，问题主要在扩散生成路径而非条件信号。
- 边界与备注：test 评估进程未设 `CUDA_VISIBLE_DEVICES`，实际落在 GPU 0 与他人
  任务共存（显存仅 ~1.3G，确定性评估数值不受影响，记录以保持透明）；
  `pre_evaluate.py` 临时改动已恢复。
- 后续（另行决策，未执行）：Phase 5 双静态 mask 输入与 remask A/B（长时效 overall
  未过 persistence，也可先做条件/残差诊断）；residual diffusion 与 full3d 讨论。
