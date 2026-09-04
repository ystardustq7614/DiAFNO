# 当前困难与下一步：确定性 1–15 天 U/V 预测

> 更新日期：2026-09-04
> 状态：**detached multi-step 主线基本执行完毕**——工作包 1–5 完成；工作包 6（full3d）
> 仅完成步骤 1–4（画像/资源 probe/K1 smoke/1-epoch pilot），K3 被预注册门槛阻塞。
> 三项待办已全部闭环（2026-09-04）：
> 1. **实验 11 middle gate-5 裁定**：勘误修正已执行（正式 Ep4 test 0.851，test 门槛
>    全过），gate 5 d15 边缘缺陷已裁定接受（见 §4）；
> 2. **full3d 预算路径**：已定 **Path B——冻结待独立正式预算**，重启前置项见 §5；
> 3. **后续模型分支准入**：评估完成，**六分支全部不满足准入**，无新立项（见 §6）。
> 当前无待执行实验：等待 full3d 独立预算，或新证据按 §6 触发条件重开分支评估。
> 历史实施计划（含已完成的算法/文件/运行入口原文）归档于
> [archive/MULTISTEP_PLAN_20260901.md](./archive/MULTISTEP_PLAN_20260901.md)。
> 实验数字一律以下列 `RESULTS.md` 为准，本文不复制结果表。

## 1. 任务语义（冻结）

- 数据：1994–2022 共 10,591 个连续日平均 ROMS/COAWST 场，`400×441×30` 地形追随
  sigma 层（k=0 海底、k=29 海面）；预测对象是共定位后**曲线网格方向** u/v（非
  east/north），正式指标映射回原生 C-grid、用未裁剪物理真值计算 masked RMSE/MAE。
- 输入：过去 7 天 u/v（day-major 14 通道）；正式评估：15 天自回归 rollout。
- 切分：train `[0,8401)` / val `[8401,9496)` / test `[9496,10591)`；任何窗口不跨 split。
- 数据语义详见 [PRE 数据说明](../data/PRE_ocean_data.md)。

## 2. 当前证据（全部已完成，数字见各 RESULTS.md）

| 实验 | 一句话结论 |
|---|---|
| [07 单步 residual 基线](../experiments/07_residual_baseline/RESULTS.md) | day-1 优于 persistence；15-day overall 持平略差，d4–5 后 crossover |
| [08 静态 mask 输入 A/B](../experiments/08_static_mask_ablation/RESULTS.md) | 不保留 `_MSK`：原 14 通道更好 |
| [09 remask feedback A/B](../experiments/09_remask_feedback_ablation/RESULTS.md) | 保持 `rf0`：中段改善但长段转差，overall 无增益 |
| [10 detached multi-step](../experiments/10_multistep_deterministic/RESULTS.md) | 假设成立：MS5/MS10 消除 crossover，test overall ratio 1.018→0.871→0.838 |
| [11 代表层 middle/bottom](../experiments/11_representative_layers/RESULTS.md) | bottom 全门槛 Go；middle 勘误修正已执行（正式 Ep4 test 0.851，test 门槛全过），但 gate 5 corr 在 d15 边缘未过（见下） |
| [06 full3d](../experiments/06_full3d/RESULTS.md) | probe/K1 smoke/1-epoch pilot 完成；pilot 无逐层 day-1 信号，K3 阻塞 |

**当前最优模型**：surface **MS10 `Ep2.pth`**
（`checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS10/Ep2.pth`，
test 15-day overall ratio 0.838、day-1 0.833）。
⚠️ run 目录内的 `best.pth` 按训练期 `val_masked_relL2` 产生（MS10 对应 Ep3），
**不是**选型产物，禁止用于正式评估——正式选型协议见 Runbook 第 4 节。

**已定的消融结论（不再重开）**：14 动态条件通道、无静态 mask 输入、
`remask_feedback=False`（rf0）。

## 3. 未解决问题（后续工作的指向证据）

1. **方差塌缩**：MS 臂 var_ratio 在 d15 降至 ~0.3（u 更明显），预测偏平滑。
2. **d15 附近 ratio 回升**：test 最差 lead ~0.894（MS10 Ep2），仍是全 rollout 最弱点。
3. **轻微 bias 漂移**：晚段 u/v 各有系统偏移（幅度与符号见实验 10/11 诊断）。
4. **full3d 1-epoch pilot 无逐层信号**：60 个 layer×变量 day-1 ratio 全部 ≈1.000，
   属证据不足而非证伪（对照代表层"Ep1 无技能"轨迹）。
5. **实验 11 middle 选型勘误（2026-09-04，修正已执行）**：middle probe day-1 原记录
   0.770 无法从当前仓库可用的归档 NPZ 复现（真值 0.582），导致 MS5 day-1 门槛误算
   （0.785）；复核后门槛内只有 Ep4/Ep5，预注册规则正式选中 Ep4 而非已执行的 Ep2。
   **已执行**：正式 Ep4 test 0.851（test 门槛全部通过），Ep2 test 仅保留为探索性
   结果；但 Ep4 的 val 结构诊断 gate 5 corr 在 d15 边缘未过（u 0.428 vs 0.430、
   v 0.417 vs 0.423），按预注册字面不记"全门槛 Go"→ 转入 §4 裁定。
   bottom 层不受影响。详见[实验 11 勘误节](../experiments/11_representative_layers/RESULTS.md)。
6. **工程注意（不改变算法）**：detached 反馈 forward 必须包在
   `autocast(enabled=False)` 内（autocast 权重缓存会使 fp16 副本 detached、DDP 梯度
   规约失败；2026-09-03 已修复，DDP2 smoke 通过）。新增多步相关代码时保持该约束。

## 4. 已裁定 1：实验 11 middle gate-5 接受边缘缺陷（2026-09-04）

勘误修正已执行（2026-09-04）：正式选型 Ep4 的 test 0.851 全部通过 test 门槛
（day-1 0.665、u/v 各自 < 1.0、d10–15 每日 < 1.0、无 crossover），middle 的 MS5
长时效修复在合规选型下成立；Ep2（0.830）仅作探索性证据保留。

gate 5 裁定（2026-09-04）：**接受边缘结构缺陷**——Ep4 的 val corr 仅在 d15 以
极小差距低于 persistence（u 0.428 vs 0.430、v 0.417 vs 0.423；77 窗均值口径，
lead 1–14 全占优），middle 层记"gate 1–4 + test 全过、gate 5 边缘未过"。不通过
事后放宽容差改写预注册门槛；不改选 Ep5（预注册选型规则只定义"过 day-1 门槛者中
val h15 overall 最低"，无 fallback）；不回退 Ep2（day-1 未过门槛，已被勘误排除）。
作为既存事实转入 full3d/分支决策。

bottom 层不受影响，无需重做。

## 5. 已决策 2：full3d 选定 Path B——冻结待独立正式预算（2026-09-04）

预注册准入条件"训练健康 **且** 逐层 day-1 有可预测信号"只满足前半；三条候选路径
（A 追加 single-step epochs / B 冻结待预算 / C 调参重跑 pilot）中已选定 **Path B**：
full3d 冻结，不排任何训练，待独立正式预算（实测 ≈5 天/50 epoch，峰值 22.6 GB、
24 GB 卡无同卡推理余量）落实后按完整协议重立项（K3 仍按预注册条件阻塞）。

重启前置项（预算落实后、重立项前完成，均不占训练卡）：
- 单步峰值显存（22.6 GB）与逐 epoch 评估成本（val h15 ≈2 h05m）的压缩方案；
- 复核统一 min-max 归一化对底层主体分布的强压缩（底层归一化 std 约为海面 1/3，见
  [实验 06 全层数据画像](../experiments/06_full3d/RESULTS.md)）是否需要 per-band 归一化
  （当前未改动）。

## 6. 已评估 3：后续分支准入——六分支全部不满足（2026-09-04，纯归档证据复算）

评估方法：只读归档 NPZ（surface MS10 Ep2 与单步 Ep10、middle Ep4、bottom Ep5 的
test eval NPZ 与 val `leadtime_diag` NPZ；单步 Ep10 旧格式 diag NPZ 按 2026-09-02
勘误纪律不复用，引[实验 07 RESULTS](../experiments/07_residual_baseline/RESULTS.md)
归档数字），不跑任何新评估/训练。关键数字：

- **u/v 不对称已被 MS 大幅消解**（test h15 overall u / v）：单步 Ep10
  1.014 / 1.031（长 lead v 明显更差，d14 v 1.188 vs u 1.145）→ MS 后
  surface 0.842 / 0.824（gap −0.018）、middle 0.851 / 0.850（−0.000）、
  bottom 0.811 / 0.820（+0.009）；逐 lead |v−u| ≤ 0.058 且三层方向不一致。
- **detached 已稳定越过长 lead persistence**：三层 test 全部 15 天 ratio < 1
  （最差 surface 0.906 / middle 0.979 / bottom 0.880，均在 d15），val 无
  crossover，corr 除 middle d15 边缘（−0.002/−0.006）外全 lead 占优。
- **残余缺陷为两变量共有而非不对称**：val var_ratio@d15 surface u 0.337 / v 0.425、
  middle u 0.262 / v 0.372、bottom u 0.305 / v 0.371；bias 漂移最大为 middle u
  −0.050@d15（相对 persistence d15 误差 0.218 仍小）。

| 分支 | 准入条件（全部满足才立项） | 判定 |
|---|---|---|
| 物理单位 loss weighting | MS 后 u/v 改善明显不对称，且需优化 native m/s 目标 | **不成立**：MS 后 overall gap ≤0.018、逐 lead ≤0.058 且三层方向不一致；"native m/s 目标"虽为真，但无不对称证据支持重加权收益 |
| direct multi-horizon head | detached/TBPTT 不能稳定越过长 lead persistence | **不成立**：三层 test 全部 15 天 < 1 |
| Truncated BPTT（最近 2–3 步） | MS 明显改善 long lead 但 detached 版本停滞 | **不成立**：detached 长 lead 明显改善且无停滞（选型 epoch 趋势仍在改善） |
| 额外输入变量（zeta/temp/salt/rho 等） | multi-step 仍无法控制 bias/correlation，且数据审计支持增益假设 | **不成立**：corr 全 lead 占优（middle d15 边缘除外）、ratio 全 <1、bias 可控；数据审计增益假设未做 |
| residual diffusion | 确定性 mean forecast 已稳定，且项目明确需要概率预测 | **不成立**：任务语义为确定性点预报（§1 冻结），无概率预测需求 |
| full BPTT | 短 TBPTT 有稳定增益且显存/梯度诊断证明继续加长值得 | **不成立**：前提（TBPTT 立项）缺失 |

再评估触发条件（新证据出现时按新预注册重开，不追溯）：若后续工作专攻 u 分量
d15 rebound（surface u 0.906 @d15）并产生新的 u/v 不对称证据，loss weighting
可重新评估。

## 7. 执行约定（不变）

- test 在配置与 checkpoint 冻结后只运行一次；不用 test 选 epoch/超参/Go-No-Go。
- 任何新分支先过 `python pre_smoke_test.py` 全部回归，再单卡 smoke → DDP smoke →
  正式短训；运行方法见 [PRE 运行手册](../operations/PRE_runbook.md)。
- 代码/文档实现变化记入 [Changelog](./CHANGELOG.md)；实验结果只写入对应实验目录。
