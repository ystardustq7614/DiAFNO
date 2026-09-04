# 当前困难与下一步：确定性 1–15 天 U/V 预测

> 更新日期：2026-09-04
> 状态：**detached multi-step 主线基本执行完毕**——工作包 1–5 完成；工作包 6（full3d）
> 仅完成步骤 1–4（画像/资源 probe/K1 smoke/1-epoch pilot），K3 被预注册门槛阻塞。
> 当前按顺序等待一项修正执行与两项决策：
> 1. **实验 11 middle 勘误修正**：按当前仓库可用归档数据和预注册规则正式改选
>    MS5 Ep4，补跑一次 test；已执行的 Ep2 test 仅保留为探索性结果；
> 2. **full3d K3 / 正式预算路径**（候选 A/B/C）；
> 3. **后续模型分支准入**（loss weighting、direct multi-horizon head 等）。
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
| [11 代表层 middle/bottom](../experiments/11_representative_layers/RESULTS.md) | bottom 全门槛 Go；middle 存在选型门槛勘误（见下） |
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
5. **实验 11 middle 选型勘误（2026-09-04）**：middle probe day-1 原记录 0.770 无法从
   当前仓库可用的归档 NPZ 复现（真值 0.582），导致 MS5 day-1 门槛误算（0.785）；
   复核后门槛内只有 Ep4/Ep5，预注册规则正式选中 Ep4 而非已执行的 Ep2。**待执行**：
   冻结 Ep4 并补一次 test；Ep2 test 仅保留为探索性结果。bottom 层不受影响。
   详见[实验 11 勘误节](../experiments/11_representative_layers/RESULTS.md)。
6. **工程注意（不改变算法）**：detached 反馈 forward 必须包在
   `autocast(enabled=False)` 内（autocast 权重缓存会使 fp16 副本 detached、DDP 梯度
   规约失败；2026-09-03 已修复，DDP2 smoke 通过）。新增多步相关代码时保持该约束。

## 4. 下一执行 1：实验 11 middle 勘误修正（排最前）

middle 的 MS5 长时效修复本身成立（Ep2 test 0.830、结构诊断见实验 11），但其正式
选型合规性被勘误破坏（见 §3.5）。按原预注册规则，正式 checkpoint 固定为 Ep4，
下一步补跑一次 test（test 只运行一次的纪律对该新冻结重新计算）。Ep2 已经看过 test
结果，不能通过事后放宽容差追溯性变成预注册选型，只作为方案偏离后的探索性证据保留。

bottom 层不受影响，无需重做。

## 5. 下一决策 2：full3d K3 / 正式预算（实验 06）

预注册准入条件"训练健康 **且** 逐层 day-1 有可预测信号"只满足前半。
三条候选路径（均需单变量论证，不得与其他改动叠加）：

- **路径 A**：追加 single-step epochs（实测 ≈2.3 h/epoch）直到出现逐层 day-1 信号，
  再进 K3 pilot；成本低但可能多次空转。
- **路径 B**：冻结 full3d，待独立正式预算（实测 ≈5 天/50 epoch，峰值 22.6 GB、
  24 GB 卡无同卡推理余量）后按完整协议重立项；单步峰值显存与逐 epoch 评估成本
  （val h15 ≈2 h05m）需先解决。
- **路径 C**：调参（如 lr/batch）重跑 pilot；破坏与代表层的预算可比性，需单独预注册。

决策时需一并复核：统一 min-max 归一化对底层主体分布的强压缩（底层归一化 std
约为海面 1/3，见[实验 06 全层数据画像](../experiments/06_full3d/RESULTS.md)）是否需要
per-band 归一化（当前未改动）。

## 6. 下一决策 3：后续分支准入（单变量实施，互不叠加）

| 分支 | 准入条件（全部满足才立项） |
|---|---|
| 物理单位 loss weighting | MS 后 u/v 改善明显不对称，且需优化 native m/s 目标 |
| direct multi-horizon head | detached/TBPTT 不能稳定越过长 lead persistence |
| Truncated BPTT（最近 2–3 步） | MS 明确改善 long lead 但 detached 版本停滞 |
| 额外输入变量（zeta/temp/salt/rho 等） | multi-step 仍无法控制 bias/correlation，且数据审计支持增益假设 |
| residual diffusion | 确定性 mean forecast 已稳定，且项目明确需要概率预测 |
| full BPTT | 短 TBPTT 有稳定增益且显存/梯度诊断证明继续加长值得 |

当前指向证据：§3.1–3.3（方差塌缩、d15 回升、bias 漂移）主要支持
loss weighting 与 direct multi-horizon 的评估。

## 7. 执行约定（不变）

- test 在配置与 checkpoint 冻结后只运行一次；不用 test 选 epoch/超参/Go-No-Go。
- 任何新分支先过 `python pre_smoke_test.py` 全部回归，再单卡 smoke → DDP smoke →
  正式短训；运行方法见 [PRE 运行手册](../operations/PRE_runbook.md)。
- 代码/文档实现变化记入 [Changelog](./CHANGELOG.md)；实验结果只写入对应实验目录。
