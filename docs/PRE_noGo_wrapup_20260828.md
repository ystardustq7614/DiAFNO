# DiAFNO / PRE surface SD2 重跑：负面结果收尾报告（No-Go wrap-up）

> 日期：2026-08-28 ｜ 分支：`adapt-weather-ocean`（HEAD `52c0113`，工作树含未提交改动，见 §7）
> T0 = 07:44，收尾 ≈ 17:40（全程 ~10h，其中等卡 1h48m）。
> 前置文档：`docs/PRE_rerun_report_20260828.md`（v2 执行计划，本报告是其执行结果与诊断续篇）。

## 0. 一句话结论

**SD2（sigma_data 0.1712）重跑解决了旧跑的失控尖刺，但未解决"输给 persistence"的根本问题**：
val day-1 最优配置 model/pers ≈ **2.0×**，正式 test 全部 15 个 lead day 均未达标（overall 1.64×）。
经系统诊断，**条件通路本身正确且含足够赢过 persistence 的信息**（线性探针 0.1177 < pers 0.1293），
失败根因指向 **EDM 迭代采样框架无法兑现条件均值 + 模型欠训练**，而非数据/通道/mask/尺度 bug。

## 1. 执行时间线

| 阶段 | 内容 | 结果 |
|---|---|---|
| 0 | 环境核对 + 双冒烟 | PASS |
| — | 等卡（8 卡全满，GPU2/3 被他人占用） | 1h48m 后 GPU2/3 同时空 |
| 1 | 主训练 lr=1e-3，10 epoch（GPU2） | **早停 @5 epoch**，best val_rel **1.5296 @Ep3** |
| — | lr=3e-4 对照（用户自建副本 `DiAFNO_lr3e4`，GPU3） | 早停 @3（best 2.1419 @Ep1）；后经手动续训至 10 epoch，best **2.3331 @Ep7** |
| 2 | checkpoint 筛选（val day-1 ch0_e1） | **Ep3 最优 0.2584**（Ep2 0.2991 / Ep4 0.3305） |
| 3 | sampler 消融（val day-1，Ep3） | ch0_e1 **0.2584** ＜ ch80_e1 0.3234（churn 有害 25%）；ch0_e4 0.2471（仅 +4.4%＜10% → 锁定 E=1） |
| 4 | val day-3 稳定性 | 无发散（model 增长 1.36× vs pers 1.65×，ratio 随 lead 改善） |
| 5 | **正式 test**（唯一一次，ch0_e1，全 154 窗） | **FAIL**：全 15 lead day model/pers ＞1（§3 表） |
| — | σ_max=3 修复实验（val day-1，No-Go 诊断驱动） | **0.2851，比 σ_max=80 更差 → 证伪** |
| — | No-Go 诊断（代码审查 + 4 组探针） | §4-§5 |
| 6 | full3d | **未进入**（test 未通过，按计划取消） |

## 2. 训练曲线（loss.dat，epoch 耗时 s / train_loss / val_masked_relL2）

主训练 lr=1e-3（早停 @5）：

| ep | 1 | 2 | 3(best) | 4 | 5 |
|---|---|---|---|---|---|
| val_rel | 1.651 | 1.683 | **1.530** | 2.181 | 1.581 |

对照 lr=3e-4（续训至 10）：

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7(best) | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val_rel | 2.142 | 3.629 | 2.532 | 2.861 | 2.600 | 2.566 | **2.333** | 2.535 | 2.636 | 2.568 |

→ **低 lr 显著更差**（lr 假设证伪）；两训练均远未达健康阈值 0.8。

## 3. 评估结果汇总（native-grid pooled masked RMSE, m/s；val pers=0.1294，zero=0.2620）

val day-1：

| 配置 | RMSE | ratio |
|---|---|---|
| 主 Ep3 ch0_e1（锁定组合） | **0.2584** | 1.998 |
| 主 Ep3 ch80_e1 | 0.3234 | 2.500 |
| 主 Ep3 ch0_e4 | 0.2471 | 1.911 |
| 主 Ep3 ch0_e1 **σ_max=3** | 0.2851 | 2.204 |
| lr3e4 Ep1 / Ep7(best) / Ep10（e1） | 0.3259 / 0.3779 / 0.3558 | 2.52 / 2.92 / 2.75 |
| lr3e4 Ep7 e4 / Ep10 e4 / ch80 | 0.3057 / 0.4336 / 0.3259 | 2.36 / 3.35 / 2.52 |
| 线性探针（诊断，非模型） | **0.1177** | **0.91（赢 persistence）** |

val day-3（锁定组合）：d1 0.2584 / d2 0.3356 / d3 0.3511（ratio 1.998→1.848→1.648，无发散；
但 d2/d3 差于 zero 0.247/0.251——反馈帧陆地污染，见 §5-3）。

**正式 test（ch0_e1，154 窗 × 15 天，EVAL_SEED=123）**：

| lead | u model/pers | v model/pers |
|---|---|---|
| 1 | 0.2649/0.1387 = **1.91** | 0.2485/0.0895 = **2.78** |
| 5 | 0.4325/0.2581 = 1.68 | 0.2136/0.1403 = 1.52 |
| 10 | 0.4178/0.2801 = 1.49 | 0.2504/0.1459 = 1.72 |
| 15 | 0.4467/0.2670 = 1.67 | 0.2273/0.1463 = 1.56 |

overall：model 0.3442 vs pers 0.2098（**1.64×**）vs zero 0.2568 → **判定 FAIL**（标准：各 lead ＜1）。
输出：`eval_test_h15_ch0_e1_s123_ckptEp3.npz` + figures（路径见 §6）。

## 4. 已排除的假设（全部有证据）

1. **sigma_data 尺度**：修复生效（log `sigma_data=0.17121 (scale 2.000x)`），比旧跑改善（2.7-3.1×→2.0×）但不足。
2. **条件通道 day-major 交错顺序 / 训练-评估不一致**：bit-exact 复核一致（pre_dataset.py:288-301；
   评估反馈 `cat([cur[:,2:], p])` 同序同空间；探针复现正式值 0.2584）。
3. **IAFNODiff 拼接顺序**：cond 在前 noisy 在后（IAFNO.py:290-292），两侧一致。
4. **mask/NaN 污染（day-1）**：NaN→0 严格，loss mask 广播正确；oracle=0.0032 证明评估管线正确。
5. **lr**：3e-4 对照全面更差。
6. **sampler churn**：ch0 优于 ch80（与旧跑相反方向的有害性）。
7. **自回归发散**：day-3 检查通过。
8. **σ_max=80 起步 OOD**（诊断首要假设）：σ_max=3 实验更差 → **单纯缩 σ_max 不是解**。

## 5. 诊断核心证据与现存解释（详见 `~/checkpoints/PRE/diag_noGo_20260828/`）

1. **条件信息充足**：14 维条件线性 ridge → 0.1177（＜persistence 0.1293），权重集中于最后一天通道。
2. **网络确实消费条件**：条件全零 0.4775 / 通道反序 0.5655 / 换窗条件 0.3408，均显著劣于正常 0.2584。
3. **单步去噪在真值盆地内极好**（真 cond：σ=0.34 时 0.043），但**从不逃逸错误盆地**：输入 day-7 场则还你
   day-7 场（0.129-0.136），输入均值场则还均值场（0.235-0.295）——D_θ 学到"投影到最近数据流形"，
   不是 E[y|cond,x] 的全局搜索。
4. **采样轨迹**：高 σ 段 D 保持条件均值质量（0.114-0.12），endgame（σ 2→0.1）D 退化到 ≈RMSE(x)，
   ODE 冻结进坏盆地，FINAL 0.21-0.39 ≫ 中段 0.11。
5. **陆地污染（day-2+）**：采样输出在 land 格（30% 网格）|pred|≈0.56-0.81，而训练 cond 陆地恒 0；
   反馈帧陆地置零消融 d2 −7.7%（pre_rollout.py:50 处一行修复，未应用）。
6. **欠训练**：5/10 epoch 早停，patch_embed 权重仍近初始化（mean|w|≈0.037 vs init≈0.036），train_loss 仍在降。

**工作解释**：近确定性回归问题（线性即可赢）+ 扩散采样从纯噪声出发的脆弱轨迹 + 欠训练的弱条件化
→ 单样本落在"条件均值场 + 噪声盆地"附近，RMSE ≈ 均值场水平（0.253-0.26），ensemble 平均收益天然有限（4.4%）。

## 6. 交付物清单（实测存在）

| 物 | 路径 |
|---|---|
| 主训练 checkpoints + 曲线 | `~/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/{Ep1-5,best}.pth, loss.dat` |
| checkpoint 筛选（≤3 候选） | 同目录 `eval_val_h1_ch0_e1_s123_ckptEp{2,3,4}.npz` + figures |
| sampler 消融 ①②③ | 同目录 `...ch0_e1.../ch80_e1.../ch0_e4...Ep3.npz` + figures |
| σ_max=3 实验 | 同目录 `eval_val_h1_ch0_e1_s123_ckptEp3_sigmax3.npz` + figures |
| val day-3 | 同目录 `eval_val_h3_ch0_e1_s123_ckptEp3.npz` + figures |
| **正式 test（FAIL 记录）** | 同目录 `eval_test_h15_ch0_e1_s123_ckptEp3.npz` + `figures_h15_.../` |
| lr=3e-4 对照全套（含续训 Ep4-10 与其消融） | `~/checkpoints/PRE_lr3e4/surface_smoke_..._SD2/`（Ep1-10、loss.dat、eval npz×7、figures、log×8） |
| 日志 | `~/checkpoints/PRE/{train_surface_sd2, eval_screen_val_day1, eval_ablation, eval_test_final, eval_sigmax3}.log` |
| **诊断产物（脚本+结果）** | `~/checkpoints/PRE/diag_noGo_20260828/`（自 /tmp/opencode/diag 固化，64KB） |
| full3d | 未进入（test FAIL，按计划取消；无目录） |
| 旧反面基线（未动） | `~/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7/` |

## 7. 代码与状态备注（未提交，均未 commit）

- `pre_trainer.py`：`EPOCH_OVERRIDES = {"surface_smoke": 10}`（阶段 1 临时配置）。
- `pre_evaluate.py`：新增 `SAMPLER_SIGMA_MAX`（默认 3；设 None 恢复 EDM 默认 80）+ 加载行打印 sigma_max；
  **临时配置残留**：`SPLIT="val"`、`ROLLOUT_DAYS=1`、`OUTPUT_TAG="sigmax3"`——复用前需按需重设。
- `docs/PRE_rerun_report_20260828.md`：v2 编辑。
- 独立副本 `/data2/user/zyq/projects/DiAFNO_lr3e4/`：仅 `OUT_ROOT`+`lr=3e-4` 两处 diff（另含用户续训期间的改动）。
- `RESUME_SIGMA_POLICY` 保持默认 `error`；`MAX_WINDOWS` 全程 None；test 仅阶段 5 触碰一次（另有一次
  σ_max=3 属 val——test 纪律未破坏）。
- 收尾时 GPU 2 有他人进程（12.7GB），GPU 3 空闲；无我方进程残留。

## 8. 若未来重启（建议优先级，均未实施）

1. **重训**：P_mean/P_std 上移使训练 σ 覆盖对齐 sigma_max（或反之），epoch 20+ 且放宽早停，
   考虑 cond 同映射到 [-1,1]；时间嵌入维度 16 偏小可加大。
2. **反馈帧陆地置零**（pre_rollout 一行，day-2+ 收益 ~8%，day-1 不受影响）。
3. **采样侧**：persistence-init / SDEdit 式起点（x₀=day-7+σ_noise）未测试——但盆地探针显示收益上限≈persistence 平价。
4. **基线参照**：线性 ridge 0.1177 应作为报告表格中的强基线——任何扩散方案需先赢它再谈赢 persistence。
5. 评估口径统一：训练期 val（churn=80、24 窗）与正式口径（ch0、全窗）差异大，checkpoint 选择应尽快切到正式口径。
