# 实验 10 结果：MS5 / MS10

> 运行日期：2026-09-02；单卡 RTX 4090（GPU 0）；协议 = 实验 07 冻结协议 + 训练 horizon 单变量。

## 产物

| 臂 | run 目录 | 训练日志 |
|---|---|---|
| MS5 smoke | `checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S4_C7_SD2_RES_MS5_SMOKE/` | `checkpoints/PRE/train_ms5_smoke1.log`（SMOKE PASS，max_lead=3） |
| MS5 短训 | `checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5/` | `checkpoints/PRE/train_ms5_full.log` |
| MS10 短训 | `checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS10/` | `checkpoints/PRE/train_ms10_full.log` |

评估/诊断日志：`checkpoints/PRE/eval_ms5_val15_ep{1..5}.log`、`eval_ms10_val15_ep{1..3}.log`、
`eval_ms5_test15_ep4.log`、`eval_ms10_test15_ep2.log`、`diag_ms5_ep4_val.log`、`diag_ms10_ep2_val.log`。

## 训练概况

| 臂 | epochs | 每 epoch 耗时 | val_masked_relL2（首→末） | AMP skip | 异常 |
|---|---|---|---|---|---|
| MS5 | 5 | ~30 min | 0.4238 → 0.3897（单调降） | 0 | 无 |
| MS10 | 3 | ~38 min | 0.3965 → 0.3867（单调降） | 0 | 无 |

显存：两者峰值均 ~20.1 GB（24 GB 卡），平稳无增长；温度 ≤64°C。

## validation 选型（15 天自回归，stride 7，154 窗口，native m/s；persistence d1 0.1296 / overall 0.2316）

**MS5**（day-1 守门 ≤0.1031：Ep3 0.1016 ✓ / Ep4 0.1001 ✓ / Ep5 0.0998 ✓；Ep1 0.1062、Ep2 0.1034 未过）：

| ckpt | day-1 (ratio) | 15-day overall (ratio) | u | v | max d10–15 ratio |
|---|---|---|---|---|---|
| Ep3 | 0.1016 (0.784) | 0.1930 (0.833) | 0.841 | 0.806 | 0.874 |
| **Ep4（选型）** | 0.1001 (0.773) | **0.1904 (0.822)** | 0.827 | 0.806 | 0.861 |
| Ep5 | 0.0998 (0.770) | 0.1916 (0.827) | 0.833 | 0.807 | 0.870 |

**MS10**（三个 epoch 全过 day-1 守门）：

| ckpt | day-1 (ratio) | 15-day overall (ratio) | u | v | max d10–15 ratio |
|---|---|---|---|---|---|
| Ep1 | 0.1015 (0.783) | 0.1892 (0.817) | 0.818 | 0.813 | 0.840 |
| **Ep2（选型）** | 0.0997 (0.769) | **0.184301 (0.796)** | 0.799 | 0.787 | 0.813 |
| Ep3 | 0.0992 (0.766) | 0.184418 (0.796) | 0.800 | 0.783 | 0.811 |

Ep2 与 Ep3 overall 仅差 0.06%（0.184301 vs 0.184418），按预注册规则取 overall 更低的
Ep2；Ep3 在 day-1、v、晚段 ratio 上略优，留作复核备份。

## 结构诊断（val，stride 14，77 窗口）

单步基线补测（2026-09-02，同协议跑 `diag_leadtime_residual.py` SPLIT=val，
Ep10；产物在 /tmp，日志 `checkpoints/PRE/diag_exp07_ep10_val.log`）：crossover
d13（pooled ratio 0.785→0.944@d7→1.023@d14）；u corr 0.916→0.500@d7→0.381@d15，
**d7 起低于 persistence**（0.514/0.507）；u bias -0.078@d6 漂移变号至 +0.066@d15；
v d15 ratio 1.097、corr 0.387 vs 0.522。

| 指标（val，u 除非注明） | 单步 Ep10（补测） | MS5 Ep4 | MS10 Ep2 |
|---|---|---|---|
| crossover day | **d13** | **无**（0.76–0.88） | **无**（0.74–0.83） |
| corr_m vs corr_p | **d7 起反超**（d15 0.381 vs 0.507） | 全 lead 占优 | 全 lead 占优（最小差 0.073） |
| u bias 漂移 | -0.078@d6 → +0.066@d15（变号） | 晚段负漂 -0.071@d15 | 稳定 +0.017@d15 |
| u var_ratio @ d1/d7/d15 | 0.807 / 0.541 / 0.521 | 0.808 / 0.353 / 0.302 | 0.798 / 0.417 / 0.337 |

注：MS 臂的 var_ratio 更低（更模糊）但 corr/RMSE 全面更优——多步训练以略增
平滑换取结构保持。

## test（冻结后各运行一次；h15，stride 7，154 窗口）

| 指标（test） | 单步基线（实验 07） | MS5 Ep4 | MS10 Ep2 |
|---|---|---|---|
| day-1 model / pers (ratio) | 0.0973 / 0.1167 (0.833) | 0.0984 / 0.1167 (0.843) | **0.0972 / 0.1167 (0.833)** |
| 15-day overall model / pers (ratio) | 0.2136 / 0.2098 (1.018) | 0.1827 / 0.2098 (0.871) | **0.1759 / 0.2098 (0.838)** |
| u overall ratio | 1.014 | 0.875 | **0.842** |
| v overall ratio | 1.031 | 0.855 | **0.824** |
| 最差 lead ratio | >1（多日） | 0.963 (d15) | **0.894 (d15)** |

## 门槛核对（预注册，val 口径）

- day-1 ≤ 0.1031：MS5 Ep3/4/5、MS10 Ep1/2/3 全过 ✓
- 15-day overall ratio < 0.941：MS5 Ep4 0.822 ✓、MS10 Ep2 0.796 ✓
- u/v 各自 overall ratio < 1.0：全过 ✓
- day 10–15 每日 ratio < 1.0：全过（MS5 最差 0.861、MS10 最差 0.813）✓
- 结构检查：crossover 消除、corr 全面占优；variance collapse 与晚段 bias 保留为观察项 ✓

**结论：首要假设成立**——detached autoregressive multi-step 在近单步显存（20 GB，
与单步同量级）下消除了 day 4–5 crossover；MS10 进一步改善晚段（test overall ratio
1.018 → 0.871 → 0.838；day-1 保持 0.833 不退化）。

## 问题与备注

1. **DDP2 smoke：根因定位并修复后通过（2026-09-03）**。首次尝试（GPU 0+共享
   GPU 1）rank 1 OOM——per-rank 真实峰值 ~17.5-20 GB，共享卡必然失败；改用两块
   全空卡（GPU 0/3）后暴露第二个问题：DDP 报 "Expected to have finished
   reduction in the prior iteration"，缺失梯度的 27 个参数恰为 Linear 族
   （autocast 可缓存权重）而 AFNO 频域层完好。根因 = autocast 权重缓存：no_grad
   反馈 forward 在 autocast 内把 fp16 权重以 detached 状态写入缓存，同批次最终
   带梯度 forward 复用后损失图与这些参数断连。修复 = 反馈 forward 用嵌套
   `autocast(enabled=False)`（fp32 推理，见 Changelog 2026-09-03）。修复后
   **SMOKE PASS**（GPU 0/3，4 updates/rank、无 AMP skip、max_lead=3、
   effective batch 8）。
2. **MS5 Ep1/Ep2 未过 day-1 守门**：低 epoch checkpoint 的 day-1 尚未恢复到基线
   水平即已改善 long lead——多步训练早期以牺牲 day-1 换取反馈适应。
3. **评估驱动器**：`pre_evaluate.py` 全程未被修改（scratch 驱动器内存补丁常量）；
   `EPOCH_OVERRIDES` 为 MS10 临时设 3，已还原 `{}`。
4. 方差塌缩与 d15 附近 ratio 回升（test 0.894）仍存在——后续分支（物理单位 loss
   weighting、direct multi-horizon head）的准入证据见方向文档 §10。
