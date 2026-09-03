# 实验 11 结果：代表层（middle=14 / bottom=0）

> 运行日期：2026-09-03；单卡 RTX 4090；协议 = 实验 07/10 冻结协议，单变量 = depth_index。
> 预注册见 `EXPERIMENT.md`（门槛在开跑前冻结）。

## 产物

| 臂 | run 目录 | 日志 |
|---|---|---|
| middle 单步 probe | `checkpoints/PRE/middle_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/` | `train_mid_probe.log` |
| bottom 单步 probe | `checkpoints/PRE/bottom_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/` | `train_bottom_probe.log` |
| middle MS5 | `checkpoints/PRE/middle_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5/` | `train_mid_ms5.log` |
| bottom MS5 | `checkpoints/PRE/bottom_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5/` | `train_bottom_ms5.log` |

per-layer stats 缓存：`stats_d14_clipnone.npz` / `stats_d0_clipnone.npz`（各现算 37 s）。

评估/诊断日志（`checkpoints/PRE/`）：单步 probe 逐 epoch 选型
`eval_{mid,bottom}_val1_ep{1..10}.log`、`eval_{mid,bottom}_val15_ep10.log`、
test `eval_{mid,bottom}_test15_ep10.log`；MS5 逐 epoch 选型
`eval_{mid,bottom}ms5_val15_ep{1..5}.log`、test `eval_midms5_test15_ep2.log` /
`eval_bottomms5_test15_ep5.log`；结构诊断 `diag_{mid,bottom}_val.log`、
`diag_{mid,bottom}_ms5_val.log`。

## 单步 probe（10 epochs，day-1 选型 = val h1 stride 7）

| 层 | 训练 | best | val day-1 ratio（u / v / overall） | test h15 overall ratio (u / v) |
|---|---|---|---|---|
| middle | 10/10，无早停 | Ep10 | 0.754 / 0.803 / **0.770** ✅ | **1.183**（1.138 / 1.451）❌ |
| bottom | 10/10，无早停 | Ep10 | 0.550 / 0.624 / **0.568** ✅ | **0.930**（0.920 / 0.966）⚠️ |

- 两层都过单步门槛（day-1 ratio < 1.0）→ **均进入 MS5**。
- middle 单步存在与 surface 实验 07 相同的长时效失效：test crossover d5、d15 pooled ratio 1.27、v 恶化最快（1.45）。
- bottom 单步明显更强（更易预测，与 WP1 画像一致），val 无 crossover，但 test d15 仍略破 1（v 主导）。

## MS5（K=5，从各层 probe Ep10 weights-only 初始化，5 epochs）

选型 = 过 day-1 门槛 epoch 中 val h15 overall ratio 最低者：

| 层 | 逐 epoch day-1 ratio（门槛） | 选型 | val h15 overall (u / v) | test h15 overall (u / v) |
|---|---|---|---|---|
| middle | 0.619/0.621/0.605/0.590/0.590（≤0.785 全过） | **Ep2** | 0.814（0.813 / 0.826）✅ | **0.830**（0.825 / 0.864）✅ |
| bottom | 0.601/0.588/0.596/0.576/0.575（≤0.5796，Ep4/5 过） | **Ep5** | 0.790（u 0.845 / v 0.820 @d10-15 max）✅ | **0.813**（0.838 / 0.864 @d10-15 max）✅ |

### 门槛核对（val；全部预注册）

| 门槛 | middle MS5 Ep2 | bottom MS5 Ep5 |
|---|---|---|
| day-1 ≤ probe最优 × 1.02 | 0.6205 ≤ 0.785 ✅ | 0.575 ≤ 0.5796 ✅（余量小） |
| 15-day overall ratio < 0.941 | 0.814 ✅ | 0.790 ✅ |
| u / v 各自 < 1.0 | 0.813 / 0.826 ✅ | ✅ |
| day 10–15 每日 < 1.0 | max 0.904 (d15) ✅ | max ~0.87 (d15) ✅ |
| 结构：crossover / corr | 无 crossover；corr 全 lead 占优 | 无 crossover；corr 全 lead 占优 |

### test h15 与单步 probe 对比（关键修复证据）

| | 单步 probe | MS5 | 变化 |
|---|---|---|---|
| middle test overall ratio | 1.183 | **0.830** | crossover 消除（d15 1.27→0.96） |
| bottom test overall ratio | 0.930 | **0.813** | v d15 1.15 → 0.880 |
| bottom test v d15 ratio | 1.150 | **0.880** | probe 的唯一短板被修复 |

## 结构诊断（val，77 窗口，选型 checkpoint）

- middle MS5 Ep2：pooled ratio 0.62→0.90 单调（无 crossover）；u corr 0.928→0.469（全 lead > persistence）；u bias +0.043@d15；var_ratio u 0.286@d15（平滑化）。
- bottom MS5 Ep5：pooled ratio 0.60→0.84（无 crossover）；u corr 0.940→0.538、v 0.918→0.557（全 lead 占优）；v bias 全程 |·|≤0.005；var_ratio 0.31–0.37@d15。

## 结论

1. **垂向泛化成立**：detached MS5 的修复效果跨深度成立——middle test overall 1.183→0.830、bottom 0.930→0.813，与 surface（1.018→0.871）同模式。
2. **垂向难度排序与 WP1 画像一致**：bottom 最易（MS5 后 test d1 ratio 0.676）、middle 居中（0.62 val）、surface 最难；层 MS5 的 day-1 门槛余量也按此排序（bottom 余量最小——其 persistence 本身已很强）。
3. bottom 单步的 v 分量长时效短板被 MS5 完全修复（1.15→0.880）。
4. 遗留（与 surface 一致）：方差塌缩（var_ratio ~0.3@d15）、d15 ratio 回升（0.88–0.96）；bias 轻微正向漂移。
5. 代表层证据支持 full3d 投资（垂向各层均有可利用信号 + MS 路径可迁移），但 full3d 的 1-epoch pilot 尚无信号（见实验 06 RESULTS），K3 按预注册条件阻塞。

## 备注

- MS5 epoch 实测 ~46-47 min（probe 单步 ~21.6 min）——高于实验 10 surface 的 ~30 min，原因待查（同为 batch 4；可能与评估共卡时长有关，非科学问题）。
- bottom MS5 的 day-1 门槛余量最小（Ep1–3 未过 0.5796），提示对该层可考虑放宽容差或更多 epochs——本轮不追加（预注册纪律）。
- 所有评估/诊断由值守 agent 按给定命令执行；`pre_evaluate.py`/诊断脚本零修改。
