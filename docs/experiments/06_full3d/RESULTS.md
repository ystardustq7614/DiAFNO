# 实验 06 结果：full3d 30 层训练与评估

> 状态：**部分执行（工作包 6 第 1–4 步完成；K3 按预注册条件阻塞）**
> 更新日期：2026-09-03

## 已完成步骤与实测结果

### 1. 全层数据画像（工作包 1，2026-09-01）

`checkpoints/PRE/diag_uv_predictability_20260901/`：门禁四项全 PASS（0 动态缺失、
逐层有效计数充足）。val persistence d1（u）：bottom band 0.068 / middle 0.105 /
upper 0.137 m/s；统一 min-max 无截断，底层归一化 std 约为海面 1/3。

### 2. 资源 probe（2026-09-03，GPU 4，实测）

| 项 | 实测值 |
|---|---|
| stats cache | `stats_all_clipnone.npz`（WP1 顺带生成） |
| 单 `__getitem__` condition（7 天 × 2 变量 = 14 通道 × 30 层） | ~1.6 s（≈296 MB/样本，冷热一致） |
| batch 1 单步 train step（fwd+bwd+step） | 0.97 s |
| **单步训练峰值显存** | **20.7 GB allocated / 22.6 GB reserved**（24 GB 卡，无余量做同卡推理） |
| 单步 1 epoch（8394 步） | ≈ 2.3 h → **正式 50 epoch ≈ 5 天**（预算决策的关键数字） |
| OOM 预案（embed 128→96 / implicit 2→1） | 未触发 |

I/O 与 GPU 大致平衡（2 workers 有效喂入 ~0.8 s/步 < 0.97 s/步），非瓶颈。

### 3. K1 smoke（2026-09-03）

`full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE`（日志 `checkpoints/PRE/train_f3d_k1_smoke.log`）：
**SMOKE PASS**（4 updates、无 AMP skip、finite、checkpoint 齐全）。

### 4. 1-epoch single-step pilot（2026-09-03，GPU 4）

- 产物：run 目录 `checkpoints/PRE/full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES/`
  （Ep1/best/loss.dat/eval npz/figures）；日志 `checkpoints/PRE/train_f3d_pilot.log`、
  `checkpoints/PRE/eval_f3d_val15_ep1.log`。
- 训练健康：7701 s（2h08m）、8394/8394 步、峰值 22.2 GB 平坦、无 OOM/skip/非有限、
  best_val=0.53831。
- **逐层 day-1 信号：无**。60 个（u/v × 30 层）day-1 ratio 全部落在
  [0.9984, 0.9999]，无 band 结构；residual 输出 ≈6e-4 m/s，仅为 day-1 persistence
  误差（0.0655 m/s）的 ~1%；15-day overall ratio ≈ 0.999。
- 解读：与 surface/middle/bottom 轨迹的"Ep1 尚无技能"状态一致（middle probe
  Ep1 day-1 ratio 1.036 → Ep10 0.582），**1 epoch（batch 1）不足以出现信号**，
  属证据不足而非否定。
- val h15 评估耗时 2h05m（batch 1 × 154 窗；前段受 CPU 争用拖慢）。

### 5. K3 pilot：按预注册条件阻塞

[历史实施计划 §6 工作包 6 第 5 项](../../project/archive/MULTISTEP_PLAN_20260901.md)的
准入门（"训练健康且逐层 day-1 有可预测信号后，才做 K3"）
**未满足**（无逐层信号）→ K3 未启动。候选路径（需决策，均未执行）：

- A：追加 single-step epochs（每 epoch ≈ 2.3 h）直至逐层 day-1 信号出现，再进 K3；
- B：接受"1 epoch 无信号"为阶段性结论，full3d 冻结待正式预算（≈5 天/50 epoch 实测）
  另行立项；
- C：提高 pilot 学习率/加大 batch（需单变量论证，破坏与代表层的预算可比性）。

### 恢复与准入条件（原清单状态）

1. 全层数据画像 ✅；2. 资源实测 ✅；3. K1 smoke ✅；4. single-step pilot 逐层信号 ❌（未出现）；5. K3 ❌（被 4 阻塞）；6. 正式预算未冻结（实测数据已备：≈5 天/50 epoch）。

## 结果记录规则

后续运行后在本文追加逐层/分 band 指标与资源数据；代码实现及测试结果写入项目
Changelog，不在本文记录。
