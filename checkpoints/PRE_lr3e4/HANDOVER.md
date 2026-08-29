# 交接文档：DiAFNO lr=3e-4 surface SD2 对照实验（PRE 海流预报）

> 交接时间：2026-08-28 14:1x ｜ 交接范围：本对照实验全部工作 ｜ 读者：后续接手的 agent
> 工作区：`/data2/user/zyq/projects/DiAFNO_lr3e4`（对照工作区，允许修改）
> 产物根目录：`/data2/user/zyq/checkpoints/PRE_lr3e4`（独立 OUT_ROOT）

## 0. 必须遵守的边界（先读这段）

- **禁止修改**主工作区 `/data2/user/zyq/projects/DiAFNO` 与主实验目录 `/data2/user/zyq/checkpoints/PRE`（主实验 lr=1e-3 的 SD2 训练/评估在其中进行，且**有并行会话正在使用**）。读是可以的。
- 本对照工作区内**有未提交的改动**（3 个文件，见 §4），按任务要求**不要 commit / push**。
- 本机器多会话并行作业，GPU 归属动态变化：**启动任何 GPU 任务前先 `nvidia-smi` 确认空闲卡**；交接时主工作区会话正在 GPU 3 跑主实验的 test 评估（`.../PRE/eval_test_final.log`）。
- `pre_trainer.py` / `pre_evaluate.py` 是脚本式模块级常量配置（无 CLI 参数）：**改常量 → py_compile → 运行**。共享无副作用配置在 `pre_config.py`。不要 import 这两个脚本。
- 评估输出**拒绝覆盖**：`eval_*.npz` 已存在或 figures 目录非空会直接 raise；换配置用 `OUTPUT_TAG`，重跑先删旧产物。

## 1. 实验设定（与主实验唯一差异是 lr）

| 项 | 值 |
|---|---|
| 任务 | PRE 海流：7 天条件（14 ch，day-major u/v 交错）→ 次日 u/v（2 ch），条件扩散单步预测 |
| preset | `surface_smoke`（depth_index=29 表层，400×441×1，patch (4,3,1)，embed 180，I4/E4，BS4，sampling_steps 32）|
| **lr** | **3e-4**（主实验 1e-3）——本对照的唯一变量 |
| sigma_data | **0.1712084**（stats sigma 0.0856042 × 2.0，即 SD2 固定尺度；checkpoint config 内有记录）|
| 数据/归一化 | `~/data_processed/PRE/aligned/{u,v}_rho.npy`；stats 缓存 `~/data_processed/PRE/norm/stats_d29_clipnone.npz`；train/val/test = [0,8401)/[8401,9496)/[9496,10591) 天 |
| 运行环境 | `/data2/user/zyq/miniconda3/envs/diafno/bin/python`（torch 2.4.1+cu124）；训练入口 `pre_trainer.py`，评估入口 `pre_evaluate.py` |
| 训练日志 | `PRE_lr3e4/surface_lr3e4_sd2_train.log`（PID 文件同目录，进程均已结束）|

run 目录（下称 RUN_DIR）：
`/data2/user/zyq/checkpoints/PRE_lr3e4/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/`

## 2. 已完成的四阶段工作

### 阶段一：训练（已完成，但被 early-stop 提前终止）
- 2026-08-28 10:57 启动（GPU 3，`setsid nohup` 后台），12:19 结束。
- 要求 10 epoch，**实际只跑了 3/10**：`pre_trainer.py` 的 early-stop（val 连续 2 epoch 变差即停）在 Ep3 后触发。
- loss.dat（RUN_DIR/loss.dat，列 = 时间/ train_loss / val_masked_relL2）：

  | epoch | val_masked_relL2 | 备注 |
  |---|---|---|
  | 1 | **2.14188** | **best（best.pth = Ep1.pth）** |
  | 2 | 3.62861 | best 的相邻 epoch |
  | 3 | 2.53195 | last epoch |

- ⚠️ **重要判断：该 early-stop 很可能是误触发**——见 §5，单种子验证采样的方差极大，val 指标波动（2.14→3.63→2.53）不代表真实退化。
- 训练期间 train_loss：0.178 → 0.052 → 0.040（EDM 加权去噪损失，平台化≈气候态 score 已学出）。

### 阶段二：day-1 并行评估（已完成，全部在 val split）
协议：`SPLIT=val, ROLLOUT_DAYS=1, EVAL_STRIDE=7, MAX_WINDOWS=None, BATCH_SIZE=4, SAMPLING_STEPS=32, EVAL_SEED=123`（逐窗口种子 = 123 + 窗口起始日，与 batch 无关），156 个窗口。结果（native 网格 pooled RMSE, m/s）：

| 配置 | day-1 model | persistence | model/pers | u | v | npz（RUN_DIR 下）|
|---|---|---|---|---|---|---|
| churn=0, E=1 | 0.3259 | 0.1294 | 2.520 | 0.3592 | 0.2889 | `eval_val_h1_ch0_e1_s123_ckptbest.npz` |
| churn=80, E=1 | 0.4336 | 0.1294 | 3.352 | 0.4556 | 0.4105 | `eval_val_h1_ch80_e1_s123_ckptbest.npz` |
| **churn=0, E=4**（胜出配置）| **0.3057** | 0.1294 | **2.363** | 0.3379 | 0.2698 | `eval_val_h1_ch0_e4_s123_ckptbest.npz` |

- churn 选择：churn=0 优于 80；E=4 仅再提升 ~6%（误差由偏差主导，集成帮不了太多，见 §5）。
- 结论：**模型 day-1 显著劣于 persistence（2.4 倍），甚至与 zero-current baseline（0.2620）同档**。主实验 lr=1e-3 也有同样问题（Ep3: 0.2584, ratio 2.0；E=4: 0.2471, ratio 1.91），故**不是 lr=3e-4 造成的，是系统性欠训练/条件学习不足**。

### 阶段三：劣于 persistence 的根因诊断（已完成，证据充分）
诊断脚本：`PRE_lr3e4/diag_day1.py`（从 /tmp 拷贝存档；跑法见文件头注释，需 GPU，约 3 分钟）。核心证据（对照 best.pth，12 个 val 窗口，物理单位，rho 网格）：

1. **模型无视条件（决定性证据）**：同一种子、换用窗口 A/B 两个完全不同的 7 天条件，输出相关 corr = **0.96/0.98**（u/v）——输出几乎完全由初始噪声决定，条件只贡献 ~2-4%。
2. **输出 ≈ 气候平均场 + 过大散度**：corr(pred, clim)=0.69/0.63 ≥ corr(pred, day7)=0.69/0.60 > corr(pred, tgt)=0.58/0.59；**corr(pred−day7, tgt−day7) ≈ 0（−0.02/+0.07）**——模型相对 persistence 的增量与真实 day-8 变化零相关（无订正技巧）。
3. **采样散度被高估 ~2 倍**：pred std（0.445/0.358）≈ 2× tgt std（0.217/0.156）；真实条件不确定度（persistence RMSE 0.109）远小于模型样本间散度（~0.18-0.27）。
4. RMSE 分解：model 0.333 ≫ climatology 0.177 ≫ persistence 0.109 —— 模型连气候平均都不如（散度噪声所致）。
5. **已排除**：采样/反归一化 bug（sample() 正确映射回 [0,1]）、网格转换误差（rho-oracle 0.0032）、条件未接入网络（`IAFNODiff.forward` 第 288-292 行确实 concat 条件，in_chans=16）、mask/指标错误（各基线自洽）、trainer 验证指标 bug（其 2.14 与诊断实测 1.78 同量级，差异来自种子抽样方差）。
6. 机理：条件仅经 patch-embed 通道拼接进入网络（无独立条件调制通路，只有噪声水平 σ 有 FiLM）；EDM 损失前期主要学边际分布（气候态）score，条件结构学习慢得多。3 epoch（6.3k updates）远不够。

## 3. 当前产物清单（截至交接时全部进程已结束）

```
PRE_lr3e4/                              # 独立 OUT_ROOT（对照实验专用）
├── HANDOVER.md                         # 本文档
├── diag_day1.py                        # 根因诊断脚本（§5）
├── surface_lr3e4_sd2_train.log/.pid    # 阶段一训练日志（PID 已失效，仅存档）
├── eval_val_h1_ch{0,80}_e{1,4}.log     # 阶段二评估日志（含完整 RMSE 表）
├── eval_{ch0_e1,ch80_e1,ch0_e4}.pid    # 评估 PID 存档（已失效）
├── eval_screen_val_day1.log            # ⚠️ 并行会话所写（非本会话产物）
└── surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/   # RUN_DIR
    ├── Ep1.pth Ep2.pth Ep3.pth         # 各 epoch checkpoint（含 config: sigma_data 等）
    ├── best.pth                        # = Ep1（val 最低）
    ├── loss.dat                        # 3 行完整历史
    ├── eval_val_h1_ch0_e1_s123_ckptEp{1,2}.npz   # ⚠️ 并行会话所写（Ep1/Ep2 对比评估）
    ├── eval_val_h1_{ch0_e1,ch80_e1,ch0_e4}_s123_ckptbest.npz   # 本会话三组评估
    └── figures_*/                      # 对应 figures 目录
```

## 4. 对照工作区的代码改动（未提交，勿 commit）

| 文件 | 改动 | 目的 |
|---|---|---|
| `pre_config.py` | `OUT_ROOT` → `.../PRE_lr3e4`；surface_smoke `lr` 1e-3→**3e-4** | 独立输出目录；对照 lr |
| `pre_trainer.py` | `EPOCH_OVERRIDES = {"surface_smoke": None}`（原为 4）→ 走 preset 的 10 epoch | 要求训 10 epoch |
| `pre_evaluate.py` | CHECKPOINT 显式指向对照 best.pth；SPLIT=val；ROLLOUT_DAYS=1；SAMPLER_S_CHURN=0；ENSEMBLE_SIZE=4；SAMPLING_STEPS=32；EVAL_SEED=123（EVAL_STRIDE=7/MAX_WINDOWS=None/BATCH_SIZE=4 沿用默认）| 阶段二协议；**当前停在 churn=0/E=4 状态** |

git 状态：detached HEAD（52c0113），上述 3 文件 modified，无其他改动。

## 5. 待办与建议（2026-08-28 傍晚更新：原 §5.1/5.2 已完成，见 §7）

1. ~~续训至 10 epoch~~ **已完成（阶段四，§7）**——结论：条件仍未学出，且点预测技巧反而退化。**不要再单纯加 epoch**。
2. ~~复评 day-1~~ **已完成（§7）**。
3. **【新的核心待办】结构/目标改造后再训**，当前证据下单纯续训无意义：
   - 给条件加显式通路：对 AFNO block 增加条件 FiLM（scale/shift 由 14ch 条件嵌入生成）或 cross-attention；对照 `IAFNODiff.forward` 目前条件只经 patch-embed 通道拼接、仅 σ 有 FiLM。
   - 或改任务形式：预测相对 day-7 的残差（residual head），把"persistence 基线"内置进目标。
   - 或训练目标加权：加大低 σ 段权重 / 辅助条件一致性损失。
   - 改动只做在本对照工作区；改完用 `diag_day1.py`（支持传 checkpoint 路径参数）快速验证 anomaly corr 是否脱离 0。
4. E=4 test split 终评：等条件学出后再做（当前模型 day-1 劣于 zero，上 test 无意义）。
5. 长任务一律 `setsid nohup ... > log 2>&1 < /dev/null &`，并用 `pgrep -f` + `/proc/PID/cwd|fd/1` 核实真正主进程 PID（setsid 包装进程会退出）。监控用周期性 `tail -n`，不要 `tail -f`。
6. 机器上多会话共享 GPU：先 `nvidia-smi` 挑空闲卡；评估互相拖慢但不影响正确性（逐窗口播种保证可复现）。

## 6. 关键数字速查

- sigma_data = 0.1712084（SD2 = 2.0 × stats sigma 0.0856042）；stats 缓存：`~/data_processed/PRE/norm/stats_d29_clipnone.npz`
- 对照 run tag：`surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2`；主实验同名 tag 在 `.../PRE/` 下（勿混淆）
- day-1 native RMSE：persistence 0.1294 ｜ zero 0.2620 ｜ rho-oracle 0.0032 ｜ 对照 best E=1 0.3259 / E=4 0.3057 ｜ 主实验 Ep3 E=1 0.2584 / E=4 0.2471
- 训练耗时：~23 min/epoch（独占 RTX 4090 时）；day-1 全量 val 评估（156 窗口）：E=1 ~6-7 min，E=4 ~15-25 min（视 GPU 共享情况）

## 7. 阶段四（2026-08-28 16:00-19:00）：续训至 10 epoch + 复评 —— 条件仍未学出，技巧退化

### 7.1 续训（handover 原 §5.1）
- 从 `Ep3.pth` 续训 epoch 4-10（GPU 4 独占，`setsid nohup`，PID 2778401 已正常退出）。
- `pre_trainer.py` 改动：`checkpoint_path` 指向 Ep3.pth；early-stop patience `worse_epochs >= 2` → `>= 8`（剩余 7 epoch 内不可能触发）。
- 全程无异常；train_loss 单调下降 0.0321 → 0.0162；单种子 val rel-L2 在 2.33-2.86 波动，**始终未超过 Ep1 的 2.14188 → best.pth 仍为 Ep1**（该 val 指标噪声大，不代表真实技巧，见 §5 旧文与诊断）。
- 新产物：`Ep4.pth`…`Ep10.pth`、loss.dat（10 行完整历史）、`surface_lr3e4_sd2_resume_ep4-10.log`。

### 7.2 day-1 复评（handover 原 §5.2；协议同 §2，val split，churn=0，seed 123）

| checkpoint | E=1 | E=4 | 说明 |
|---|---|---|---|
| Ep1（=best.pth）| 0.3259 | 0.3057 | 阶段二结果 |
| **Ep10** | 0.3779 | 0.3558 | **比 Ep1 差**（ratio 2.92 / 2.75 vs 2.52 / 2.36）|
| persistence | 0.1294 | — | |
| zero | 0.2620 | — | Ep10 E=1 已劣于 zero |

npz：`eval_val_h1_ch0_{e1,e4}_s123_ckptEp10.npz`；日志 `eval_val_h1_ch0_{e1,e4}_ckptEp10.log`。

### 7.3 条件学出信号对比（`diag_day1.py <ckpt>`，12 窗口，rho 网格物理单位）

| 指标 | Ep1 | Ep10 | 解读 |
|---|---|---|---|
| corr(pred−day7, tgt−day7) u/v | −0.02/+0.07 | −0.05/+0.05 | **增量技巧仍为零**——条件订正完全没学出 |
| corr(pred, tgt) u/v | 0.58/0.59 | 0.59/**0.20** | **v 通道崩塌** |
| pred mean − tgt mean（u/v bias）| +0.01/−0.03 | **−0.107/−0.145** | bias 显著增大 |
| pred std / tgt std（u）| 2.05× | 2.10× | 散度依旧 ~2 倍过_dispersion |
| member spread u/v（4 种子）| ~0.27（pooled, 旧口径）| 0.158/0.107 | ≈ 真实条件不确定度（pers 0.135/0.075）的 1.2-1.4 倍 |
| 同种子换条件输出 corr u/v | 0.96/0.98 | 0.84/0.96 | 条件影响仍只占小部分 |
| trainer 式 relL2（physical）| 1.78 | 2.10 | 与全量 val 日志量级一致 |

### 7.4 阶段四结论
1. **EDM 目标与预报技巧错位**：train_loss 再降一半（0.040→0.016）的同时 day-1 RMSE 上升（0.326→0.378 E=1）。模型在精化"气候态 + 宽条件分布"的 score，而不是条件均值。
2. **条件依赖没有随训练涌现**（anomaly corr 恒为 0），且 v 通道出现相关崩塌 + 系统性 bias —— 继续加 epoch 无益甚至有害。
3. **必须改结构/目标**（见 §5.3：条件 FiLM/cross-attention、residual 形式、低 σ 加权），改完用 `diag_day1.py` 先验 anomaly corr 再上全量评估。
4. 当前 `pre_trainer.py` 停在"resume Ep3 + patience 8"状态、`pre_evaluate.py` 停在"Ep10 + E=4"状态；新会话接手时按需复位这些常量。
