# DiAFNO / PRE_ocean_data 重跑执行报告

> 日期：2026-08-28 ｜ 分支：`adapt-weather-ocean` ｜ HEAD：`6cc418b`（烟测脚本修改）+ 2 个未提交修复（见 §2）
> 用途：本报告自包含，交给执行 agent 按 §5 分阶段执行。所有路径为绝对路径。
> 背景文档：`docs/PRE_runbook.md`（运行手册）、`docs/PRE_ocean_data.md`（数据分析）、`AGENTS.md`（仓库约定）。

---

## 0. 执行摘要

- 旧冒烟实验（8月27日）泛化失败：test 集 day-1 RMSE 是 persistence 的 **2.7~3.1 倍**，15 天所有 lead day 均未赢过 persistence。
- 根因已定位并已在代码中修复（commit `6cc418b`）：**sigma_data 尺度 bug**（用了 [0,1] 空间的 0.0856，EDM 实际工作在 [-1,1] 空间，正确值为 **0.1712**）+ 训练卫生问题。采样侧（S_churn=80 满幅噪声 + 单轨迹）尚未消融，是本次重跑要解决的另一半。
- 冒烟回归测试此前有 2 个失败项，已由本报告作者修复（§2），当前 **29 项全 PASS**。
- 资源：保底 1 张 GPU（**GPU 3**，24GB 空闲；其余被其他用户任务占满）。时间预算 **24 小时**。
- 交付优先级：**surface 保底**（完整训练 + 采样消融 + 15 天正式评估），full3d 尽力而为（§5 阶段 4）。

---

## 1. 现状诊断

### 1.1 旧实验结果（反面基线，保留勿删）

目录：`/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7/`（无 `_SD2` 后缀 = 旧尺度实验）

训练曲线（`loss.dat`，列 = epoch 耗时 s / train_loss / val_masked_relL2）：

| epoch | 1 | 2 | 3 | 4 | ... | 10 |
|---|---|---|---|---|---|---|
| val_masked_relL2 | 1.956 | 1.577 | **1.567**(best) | 2.193 | 振荡上升 | 2.393 |

- val_masked_relL2 > 1 意味着比"全预测零"还差；第 3 epoch 后持续恶化。
- 每 epoch ≈ 1580~1607 s（≈ 24 min，含验证采样）。

test 集正式评估（`eval_test.npz`，154 窗口 × 15 天 rollout，原生网格 masked RMSE，m/s）：

| lead day | u: model / pers | v: model / pers |
|---|---|---|
| 1 | 0.377 / 0.139 = **2.72** | 0.279 / 0.090 = **3.11** |
| 5 | 0.628 / 0.258 = 2.43 | 0.214 / 0.140 = 1.53 |
| 10 | 0.488 / 0.280 = 1.74 | 0.232 / 0.146 = 1.59 |
| 15 | 0.531 / 0.267 = 1.99 | 0.242 / 0.146 = 1.66 |

失败模式（`figures/d01_s00_u.png`）：开阔大洋系统性负偏置 + 近岸/边界格点 +2 m/s 极端尖刺 → 采样失控，不是"差一点"。

### 1.2 根因分析

1. **sigma_data 尺度 bug（主因，已修）**：`pre_dataset.py` 的 stats 缓存存的是 [0,1] 归一化空间的 pooled sigma（surface = 0.08560）；但 `diffusion.py` 训练时把图像 `images*2-1` 映射到 [-1,1]，数据 std 实为 2 倍。旧跑训练+采样统一用了 0.0856（一半），EDM 预条件 c_skip/c_out/c_in 整体失准，经 Heun 采样与自回归 rollout 逐级放大。修复：`pre_config.py` 的 `SIGMA_DATA_SCALE=2.0` + `sigma_data_from_stats()`，训练/评估统一换算；新 checkpoint 落 `_SD2` 后缀目录，与旧实验物理隔离。
2. **训练卫生（已修）**：旧 AMP API、AMP 跳过的 update 也推进 scheduler、无非有限 loss 中止、无早停、best.pth 状态过期 bug。均已修复（见 `pre_trainer.py`）。
3. **采样配置未消融（本次要解决）**：旧评估 S_churn=80（32 步下每步 gamma 封顶 0.414，每步注入满幅噪声）+ 单轨迹。对 RMSE 类点预测指标，S_churn=0（确定性 Heun）和/或 ensemble 均值通常显著更优。runbook §4 已写明消融方案，本次执行。

---

## 2. 代码现状与已完成的修复（未提交，执行前必须核对）

`git status` 应显示恰好 2 个修改文件（`git diff` 核对）：

1. **`pre_smoke_test.py`**（测试 bug 修复）：`test_ensemble_rollout_uses_autocast` 原断言依赖全局 `torch.cuda.is_available()`，在"有 GPU 但张量在 CPU"的环境下必挂。已改为按张量设备断言，并补充 CUDA 张量路径的断言。
2. **`scripts/preprocess_align_uv.py`**（真实代码 bug 修复）：`torch_colocate_u/v` 原实现输出 shape 硬编码全局常量 `S,H,W`（30,400,441），对非生产 shape 的输入直接报错。已改为从输入 shape 推导（u: `(t,s,r,c)→(t,s,r,c+1)`；v: `(t,s,r,c)→(t,s,r+1,c)`）。
   - **注意**：已生成的对齐数据 `~/data_processed/PRE/aligned/{u_rho,v_rho}.npy` 是修复前生产的，但生产 shape 恰好匹配硬编码常量，**数据本身是正确的，无需重跑预处理**。

验证命令（全部应通过）：

```bash
cd /data2/user/zyq/projects/DiAFNO
source ~/miniconda3/etc/profile.d/conda.sh && conda activate diafno
CUDA_VISIBLE_DEVICES=3 python pre_smoke_test.py   # 结尾打印 "pre_smoke_test passed"
python smoke_test.py                              # 打印 "CPU smoke test passed"
```

---

## 3. 资源与实测耗时

| 项 | 数值 | 来源 |
|---|---|---|
| 可用 GPU | **GPU 3**（24GB 空闲）；其余 7 张被其他用户任务占用 | nvidia-smi 实测 |
| CPU / 内存 | 256 核 / 935GB 可用（full3d 数据 418GB 可进页缓存） | 实测 |
| 磁盘 | /data2 余 5.1TB | 实测 |
| surface epoch | **≈ 24 min**（B=4，2099 batch/epoch，含验证采样） | 旧跑 loss.dat 实测 |
| surface 完整评估 | **≈ 2~2.5h**（154 窗口 × 15 天 × 32 步 × Heun 2 次前向，E=1） | 旧跑时间戳推算 |
| full3d 统计缓存 | **不存在**，首次启动自动计算（3 遍流式扫描 ~530GB，15~30 min，一次性） | 缓存目录实测 |
| full3d epoch | **未知，估计 5~7h**（token 数 15×、batch 数 4×于 surface），**必须实测**（§5 阶段 4 探针） | 估算 |

---

## 4. 需要执行的代码修改（按阶段）

> 原则：除下列修改外不动任何代码。所有修改都是模块级常量，改完直接运行脚本即可。

### 4.1 阶段 1（surface 重训）：`pre_trainer.py`

```python
# 第 45 行附近，原：
EPOCH_OVERRIDES = {"surface_smoke": 4}
# 改为：
EPOCH_OVERRIDES = {"surface_smoke": 10}
```

**不要**先跑 4 epoch 再续训到 10：cosine scheduler 的 T_max 按 4 epoch 建，lr 已衰减到 0，续训等于白跑。一次定 10 epoch，早停（连续 2 epoch 验证恶化）自动兜底。

### 4.2 阶段 2（采样消融）：`pre_evaluate.py` 顶部常量

固定：`PRESET="surface_smoke"`、`CHECKPOINT=None`（自动取 `_SD2` 目录的 best.pth）、`SPLIT="test"`、`ROLLOUT_DAYS=1`、`EVAL_STRIDE=7`、`MAX_WINDOWS=None`、`BATCH_SIZE=4`。逐组改两个常量：

| 组 | SAMPLER_S_CHURN | ENSEMBLE_SIZE | 预计耗时 |
|---|---|---|---|
| ① ch0_e1 | 0 | 1 | ~10 min |
| ② ch0_e4 | 0 | 4 | ~40 min |
| ③ ch80_e1 | 80 | 1 | ~10 min |
| ④ ch80_e4 | 80 | 4 | ~40 min |

输出 tag 含 `h1_ch{churn}_e{es}_s123_ckptbest`，4 组天然不冲突，**不要设 OUTPUT_TAG**（输出拒绝覆盖）。

### 4.3 阶段 3（surface 正式评估）：`pre_evaluate.py`

`ROLLOUT_DAYS=15`，S_churn/ENSEMBLE_SIZE 用阶段 2 胜者。
- 胜者 E=1：`MAX_WINDOWS=None`（全 154 窗口，~2.5h）；
- 胜者 E=4：`MAX_WINDOWS=48`（~3h，否则 9h 超预算）。

### 4.4 阶段 4（full3d）：`pre_trainer.py` + `pre_evaluate.py`

- 两个文件的 `PRESET` 都改为 `"full3d"`；
- 探针实测 epoch 时间后，`EPOCH_OVERRIDES` 加一项 `"full3d": N`（N 的计算见 §5 阶段 4）；
- 评估侧：`BATCH_SIZE=1`、`EVAL_STRIDE=14`、`MAX_WINDOWS=24~32`、`ENSEMBLE_SIZE=1`、S_churn 用阶段 2 胜者；时间再紧则 `SAMPLING_STEPS=18`；
- 若探针显示 I/O 瓶颈（batch 耗时波动大、GPU 利用率低）：`pre_config.py` full3d 的 `num_workers` 2→8（低风险，可选）。

---

## 5. 24 小时分阶段执行方案

> 记启动时刻为 T0。所有训练/评估用 `nohup ... &` + log 文件，夜间无人值守。
> 每个阶段开头先 `source ~/miniconda3/etc/profile.d/conda.sh && conda activate diafno && cd /data2/user/zyq/projects/DiAFNO`。

### 阶段 0｜T0+0:00–0:10 — 环境核对

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 确认 GPU 3 空闲
cd /data2/user/zyq/projects/DiAFNO && git status --short                # 应只有 §2 的 2 个文件
CUDA_VISIBLE_DEVICES=3 python pre_smoke_test.py && python smoke_test.py # 全 PASS 才继续
```

### 阶段 1｜T0+0:10–4:30 — surface 重训（SD2，10 epoch）

完成 §4.1 修改后：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u pre_trainer.py \
  > /data2/user/zyq/checkpoints/PRE/train_surface_sd2.log 2>&1 &
tail -f /data2/user/zyq/checkpoints/PRE/train_surface_sd2.log
```

**启动后 5 分钟内必查**（log 第一屏）：
- `sigma_data=0.17120 (scale 2.000x)` —— 修复生效；若为 0.08560 说明改错文件/环境；
- `run_dir=/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2` —— 新目录；
- 无 pre-flight 报错（Ep 碰撞 / loss.dat 截断）。

**Go/No-Go 闸门（约 T0+1:00，epoch 2 落盘后）**：
- ✅ epoch 2 的 `val_masked_relL2` **< 0.8 且较 epoch 1 下降**（旧跑同期 1.58）→ 继续跑满/等早停；
- ❌ 仍在 1.5 附近或不降 → **立即 kill**，不进入后续阶段；回头排查数据通路（条件通道 day-major 交错顺序、mask 广播、归一化方向），sigma 尺度不是唯一病根。

注意：训练内验证用模型默认 S_churn=80，数值偏悲观，与旧曲线同口径可比；最终技能以阶段 2/3 评估为准。训练完成标志：log 末尾 `done. best val_masked_relL2 = ...` 或早停信息。

### 阶段 2｜T0+4:30–6:30 — 采样消融（day-1，4 组）

按 §4.2 逐组改常量并运行（顺序 ①→②→③→④）：

```bash
CUDA_VISIBLE_DEVICES=3 python -u pre_evaluate.py 2>&1 | tee -a /data2/user/zyq/checkpoints/PRE/eval_ablation.log
```

每组结尾打印 `=== day-1 and overall comparison table ===`，记录 4 组的 day-1 pooled RMSE。
**捷径**：若 ①(ch0_e1) 已明显优于 ③(ch80_e1)（>20%），churn=80 路线判死，可跳过最贵的 ④。
**选取规则**：day-1 RMSE 最低者；若 ch0_e1 与 ch0_e4 差距 <3%，选 E=1（阶段 3 省 4 倍时间）。

### 阶段 3｜T0+6:30–9:30 — surface 正式 15 天评估（保底交付物）

按 §4.3 改常量后运行同一命令。**通过标准**：控制台汇总表中各 lead day `model/pers < 1`（day-1 目标 ≤ 0.8）。
产出（在 `_SD2` 目录）：`eval_test_h15_ch*_*_ckptbest.npz` + `figures_*/`。
**至此保底交付物完成（约 T0+9:30）。若未通过，记录结果并停止，不进 full3d。**

### 阶段 4｜T0+9:30–24:00 — full3d（尽力而为）

1. 改 `PRESET="full3d"`（训练+评估两个文件），启动训练——先自动算全 30 层统计（~30 min，一次性）；
2. **探针**：盯 log 的 `[ep 1] batch 100/8394 ... Xs/batch`，epoch 时间 ≈ X × 8394 + 验证采样（16 窗口 × 64 前向）。首 epoch 含冷 I/O，偏保守正好。记录后 **kill**（未写 checkpoint，目录干净）；
3. 计算 `N = floor((24:00 − 当前时刻 − 3h 评估预留 − 1h 缓冲) / epoch 小时数)`，写入 `EPOCH_OVERRIDES["full3d"]=N`，重启正式训练；
4. 训练完成后按 §4.4 跑粗评估；
5. **决策规则**：
   - N ≥ 2 → 正常执行；
   - N = 1 → 单 epoch 价值有限：跑 1 epoch + `MAX_WINDOWS=8` 迷你评估，仅验证管线通畅；
   - N < 1（epoch > ~9h）→ 放弃 full3d 训练，改为：surface 的 val 集补充评估（`SPLIT="val"`）或更大窗口的 ensemble 评估，充实保底结果。

**OOM 处理顺序**（full3d 显存紧张时）：`embed_dim` 128→96 → `implicit_layer` 2→1；不要动 patch（441 只能被 1/3/7/9 整除）。

---

## 6. 风险与注意事项

- **旧目录勿动**：`surface_smoke_..._C7`（无 _SD2）是反面基线；新实验全部落 `_SD2` 目录，物理隔离。
- **评估输出拒绝覆盖**：同一 tag 重跑会报错；消融 4 组 tag 天然不同。如需重跑同配置，先删旧输出或设 `OUTPUT_TAG`。
- **断点续训**：`RESUME_SIGMA_POLICY` 保持默认 `"error"`；本方案不涉及跨尺度续训，不要改成 migrate/adopt。
- **GPU 被抢**：执行中若 GPU 3 被其他任务占用，先 `nvidia-smi` 另选空闲卡改 `CUDA_VISIBLE_DEVICES`；训练脚本单卡即可。
- **若中途腾出第二张卡**：阶段 2 的 4 组消融可两卡对半分（省 ~1h）；或阶段 1 与 full3d 统计/探针并行。
- **随机性**：训练 seed 123、验证 seed 1234、评估逐窗口 seed（EVAL_SEED+start_day）均已内建，评估轨迹与 batch 大小无关，可放心调 BATCH_SIZE/MAX_WINDOWS。
- 本报告未改动 `AGENTS.md` 所述的任何行为，无需更新文档；执行完成后建议把 §2 的两个修复与阶段 1 的 `EPOCH_OVERRIDES` 改动一起提交。

## 7. 交付物清单（执行完成后应存在）

| 物 | 路径 |
|---|---|
| surface SD2 checkpoint + 训练曲线 | `~/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/{Ep*.pth,best.pth,loss.dat}` |
| 采样消融 4 组（或按捷径删减） | 同上目录 `eval_test_h1_ch*_e*_s123_ckptbest.npz` |
| **surface 正式 15 天评估（保底）** | 同上目录 `eval_test_h15_ch*_e*_s123_ckptbest.npz` + `figures_h15_*/` |
| full3d checkpoint（若执行） | `~/checkpoints/PRE/full3d_BS1_EMD128_I2_E4_S32_C7_SD2/` |
| full3d 粗评估（若执行） | 同上目录 `eval_test_h15_*.npz` |
| 全部日志 | `~/checkpoints/PRE/train_surface_sd2.log`、`eval_ablation.log` 等 |

## 附录：关键路径速查

- 仓库：`/data2/user/zyq/projects/DiAFNO`（分支 `adapt-weather-ocean`）
- 环境：`conda activate diafno`（Python 3.10, torch 2.4.1+cu124）
- 对齐数据：`~/data_processed/PRE/aligned/`（u_rho/v_rho 各 209GB，已生产，勿重跑）
- 归一化缓存：`~/data_processed/PRE/norm/stats_d29_clipnone.npz`（surface 已有；full3d 首次自动算）
- checkpoint 根：`~/checkpoints/PRE/`
- 运行手册：`docs/PRE_runbook.md`
