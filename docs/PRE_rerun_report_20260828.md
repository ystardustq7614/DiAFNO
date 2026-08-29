# DiAFNO / PRE_ocean_data 重跑执行报告（v2）

> 日期：2026-08-28（v2 修订） ｜ 分支：`adapt-weather-ocean`
> 用途：本报告自包含，交给执行 agent 按 §5 分阶段执行。所有路径为绝对路径。
> 背景文档：`docs/PRE_runbook.md`（运行手册）、`docs/PRE_ocean_data.md`（数据分析）、`AGENTS.md`（仓库约定）。
>
> **v2 修订要点**（相对 v1）：① 采样消融移到 val，test 只在最终配置锁定后跑一次；
> ② 新增 checkpoint 筛选步骤，不再默认评估 best.pth；③ 废止 `MAX_WINDOWS` 截断
> （其取最前面窗口，有抽样偏差），正式评估恒为全窗口；④ epoch-2 硬闸门改为软目标 +
> 硬中止条件表；⑤ full3d 明确定位为 preliminary；⑥ num_workers 先 2→4 实测。

---

## 0. 执行摘要

- 旧冒烟实验（8月27日）泛化失败：test 集 day-1 RMSE 是 persistence 的 **2.7~3.1 倍**，15 天所有 lead day 均未赢过 persistence。
- 首要候选根因：**sigma_data 尺度 bug**（用了 [0,1] 空间的 0.0856，EDM 实际工作在 [-1,1] 空间，正确值为 **0.1712**），机制明确且与旧实验全部症状相容，但**因果性由本次 SD2 重跑验证**——若重跑后 val 仍 >1，按阶段 1 的 No-Go 分支排查备选假设。次要因素：训练卫生问题（已修）+ 采样配置（S_churn=80 满幅噪声 + 单轨迹，本次消融解决）。
- 冒烟回归测试的 2 个失败项已修复，当前 **29 项全 PASS**。
- 资源：保底 1 张 GPU（**GPU 3**，24GB 空闲；其余被其他用户任务占满）。时间预算 **24 小时**。
- 交付优先级：**surface 保底**（完整训练 + checkpoint 筛选 + val 消融 + 一次正式 test）；**full3d 为 preliminary**，仅在第 5 阶段 test 通过且有剩余时间时执行。
- 预算：E=1 胜出 → surface 约 **9~11h**，full3d 剩 ~14h；E=4 胜出且坚持全量 test → surface 约 **16~18h**，full3d 只剩统计/探针/单 epoch 管线验证。

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

### 1.2 根因分析（候选假设 + 已修问题）

1. **sigma_data 尺度 bug（首要候选根因，已修，因果性待重跑验证）**：`pre_dataset.py` 的 stats 缓存存的是 [0,1] 归一化空间的 pooled sigma（surface = 0.08560）；但 `diffusion.py` 训练时把图像 `images*2-1` 映射到 [-1,1]，数据 std 实为 2 倍。旧跑训练+采样统一用了 0.0856（一半），EDM 预条件 c_skip/c_out/c_in 整体失准，经 Heun 采样与自回归 rollout 逐级放大。修复：`pre_config.py` 的 `SIGMA_DATA_SCALE=2.0` + `sigma_data_from_stats()`，训练/评估统一换算；新 checkpoint 落 `_SD2` 后缀目录，与旧实验物理隔离。
   **若 SD2 重跑后 val 仍 >1**：说明 sigma 不是唯一病根，按 No-Go 分支排查备选假设——条件通道 day-major 交错顺序、mask 广播、归一化方向、AMP 数值路径。
2. **训练卫生（已修）**：旧 AMP API、AMP 跳过的 update 也推进 scheduler、无非有限 loss 中止、无早停、best.pth 状态过期 bug。均已修复（见 `pre_trainer.py`）。
3. **采样配置未消融（本次解决）**：旧评估 S_churn=80（32 步下每步 gamma 封顶 0.414，每步注入满幅噪声）+ 单轨迹。对 RMSE 类点预测指标，S_churn=0（**无逐步注噪的 Heun——注意初始噪声仍随机，逐窗口 seed 固定后才可复现，并非完全确定性**）和/或 ensemble 均值通常显著更优。
4. **checkpoint 选择标准错位（本次解决）**：best.pth 按"单步、S_churn=80、24 窗口"的 val_masked_relL2 选出，与最终"多步 rollout、最终 sampler"的目标准则不一致，需显式筛选（阶段 2）。

---

## 2. 代码现状（执行前必须核对，以执行环境为准）

> 本报告不硬编码提交状态：**执行前以执行环境的 `git rev-parse --short HEAD` 和 `git status --short` 为准**（本地 Windows 副本与服务器可能不同步）。
> 服务器参考状态（2026-08-28）：HEAD `52c0113`、工作树干净——冒烟测试的两个修复与 v1 报告均已提交。

关键修复内容（应在当前 HEAD 中包含）：

1. **`pre_smoke_test.py`**（测试 bug 修复）：`test_ensemble_rollout_uses_autocast` 原断言依赖全局 `torch.cuda.is_available()`，在"有 GPU 但张量在 CPU"的环境下必挂。已改为按张量设备断言，并补充 CUDA 张量路径的断言。
2. **`scripts/preprocess_align_uv.py`**（真实代码 bug 修复）：`torch_colocate_u/v` 原实现输出 shape 硬编码全局常量 `S,H,W`（30,400,441），对非生产 shape 的输入直接报错。已改为从输入 shape 推导（u: `(t,s,r,c)→(t,s,r,c+1)`；v: `(t,s,r,c)→(t,s,r+1,c)`）。
   - **注意**：已生成的对齐数据 `~/data_processed/PRE/aligned/{u_rho,v_rho}.npy` 是修复前生产的，但生产 shape 恰好匹配硬编码常量，**数据本身是正确的，无需重跑预处理**。

验证命令（全部应通过，否则先停下排查）：

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
| surface 完整评估 | **≈ 2~2.5h / 154 窗口 × 15 天 × 32 步 × Heun 2 次前向（E=1）**；E=4 约 ×4 | 旧跑时间戳推算 |
| val 集评估规模 | val 共 1095 天，stride 7 → **~156 窗口**，day-1 E=1 ≈ 10 min/次 | 切分推算 |
| full3d 统计缓存 | **不存在**，首次启动自动计算（3 遍流式扫描 ~530GB，15~30 min，一次性） | 缓存目录实测 |
| full3d epoch | **未知，估计 5~7h**（token 数 15×、batch 数 4×于 surface），**必须实测**（阶段 6 探针） | 估算 |

---

## 4. 需要执行的代码修改（按阶段）

> 原则：除下列修改外不动任何代码。所有修改都是模块级常量，改完直接运行脚本即可。
> `pre_evaluate.py` 输出 tag 自带 split/sampler/checkpoint 标识（`eval_<split>_h*_ch*_e*_s*_ckpt<stem>`），
> 各阶段/各组之间天然不冲突，**一律不要设 OUTPUT_TAG**；输出已存在时脚本拒绝覆盖（重跑需先删旧输出）。

### 4.1 阶段 1（surface 重训）：`pre_trainer.py`

```python
# 第 45 行附近，原：
EPOCH_OVERRIDES = {"surface_smoke": 4}
# 改为：
EPOCH_OVERRIDES = {"surface_smoke": 10}
```

**不要**先跑 4 epoch 再续训到 10：cosine scheduler 的 T_max 按 4 epoch 建，lr 已衰减到 0，续训等于白跑。一次定 10 epoch，早停（连续 2 epoch 验证恶化）自动兜底。

### 4.2 阶段 2（checkpoint 筛选，val day-1）：`pre_evaluate.py`

| 常量 | 值 | 说明 |
|---|---|---|
| `SPLIT` | `"val"` | 筛选/消融一律在 val |
| `ROLLOUT_DAYS` | `1` | |
| `SAMPLER_S_CHURN` | `0` | 筛选用先验赢家 sampler（见 §5 阶段 2 的循环依赖规则） |
| `ENSEMBLE_SIZE` | `1` | |
| `CHECKPOINT` | `"~/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/EpN.pth"` | **显式指定**，逐候选改 |
| `EVAL_STRIDE` / `MAX_WINDOWS` / `BATCH_SIZE` | `7` / `None` / `4` | |

### 4.3 阶段 3（sampler 消融，val day-1）：`pre_evaluate.py`

同 §4.2 的 SPLIT/ROLLOUT_DAYS/CHECKPOINT（锁定为阶段 2 选出的 EpN\*），逐组改：

| 组 | SAMPLER_S_CHURN | ENSEMBLE_SIZE | 预计耗时 |
|---|---|---|---|
| ① ch0_e1 | 0 | 1 | ~10 min（**可复用阶段 2 中 N\* 的筛选结果，若已跑则跳过**） |
| ② ch80_e1 | 80 | 1 | ~10 min |
| ③ 胜者 e4 | 胜者 | 4 | ~40 min |

### 4.4 阶段 4（val day-3 稳定性检查）：`pre_evaluate.py`

`SPLIT="val"`、`ROLLOUT_DAYS=3`、锁定组合（胜者 churn × 胜者 E）、`CHECKPOINT` 显式 EpN\*。E=1 ≈ 30 min；E=4 ≈ 2h。

### 4.5 阶段 5（正式 test，仅一次）：`pre_evaluate.py`

| 常量 | 值 |
|---|---|
| `SPLIT` | `"test"`（**全流程唯一一次触碰 test**） |
| `ROLLOUT_DAYS` | `15` |
| `CHECKPOINT` | 显式 `"…/EpN*.pth"`（即使 N\* 与 best.pth 同 epoch 也显式写，自文档化） |
| `SAMPLER_S_CHURN` / `ENSEMBLE_SIZE` | 阶段 3/4 锁定的胜者组合 |
| `EVAL_STRIDE` / `MAX_WINDOWS` | `7` / `None` —— **恒为全 154 窗口，禁止 MAX_WINDOWS 截断**（`starts[:max_windows]` 取最前面窗口，会造成抽样偏差） |

E=4 时全量 154 窗口 ≈ 10h，是否接受由阶段 3 的决策规则确定（见 §5）。

### 4.6 阶段 6（full3d，preliminary）：`pre_trainer.py` + `pre_evaluate.py`

- 两个文件的 `PRESET` 都改为 `"full3d"`；
- 探针实测 epoch 时间后，`EPOCH_OVERRIDES` 加一项 `"full3d": N`（N 的计算见 §5 阶段 6）；
- 评估侧：`BATCH_SIZE=1`、`EVAL_STRIDE=14`、`MAX_WINDOWS=24~32`、`ENSEMBLE_SIZE=1`、S_churn 用胜者；时间再紧则 `SAMPLING_STEPS=18`；
- **num_workers 调整须实测驱动**：若探针显示 I/O 瓶颈（batch 耗时波动大、GPU 利用率低），先 `pre_config.py` full3d `num_workers` 2→4 重测吞吐，确认提升后再试 8；不要直接跳 8（340MB/样本 × 多 worker 的预取与 pinned-memory 压力未验证）。

---

## 5. 24 小时分阶段执行方案

> 记启动时刻为 T0。所有训练/评估用 `nohup ... &` + log 文件，夜间无人值守。
> 每个阶段开头先 `source ~/miniconda3/etc/profile.d/conda.sh && conda activate diafno && cd /data2/user/zyq/projects/DiAFNO`。
> **test 纪律：test 只在阶段 5 触碰一次；阶段 0~4 的所有选择（checkpoint、sampler、ensemble）只用 val。**

### 阶段 0｜T0+0:00–0:10 — 环境核对

```bash
git -C /data2/user/zyq/projects/DiAFNO rev-parse --short HEAD && git -C /data2/user/zyq/projects/DiAFNO status --short
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 确认 GPU 3 空闲
CUDA_VISIBLE_DEVICES=3 python pre_smoke_test.py && python smoke_test.py # 全 PASS 才继续
```

### 阶段 1｜T0+0:10–4:50 — surface 重训（SD2，10 epoch）

完成 §4.1 修改后：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u pre_trainer.py \
  > /data2/user/zyq/checkpoints/PRE/train_surface_sd2.log 2>&1 &
tail -f /data2/user/zyq/checkpoints/PRE/train_surface_sd2.log
```

**启动后 5 分钟内必查（硬中止条件）**（log 第一屏）：
- `sigma_data=0.17120 (scale 2.000x)` —— 修复生效；**若为 0.08560 → 立即 kill**（改错文件/环境）；
- `run_dir=/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2` —— 新目录；
- 无 pre-flight 报错（Ep 碰撞 / loss.dat 截断）；non-finite loss 会自动 abort；skipped updates 占比异常高（如 >10%）→ 停下查 AMP 数值。

**训练中闸门（取代 v1 的 epoch-2 硬阈值）**：

| 类型 | 条件 | 动作 |
|---|---|---|
| 硬中止 | 上表机械性故障（sigma 错 / non-finite / skipped 异常） | 立即停，修复后重来 |
| 继续 | val_masked_relL2 仍在下降（**哪怕 epoch 2 只有 0.9 甚至 1.1**） | 继续跑 |
| 兜底 | 连续 2 epoch 恶化 | 交给现有 early stop |
| 软目标 | < 0.8 | 达到即健康信号；未达**不作 kill 依据**（旧跑同期 1.58，参考对照） |

注意：训练内验证用模型默认 S_churn=80、仅 24 个均匀窗口，数值偏悲观且噪声大，与评估口径不同——**不得用它做最终判断**，最终判断在阶段 2~5。训练完成标志：log 末尾 `done. best val_masked_relL2 = ...` 或早停信息。

### 阶段 2｜T0+4:50–5:30 — checkpoint 筛选（val day-1，ch0_e1）

按 §4.2 配置，对候选 checkpoint 逐个跑 val day-1（每个 ~10 min）：

- **N\* 选取**：`loss.dat` 第 3 列（val_masked_relL2）取 argmin，限实际跑完的 epoch；
- **候选集**：{N\*−1, N\*, N\*+1} ∩ [1, epochs_run]（N\*=1 时只有 {1,2}；早停时 N\*+1 可能不存在），最多 3 个；
- **动机**：best.pth 的选择标准（S_churn=80、单步、24 窗口）与最终目标准则错位，且单 epoch val 值噪声大，argmin 不可靠——用最终将用的 sampler 在 val 上直接比较；
- 选出 day-1 RMSE 最低的 **EpN\*.pth**，后续所有阶段 `CHECKPOINT` 显式指向它。

### 阶段 3｜T0+5:30–6:40 — sampler 消融（val day-1）

按 §4.3 顺序 ①→②→③：

```bash
CUDA_VISIBLE_DEVICES=3 python -u pre_evaluate.py 2>&1 | tee -a /data2/user/zyq/checkpoints/PRE/eval_ablation.log
```

每组结尾打印 `=== day-1 and overall comparison table ===`，记录各组 day-1 pooled RMSE。

- **循环依赖规则**：阶段 2 筛选已用 ch0_e1，其 N\* 结果直接复用为①；若②(ch80_e1) 反而胜出且优势 >10%，需用 ch80_e1 对阶段 2 的候选组快速复核一遍（+10 min 保险），确认 N\* 不变；
- **E=4 资源决策（在此锁定，供阶段 5 执行）**：
  - 胜者 E 相对另一者（day-1）优势 **< 10%** → 正式 test 用 **E=1**（全 154 窗口 ~2.5h），E 的优势作为 val 上的补充结论；
  - 优势 **≥ 10%** → 正式 test 投 ~10h 跑 **E=4 全 154 窗口**，**full3d 从计划中移除**（只剩统计+探针级别动作）。

### 阶段 4｜T0+6:40–7:30 — val day-3 自回归稳定性检查

按 §4.4 用锁定组合跑 val day-3（E=1 ≈ 30 min；E=4 ≈ 2h）。
**通过标准**：day-3 RMSE 未出现相对 day-1 的异常发散（如 day-3/day-1 比值显著劣于 persistence 的同比值），确认自回归不立即崩溃。

### 阶段 5｜T0+7:30 起 — 正式 test（全流程唯一一次）

按 §4.5 运行。E=1 约 2.5h（≈T0+10:00 完成）；E=4 约 10h（≈T0+17:30 完成）。
**通过标准**：控制台汇总表中各 lead day `model/pers < 1`（day-1 目标 ≤ 0.8）。
产出（在 `_SD2` 目录）：`eval_test_h15_ch*_e*_s123_ckptEpN.npz` + `figures_*/`。
**此结果同时是 full3d 的 go/no-go 依据；未通过则记录结果并停止，不进阶段 6。**

### 阶段 6｜test 通过且有剩余时间 — full3d（**preliminary**）

> **报告口径：full3d 结果一律标注 preliminary。** 1~2 epoch 的欠训练模型，其自回归 rollout 会高估崩溃程度（单步 val 尚可而 rollout 崩溃 ≠ 充分训练后的泛化表现），不得宣称"验证了泛化"。

1. 改 `PRESET="full3d"`（训练+评估两个文件），启动训练——先自动算全 30 层统计（~30 min，一次性）；
2. **探针**：盯 log 的 `[ep 1] batch 100/8394 ... Xs/batch`，epoch 时间 ≈ X × 8394 + 验证采样（16 窗口 × 64 前向）。首 epoch 含冷 I/O，偏保守正好。记录后 **kill**（未写 checkpoint，目录干净）；
3. 计算 `N = floor((24:00 − 当前时刻 − 3h 评估预留 − 1h 缓冲) / epoch 小时数)`，写入 `EPOCH_OVERRIDES["full3d"]=N`，重启正式训练；
4. 训练完成后按 §4.6 跑粗评估；
5. **决策规则**：
   - N ≥ 2 → 正常执行（结果仍标 preliminary）；
   - N = 1 → 单 epoch 价值有限：跑 1 epoch + `MAX_WINDOWS=8` 迷你评估，仅验证管线通畅；
   - N < 1（epoch > ~9h）→ 放弃 full3d 训练，改为：surface 的 val 集补充评估（`SPLIT="val"`、`ROLLOUT_DAYS=15`）充实保底结果。

**OOM 处理顺序**（full3d 显存紧张时）：`embed_dim` 128→96 → `implicit_layer` 2→1；不要动 patch（441 只能被 1/3/7/9 整除）。

### 时间预算汇总

| 情形 | 累计到 surface 正式结果 | full3d 可用 |
|---|---|---|
| E=1 胜出 | T0+9~11h | ~14h（统计+探针+N epoch+粗评估） |
| E=4 胜出、全量 test | T0+16~18h | 仅统计+探针或单 epoch 管线验证 |

---

## 6. 风险与注意事项

- **旧目录勿动**：`surface_smoke_..._C7`（无 _SD2）是反面基线；新实验全部落 `_SD2` 目录，物理隔离。
- **评估输出拒绝覆盖**：同一 tag 重跑会报错。如需重跑同配置，先删旧输出或设 `OUTPUT_TAG`（正常流程不需要）。
- **断点续训**：`RESUME_SIGMA_POLICY` 保持默认 `"error"`；本方案不涉及跨尺度续训，不要改成 migrate/adopt。
- **GPU 被抢**：执行中若 GPU 3 被其他任务占用，先 `nvidia-smi` 另选空闲卡改 `CUDA_VISIBLE_DEVICES`；训练脚本单卡即可。
- **若中途腾出第二张卡**：阶段 2/3 的 val 评估可两卡并行（省 ~1h）；或阶段 1 与 full3d 统计/探针并行。
- **随机性**：训练 seed 123、验证 seed 1234、评估逐窗口 seed（EVAL_SEED+start_day）均已内建，评估轨迹与 batch 大小无关，可放心调 BATCH_SIZE。S_churn=0 下初始噪声仍随机，复现依赖逐窗口 seed。
- **num_workers**：实测驱动，2→4 确认吞吐提升后再试 8（见 §4.6）。
- 执行完成后建议提交本报告对应的一切代码改动（阶段常量属临时配置，可不提交或注明）。

## 7. 交付物清单（执行完成后应存在）

| 物 | 路径 |
|---|---|
| surface SD2 checkpoint + 训练曲线 | `~/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/{Ep*.pth,best.pth,loss.dat}` |
| checkpoint 筛选结果（≤3 个，val day-1） | 同上目录 `eval_val_h1_ch0_e1_s123_ckptEpN*.npz` |
| sampler 消融（val day-1，①②③） | 同上目录 `eval_val_h1_ch*_e*_s123_ckptEpN*.npz` |
| val day-3 稳定性检查 | 同上目录 `eval_val_h3_ch*_e*_s123_ckptEpN*.npz` |
| **surface 正式 test（保底，仅一次）** | 同上目录 `eval_test_h15_ch*_e*_s123_ckptEpN.npz` + `figures_*/` |
| full3d checkpoint + 粗评估（若执行，**标 preliminary**） | `~/checkpoints/PRE/full3d_BS1_EMD128_I2_E4_S32_C7_SD2/` |
| 全部日志 | `~/checkpoints/PRE/train_surface_sd2.log`、`eval_ablation.log` 等 |

## 附录：关键路径速查

- 仓库：`/data2/user/zyq/projects/DiAFNO`（分支 `adapt-weather-ocean`）
- 环境：`conda activate diafno`（Python 3.10, torch 2.4.1+cu124）
- 对齐数据：`~/data_processed/PRE/aligned/`（u_rho/v_rho 各 209GB，已生产，勿重跑）
- 归一化缓存：`~/data_processed/PRE/norm/stats_d29_clipnone.npz`（surface 已有；full3d 首次自动算）
- checkpoint 根：`~/checkpoints/PRE/`
- 运行手册：`docs/PRE_runbook.md`
