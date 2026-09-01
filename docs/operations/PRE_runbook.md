# PRE_ocean_data 7→15 天海流预测运行手册

> 本文只记录可执行流程和固定约定；实验目的、对照、预期和实际结果分别存放在
> [实验索引](../experiments/README.md)下各实验的 `EXPERIMENT.md` 与 `RESULTS.md`。
> 当前 surface 确定性基线已通过 day-1 门槛，但 15-day overall 尚未优于 persistence；
> full3d 正式长训未准入。本手册中的 full3d 命令表示当前 K1 管线能力，不表示实验已完成。

> 任务：用连续 7 天的日平均三维 u/v（方案 A：原始交错网格自对齐到 rho 网格，保留网格方向分量，**不旋转**），
> 通过 DiAFNO 单步条件模型（扩散或确定性 persistence-residual）预测第 8 天，再自回归
> 滚动 15 次得到未来 1~15 天；
> 按预测天数 × u/v × 垂向层统计 masked RMSE/MAE，并与 persistence baseline 比较。
> 正式指标一律在**原生 u/v 交错网格**上、对**未经裁剪的原始物理真值**计算。

## 0. 网格、mask、归一化与时间约定

### 0.1 原生与 rho 网格 shape

| 网格 | u | v |
|---|---|---|
| 原生（ROMS C-grid 交错） | `(T, 30, 400, 440)`，`u[r,c]` 位于 rho(r,c) 与 rho(r,c+1) 之间 | `(T, 30, 399, 441)`，`v[r,c]` 位于 rho(r,c) 与 rho(r+1,c) 之间 |
| rho（模型网格） | `u_rho.npy` `(10591, 30, 400, 441)` | `v_rho.npy` 同 shape |

### 0.2 不旋转的插值规则（方案 A）

```text
u_rho[r,c] = mean_valid(u[r,c-1], u[r,c])   (c=1..439)；边界列直接复制
v_rho[r,c] = mean_valid(v[r-1,c], v[r,c])   (r=1..398)；边界行直接复制
```

陆地保持 NaN；**不**做 angle 旋转，u/v 仍是网格 xi/eta 方向分量。

### 0.3 双变量 mask

- `mask_u_rho.npy` / `mask_v_rho.npy`（均为 `(400, 441)`）：用与插值相同的模板从 `mask_u`/`mask_v` 构造
  （rho 点有效 ⟺ 相邻两个 face 中至少一个有效，边界单侧），保证对齐后 NaN 位置 == mask==0。
- `mask_uv.npy` = 两者交集（仅兼容/选用）。
- 预处理以提供的 mask 为准，对**每一天、全部 30 层**分两个方向处理：`mask==1`（海洋）处出现 NaN
  属于动态缺测，报首个 `(t, s, r, c)` 并立即停止，绝不用静态 mask 掩盖；`mask==0`（陆地）处带有的值
  （本数据集的 `u` 有 45 个静态陆侧边界 face，v 无）在共定位前直接置 NaN 丢弃并逐变量计数汇报。
  强制执行后对齐结果 NaN 位置 == `mask==0` 严格成立（首个 chunk 断言）。

### 0.4 归一化与裁剪决策

- 每变量独立 min-max 到 `[0,1]`，统计量只用 **train 段该变量的海洋点**（u 用 `mask_u_rho`，v 用 `mask_v_rho`）；
  归一化后陆地填 0；loss/指标全部 masked。
- **默认不做 percentile clipping（`clip_pct = None`）**；如需裁剪必须显式配置。裁剪策略、depth preset、
  split 边界和 mask 哈希都写入统计缓存文件名与内容，任何一项变化自动重算（包括 `splits` 字段缺失或与当前
  切分不一致时视为过期）。`hi <= lo` 直接报错。
- **pooled sigma 在归一化之前先 clip**：`x = clip(vals, lo, hi); x = (x-lo)/(hi-lo)` 之后才对 u+v 拼接
  计算总体标准差 —— 与 Dataset 的归一化完全一致。
- stats 缓存保存的是 **[0,1] 归一化空间**的 pooled sigma（surface 为 `0.08560`）；但 EDM 内部把图像
  归一化到 `[-1,1]`（`diffusion.py` 的 `images*2-1`），因此**训练/评估的 `sigma_data = 2.0 * stats["sigma"]`
  （`0.17120`）**。换算函数统一放在 `pre_config.py`（`SIGMA_DATA_SCALE`、`sigma_data_from_stats`、
  `sigma_data_from_checkpoint`），训练与评估必须调用同一个实现；**不要修改 stats 缓存**。
- 新 checkpoint 的 `config` 保存 `stats_sigma`、`sigma_data_scale`、`sigma_data` 三个字段；评估优先读
  checkpoint 内的 `sigma_data`。旧 checkpoint（无该字段）评估时回退旧尺度 `stats["sigma"]` 并打印明确提示。
  断点续训的尺度策略由 `RESUME_SIGMA_POLICY` 决定（`"error"` 默认直接报错 / `"migrate"` 显式迁移到 SD2 /
  `"adopt"` 旧尺度续训到独立子目录），详见第 6 节。
- **正式评估使用未经裁剪的原生物理真值**（`NativeUVReader` 直接读原始 `u.npy`/`v.npy`）；
  禁止用归一化 target 反归一化冒充原始真值。

### 0.5 连续时间切分

- 从权威 `ocean_time` 元数据逐文件校验：共 10591 个时刻、严格递增、相邻间隔**恰好 24 小时**
  （在 `datetime64[s]` 精度校验，23/25 小时间隔必须失败；错误报告索引、前后时间与实际间隔）。
  原始时间读取阶段不下压到天；缓存两份：
  `aligned/ocean_time_seconds.npy`（精确 `datetime64[s]`）与 `aligned/ocean_time.npy`
  （日期视图 `datetime64[D]`，兼容旧代码）。Dataset 启动时校验时间文件与数组长度一致。
- 切分（连续、不重叠、窗口不跨界）：train `[0, 8401)` / val `[8401, 9496)` / test `[9496, 10591)`。

### 0.6 rho→原生 评估映射（固定规则，无旋转）

```text
rho u → native u：沿 xi 对相邻 rho 点取平均 → (400, 440)
rho v → native v：沿 eta 对相邻 rho 点取平均 → (399, 441)
```

rollout 在 rho 网格上做；每个 lead day 的预测经此规则映射回原生网格后，
用原生 `mask_u`/`mask_v` 与原始物理真值比较。**只保存正式原生网格指标**，不保存 rho 补充指标。

### 0.7 原生 reader 统一布局

`NativeUVReader.get(day, days)` 对两套预设返回同一布局（sigma 轴在最后，与模型网格一致）：

```text
full3d (depth_index=None):  u_sel (days, 400, 440, 30), v_sel (days, 399, 441, 30)
surface (depth_index=29):    u_sel (days, 400, 440, 1),  v_sel (days, 399, 441, 1)
```

评估端只创建**一个** reader，每个窗口起点只 `get()` 一次并解包 `(u, v)`，不再转置。

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `scripts/profile_preprocess_align_uv.py` | 不触碰生产输出的性能探针：抽取真实 mmap chunk 到私有临时目录，分别统计 CPU、I/O；可用 `--gpu-compare` 调正式 CUDA 实现并与 NumPy 参考逐项核对，默认结束后删除 scratch |
| `scripts/preprocess_align_uv.py` | 方案 A 预处理：**单卡 CUDA 必需**；原始 u/v → rho 网格、双变量 mask、极值定位、ocean_time 精确校验（`verify_daily_time`，24h 间隔）；输出 `u_rho.npy`/`v_rho.npy`/`mask_u_rho.npy`/`mask_v_rho.npy`/`mask_uv.npy`/`ocean_time.npy`/`ocean_time_seconds.npy` |
| `pre_config.py` | 两套预设 `surface_smoke` / `full3d`（patch、embed_dim、batch、`val_windows` 等）+ 共享 sigma_data 换算（`SIGMA_DATA_SCALE`、`sigma_data_from_stats`、`sigma_data_from_checkpoint`、`run_tag_for(sd2=True, objective=...)`）+ objective 配置（`OBJECTIVES`、`validate_objective`、`objective_from_checkpoint`、`ensure_objective_compatible`、`RESIDUAL_TIME_SIGMA`、`MASK_SCHEME`）+ rank-0 终端进度辅助（`ProgressReporter`、`format_progress`、30s 间隔 `PROGRESS` 状态行），无副作用，可安全 import |
| `pre_models.py` | 确定性 persistence-residual 基线：`PersistenceResidualIAFNO`（包装与 EDM 相同的 `IAFNODiff`，`prediction = 条件最后一天 + 零初始化残差`，未训练时严格等于 persistence；`sample()` 忽略采样步数、与 rollout duck-type 兼容）+ `masked_mse_loss`（逐样本有效格点均值，与 diffusion.forward 的 mask 语义一致）；无副作用，可安全 import |
| `pre_dataset.py` | Dataset（滑窗/连续切分/逐变量 min-max/陆地填 0）、统计量计算与缓存（裁剪+clip 后 pooled sigma、split/mask 哈希过期检测）、`NativeUVReader`（统一布局原生真值）、双变量 mask 构造 |
| `pre_metrics.py` | 训练/评估/冒烟共享的纯指标函数：`rho_to_native`、`masked_error_sums`、`pooled_rmse`、`masked_rel_l2`、`oracle_native_error_sums`（rho-oracle 诊断基线）（无副作用，可安全 import） |
| `pre_rollout.py` | 无副作用自回归 rollout：`ensemble_rollout`（ensemble 成员完全独立、autocast 包裹、标量 seed 或**逐窗口 seeds**、`remask_feedback`+`ocean_mask` 可选陆地回灌重 mask；模型 duck-type：EDM 与确定性 residual 模型均可直接使用）、`expand_ensemble`、`ensemble_mean`（不依赖数据/模型模块，可安全 import） |
| `pre_trainer.py` | 单步训练入口（objective 可选：扩散 EDM masked 去噪损失 或 persistence-residual `masked_mse_loss`；双变量 masked loss、新 AMP API `torch.amp.GradScaler/autocast`、仅在实际 update 后 `scheduler.step()`、skipped-update/scale/lr 统计、cosine scheduler、`fork_rng` 隔离的均匀验证窗口、best/last checkpoint 共享同一 state、断点续训校验 objective/结构参数并严格采用 checkpoint 的 sigma_data（仅扩散）、residual 启动时零初始化 == persistence 自检、rank-0 tqdm/PROGRESS 进度、可选的一次性 `EPOCH_OVERRIDES`、连续 2 epoch 恶化提前停止） |
| `pre_evaluate.py` | 自回归 rollout（`ROLLOUT_DAYS`/`ENSEMBLE_SIZE`/`SAMPLER_S_CHURN`/`SAMPLER_SIGMA_MAX`/`EVAL_SEED`/`REMASK_FEEDBACK`）+ 按 checkpoint `config.objective` 重建扩散或确定性模型 + persistence/zero/rho-oracle 基线 + 原生网格正式指标 + 复现元数据（objective/residual_base/remask_feedback/sampler 等；确定性模型把采样参数显式记为不适用）+ 代表性图（逐窗口 seed、输出带 tag 且拒绝覆盖、单 reader、`masked_error_sums` 累加、rank-0 tqdm/PROGRESS 进度） |
| `pre_smoke_test.py` | 无额外依赖的 assert 回归测试（纯合成数据，直接调用正式实现） |
| `scripts/diag_leadtime_residual.py` | 长时效诊断：重放官方 rollout 协议，逐 lead day 统计 bias/方差比/逐窗口空间相关（评估 NPZ 不含的量） |
| `scripts/diag_region_breakdown.py` | 区域分解：validation day-1 协议下按 coastal（距陆地 ≤5 格）/offshore 报告 model 与 persistence 的 pooled RMSE |

数据产物（均在仓库外）：
- 对齐数据：`~/data_processed/PRE/aligned/{u_rho,v_rho}.npy`（各 209GB，float32，陆地 NaN）、双变量 mask、`ocean_time.npy`（日期视图）+ `ocean_time_seconds.npy`（精确时间）
- 归一化缓存：`~/data_processed/PRE/norm/stats_d{29|all}_clip{none|p0.1}.npz`（含裁剪策略、split、mask 哈希；删除或失配即重算）
- checkpoint：`~/checkpoints/PRE/<run_tag>/{Ep{n}.pth,best.pth,loss.dat}`（`run_tag_for` 默认带 `_SD2` 后缀，
  与旧 sd1 目录不重叠）；评估输出 `eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_ckpt{stem}[_tag].npz` + `figures_<tag>/`，
  输出已存在时**拒绝覆盖**

关键约定：
- **训练 objective**（`DIAFNO_OBJECTIVE`，默认 `diffusion`）：`diffusion` = 条件 EDM（历史路径，不变）；
  `persistence_residual` = 确定性基线 `PersistenceResidualIAFNO`——预测 = 条件第 7 天（persistence）+ 零初始化残差头输出，
  训练目标是双变量 mask 下的 `masked_mse_loss`，run 目录追加 `_RES` 后缀（如 `surface_smoke_..._SD2_RES`），
  绝不与扩散实验共用目录。零初始化保证未训练模型严格等于 persistence（trainer 启动时自检，评估端无法验证已训练模型）。
- **checkpoint 元数据**：所有新 checkpoint 的 `config` 记录 `objective`、`cond_chans`、`target_ch`、`mask_scheme`、`world_size`、
  `preset`、`train_mode`、`stats_sigma`，以及数据语义指纹 `norm_lo`/`norm_hi`（归一化范围）与
  `mask_version`（mask SHA-256 前 16 位）；扩散另存 `sigma_data_scale`/`sigma_data`，residual 另存 `residual_base`/`time_sigma`。
  评估端按 `config.objective` 重建对应模型类（旧 checkpoint 无该字段 → 一律按 diffusion 处理）；
  断点续训与评估重建都会校验 objective、`cond_chans/target_ch/mask_scheme/residual_base` 与语义指纹
  （归一化范围或 mask 版本与当前不一致 → 拒绝；residual 的 `time_sigma`/`stats_sigma` 不一致 → 拒绝；
  legacy checkpoint 缺这些字段只能打印告警、无法校验）——统计缓存或 mask 变化后绝不静默续训/评估。
- **rollout 陆地回灌（`REMASK_FEEDBACK`，默认 `False`）**：开启时每步预测先乘双变量 rho mask（陆地置 0）再进入下一窗口；
  关闭时保持历史行为（未 mask 的整帧回灌）。指标本就排除陆地，开关只改变模型下一步"看到"的条件；
  当前默认值 `False` 由实验 09 的单变量 A/B 确认；输出 tag 携带 `rf{0|1}`、metadata 记录 `remask_feedback`。
- **条件通道顺序**：day-major 交错 —— ch0=u(d0), ch1=v(d0), ch2=u(d1), …, ch13=v(d6)；rollout 时去掉最旧 2 通道、追加新帧 2 通道
- **垂向索引**：层 0=海底，层 29=海面；mask 是二维的（30 层 NaN 位置一致，预处理已全量校验）
- **loss/指标 mask**：训练 loss 用双变量 rho mask（`(1,2,H,W,Z)` 广播，逐样本除以各自有效点数）；
  验证相对 L2 = `sqrt(Σ((pred−tgt)²·mask)) / sqrt(Σ(tgt²·mask))`，逐样本后取批次均值（`pre_metrics.masked_rel_l2`）
- **总体 RMSE** = `sqrt(Σ总平方误差 / Σ总有效点数)`，不是逐层 RMSE 的算术平均；控制台摘要同样用
  `pre_metrics.pooled_rmse` 按 lead day 与按变量聚合

## 2. 步骤 0：预处理（一次性，CUDA 必需）

生产运行前建议先在目标磁盘做 scratch 探针；下例会抽取 3 个 50 天 chunk，比较正式 CUDA 实现与 NumPy 参考，
并输出可机读报告。临时数据默认自动删除，不会写 `aligned/`：

```bash
cd ~/projects/DiAFNO
GPU_ID=3  # 示例；先用 nvidia-smi 选择空闲卡
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/profile_preprocess_align_uv.py \
  --src /data2/user/zyq/datasets/PRE/processed \
  --scratch-root /data2/user/zyq/data_processed/PRE/profile_scratch \
  --gpu-compare \
  --raw-dyn /data2/user/zyq/datasets/PRE/raw/dyn \
  --profile-time-metadata \
  --report-json /data2/user/zyq/data_processed/PRE/profile_report.json
```

生产脚本固定使用过滤后可见的逻辑 `cuda:0`（`DEVICE_INDEX=0`），运行耗时由源盘读取、目标盘 flush、GPU 和
10591 个 NetCDF 时间元数据扫描共同决定，应以前述报告为准。**脚本会以 `w+` 打开 `u_rho.npy` / `v_rho.npy`，
已有同名生产结果会被覆盖；确认确实需要重跑后再执行。**

```bash
cd ~/projects/DiAFNO
GPU_ID=3  # 示例；先用 nvidia-smi 选择空闲卡
CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python scripts/preprocess_align_uv.py > ~/data_processed/PRE/aligned/preprocess.log 2>&1 &
tail -f ~/data_processed/PRE/aligned/preprocess.log
```

完成后检查日志：
- `[time] verified 10591 strictly increasing daily timestamps ...` 存在；
- mask 校验：开头出现 `[mask] day0/layer0 probe discards ... {'u': 45}`（45 个静态陆侧边界 u-face
  的值被丢弃，属预期）；结尾 `[mask] u: discarded ... values ...`（45×30 层×10591 天 = 14297850）。
  若 mask==1 的海洋点出现 NaN（动态缺测）会报 `(t, s, r, c)` 并抛错终止；
- `[extrema]` 四行给出原始与对齐后 u/v 极值及其 `(t, s, r, c)` 位置——只记录，不擅自判定异常值。

## 3. 步骤 1：真实数据训练烟测与正式训练

```bash
cd ~/projects/DiAFNO
GPU_ID=3  # 示例；先用 nvidia-smi 选择空闲卡
CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_trainer.py
```

- 不设置环境变量时是安全的 `DIAFNO_TRAIN_MODE=smoke`：模型结构和 400×441×1 网格与正式
  `surface_smoke` 完全相同，但只运行每卡 4 个 batch、1 epoch、4 个 sampler step 和少量验证窗口。
  输出目录带 `_S4_..._SMOKE`，不会与正式 checkpoint 混用。
- 烟测以末尾出现 `SMOKE PASS` 为通过：train/val 数值有限、每卡至少一次真实 optimizer update、
  无 AMP skipped update，且 `Ep1.pth`、`best.pth`、`loss.dat` 均已写出。它验证数据/I/O/显存/AMP/
  采样/checkpoint 链路，不是模型精度结论；科学门槛仍是第 4 节的 native RMSE 优于 persistence。
- 首次运行先计算归一化统计量（表层只读层 29，约 1 分钟），缓存到 `~/data_processed/PRE/norm/`。
- `full` 表层预设：400×441×1、patch (4,3,1)（整除，无 padding）、14.7k token、每卡 B=4、验证窗口 24 个
  （`np.linspace` 均匀覆盖整个 val 期，固定 seed 1234，跨 epoch 可比）。
- `full` 模式的正常训练轮数只由 `pre_config.py` 管理：`surface_smoke` 最多 10 epoch，`full3d` 最多 50 epoch；
  `EPOCH_OVERRIDES = {}` 默认不覆盖，仅供显式的一次性短跑。`sigma_data = 2.0 * stats["sigma"] = 0.17120`
  （见 0.4）；checkpoint 落在带 `_SD2` 的目录。
- 每个 epoch 保存 `Ep{n}.pth` + （新最佳时）`best.pth`，两者共享同一 state；连续 2 个 epoch 验证指标恶化
  自动提前停止；每个 epoch 打印 `train_loss`、`val_masked_relL2`、成功/跳过更新数、`grad_scale`、`lr`；
  每 100 个 batch 打印进度与 batch 耗时。
- 检查点：`train_loss` 应逐 epoch 下降；`val_masked_relL2`（物理单位、双变量 rho mask、扩散采样）应明显低于 1 且下降。
- 单卡正式训练：`DIAFNO_TRAIN_MODE=full CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_trainer.py`。
- 单机多卡正式训练：`DIAFNO_TRAIN_MODE=full CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone
  --nproc_per_node=4 pre_trainer.py`。实现为一进程一卡 DDP；stats 只由 rank 0 首次生成，训练和验证均分片，
  checkpoint 只由 rank 0 写。`batch_size` 是**每卡** batch，多卡不自动缩放 lr；checkpoint 记录
  `world_size/per_device_batch_size/effective_batch_size`，输出目录带 `_DDP4`。
- 切换 preset 用 `DIAFNO_PRESET=surface_smoke|full3d`。切换 objective 用
  `DIAFNO_OBJECTIVE=diffusion|persistence_residual`（默认 `diffusion`，行为与历史完全一致）；
  `persistence_residual` 走确定性 `masked_mse_loss`（无采样、无 sigma 调度），run 目录带 `_RES`。
  **实验 08 静态 mask 输入（arm B）**：`DIAFNO_STATIC_MASK=1` 把双变量 rho mask 的
  2 个通道经 `pre_rollout` 的 `static_cond` 单独拼入 backbone 条件（动态滑窗保持纯 14
  通道，persistence base 语义不变），仅限 `persistence_residual`，run 目录追加 `_MSK`，
  checkpoint 记录 `static_mask_input`/`model_cond_chans`，评估端按元数据自动重建；
  A/B 结论为不保留（2026-08-31），路径保留供复核。
  断点续训用 `DIAFNO_CHECKPOINT=/abs/path/EpN.pth`；必须保持原 `train_mode`、`DIAFNO_OBJECTIVE`
  和 GPU 数，避免改变 scheduler/有效 batch 语义；checkpoint 的 objective / 输入通道 / mask 方案
  与当前不一致会直接拒绝（绝不把一类 checkpoint 装进另一类模型）。扩散尺度策略见第 6 节
  `RESUME_SIGMA_POLICY`（默认 `"error"`：尺度不一致直接报错；`"adopt"` 可继续旧尺度（输出写入独立的
  `legacy_resume/` 子目录），`"migrate"` 可显式迁移到 SD2）；residual objective 无 sigma 调度，该策略不适用。
  旧 checkpoint 没有 `world_size` 元数据，只允许单卡续训。多卡产物评估时应在 `pre_evaluate.py`
  显式设置 `CHECKPOINT`，因为自动 run tag 不猜测训练时的 DDP world size。

## 4. 步骤 2：冒烟评估（rollout + persistence，原生网格）

```bash
GPU_ID=3  # 示例；先用 nvidia-smi 选择空闲卡
CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_evaluate.py   # PRESET 与训练一致
```

- 默认 test 段、每 7 天一个起点（~154 个窗口）、每窗口 `ROLLOUT_DAYS`（默认 15）步 × 32 采样步 × Heun 2 次前向，
  全程 `autocast`（AMP，与旧评估数值路径一致）。
- 采样配置为模块常量：`ROLLOUT_DAYS` / `ENSEMBLE_SIZE`（成员各自独立自回归，最后取均值）/ `SAMPLER_S_CHURN` /
  `SAMPLER_SIGMA_MAX` / `EVAL_SEED` / `OUTPUT_TAG`（可选后缀）。`SAMPLER_SIGMA_MAX=None` 保持 EDM 默认值 80；
  显式覆盖会自动进入输出 tag 和 `.npz` 元数据。当前默认 `SAMPLER_S_CHURN=0`，来自 surface SD2 validation
  消融；复现旧 `churn=80` 实验时必须显式改回 80。`ENSEMBLE_SIZE=1` 等价于原单轨迹 rollout。
- **逐窗口 seed**：每个窗口用自己的种子 `EVAL_SEED + start_day`，轨迹只取决于窗口本身，与 `BATCH_SIZE`、
  loader 分组无关（`BATCH_SIZE` 仍写入元数据）。
- 想先快速验证：`MAX_WINDOWS = 8`。
- **正式指标**：rho 预测 → 原生 u/v 重采样（`rho_to_native`）→ 与原始 `u.npy`/`v.npy`（未裁剪）比较，
  用 `mask_u`/`mask_v` 与 `masked_error_sums` 累加；persistence = 第 7 天**原生物理** u/v 重复 `ROLLOUT_DAYS` 次。
- **诊断基线**：`zero`（原生网格全零预测）与 `rho-oracle`（数据集真实 rho target 反归一化 → `rho_to_native`，
  度量 native→rho→native 转换本身的不可逆误差）。
- 输出 `<ckpt_dir>/eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_ckpt{stem}[_tag].npz`（输出已存在则**拒绝覆盖**）：
  - 正式原生网格指标：`rmse_model` / `mae_model` / `rmse_persistence` / `mae_persistence` / `rmse_zero` /
    `rmse_oracle` / `mae_*` / `valid_count`，shape `(ROLLOUT_DAYS, 2, Z)` —— surface（`surface_smoke`）`Z=1`，full3d `Z=30`；
  - 复现元数据：`rollout_days`、`ensemble_size`、`S_churn`、`sigma_max`、`seed`、`seed_scheme`、`batch_size`、`sigma_data`、
    `checkpoint_path`、`checkpoint_epoch`、`preset`、`sampling_steps`、`stride`、`window_start_indices`、
    `norm_lo/norm_hi/norm_sigma`、`grid_mapping_rule`。
- 代表性图输出到 `figures_<tag>/d{1,3,5,7,10,15}_s{layer}_{u|v}.png`（truth/prediction/error 三面板，
  表层/中层/底层，u、v）。
- **通过标准**：模型在各 lead day 的原生 masked RMSE 稳定低于 persistence（model/pers 比值 < 1）。
- **objective 感知重建**：评估先读 checkpoint 的 `config.objective` 再决定重建哪类模型——`diffusion` 走
  EDM 采样；`persistence_residual` 走确定性前向（忽略采样步数、不消耗 RNG，seed 无关）。确定性 checkpoint
  的评估必须显式设置 `CHECKPOINT` 指向 `*_RES` 目录里的 `best.pth`/`Ep{n}.pth`（自动 run tag 不猜测 objective）；
  `ENSEMBLE_SIZE > 1` 会被强制回 1 并打印 NOTE（确定性成员完全相同，平均无意义）。输出 metadata 新增
  `objective`/`residual_base`/`remask_feedback`/`sampler`/`sampler_note`/`time_sigma`；确定性模型的采样相关
  参数（`S_churn`/`sigma_max`/`sampling_steps`/`seed`）在 `sampler_note` 中**显式记为不适用**
  （数值字段写 `sigma_data=nan`、`sampling_steps=-1`）。
- **remask A/B**：`REMASK_FEEDBACK = True` 时每步预测先重应用双变量 rho mask（陆地置 0）再回灌下一窗口；
  默认 `False` 保持历史行为。输出 tag 携带 `rf{0|1}`，A/B 两组产物互不覆盖；实验 09 的结论是维持该默认值。
- **validation 选型（checkpoint 选择协议）**：正式选择指标是 **validation day-1 native RMSE**：对候选
  `Ep{n}.pth` 逐个用 `SPLIT = "val"`、`ROLLOUT_DAYS = 1` 跑 `pre_evaluate.py`（确定性模型每窗口仅一次前向，
  开销很小），比较 day-1 pooled RMSE 与 persistence 比值后选型；test 只在配置冻结后报告一次。
  **`best.pth` 是按训练日志的 `val_masked_relL2`（rho 网格相对 L2）产生的，不是 day-1 native RMSE
  最优的 checkpoint**——选型时禁止直接取用 `best.pth`，必须逐个评估 `Ep{n}.pth` 后按 native RMSE 挑选；
  训练日志里的 `val_masked_relL2` 只用于 early stop 与粗粒度监控，**不是**选型指标。
- **历史采样消融**：`ROLLOUT_DAYS=1` 下实际比较了 `churn=0/E=1`、`churn=80/E=1` 和
  `churn=0/E=4`，再以选出的 `churn=0/E=1` 跑完整 15 天。`sigma_data` 不在消融中手工指定：新 checkpoint 使用其
  `config.sigma_data`（surface SD2 通常为 0.17120）；无该字段的旧 checkpoint 才按兼容策略回退到旧尺度并告警。
- 历史 `sigma_max=3` 诊断产物来自提交 `fd3fc4c`；当前 HEAD 已正式提供 `SAMPLER_SIGMA_MAX`。
  设为 `3` 即可重跑，输出会自动带 `sm3` tag；仍须遵守拒绝覆盖规则。

## 5. 步骤 3：全 30 层全量

1. 当前先运行 `DIAFNO_PRESET=full3d DIAFNO_OBJECTIVE=persistence_residual python pre_trainer.py`
   做 K1 同结构真实数据烟测；正式长训必须再满足实验 06 的数据/资源/pilot 门槛，不因 smoke
   通过自动启动。评估端需把 `pre_evaluate.py` 的 `PRESET` 设为 `"full3d"`；
2. 首次运行会重算全 30 层归一化统计（3 遍流式扫描 ~530GB，约 15~30 分钟，一次性）；
3. 预设：patch (4,3,2) → 100×147×15 = 220.5k token、embed_dim 128、implicit 2、B=1、验证窗口 16 个。
   **若 OOM，按此顺序调**：`embed_dim` 128→96 → `implicit_layer` 2→1 → `batch_size` 已为 1；不要轻易改 patch（441 只能被 1/3/7/9 整除）；
4. 训练窗口逐样本读取 ~340MB（两个变量 8 天）；数据总量 418GB < 内存 943GB，首轮后主要靠页缓存；
5. 评估时 `BATCH_SIZE` 建议 1~2；`EVAL_STRIDE` 可加大到 14 先出粗结果。

## 6. 复现与注意事项

- `pre_trainer.py` / `pre_evaluate.py` 都是**脚本**（模块顶层执行），不要 import 它们；共享配置一律从 `pre_config.py` 取。
- 断点续训：设置 `DIAFNO_CHECKPOINT`；保存含 model/optimizer/scheduler/scaler/epoch/`best_val`，
  续训时恢复或从 `loss.dat` 重算 `best_val`，`loss.dat` 始终写完整历史（不静默覆盖）。
  **尺度策略 `RESUME_SIGMA_POLICY`**：`"error"`（默认——checkpoint 的 `sigma_data` 与当前 SD2 尺度不同
  时**直接报错**，绝不静默混合尺度）；`"migrate"`（显式尺度迁移——保留当前 SD2 尺度，继续写在 SD2 目录）；
  `"adopt"`（显式旧尺度续训——采用 checkpoint 的旧 `sigma_data`，**输出写入 checkpoint 目录旁的独立子目录
  `legacy_resume/`**（从 `legacy_resume` 内 checkpoint 续训时直接复用该目录，不嵌套）；**历史（含 best_val
  重算）从原实验目录的 `loss.dat` 读取**，因此续训写出的 `loss.dat` 始终是**完整历史**（旧历史 + 续训部分），
  保存的 config 记录**实际 scale=1.0**；原实验的 `Ep{n}.pth`/`loss.dat` 绝不会被触碰，续训产物
  永远不会被误认为 SD2）。训练循环前有预检：目标 `Ep{n}.pth` 已存在或 `loss.dat` 会被截断 → 直接拒绝
  （不白跑一轮）；每 epoch 保存前还会复查 `loss.dat` 截断（覆盖提前停止场景）。
  `torch.load` 默认 `weights_only=True`，旧 checkpoint 若确实不兼容只能对明确可信的项目 checkpoint 显式关闭。
  新最佳 epoch 的 `Ep{n}.pth` 与 `best.pth` 共享**同一个** state（先判 `is_best`、更新 `best_val`，
  再构造 state 并保存）。旧实现先保存 `Ep{n}.pth` 再更新 `best_val` 后保存 `best.pth`，
  导致新最佳 epoch 的 `Ep{n}.pth` 里记录的 `best_val` 是更新前的旧值（过期），`best.pth` 反而是新的。
- 扩散采样有随机性：训练固定 `torch.manual_seed(123)`；验证阶段整体包在 `torch.random.fork_rng()` 内，
  在上下文内固定 `VAL_SEED=1234`（CPU 始终隔离；CUDA 时 fork 当前 device），验证结束后训练 CPU/CUDA RNG
  状态恢复，验证采样不会扰动训练随机流。评估为**逐窗口 seed**（`EVAL_SEED + start_day`），轨迹与 batch 大小无关；
  正式报告请记录 seed、sampling_steps、rollout_days、ensemble_size、S_churn 与 checkpoint
  （评估元数据中已自动记录）。
- persistence baseline 已内建于 `pre_evaluate.py`，无需单独运行。
- 回归测试：`python smoke_test.py`（CPU 设备/checkpoint）+ `python pre_smoke_test.py`（PRE 管线纯合成数据断言）。
- 原始仓库的 `trainer.py` 保持未动；对 `IAFNO.py`/`diffusion.py` 的改动向后兼容（`cond_chans=None` 时行为同旧版 doubling；loss 不传 mask 时与原逻辑一致）。
- 本任务不再声称"clip 归一化已能吸收异常值"：默认不裁剪；预处理只记录极值位置，评估只用未经裁剪的原始真值。

## 7. 终端进度与监控约定（pre_trainer / pre_evaluate）

实现位于 `pre_config.py`（`ProgressReporter` + `format_progress`），只复用既有 `tqdm` 与标准库
`time.perf_counter()`，无新依赖、无监控服务：

- **交互式终端**：单层 `tqdm` 进度条（每 epoch / 每评估阶段各一条，leave=False），显示计数/总数、
  已运行时间、ETA 与速率；postfix 显示当前 loss、lr、optimizer update 数与 AMP skipped 数
  （评估 postfix 为已完成窗口数与 running day-1 model/persistence RMSE 及比值）。关键 epoch summary、
  `SMOKE PASS`、汇总表仍然按原样整行输出（经 `tqdm.write` 写出，不会被进度条打碎）。
- **非交互终端 / 日志重定向 / 监控 agent**：不输出回车覆盖的动态条；开始、结束、失败必各输出一条，
  运行期间**至少每 30 秒**输出一条完整换行并立即 flush 的状态行。周期状态行是**时间驱动**的：
  守护心跳线程按间隔补发，即使单个 batch / rollout 步阻塞超过间隔（如扩散 15 天 × Heun 采样）也不会
  中间静默。格式为可解析的 `key=value`：
  `PROGRESS phase=train epoch=1/4 step=120/2101 elapsed_s=91.2 eta_s=1506.4 step_per_s=1.31 sample_per_s=5.24 loss=0.0187 lr=1e-4 status=running`
  （`phase` 恒为首个字段、`status` 恒为末字段；**值中一切空白**（空格/换行/制表符）统一替换为 `_`，
  多行异常信息也保持单行可解析）。
- **状态词汇表（稳定解析契约）**：`start` = 阶段开始；`running` = 周期心跳/进度；
  `phase_done` = **本阶段**结束（一个训练 epoch、评估 rollout 循环），不是脚本结束；
  `completed` = **整个脚本**成功结束，只由入口脚本自身在所有产物落盘后输出（评估端在 NPZ + 汇总 +
  全部图之后），per-phase reporter 永远不会输出 `completed`；`failed` = 运行中止。
  监控端应以 `completed`/`failed` 判定运行终结，`phase_done` 仅作阶段推进信号。
- **单位与 DDP**：训练速率标注 `step_per_s`（每 rank）与 `sample_per_s`（= 每 rank batch × world size ×
  `step_per_s`，即**全局**样本吞吐）；评估标注 `window_per_s` 与 `sample_per_s`（窗口数为真实窗口数，
  按每 batch 实际窗口数推进，不足 batch 的尾批也准确）。DDP 下 train/val loader 都是 rank 分片的，
  进度行的计数/总数是**本 rank 分片**的，行内以 `scope=rank0_shard_of_4` 显式标注（单卡为
  `scope=whole_split`）；只有 rank 0 输出进度，其他 rank 不重复刷屏；异常信息保留 rank/epoch/batch 上下文。
- **生命周期**：trainer 在 pre-flight 之后输出 `PROGRESS phase=train status=start`，正常结束（含 early stop）
  在 smoke gate 后输出 `status=completed`；guarded 训练/验证块内任何异常（non-finite、OOM、SMOKE FAIL 等）
  先输出 `status=failed`（`stage=run`，含 `error=` 与已运行秒数）再退出。**不受 guarded 块保护的阶段**
  （初始化、数据/模型构建、pre-flight 拒绝、评估后处理）由 `pre_config.install_progress_failure_hook`
  安装的 `sys.excepthook` 兜底：任何未捕获异常先输出 `status=failed`（`stage=setup|data_model|rollout|
  postprocess` 标明发生位置）再打印原始 traceback，且与 guarded handler 通过去重标记保证每次运行
  只有一条 failed 行。即使 smoke 短于 30 秒，`status=start` 与终结状态也必然存在。
