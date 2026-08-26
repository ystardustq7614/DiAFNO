# PRE_ocean_data 7→15 天海流预测运行手册

> 任务：用连续 7 天的日平均三维 u/v（方案 A：原始交错网格自对齐到 rho 网格，保留网格方向分量，**不旋转**），
> 通过 DiAFNO 单步条件扩散模型预测第 8 天，再自回归滚动 15 次得到未来 1~15 天；
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
- 预处理对**每一天、全部 30 层**校验原始 NaN 模式 == `mask==0`，不一致即报首个 `(t, s, r, c)` 并停止，
  绝不用静态 mask 掩盖动态缺测。

### 0.4 归一化与裁剪决策

- 每变量独立 min-max 到 `[0,1]`，统计量只用 **train 段该变量的海洋点**（u 用 `mask_u_rho`，v 用 `mask_v_rho`）；
  归一化后陆地填 0；loss/指标全部 masked。
- **默认不做 percentile clipping（`clip_pct = None`）**；如需裁剪必须显式配置。裁剪策略、depth preset、
  split 边界和 mask 哈希都写入统计缓存文件名与内容，任何一项变化自动重算（包括 `splits` 字段缺失或与当前
  切分不一致时视为过期）。`hi <= lo` 直接报错。
- **pooled sigma 在归一化之前先 clip**：`x = clip(vals, lo, hi); x = (x-lo)/(hi-lo)` 之后才对 u+v 拼接
  计算总体标准差 —— 与 Dataset 的归一化完全一致。
- `sigma_data` = 两变量合并（u+v 拼接）归一化值的真实总体标准差，包含 u/v 均值差产生的组间方差。
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
| `scripts/preprocess_align_uv.py` | 方案 A 预处理：原始 u/v → rho 网格、双变量 mask、极值定位、ocean_time 精确校验（`verify_daily_time`，24h 间隔）；输出 `u_rho.npy`/`v_rho.npy`/`mask_u_rho.npy`/`mask_v_rho.npy`/`mask_uv.npy`/`ocean_time.npy`/`ocean_time_seconds.npy` |
| `pre_config.py` | 两套预设 `surface_smoke` / `full3d`（patch、embed_dim、batch、`val_windows` 等），无副作用，可安全 import |
| `pre_dataset.py` | Dataset（滑窗/连续切分/逐变量 min-max/陆地填 0）、统计量计算与缓存（裁剪+clip 后 pooled sigma、split/mask 哈希过期检测）、`NativeUVReader`（统一布局原生真值）、双变量 mask 构造 |
| `pre_metrics.py` | 训练/评估/冒烟共享的纯指标函数：`rho_to_native`、`masked_error_sums`、`pooled_rmse`、`masked_rel_l2`（无副作用，可安全 import） |
| `pre_trainer.py` | 单步 teacher-forcing 训练入口（双变量 masked loss、AMP、cosine scheduler、`fork_rng` 隔离的均匀验证窗口、best/last checkpoint 共享同一 state、断点续训恢复 best_val 与历史） |
| `pre_evaluate.py` | 15 步自回归 rollout + persistence + 原生网格正式指标 + 复现元数据 + 代表性图（单 reader、无转置、`masked_error_sums` 累加） |
| `pre_smoke_test.py` | 无额外依赖的 assert 回归测试（纯合成数据，直接调用正式实现） |

数据产物（均在仓库外）：
- 对齐数据：`~/data_processed/PRE/aligned/{u_rho,v_rho}.npy`（各 209GB，float32，陆地 NaN）、双变量 mask、`ocean_time.npy`（日期视图）+ `ocean_time_seconds.npy`（精确时间）
- 归一化缓存：`~/data_processed/PRE/norm/stats_d{29|all}_clip{none|p0.1}.npz`（含裁剪策略、split、mask 哈希；删除或失配即重算）
- checkpoint：`~/checkpoints/PRE/<run_tag>/{Ep{n}.pth,best.pth,loss.dat}`、`eval_test.npz`、`figures/`

关键约定：
- **条件通道顺序**：day-major 交错 —— ch0=u(d0), ch1=v(d0), ch2=u(d1), …, ch13=v(d6)；rollout 时去掉最旧 2 通道、追加新帧 2 通道
- **垂向索引**：层 0=海底，层 29=海面；mask 是二维的（30 层 NaN 位置一致，预处理已全量校验）
- **loss/指标 mask**：训练 loss 用双变量 rho mask（`(1,2,H,W,Z)` 广播，逐样本除以各自有效点数）；
  验证相对 L2 = `sqrt(Σ((pred−tgt)²·mask)) / sqrt(Σ(tgt²·mask))`，逐样本后取批次均值（`pre_metrics.masked_rel_l2`）
- **总体 RMSE** = `sqrt(Σ总平方误差 / Σ总有效点数)`，不是逐层 RMSE 的算术平均；控制台摘要同样用
  `pre_metrics.pooled_rmse` 按 lead day 与按变量聚合

## 2. 步骤 0：预处理（一次性，约 15~60 分钟 + 时间校验若干分钟）

```bash
cd ~/projects/DiAFNO
nohup python scripts/preprocess_align_uv.py > ~/data_processed/PRE/aligned/preprocess.log 2>&1 &
tail -f ~/data_processed/PRE/aligned/preprocess.log
```

完成后检查日志：
- `[time] verified 10591 strictly increasing daily timestamps ...` 存在；
- mask 一致性校验全部通过（任一 `(t, s, r, c)` 不匹配都会抛错终止）；
- `[extrema]` 四行给出原始与对齐后 u/v 极值及其 `(t, s, r, c)` 位置——只记录，不擅自判定异常值。

## 3. 步骤 1：表层冒烟训练（GPU 5/6 空闲）

```bash
cd ~/projects/DiAFNO
CUDA_VISIBLE_DEVICES=5 python pre_trainer.py    # PRESET='surface_smoke'
```

- 首次运行先计算归一化统计量（表层只读层 29，约 1 分钟），缓存到 `~/data_processed/PRE/norm/`。
- 预设：400×441×1、patch (4,3,1)（整除，无 padding）、14.7k token、B=4、10 epoch、验证窗口 24 个
  （`np.linspace` 均匀覆盖整个 val 期，固定 seed 1234，跨 epoch 可比）。
- 检查点：`train_loss` 应逐 epoch 下降；`val_masked_relL2`（物理单位、双变量 rho mask、扩散采样）应明显低于 1 且下降。
- 快速试跑可把 `pre_config.py` 中 `surface_smoke` 的 `max_train_windows` 改为 2000、`num_epochs` 改为 2。

## 4. 步骤 2：冒烟评估（rollout + persistence，原生网格）

```bash
CUDA_VISIBLE_DEVICES=5 python pre_evaluate.py   # PRESET 与训练一致
```

- 默认 test 段、每 7 天一个起点（~154 个窗口）、每窗口 15 步 × 32 采样步 × Heun 2 次前向。
- 想先快速验证：`MAX_WINDOWS = 8`。
- **正式指标**：rho 预测 → 原生 u/v 重采样（`rho_to_native`）→ 与原始 `u.npy`/`v.npy`（未裁剪）比较，
  用 `mask_u`/`mask_v` 与 `masked_error_sums` 累加；
  persistence = 第 7 天**原生物理** u/v 重复 15 次。
- 输出 `~/checkpoints/PRE/<run_tag>/eval_test.npz`：
  - 仅正式原生网格指标：`rmse_model` / `mae_model` / `rmse_persistence` / `mae_persistence` / `valid_count`，
    shape `(15, 2, Z)` —— surface（`surface_smoke`）`Z=1`，full3d `Z=30`；
  - 复现元数据：`checkpoint_path`、`checkpoint_epoch`、`preset`、`seed`、`sampling_steps`、`stride`、
    `window_start_indices`、`norm_lo/norm_hi/norm_sigma`、`grid_mapping_rule`。
- 代表性图输出到 `figures/d{1,3,5,7,10,15}_s{layer}_{u|v}.png`（truth/prediction/error 三面板，
  表层/中层/底层，u、v）。
- **通过标准**：模型在各 lead day 的原生 masked RMSE 稳定低于 persistence（model/pers 比值 < 1）。

## 5. 步骤 3：全 30 层全量

1. `pre_trainer.py` 与 `pre_evaluate.py` 中 `PRESET = "full3d"`；
2. 首次运行会重算全 30 层归一化统计（3 遍流式扫描 ~530GB，约 15~30 分钟，一次性）；
3. 预设：patch (4,3,2) → 100×147×15 = 220.5k token、embed_dim 128、implicit 2、B=1、验证窗口 16 个。
   **若 OOM，按此顺序调**：`embed_dim` 128→96 → `implicit_layer` 2→1 → `batch_size` 已为 1；不要轻易改 patch（441 只能被 1/3/7/9 整除）；
4. 训练窗口逐样本读取 ~340MB（两个变量 8 天）；数据总量 418GB < 内存 943GB，首轮后主要靠页缓存；
5. 评估时 `BATCH_SIZE` 建议 1~2；`EVAL_STRIDE` 可加大到 14 先出粗结果。

## 6. 复现与注意事项

- `pre_trainer.py` / `pre_evaluate.py` 都是**脚本**（模块顶层执行），不要 import 它们；共享配置一律从 `pre_config.py` 取。
- 断点续训：设置 `pre_trainer.py` 的 `checkpoint_path`；保存含 model/optimizer/scheduler/scaler/epoch/`best_val`，
  续训时恢复或从 `loss.dat` 重算 `best_val`，`loss.dat` 始终写完整历史（不静默覆盖）。
  新最佳 epoch 的 `Ep{n}.pth` 与 `best.pth` 共享**同一个** state（先判 `is_best`、更新 `best_val`，
  再构造 state 并保存）。旧实现先保存 `Ep{n}.pth` 再更新 `best_val` 后保存 `best.pth`，
  导致新最佳 epoch 的 `Ep{n}.pth` 里记录的 `best_val` 是更新前的旧值（过期），`best.pth` 反而是新的。
- 扩散采样有随机性：训练固定 `torch.manual_seed(123)`；验证阶段整体包在 `torch.random.fork_rng()` 内，
  在上下文内固定 `VAL_SEED=1234`（CPU 始终隔离；CUDA 时 fork 当前 device），验证结束后训练 CPU/CUDA RNG
  状态恢复，验证采样不会扰动训练随机流。正式报告请记录 seed、sampling_steps 与 checkpoint
  （评估元数据中已自动记录）。
- persistence baseline 已内建于 `pre_evaluate.py`，无需单独运行。
- 回归测试：`python smoke_test.py`（CPU 设备/checkpoint）+ `python pre_smoke_test.py`（PRE 管线纯合成数据断言）。
- 原始仓库的 `trainer.py` 保持未动；对 `IAFNO.py`/`diffusion.py` 的改动向后兼容（`cond_chans=None` 时行为同旧版 doubling；loss 不传 mask 时与原逻辑一致）。
- 本任务不再声称"clip 归一化已能吸收异常值"：默认不裁剪；预处理只记录极值位置，评估只用未经裁剪的原始真值。