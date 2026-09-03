# 当前困难与下一步：确定性 multi-step U/V 预测

> 状态：**工作包 1–6 均已启动：1–4 完成；5（代表层）完成且全门槛 Go；
> 6（full3d）完成 probe/K1/pilot，K3 按预注册条件阻塞，待预算决策**
> 制定日期：2026-09-01
> 首要科学目标：提高未来 1–15 天区域海流 `u/v` 的确定性点预测能力
> 当前起点：实验 07 surface `PersistenceResidualIAFNO` Ep10
> 适用顺序：全层数据基线 → surface MS5 → 条件式 MS10 → 代表层 → full3d probe/pilot

## 1. 决策摘要

当前工作不再以 residual diffusion 为默认主线。surface 模型采用单步 teacher forcing
训练，却在评估时执行 15 步自回归；现有证据显示 day-1 已优于 persistence，但优势在
day 4–5 后消失，并伴随方差塌缩、空间相关衰减和偏差漂移。因此第一优先级是让确定性
模型在训练时接触自己的预测反馈，直接检验并缓解 exposure bias。

主线采用最小的 **detached autoregressive multi-step**：前若干步仍完整自回归，但在
`torch.no_grad()` 下生成并回灌预测；只对选定 lead 的最后一步反向传播。它不是取消
自回归，也不是首轮启用 full BPTT，而是在接近单步显存的条件下把训练输入分布向正式
rollout 对齐。

```text
现有：真实 7 天 → day1 → day1 loss

MS5：真实 7 天 → 预测 day1(detach) → 回灌 → ... → 预测 dayJ → dayJ loss
      J 在 1..5 中按固定 schedule 选择；50% batch 保留 J=1

评估：真实 7 天 → 完整自回归 day1..day15（不使用未来真值，协议不变）
```

residual diffusion、完整 BPTT、新输入变量和 loss weighting 均保留为条件分支；在 MS5
因果检验完成前不与主变量叠加。

## 2. 已冻结的证据与任务语义

### 2.1 数据和目标

- 数据为 1994–2022 的 10,591 个连续日平均 ROMS/COAWST 场，水平 `400×441`，30 个
  地形追随 sigma 层；`k=0` 为海底，`k=29` 为海面。
- 当前模型预测经共定位后的曲线网格方向 `u/v`，正式指标再映射到原生 C-grid，以未经
  裁剪真值计算物理单位 m/s 的 masked RMSE/MAE；不是 east/north 分量。
- condition 固定为过去 7 天 `u/v`（14 通道）；最终评估固定为 15 天自回归。
- train/val/test 仍为连续时间切分 `[0,8401)` / `[8401,9496)` / `[9496,10591)`，任何
  multi-step 窗口不得跨 split 边界。

### 2.2 当前基线

主要证据来自 [实验 07](../experiments/07_residual_baseline/RESULTS.md)、
[实验 08](../experiments/08_static_mask_ablation/RESULTS.md) 和
[实验 09](../experiments/09_remask_feedback_ablation/RESULTS.md)：

- surface Ep10 validation day-1：`0.1011 m/s`，persistence `0.1294`，ratio `0.781`。
- surface Ep10 test day-1：`0.0973 m/s`，persistence `0.1167`，ratio `0.833`。
- test 15-day overall：`0.2136` vs `0.2098`，ratio `1.018`。
- test 15-day 分变量 overall：
  - `u`：`0.2651` vs `0.2615`，ratio `1.014`；
  - `v`：`0.1449` vs `0.1405`，ratio `1.031`。
- crossover 约在 day 4–5；实验 08 的静态 mask 输入和实验 09 的每步 remask 均未带来
  稳定 overall 改善，后续保持 14 动态通道、无静态 mask、`rf0`。

## 3. 当前困难

| 困难 | 直接证据 | 对下一步的约束 |
|---|---|---|
| 训练—评估分布不一致 | 训练只看真实 7 天并优化 day-1；评估连续回灌预测到 day-15 | 先让训练接触自己的预测反馈 |
| 长时效结构退化 | day 4–5 后优势消失，伴随方差塌缩、相关衰减和偏差漂移 | 不能只看 pooled RMSE，必须跟踪结构诊断 |
| u/v 退化机制不同 | test overall ratio：u 1.014，v 1.031；v 更受相关损失和正偏差影响 | u/v 必须分别验收，不能用合并指标掩盖失败 |
| 垂向可预测性未知 | 当前正式证据只有 surface；30 层 full3d 尚未运行 | 先做全层零训练画像和代表层实验 |
| full3d 资源成本未知 | 现有“约 30×”只是尺寸外推，没有实测峰值显存和吞吐 | 先做资源 probe、K1 smoke、K3 pilot |
| 一项诊断产物不完整 | `diag_leadtime_residual.py` 的 NPZ key 会让 persistence 覆盖 model | 复用诊断前先修复并加回归测试 |

当前首要假设是：**detached autoregressive multi-step 能在接近单步显存的条件下缓解
exposure bias，并延后或消除 day 4–5 crossover**。所有下一步都围绕这一假设逐项证伪，
不同时叠加新输入、loss、mask、扩散或 backbone 改动。

## 4. 范围与非目标

### 4.1 本轮必须完成

1. 全 30 层 `u/v` persistence、增量和归一化尺度画像；
2. 可测试、可恢复、DDP 安全的 detached multi-step 训练路径；
3. surface MS5 smoke、短训、逐 epoch validation 15-day 选型；
4. 通过预设门槛后才执行 MS10；
5. 建立代表层和 full3d 的数据/资源准入证据。

### 4.2 首轮明确不做

- 不训练 residual diffusion，不重跑失败的 direct diffusion；
- 不做 full BPTT，不让梯度一次穿过 5–15 个 IAFNO forward；
- 不同时加入 `zeta/temp/salt/rho/ubar/vbar`；
- 不同时改变 mask、remask、clipping、loss weighting、backbone 容量或 patch；
- 不用 test 选 epoch、超参数或 Go/No-Go；
- 不直接启动 full3d 50 epoch 正式训练。

## 5. 训练算法

令 `cond` 为 `(B,14,H,W,Z)`，`targets` 为 `(B,K,2,H,W,Z)`，选定训练 lead `J`：

```python
cur = cond
for _ in range(J - 1):
    with torch.no_grad():
        pred = train_model(cur).clamp(0.0, 1.0)
    cur = torch.cat([cur[:, 2:], pred], dim=1)

pred = train_model(cur)
loss = masked_mse_loss(pred, targets[:, J - 1], mask)
```

约束：

- 前 `J-1` 步确实执行自回归并使用模型自己的预测，不做 teacher forcing；
- detached 预测参与后续 condition，但早期计算图不保留；最后一步正常 AMP/backward；
- `clamp=[0,1]`、陆地反馈和滑窗更新必须与正式 deterministic rollout 一致；
- 实验 09 已决定维持 `rf0`，所以首轮不在 feedback 上重新应用 mask；
- multi-step 仅支持 `objective=persistence_residual` 且 `static_mask_input=False`；其他组合显式拒绝；
- DDP 各 rank 的 `J` 必须相同，避免 forward 次数不一致。首版不用随机通信，按 batch index
  生成固定 schedule。

### 5.1 Lead schedule

MS5 固定循环：

```text
1, 2, 1, 3, 1, 4, 1, 5, ...
```

即 50% batch 保持 day-1 anchor，另外 50% 均匀覆盖 day 2–5。MS10 同理扩展到 day 2–10。
这一 schedule 可由 batch index 纯函数生成，无新增 RNG、配置文件或依赖。

### 5.2 为什么不首轮 full BPTT

BPTT 并非只有 RNN 才能使用：只要把同一个一步模型展开多步，并把前一步预测作为下一步
输入，整体计算图就形成了时间上的循环。这里 IAFNO 本身没有 RNN hidden state，但 15 天
rollout 对预测场的反复回灌仍构成递归；detached 版本保留这个真实递归过程，只切断步间梯度。

完整 BPTT 会保存每个时间步的 IAFNO 激活，使显存随 rollout 长度增长；当前单个 surface
状态已为 `14×400×441×1`，full3d 更大。首轮只回答“训练时看见自己的预测反馈是否改善
长 lead”。如果 detached MS5 明确有效但改善不足，下一阶段才考虑让梯度穿过最近 2–3 步
的 truncated BPTT；full BPTT 没有直接准入资格。

## 6. 工作包与准入门槛

### 工作包 1：全层零训练画像

新增只读诊断脚本，使用 train-only 数据统计尺度，使用 validation 报告可预测性：

- 每层、每变量的 mean/std、p0.1/p1/p50/p99/p99.9、min/max；
- train 一日增量 `x[t+1]-x[t]` 的 bias/std/分位数；
- validation day 1–15 persistence RMSE/MAE；
- `u/v` 分项、30 个 sigma layer、coastal/offshore 和三个 sigma index band：
  `bottom=0..9`、`middle=10..19`、`upper=20..29`；
- 当前全层逐变量统一 min-max 对各层主体分布的压缩程度。

输出 NPZ + CSV/Markdown 摘要，记录数据路径、split、stride、mask fingerprint、统计日期；
产物写服务器 run/output 目录，不把大数组提交到 git。

**门槛**：数据连续性、mask、finite 值和各层 valid count 全部通过；否则停止模型训练。

### 工作包 2：代码与 CPU/合成回归

实现 MS 路径后至少验证：

1. `K=1` 的 batch lead、condition、loss 与历史单步路径一致；
2. `J=3` 时两次预测确实进入后续 condition，且早期 prediction 无梯度、最后一步有梯度；
3. 未训练的 zero-init residual 模型在 multi-step rollout 中始终退化为 persistence；
4. schedule 在单卡/DDP 各 rank 一致；
5. dataset `horizon=K` 不跨 split，target 索引与 lead 对齐；
6. checkpoint roundtrip 保存并校验 `train_horizon`、`lead_schedule`、`feedback_detach`、
   `init_checkpoint`；
7. `_MS5`/`_MS10` 与历史 `_RES`、`_RES_MSK` 目录隔离；已有文件仍拒绝覆盖。

**门槛**：`python pre_smoke_test.py` 全部通过，且新增测试能够在错误 feedback/索引下失败。

### 工作包 3：surface MS5

初始化：只从实验 07 Ep10 加载 model weights；不恢复旧 optimizer/scheduler/scaler/epoch，
因为原 cosine schedule 已结束。新优化器从低学习率开始。

建议冻结配置：

| 项目 | 值 |
|---|---|
| preset/objective | `surface_smoke` / `persistence_residual` |
| init | 实验 07 `Ep10.pth`，weights-only |
| train horizon | 5 |
| lead schedule | `1,2,1,3,1,4,1,5,...` |
| feedback | autoregressive、detached、clamp `[0,1]`、`rf0` |
| loss | 现有 normalized `masked_mse_loss` |
| LR | `1e-4`，fresh optimizer/scheduler |
| epochs | 最多 5；保持原 non-finite/AMP skip/early-stop 保护 |
| static mask | False |

执行顺序：单卡 real-data smoke → DDP2 smoke → 单卡或固定 world-size 短训。smoke 必须检查
实际出现 `J>1` 的 batch，而不是 4 个 batch 恰好只覆盖 day1。

每个 epoch 使用冻结的 validation 15-day deterministic 协议评估，不能用训练期 24-window
`val_relL2` 代替正式选型。候选 checkpoint 先过 day-1 守门，再按 15-day overall 排名：

- day-1 native RMSE 不高于 `0.1031 m/s`（现有 `0.1011` 的预注册 2% 容差）；
- 选择 validation 15-day overall RMSE 最低者；
- Go：overall ratio `<0.941`，且 `u`/`v` 各自 overall ratio `<1.0`；
- Go：day 10–15 每日 ratio `<1.0`；
- 同时检查 bias、variance ratio、spatial correlation，防止 RMSE 小幅改善掩盖结构退化。

test 在配置与 checkpoint 冻结后只运行一次。

### 工作包 4：条件式 MS10

仅当 MS5 满足以下任一证据才执行：

- 达到全部 Go 门槛；或
- overall、crossover day、late-lead correlation 三项均明显改善，但 day 10–15 尚未全部过线。

从最佳 MS5 weights-only 初始化，`K=10`，fresh optimizer，最多 2–3 epochs；其他变量不变。
若 MS5 没有推迟 crossover 或造成 day-1 明显退化，MS10 **No-Go**。

### 工作包 5：代表层基线

全层画像完成后建立三个独立单层 deterministic probe：

| 名称 | depth index | 含义 |
|---|---:|---|
| surface | 29 | 海面 sigma 层，已有正式基线 |
| middle | 14 | 中部代表 sigma 层 |
| bottom | 0 | 海底 sigma 层 |

middle/bottom 使用与 surface 相同的 `400×441×1` 架构、patch、预算和协议，先做单步
`persistence_residual`；只有 day-1 优于对应层 persistence 才进入 MS5。这三个模型用于判断
垂向难度和 full3d 投资价值，不充当最终 full3d 结果。sigma index 随水深变化，报告中不得
把 index 直接写成固定米深。

### 工作包 6：full3d probe 与 pilot

full3d 不再以 residual diffusion 成功为前提，但仍分级准入：

1. 全层画像通过；
2. stats cache、单 `getitem`、单 batch I/O/峰值显存探针；
3. `batch_size=1`、`K=1` real-data smoke；
4. 一个 epoch deterministic single-step pilot；
5. 训练健康且逐层 day-1 有可预测信号后，才做 `K=3` detached multi-step pilot；
6. 正式训练预算另行冻结，不默认执行现有 50 epoch。

full3d 结果必须报告 30 层逐层指标及 upper/middle/bottom band，不允许只用 pooled overall
宣称成功；容量或 patch 的任何调整单独记录。

## 7. 计划修改的文件

| 文件 | 最小修改 | 必须 |
|---|---|---|
| `pre_config.py` | `DIAFNO_TRAIN_HORIZON`、weights-only init 语义、`_MS{K}` run tag、metadata/guards | 是 |
| `pre_dataset.py` | 复用已有 `horizon` 能力；仅补必要验证/注释，不另建 dataset | 视测试结果 |
| `pre_models.py` 或 `pre_rollout.py` | 放置一个可单测的 detached condition rollout helper；只选一个位置 | 是 |
| `pre_trainer.py` | 构造 K-day target、固定 lead schedule、detached feedback、weights-only init、日志/进度 | 是 |
| `pre_evaluate.py` | 复用现有 deterministic 15-day 协议；仅补选型所需 metadata/输出，不复制 evaluator | 视缺口 |
| `pre_smoke_test.py` | K=1 回归、J>1 feedback/gradient、schedule、checkpoint guards | 是 |
| `scripts/diag_uv_predictability.py` | 全 30 层零训练画像 | 是 |
| `scripts/diag_leadtime_residual.py` | 修复 NPZ key 未区分 model/persistence 导致覆盖的问题，再用于 multi-step 对比 | 是 |

计划实现不增加第三方依赖、不新建模型 class、不复制 trainer/evaluator。

## 8. 计划中的运行入口

以下命令依赖上述代码实现，当前尚不可执行；实现后仍从仓库根目录运行。

```bash
# 工作包 1：全层只读画像
python -u scripts/diag_uv_predictability.py

# 工作包 3：surface MS5 real-data smoke
DIAFNO_PRESET=surface_smoke \
DIAFNO_OBJECTIVE=persistence_residual \
DIAFNO_TRAIN_HORIZON=5 \
DIAFNO_INIT_CHECKPOINT=/data2/user/zyq/checkpoints/PRE/<RES_RUN>/Ep10.pth \
python -u pre_trainer.py

# 工作包 3：surface MS5 正式短训
DIAFNO_PRESET=surface_smoke \
DIAFNO_OBJECTIVE=persistence_residual \
DIAFNO_TRAIN_HORIZON=5 \
DIAFNO_TRAIN_MODE=full \
DIAFNO_INIT_CHECKPOINT=/data2/user/zyq/checkpoints/PRE/<RES_RUN>/Ep10.pth \
python -u pre_trainer.py
```

DDP smoke 使用相同参数配合 `torchrun --standalone --nproc_per_node=2`；world size、GPU、
effective batch、LR、吞吐和峰值显存必须写入实验记录。

## 9. 产物与实验记录

进入训练执行时新建实验 10 目录，遵循既有约定：

```text
docs/experiments/10_multistep_deterministic/
├── EXPERIMENT.md
└── RESULTS.md
```

至少归档：

- 全层画像摘要及服务器产物路径；
- 训练与逐 epoch validation 日志；代码回归和 smoke 验证记录归 Changelog；
- checkpoint config、loss.dat、选型表；
- validation/test 逐 lead × u/v × layer RMSE/MAE；
- bias/variance ratio/correlation 与 crossover day；
- 失败、OOM、AMP skip、early stop 和任何配置偏离。

## 10. 后续分支的准入条件

- **Truncated BPTT（最近 2–3 步）**：MS5 明确改善 long lead，但 detached 版本仍停滞；
- **物理单位 loss weighting**：MS5 后 `u/v` 改善明显不对称，且需优化 native m/s 目标；
- **额外输入变量**：multi-step 仍无法控制 bias/correlation，且数据审计支持 `zeta`、
  `temp/salt/rho` 等物理状态有增益假设；
- **direct multi-horizon head**：detached/TBPTT 不能稳定越过长 lead persistence；
- **residual diffusion**：确定性 mean forecast 已稳定，项目明确需要概率分布或扩散在同一
  U/V 点预测门槛下有独立可证伪假设；
- **full BPTT**：短 TBPTT 有稳定增益且显存/梯度诊断证明继续加长值得。

任何后续分支都必须单变量实施，不能与 MS horizon、loss、input、mask 同时改变。

## 11. 当前执行清单

- [x] 科学目标与当前主线讨论完成；
- [x] 修改与执行计划成文；
- [x] 工作包 1：全层画像实现与运行（门禁四项全 PASS，产物见 §12）；
- [x] 工作包 2：multi-step 代码与 CPU 回归（`pre_smoke_test.py` 55/55）；
- [x] MS5 单卡 smoke（SMOKE PASS）+ DDP2 smoke（2026-09-03 修复 autocast 缓存
  问题后 SMOKE PASS，见 Changelog）；
- [x] MS5 短训与逐 epoch validation 选型（Ep4；全门槛 Go；test 一次 0.871）；
- [x] 根据门槛决定 MS10（获准并完成；选型 Ep2；test 一次 0.838）；
- [x] 代表层单步/MS5（实验 11：middle/bottom probe+MS5 全门槛 Go，垂向泛化成立）；
- [x] full3d 资源 probe/K1 smoke/pilot（实验 06：实测 22.6 GB、0.97 s/步、
  1 epoch ≈ 2.3 h；pilot 健康但 1-epoch 无逐层信号）；
- [ ] full3d K3 pilot：按预注册条件阻塞（无逐层 day-1 信号），路径决策待定
  （加 single-step epochs ≈ 2.3 h/个 / 冻结 full3d 待正式预算 ≈ 5 天/50 epoch / 调参）；
- [x] 冻结后 test 报告（MS5 Ep4 与 MS10 Ep2 各一次，见实验 10 RESULTS）；
- [ ] 决定 TBPTT、额外变量、direct multi-horizon、diffusion 是否立项。

## 12. 事后回顾记录

每个工作包完成后只在这里记录“证据链接与决策”，详细数字留在对应实验的 `RESULTS.md`，
代码实现与验证留在 `docs/project/CHANGELOG.md`，不在三处复制同一张结果表。

| 工作包 | 实验/结果证据 | 实际结论 | 是否达到门槛 | 与计划偏差 | 后续动作 |
|---|---|---|---|---|---|
| 全层画像 | `checkpoints/PRE/diag_uv_predictability_20260901/`（npz/csv/SUMMARY.md，2026-09-01） | 门禁四项全 PASS（0 动态缺失、逐层有效计数充足）；val persistence d1：u bottom/middle/upper = 0.068/0.105/0.137 m/s，v = 0.039/0.054/0.087；d15 = 0.130/0.204/0.281 与 0.066/0.087/0.149；surface 最难、底层最易；统一 min-max 无截断，底层归一化 std 约为海面 1/3 | 是（连续性/mask/finite/valid count 全过） | 无（脚本按 §7 计划新建） | MS5 可启动；full3d 画像证据就绪；注意底层在统一归一化下被强压缩，full3d 立项时复核是否需要 per-band 归一化（当前不改动） |
| multi-step 代码与回归 | Changelog 2026-09-01（工作包 2 条目）；`pre_smoke_test.py` 55/55 | detached MS 路径（`DIAFNO_TRAIN_HORIZON`/`_MS{K}` tag/weights-only init/守卫/元数据）实现完成；K=1 与历史单步逐位一致；schedule 纯函数 DDP 一致；未训练模型 multi-step 恒等于 persistence | 是（新增 9 项测试全过，含错误反馈/索引下的失败用例） | 无（MS 超参以内置默认 `MS_DEFAULTS` 落地，未新增环境变量；val 期 `val_masked_relL2` 仍为单步训练健康信号） | MS5 real-data smoke（单卡→DDP2）→ 短训与逐 epoch validation 15-day 选型 |
| surface MS5 | 实验 10 `RESULTS.md`（2026-09-02） | 全门槛 Go：val 选型 Ep4（overall ratio 0.822，day-1 0.773），crossover 消除、corr 全 lead 占优；test 一次 overall ratio **0.871**（单步基线 1.018），day-1 0.843 | 是（§6 WP3 全部门槛） | 单卡执行；DDP2 smoke 因无双卡空闲未做 | 冻结 Ep4；晚段 u bias -0.071 与方差塌缩记为观察项 |
| 条件式 MS10 | 实验 10 `RESULTS.md`（2026-09-02） | 从 MS5 Ep4 weights-only 续训 K=10 × 3 epochs；val 选型 Ep2（overall ratio 0.796）；结构诊断晚段 bias 稳定（+0.017）；test 一次 overall ratio **0.838**，day-1 0.833，最差 lead 0.894 | 准入证据满足（MS5 全门槛 Go）；MS10 无预注册数值门槛，val/test 全面优于 MS5 | `EPOCH_OVERRIDES` 临时设 3，已还原 | MS10 Ep2 为当前最优冻结 checkpoint；d15 ratio 回升与方差塌缩 → §10 分支准入证据 |
| 代表层 | 实验 11 `RESULTS.md`（2026-09-03） | 两层 probe 过 day-1 门槛（middle 0.770 / bottom 0.568）；层 MS5 全门槛 Go：middle Ep2 test overall **0.830**（单步 1.183，crossover 消除）、bottom Ep5 **0.813**（v d15 1.15→0.880 修复）；难度排序与 WP1 画像一致 | 是（镜像 surface 预注册） | MS5 epoch 实测 ~46 min（预估 30）；bottom day-1 门槛余量小（Ep1–3 未过） | 垂向泛化成立；代表层证据支持 full3d 投资 |
| full3d probe/pilot | 实验 06 `RESULTS.md`（2026-09-03） | probe/K1 smoke/pilot 完成：实测 22.6 GB、0.97 s/步、1 epoch ≈ 2.3 h（50 epoch ≈ 5 天）；pilot 健康但 1-epoch 无逐层 day-1 信号（60 ratio ≈1.000） | 第 1–4 步达成；K3 门槛（逐层信号）未满足 → 阻塞 | 评估耗时 2h05m（batch 1）；OOM 预案未触发 | K3 路径决策：加 epochs / 冻结待正式预算 / 调参（需单变量论证） |
