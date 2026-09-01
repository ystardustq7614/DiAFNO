# 项目 Changelog

本文件记录本地适配仓库中与 PRE 模型、训练和评估有关的实际变更与已确认计划。

- **已完成**：代码或文档已经存在于当前工作区，并有相应验证或状态证据；
- **计划中**：只完成设计，尚未修改代码或运行实验；
- 每次实施后将条目从“计划中”迁移到对应日期，并补充验证命令与结果；
- 未执行、失败或缺少产物的实验不会记为完成。

## Unreleased — 计划中

### Proposed

- Phase 6 计划文档待另立（旧计划已归档至
  [archive/CODE_MODIFICATION_PLAN_20260830.md](./archive/CODE_MODIFICATION_PLAN_20260830.md)）：
  residual diffusion 的残差 sigma 实测定标、短训冒烟与验收门槛
  （overall ratio < 0.941 且 day 10-15 < 1.0）。
- Phase 5③（可选）：分段 remask 变体与近岸误差靶点（coastal 0.867 vs offshore
  0.777）是否立项待讨论。

以上条目均**尚未实施**。

## 2026-09-01 — 已完成（文档归档与交接同步）

- `docs/project/CODE_MODIFICATION_PLAN.md` 归档为
  `docs/project/archive/CODE_MODIFICATION_PLAN_20260830.md`（补执行完毕状态头；
  Phase 6 计划将另立新文档，避免同处冲突）。
- `PROJECT_HANDOFF_SUMMARY.md` 更新至 2026-09-01 现状：一句话结论改为
  persistence-residual 基线 Go；§5 mask 输入建议改写为 A/B 结论（不保留）；
  §6/§7 并入基线成绩与 day-2 7.7% 声明更正；新增第 9 节（基线/诊断/Phase 5
  决策/Phase 6 含义）；未完成项清单 8 条逐项标注完成状态。
- `docs/README.md`、`docs/experiments/README.md`（实验 07 状态与决策树）、
  `docs/operations/PRE_runbook.md`（`DIAFNO_STATIC_MASK`/`_MSK` 约定与两个
  诊断脚本）、`AGENTS.md`（static mask 约定）同步。

## 2026-09-01 — 已完成（Phase 5② remask_feedback A/B，评估-only）

- 同一 checkpoint（A 臂 Ep10）在 validation 15 天确定性 rollout 下对比
  rf0（历史整帧回灌）与 rf1（每步预测重应用海洋 mask 后回灌）；
  `pre_evaluate.py` 原生支持，统一 `OUTPUT_TAG="rfab"` 避免与 test 图目录冲突；
  两臂 exit=0，产物 `eval_val_h15_..._rf{0,1}_..._rfab.npz`。
- **决策：默认维持 rf0（历史行为）**——rf1 分段效应：day 2-8 改善
  （-0.5%~-7.9%），day 9-15 转差（+1.5%~+7.8%），overall 持平略差
  （0.2183 vs 0.2180）；不满足"稳定改善才保留"。
- **HANDOFF 未完成项 5 复现结论**：远端"day-2 改善约 7.7%"未在 day-2 复现
  （实际 -0.49%）；同量级改善实际位于 day 4-7（-5.9%~-7.9%），方向一致、
  数值归属更正（详见实验 07 RESULTS.md 追加节）。

## 2026-08-31 — 已完成（Phase 5① 双静态 mask 输入 A/B）

### Static mask input support（arm B 代码路径）

- `pre_config.py`：`DIAFNO_STATIC_MASK` 开关（`static_mask_input()`）、
  `STATIC_MASK_CHANNELS=2`、run tag `_MSK` 后缀（B 臂绝不与 A 臂共用目录）。
- `pre_models.py`：`PersistenceResidualIAFNO.forward/sample` 增加可选
  `static_cond`——仅拼入 backbone 的 x_self_cond；DYNAMIC 窗口保持纯 14 通道，
  persistence base 语义不变；批次广播与形状/通道错误显式拒绝。
- `pre_rollout.py`：`ensemble_rollout/_rollout_one/_sample` 增加 `static_cond`
  透传（None 时逐位保持历史行为，EDM 调用签名不受影响；滑窗切片不变）。
- `pre_trainer.py`：objective 守卫（static mask 仅限 persistence_residual）、
  `MODEL_COND_CH` 16 通道建模、零初始化 identity 探针含静态通道、train/val
  传入 `static_cond`、checkpoint 记录 `static_mask_input`/`model_cond_chans`、
  resume 结构守卫拒绝跨配置续训。
- `pre_evaluate.py`：按 checkpoint `config.static_mask_input` 自动重建（元数据
  驱动，不读环境变量）、输出 tag `msk{0|1}`、npz 元数据记录该字段。
- `pre_smoke_test.py` 新增 2 项测试（wrapper 静态拼接/恒等/错误拒绝、rollout
  static_cond 透传与滑窗纯度），47 项全部 PASS；legacy `smoke_test.py` 通过。

### A/B 执行与决策

- B 臂 smoke `SMOKE PASS`；10/10 epochs 训练 3 h 36 min（best 0.40038@ep10，
  全程单调改善）；validation day-1 选型 10 个 checkpoint 全部 exit=0。
- **决策：不保留静态 mask 输入**——A 最优 0.1011 < B 最优 0.1024，10 epoch 中
  9 个 A 领先；区域分解（coastal/offshore × u/v）4 项全部 A 优。B 臂产物保留
  于 `..._RES_MSK/` 供复核。
- 新增 `scripts/diag_region_breakdown.py`（coastal = 距陆地 ≤5 格的区域分解
  评估；coastal 改善幅度小于 offshore 的观察记录在案）。

### 边界

- 未执行 Phase 5②（remask A/B）；未修改 `IAFNO.py`/`diffusion.py`/
  `pre_dataset.py`/`pre_metrics.py`；`pre_evaluate.py` 选型用临时改动已恢复，
  工作区余 Phase 5① 代码路径、2 个诊断脚本与文档更新。

## 2026-08-31 — 已完成（persistence-residual 真实数据 smoke、短训练与 Phase 3 Go）

### Real-data smoke（Phase 2）

- 单卡（GPU 1，默认 smoke 模式，`DIAFNO_OBJECTIVE=persistence_residual`）：`SMOKE PASS`；
  4 updates/rank、无 AMP skip、零初始化 identity 自检通过；产物
  `surface_smoke_..._S4_C7_SD2_RES_SMOKE/`。
- DDP world size 2（GPU 1+2，`torchrun --standalone`）：`SMOKE PASS`；每 rank 4 updates、
  skipped 0；进度行 `scope=rank0_shard_of_2`、仅 rank 0 写 checkpoint；产物
  `..._RES_SMOKE_DDP2/`。smoke 末行 `lr=0` 为 cosine T_max=4 退火到底的设计行为。

### Phase 3 surface 短训练

- 单卡 10 epochs 跑满（未触发 early stop），3 h 35 min，~1.63 step/s；
  `val_masked_relL2` 从 0.58275 单调降至 0.40325（仅 ep4 一次波动），
  checkpoint 落盘 `surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/`（Ep1~Ep10 + best + loss.dat）。

### Validation day-1 选型（Phase 3 Go/No-Go）

- 逐个 `Ep{n}.pth`（禁用 `best.pth`）以 `SPLIT="val"`、`ROLLOUT_DAYS=1` 运行
  `pre_evaluate.py`（156 窗口，确定性评估），10 轮全部 exit=0，产物
  `eval_val_h1_ch0_e1_s123_rf0_ckptEp{n}.npz`。
- **Go**：Ep10 validation day-1 native RMSE `0.1011 m/s` 严格优于 persistence
  `0.1294 m/s`（ratio 0.781），并优于 ridge probe 参考 `0.1177 m/s`；
  Ep2 起所有 epoch ratio < 1 且随训练单调下降。
- 实验记录已补录：`docs/experiments/07_residual_baseline/{EXPERIMENT,RESULTS}.md`。

### Phase 4 test 报告（同日完成）

- 配置冻结：Ep10（validation day-1 选出）、`SPLIT="test"`、`ROLLOUT_DAYS=15`、
  154 窗口（stride 7）、确定性评估；单次运行 `status=completed`（~6.7 min，零异常）。
- **test day-1 `0.0973 m/s` 优于 persistence `0.1167`（ratio 0.833，Phase 4 第一目标达成）**；
  15-day overall `0.2136` vs persistence `0.2098`（ratio 1.018，基本持平，长时效自回归
  误差累积仍未解决）。
- 对照扩散：SD2 diffusion test d1 `0.2568` / overall `0.3442`；本基线分别改善约
  2.6 倍 / 1.6 倍。产物 `eval_test_h15_ch0_e1_s123_rf0_ckptEp10.npz` + figures。
- 备注（透明记录）：test 评估进程未设 `CUDA_VISIBLE_DEVICES`，落在 GPU 0 与他人任务
  共存（~1.3G 显存，确定性评估数值不受影响）。

### 长时效误差诊断（同日追加）

- 新增 `scripts/diag_leadtime_residual.py`：复用官方 rollout 协议在 77 个
  stride-14 test 窗口上重放 Ep10 的 15 天确定性 rollout，补齐评估 NPZ 不含的
  逐 lead day bias / 方差比 / 逐窗口空间相关；产物
  `leadtime_diag_ckptEp10.npz/.png`（run 目录）。
- 结论：**方差塌缩主导**（u 方差比 d1 0.87 → d7 起 ~0.55，模糊化）；空间相关
  d7 起低于 persistence（0.48 vs 0.57，d15 0.39 vs 0.61）；bias 漂移且变号
  （u: -0.005 → -0.11 → +0.065）。d1-3 为优势期（ratio 0.85-0.93）。
- 含义：为 Phase 6 residual diffusion 讨论提供直接证据（生成式采样恢复方差）；
  Phase 5 A/B 对模糊化改善有限。详见实验 07 RESULTS.md 追加节。

### 边界

- 未执行 Phase 5 A/B 与 full3d；未声称 15-day overall 优于 persistence。
- 评估用的 `pre_evaluate.py` 常量改动（CHECKPOINT/SPLIT/ROLLOUT_DAYS）已全部恢复，
  工作区仅余文档更新与新增诊断脚本。

## 2026-08-30 — 已完成（persistence-residual 基线代码实施）

### Training / models

- 新增 `pre_models.py`：`PersistenceResidualIAFNO`（condition-only 确定性基线，
  预测 = 条件第 7 天 persistence + 零初始化残差头输出；未训练时严格等于 persistence）
  与 `masked_mse_loss`（逐样本有效格点均值，与 EDM masked loss 语义一致）。
- `pre_config.py`：objective 配置（`OBJECTIVES`/`validate_objective`/`objective_from_checkpoint`/
  `ensure_objective_compatible`、`MASK_SCHEME`、`RESIDUAL_TIME_SIGMA`）；`run_tag_for`/
  `training_run_tag` 支持 objective（persistence_residual 追加 `_RES`，绝不与扩散实验共用目录）；
  新增共享进度辅助 `ProgressReporter`/`format_progress`（交互 tqdm 条；非交互 ≥30s 一条可解析
  `PROGRESS key=value` 状态行，start/completed/failed 必发）。
- `pre_trainer.py`：`DIAFNO_OBJECTIVE` 选择训练目标（默认 `diffusion`，行为不变）；
  residual 走 `masked_mse_loss`；启动时零初始化 == persistence 严格自检；checkpoint config
  记录 `objective/cond_chans/target_ch/mask_scheme`（扩散另存 sigma 字段，residual 另存
  `residual_base/time_sigma`）；断点续训校验 objective 与关键结构参数（跨模型类/结构变化拒绝）；
  sigma 尺度决策仅适用于扩散；rank-0 每 epoch train/val 进度条 + run 级
  start/completed/failed 状态行；保留全部关键 epoch summary 与 `SMOKE PASS` 文本。
- `pre_rollout.py`：`remask_feedback`/`ocean_mask` 可选开关（启用时每步预测重应用海洋 mask
  后再回灌；默认 False 保持历史行为）；docstring 明确模型 duck-type 对确定性模型的兼容
  （无 RNG 消耗、seed 无关、成员相同）。
- `pre_evaluate.py`：按 checkpoint `config.objective` 重建 diffusion 或确定性模型（legacy →
  diffusion 并提示）；确定性评估强制 `ENSEMBLE_SIZE=1`、采样参数在 `sampler`/`sampler_note`
  中显式记为不适用（`sigma_data=nan`、`sampling_steps=-1`）；`REMASK_FEEDBACK` 配置 + `rf{0|1}`
  输出 tag + metadata（`objective/residual_base/remask_feedback/sampler/sampler_note/time_sigma`）；
  评估进度条（running day-1 RMSE/比值 postfix）与 failed/completed 状态行。

### Tests / verification

- `pre_smoke_test.py` 新增 9 项 CPU 回归测试：零初始化 persistence identity（shape/通道序/
  clamp/忽略采样步数）、一次 optimizer step（head 移动、全部参数有 `.grad`——DDP 兼容性质）、
  masked MSE 语义（参考实现、陆地不变性、全零 mask→0、广播）、checkpoint roundtrip +
  objective 守卫、rollout remask 开/关行为与 mask 必填、deterministic rollout（seed 无关、
  成员相同）、objective 配置辅助、run tag objective 后缀、`ProgressReporter` 行格式与间隔门控。
- 验证命令与结果（本地 `diafno` env，torch 2.4.1+cpu，无 CUDA）：
  - `python -m py_compile pre_models.py pre_config.py pre_rollout.py pre_trainer.py pre_evaluate.py pre_smoke_test.py` 通过；
  - `python pre_smoke_test.py` 通过：41 项测试函数全部 PASS（含 9 项新增；4 项含 CUDA 专属分支
    本机按设计 SKIP；既有 32 项结果不变，legacy diffusion 覆盖未回归）；
  - `python smoke_test.py`（legacy 路径）通过。

### Documentation

- `docs/operations/PRE_runbook.md`：文件表新增 `pre_models.py`；objective/`_RES`/元数据/续训守卫、
  `REMASK_FEEDBACK`+`rf` tag、validation day-1 native RMSE 选型协议、新增第 7 节终端进度与监控约定。
- 新增 `docs/experiments/07_residual_baseline/`（EXPERIMENT = 设计与门槛；RESULTS = 未执行，
  仅记录代码实施与验证状态）；实验索引与本索引条目同步更新。
- `AGENTS.md` 同步 PRE 路径约定（objective、`pre_models.py`、remask、`_RES`、进度约定）。

### 边界

- 本轮未运行任何真实数据训练/评估；未声称任何模型精度改善；Go/No-Go 数值留待实验执行后记录。
- 未修改 `IAFNO.py`/`diffusion.py`/`pre_dataset.py`/`pre_metrics.py`/`utilities3.py`。

### Code review fixes（评审跟进，2026-08-30 同日）

针对首轮实现评审的 3 项 P1 与进度计数 P2，按最小化原则修复（未引入新监控框架）：

- **心跳线程化（P1）**：`ProgressReporter` 的周期状态行改为**时间驱动**——非交互模式下由守护
  心跳线程按间隔补发 `status=running`，单个 batch/rollout 步阻塞超过间隔不再静默；所有发射
  经 `threading.Lock` 串行化，`close()` 停止线程。
- **生命周期语义（P1）**：引入稳定状态词汇表——`start`/`running`/`phase_done`（本阶段结束，
  reporter `close()` 的新默认）/`failed`；`completed` 只由入口脚本在全部产物落盘后输出
  （评估端移到 NPZ + 汇总 + 全部图之后），每个 epoch 不再输出误导性的 `completed`；
  新增 `install_progress_failure_hook`（`sys.excepthook` 兜底）为初始化/数据/模型/pre-flight/
  后处理等不受 guarded 块保护的异常输出标准 `status=failed`（`stage=setup|run|data_model|
  rollout|postprocess` 标明位置），与 guarded handler 通过 `mark_progress_failed()` 去重；
  `_progress_value` 把一切空白（含多行异常的换行/制表符）替换为 `_`，状态行永不被错误信息打断。
- **checkpoint 语义指纹（P1）**：新 checkpoint 记录 `norm_lo`/`norm_hi`/`mask_version`；
  续训与评估重建经 `check_norm_fingerprint` 校验归一化范围与 mask 版本（不一致拒绝，legacy 缺字段
  告警），residual 另经 `check_residual_time_sigma` 校验 `time_sigma`（缺失/不一致拒绝），
  residual 续训还校验 `stats_sigma`（无迁移策略）。
- **进度计数（P2）**：评估 reporter 改为按**真实窗口数**计数（total=`len(eval_ds)`、每 batch 按
  实际窗口数推进、`sample_per_s`=窗口×`ROLLOUT_DAYS`），不足 batch 的尾批吞吐不再失真；
  DDP 下 train/val 进度行显式标注 `scope=rank0_shard_of_<n>`（单卡 `whole_split`），分片计数
  不再被误读为全局。
- **文档状态（P3）**：`CODE_MODIFICATION_PLAN.md` 状态头更新为"首轮代码已实施"；runbook 第 7 节
  重写状态词汇表/心跳/兜底 hook/scope 约定，第 1 节与第 4 节补充指纹校验与 `best.pth` 选型警告
  （`best.pth` 按 `val_masked_relL2` 产生，禁止直接用于 day-1 native RMSE 选型）。
- `pre_smoke_test.py` 新增 4 项测试：时间驱动心跳（零 update 仍发射、close 后停止）、多行错误
  清洗、失败 hook 去重与 stage 读取、归一化/mask/time_sigma 指纹校验。
- 验证：`py_compile` 通过；`pre_smoke_test.py` 45 项测试函数全部 PASS（新增 4 项；4 项含 CUDA
  分支本机按设计 SKIP）；`smoke_test.py` 通过；`git diff --check` 干净。
- 未处理（评审 P2，按最小化原则留待后续）：coastal/open-ocean 与空间相关性指标、
  validation 选型流程自动化。

## 2026-08-30 — 已完成

### Documentation

- 新增 `docs/project/CODE_MODIFICATION_PLAN.md`，固化下一轮代码修改、烟测、单卡/DDP 和全量训练门槛。
- 新增本 changelog，并加入项目文档索引。
- 本次更新未修改 Python 源码、训练配置或模型参数。

### Repository state

- 本地 `adapt-weather-ocean` 已安全 fast-forward 到远端 `43f9813`，当前与 `origin/adapt-weather-ocean` 为 `0 ahead / 0 behind`。
- 保留 fast-forward 前的保险 stash：`codex-safe-ff-origin-adapt-weather-ocean-20260830`；尚未删除。
- 远端归档文档已按现有 `docs/` 分类结构整理；项目仍保留未提交的工作区修改。

### Training infrastructure

- `pre_config.py` 增加 smoke/full 训练 profile 和隔离的 run tag 规则。
- `pre_trainer.py` 支持真实数据 smoke、单卡训练和 `torchrun` DDP；仅 rank 0 写 checkpoint，训练状态记录 world size/profile。
- `pre_smoke_test.py` 增加训练配置相关回归覆盖。

### Verification

- Python 编译检查通过。
- legacy `smoke_test.py` 通过。
- PRE `pre_smoke_test.py` 通过；本机无 CUDA，4 项 CUDA 专属检查跳过。
- `git diff --check` 与 Markdown 本地链接检查通过。

> 注：上述训练基础设施为当前工作区已有改动，尚未据此声称模型效果改善；真实 GPU smoke 和全量训练仍需在服务器环境执行。
