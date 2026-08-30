# 项目 Changelog

本文件记录本地适配仓库中与 PRE 模型、训练和评估有关的实际变更与已确认计划。

- **已完成**：代码或文档已经存在于当前工作区，并有相应验证或状态证据；
- **计划中**：只完成设计，尚未修改代码或运行实验；
- 每次实施后将条目从“计划中”迁移到对应日期，并补充验证命令与结果；
- 未执行、失败或缺少产物的实验不会记为完成。

## Unreleased — 计划中

### Proposed

- 在服务器完成单卡和目标 world size 的真实数据 smoke（`DIAFNO_OBJECTIVE=persistence_residual`），
  再启动 surface 短训练。
- 以 validation day-1 native RMSE 优于 persistence 作为全量训练的 Go 条件（Phase 3 Go/No-Go）。
- 基线完成后，依次验证双静态 mask 输入和 rollout 陆地回灌修正（Phase 5 单变量 A/B，
  `remask_feedback` 开关已就绪，默认保持历史行为）。

以上条目均**尚未实施**；详细门槛见 [PRE 模型代码修改计划](./CODE_MODIFICATION_PLAN.md)。

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
