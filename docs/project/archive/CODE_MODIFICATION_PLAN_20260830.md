# PRE 模型代码修改计划（已归档）

> **归档状态（2026-09-01）：本计划已全部执行完毕并关闭。**
> Phase 0-2（协议冻结/CPU 回归/真实数据单卡+DDP smoke）、Phase 3（短训练与
> validation 选型，**Go**：0.1011 vs persistence 0.1294）、Phase 4（test 报告：
> day-1 0.833 / overall 1.018）与 Phase 5（双静态 mask 输入、remask 回灌两轮
> A/B，均判"不保留"）全部完成，详见 [实验 07](../../experiments/07_residual_baseline/EXPERIMENT.md)
> 与 [CHANGELOG](../CHANGELOG.md)。Phase 6（residual diffusion / full3d 决策）
> 将另立新计划文档，不与本文混用。  
> 制定日期：2026-08-30  
> 适用范围：`surface_smoke` / surface 训练路径；`full3d` 暂不启动

## 1. 目标与当前判断

当前扩散模型已经能够完成训练和评估，但正式 day-1 预测仍弱于 persistence。已有诊断表明：

- 线性/ridge condition-only probe 的 RMSE 为 `0.1177 m/s`，优于 persistence 的 `0.1293 m/s`，说明 7 天条件中存在可利用信号；
- 当前扩散模型使用真实条件时的 day-1 RMSE 为 `0.2584 m/s`，且预测与真值的空间相关性明显低于 persistence；
- 条件扰动实验说明模型会使用条件，但还没有把条件转化成足够准确的空间预测；
- 历史 rollout 的陆地点回灌没有重新应用 mask，这是需要单独验证的正确性问题，不能把未留存证据的改善幅度当作既定结论。

因此首轮代码修改只建立一个**确定性、condition-only、相对 persistence 预测残差**的 IAFNO 基线。扩散路径保留为对照，不在首轮同时修改 EDM、噪声日程或采样器。

## 2. 设计原则

1. **先 surface，后 full3d**：surface 没有稳定优于 persistence 前，不投入 full3d 全量训练。
2. **先确定性基线，后恢复扩散**：先验证 backbone 能否学到条件到下一日流场的映射。
3. **复用训练基础设施**：沿用现有 smoke/full profile、AMP、checkpoint、单卡和 `torchrun` DDP，不复制第二套 trainer。
4. **一次只改一个主要变量**：残差基线、静态 mask、rollout 陆地回灌依次验证。
5. **validation 决策，test 报告**：validation 用于 checkpoint 和 Go/No-Go；test 仅在配置冻结后报告最终结果。
6. **终端进度可观测**：人在交互终端能直接看到动态进度，监控 agent 在非交互日志中也能读取稳定、可解析的状态行。

## 3. 计划中的最小代码改动

### 3.1 新增确定性 persistence-residual 模型

计划新增 `pre_models.py`，提供一个薄封装，例如 `PersistenceResidualIAFNO`：

- 输入仍为连续 7 天、day-major 的 `u/v` 条件，共 14 通道；
- 复用 `IAFNODiff` 作为 condition-only backbone，输出 2 通道残差；
- persistence 基准取条件最后一天：`base = condition[:, -2:]`；
- 最终预测为 `prediction = base + residual`；
- 残差输出头零初始化，使未训练模型严格退化为 persistence；
- 提供与 rollout 兼容的确定性预测接口，避免复制 rollout 主流程。

首轮不增加额外损失、注意力模块或新依赖。训练目标使用现有 ocean mask 下的 masked MSE；指标继续换算到物理单位并使用 native C-grid 口径。

### 3.2 在现有训练入口选择训练目标

计划修改 `pre_config.py` 和 `pre_trainer.py`：

- 增加显式 objective：`diffusion` 或 `persistence_residual`；
- 保持现有 diffusion 行为可复现，新基线使用独立 run tag（建议 `_RES`）；
- checkpoint 写入 `objective`、`residual_base`、输入通道、mask 方案、world size 和训练 profile；
- resume 时校验 objective 和关键结构参数，禁止误加载另一类模型；
- batch size 继续表示每个 rank 的 batch size，不自动按 GPU 数线性缩放学习率；
- full run 固定 GPU 数和有效 global batch，避免不同 world size 成为隐藏实验变量。

不计划新建 `pre_residual_trainer.py`，因为这会复制数据、AMP、DDP、保存和恢复逻辑。

### 3.3 评估与 rollout 兼容

计划修改 `pre_evaluate.py` 和 `pre_rollout.py`：

- 根据 checkpoint metadata 重建 diffusion 或 deterministic 模型；
- deterministic checkpoint 不使用 sampler 参数，输出 metadata 中明确记为不适用；
- 沿用现有 15-day autoregressive rollout 和 persistence/zero/ridge 对照；
- 增加显式 `remask_feedback` 开关；启用时，每步预测进入下一窗口前重新应用 ocean mask，陆地点置零；
- 开关同时覆盖 deterministic 与 diffusion，默认行为和最终取值由 Phase 5 的独立 A/B 决定并写入输出 metadata。

### 3.4 静态 mask 输入作为第二阶段 A/B

基础 14 通道模型跑通并得到正式 validation 后，再比较：

- A：仅 14 通道历史 `u/v`；
- B：14 通道历史 `u/v` + 2 个静态 `u/v` 有效域 mask。

两组保持初始化、数据划分、训练预算和 GPU 数一致，并分别报告 overall、coastal、open-ocean 指标。首轮不把 mask A/B 与模型结构修改捆绑执行。

### 3.5 训练与评估的终端进度

计划在 `pre_trainer.py` 和 `pre_evaluate.py` 中直接复用环境已有的 `tqdm`，并用标准库 `time.perf_counter()` 计时，不增加监控依赖或独立服务：

- 交互式终端使用单层 `tqdm` 进度条，显示当前阶段、epoch、batch/update 或 evaluation window、已完成比例、已运行时间、ETA 和当前速度；
- 训练 postfix 至少包含当前 loss、learning rate、optimizer update 数和 AMP skipped-update 数；
- 评估 postfix 至少包含当前 window/lead day/ensemble 位置，以及当前可获得的 RMSE/MAE；
- 速度同时明确单位，例如 `step/s`、`sample/s` 或 `window/s`；DDP 下区分或明确标注 global throughput，避免把单 rank 速度误读为全局速度；
- DDP 只由 rank 0 输出全局进度，其他 rank 不重复刷屏；异常仍保留 rank 信息；
- 非交互终端、日志重定向或监控 agent 场景不依赖回车覆盖的动态条，而是在开始、结束、异常及运行期间至少每 30 秒输出一条完整换行并立即 flush 的状态行；
- 稳定状态行使用可解析的 `key=value` 格式，例如：

  ```text
  PROGRESS phase=train epoch=1/4 step=120/2101 elapsed_s=91.2 eta_s=1506.4 step_per_s=1.31 sample_per_s=5.24 loss=0.0187 lr=1e-4 status=running
  ```

- smoke 即使短于 30 秒，也必须输出 `status=start` 和 `status=completed`；发生 non-finite、OOM 或其他异常时输出 `status=failed` 和阶段位置后再退出；
- 保留现有关键 epoch summary 和 `SMOKE PASS` 文本，进度条不得吞掉、覆盖或打碎这些行。

实现时优先使用 `tqdm` 自带的 elapsed/remaining/rate 格式和 `tqdm.write()`；只补充上述 agent-readable 周期状态行，不设计新的日志框架。

## 4. 分阶段执行与验收门槛

### Phase 0：冻结评估协议

- 使用既有连续 train/val/test 划分和 normalization cache；
- 固定 day-1 native RMSE/MAE 为主要选择指标；
- 保存 persistence、zero 和 ridge/linear probe 数值作为基线；
- 不使用 test 集调超参数。

交付物：配置快照和基线表。未完成本阶段，不开始全量训练。

### Phase 1：CPU 合成回归测试

在 `pre_smoke_test.py` 中计划增加：

- 输入/输出 shape 与通道顺序测试；
- 零初始化时 `prediction == last_day_persistence`；
- masked loss 忽略陆地点；
- 一次 forward/backward 后参数和 loss 均为 finite；
- checkpoint 保存、重建和 objective 不匹配拒绝测试；
- rollout 每步回灌前重新应用 mask 的回归测试。

通过条件：全部 CPU 测试通过，且不改变 legacy diffusion 测试结果。

### Phase 2：真实数据 GPU smoke

先单卡，再运行目标 world size 的 DDP smoke：

- 使用完整 surface 网格和正式模型结构；
- 每个 rank 仅执行 4 个 optimizer updates、1 个 epoch；
- 验证数据读取、AMP、梯度、scheduler、checkpoint、resume 和 DDP 汇总；
- loss/metric 必须 finite，无 non-finite abort，实际 optimizer update 数正确；
- rank 0 只保存一套互不冲突的产物，并打印明确的 `SMOKE PASS`；
- 零初始化的训练前指标应与 persistence 在容差内一致。
- 交互模式能看到 `tqdm` 的 elapsed、ETA 和速度；非交互模式至少包含 start/completed 两条完整 `PROGRESS` 状态行。

说明：4-step smoke **不要求**优于 persistence，它只证明全链路正常。

### Phase 3：surface 短训练

- 使用冻结的单一配置执行短训练；
- 按 validation day-1 native RMSE 选择 checkpoint；
- 同时记录模型/persistence RMSE 比值、MAE、空间相关性，以及 coastal/open-ocean 分项；
- 保留 loss curve、配置、commit、GPU 数、world size 和有效 global batch。

Go 条件：最佳 validation day-1 native RMSE 严格优于 persistence。否则停止扩大全量预算，先诊断优化、输入或目标，而不是直接增加 epoch。

### Phase 4：surface 全量训练

只有 Phase 3 达到 Go 条件后执行：

- 固定 GPU 数、global batch、学习率、seed 和数据版本；
- 完成训练后只用冻结 checkpoint 做一次 test 报告；
- 第一目标是稳定优于 persistence `0.1293 m/s`；进一步目标是达到或优于 ridge probe `0.1177 m/s`。
- 训练与评估的终端状态至少每 30 秒可被读取；阶段结束时报告总耗时和平均吞吐。

### Phase 5：单变量改进实验

按以下顺序逐项 A/B，不并行叠加：

1. 静态双 mask 输入；
2. rollout 陆地回灌修正；
3. 必要时再检查条件归一化或损失加权。

每项只有在相同验证协议下稳定改善才保留。

### Phase 6：扩散与 full3d 决策

- 确定性 IAFNO 尚未优于 persistence：不恢复扩散、不启动 full3d；
- 确定性 IAFNO 已优于 persistence：再讨论 residual diffusion 或其他生成式目标；
- surface 的模型、评估和 DDP 路径稳定后，才为 full3d 制定显存和训练预算。

## 5. 计划修改的文件

| 文件 | 计划改动 | 首轮是否必须 |
|---|---|---|
| `pre_models.py` | 新增最薄的 persistence-residual 模型封装 | 是 |
| `pre_config.py` | objective、run tag、checkpoint 配置字段 | 是 |
| `pre_trainer.py` | 切换 objective；增加 rank-0 `tqdm` 与可解析训练状态 | 是 |
| `pre_evaluate.py` | 按 objective 重建；增加评估进度、耗时和吞吐输出 | 是 |
| `pre_rollout.py` | 兼容确定性预测，并统一处理回灌 mask | 是 |
| `pre_smoke_test.py` | 新模型、checkpoint、mask 和 rollout 回归测试 | 是 |
| `docs/operations/PRE_runbook.md` | 增加 smoke、单卡、DDP 和 full run 命令 | 实施时更新 |
| `docs/experiments/<new_experiment>/` | 配置冻结后记录实验设计和结果 | 实施时创建 |

首轮原则上不修改 `IAFNO.py` 和 `diffusion.py`；只有现有接口无法支持最薄封装时，才提出有测试覆盖的最小改动。

## 6. 完成定义

本计划的代码阶段只有同时满足以下条件才算完成：

- CPU 合成回归测试全部通过；
- 单卡真实数据 smoke 明确通过；
- 目标多卡规模 DDP smoke 明确通过；
- checkpoint 能在对应 objective 下保存、恢复和评估；
- 交互终端能读取训练/评估进度、elapsed、ETA 和速度；非交互日志能读取定期、换行且已 flush 的 `PROGRESS key=value` 状态；
- DDP 进度不被多 rank 重复输出，start/completed/failed 状态均能明确识别；
- 短训练的 validation day-1 native RMSE 优于 persistence，才允许进入全量训练；
- 文档、实验记录和 `CHANGELOG.md` 与实际代码状态一致；
- 没有把尚未执行的训练或预期数值写成已完成结果。

## 7. 本次文档更新边界

本次只新增计划和 changelog，并更新文档索引。没有执行上述代码修改，也没有启动新的训练或评估。
