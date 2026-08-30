# 实验 07：surface persistence-residual 确定性基线 — 结果

> 状态：未执行（本文件只记录真实状态，不填造任何数值）

## 当前状态

- **未执行真实数据训练与评估**。阻塞原因：
  1. 单卡真实数据 smoke（Phase 2 前半）尚未在服务器运行；
  2. 目标 world size 的 DDP smoke（Phase 2 后半）尚未运行；
  3. Phase 3 短训练与 validation 选型、Phase 4 test 报告均以此为先决条件。

## 已完成（代码与本地验证，2026-08-30）

- `pre_models.py`：`PersistenceResidualIAFNO`（零初始化残差头、persistence identity、
  rollout 兼容 `sample()`）与 `masked_mse_loss`。
- `pre_config.py`：objective 配置与校验、`_RES` run tag、`ProgressReporter` 进度辅助。
- `pre_trainer.py` / `pre_evaluate.py` / `pre_rollout.py`：objective 切换与重建、
  checkpoint objective/结构守卫、`remask_feedback` 开关、rank-0 tqdm/PROGRESS 进度。
- `pre_smoke_test.py`：新增 9 项 CPU 回归测试，全部通过（共 35 项，4 项 CUDA 专属
  本地跳过）；legacy `smoke_test.py` 与既有测试结果不变。

## 待填结果（执行后补录，绝不预先填写）

- smoke：单卡/DDP `SMOKE PASS` 证据与产物路径。
- 短训练：loss curve、每 epoch validation 指标、validation day-1 native RMSE 选型表。
- test：day-1 / 15-day native RMSE/MAE、model/persistence 比值、u/v 与 coastal/offshore 分项。
- 结论：Phase 3 Go / No-Go 与后续决策。
