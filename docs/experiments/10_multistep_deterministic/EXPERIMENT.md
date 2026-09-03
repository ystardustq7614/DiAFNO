# 实验 10：surface detached autoregressive multi-step（MS5 / MS10）

## 目标

检验首要假设（`docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md` §3）：
detached autoregressive multi-step 能在接近单步显存的条件下缓解 exposure bias，
并延后或消除 day 4–5 crossover。

- **MS5 臂**：`K=5`，schedule `1,2,1,3,1,4,1,5`，从实验 07 Ep10 weights-only 初始化，
  lr 1e-4（fresh optimizer/scheduler），最多 5 epochs。
- **MS10 臂**：`K=10`，schedule `1,2,1,3,…,1,10`，仅在 MS5 通过全部门槛后执行
  （§6 WP4），从 MS5 选型 checkpoint weights-only 初始化，最多 3 epochs。
- 其余一切与实验 07 冻结协议一致（backbone/数据/归一化/mask/loss/rf0/无静态 mask）；
  单变量 = 训练 horizon。

## 设计

- 训练：`pre_trainer.py`，`DIAFNO_TRAIN_HORIZON={5,10}` + `DIAFNO_INIT_CHECKPOINT`，
  单卡 4090（GPU 0），batch 4，AMP，按 batch index 的固定 lead schedule（50% day-1
  anchor），前 J-1 步 no_grad 自回归纳回灌（clamp [0,1]、rf0），仅第 J 步反传。
- 逐 epoch 选型：`pre_evaluate.py` 协议（val，15 天自回归，stride 7，154 窗口，
  native m/s）；先过 day-1 ≤ 0.1031 守门，再按 15-day overall 排名。
- 结构诊断：`scripts/diag_leadtime_residual.py`（val，stride 14，77 窗口）。
- test：配置与 checkpoint 冻结后各运行一次（h15，stride 7，154 窗口）。

## 状态

- MS5：**已完成，Go**（全部门槛通过）。
- MS10：**已完成**（val 全面优于 MS5）。
- DDP2 smoke：**已完成（2026-09-03，修复 autocast 权重缓存问题后 SMOKE PASS）**，
  见 `RESULTS.md` 问题 1 与 Changelog。

运行入口与产物路径见 `RESULTS.md`。
