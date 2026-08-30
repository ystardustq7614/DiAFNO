# 实验 03：day-1 采样与 checkpoint 消融

> 状态：已执行
> 数据：validation，156 个窗口
> 共同设置：day-1、32-step Heun、seed 123、原生 C-grid masked 指标

## 实验目的

判断 SD2 结果差是否主要来自 checkpoint、`S_churn`、ensemble size 或 `sigma_max`，
并为 15 天正式 rollout 选择代价合理的采样配置。

## 对照组

| 组别 | checkpoint | `S_churn` | ensemble | `sigma_max` | 目的 |
|---|---|---:|---:|---:|---|
| A | Ep2 | 0 | 1 | 默认 | 比较训练阶段 |
| B | Ep3 | 0 | 1 | 默认 | 主基准 |
| C | Ep4 | 0 | 1 | 默认 | 检查继续训练是否恶化 |
| D | Ep3 | 80 | 1 | 默认 | 测试 stochastic churn |
| E | Ep3 | 0 | 4 | 默认 | 测试 ensemble mean |
| F | Ep3 | 0 | 1 | 3 | 测试降低起始噪声上限 |

persistence 和 zero field 使用同一批窗口。除表中变量外，数据、归一化、模型、sampling
steps、seed 与指标实现保持一致。

## 记录指标

- pooled native masked day-1 RMSE/MAE，单位 m/s。
- model/persistence ratio。
- 相对单轨迹主基准的改善百分比。
- 运行成本；若 ensemble 收益小于 3%～5%，优先选 E=1。

## 执行方法

逐组设置 `CHECKPOINT`、`ROLLOUT_DAYS=1`、`SAMPLER_S_CHURN`、
`ENSEMBLE_SIZE` 和必要的 `SAMPLER_SIGMA_MAX`，每组运行一次：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u pre_evaluate.py
```

输出 tag 必须包含 checkpoint、churn、ensemble 和额外消融名；已有输出拒绝覆盖。

历史 `sigma_max=3` 组最初由提交 `fd3fc4c` 中的临时开关生成。当前 HEAD 已正式提供
`SAMPLER_SIGMA_MAX`：默认 `None` 保持 EDM 的 `sigma_max=80`；复现 F 组时设为 `3`。
显式覆盖会自动在输出 tag 中加入 `sm3`，实际值也写入 `.npz` 的 `sigma_max` 元数据；
若要与历史文件名完全区分，可另设新的 `OUTPUT_TAG`。

## 预期结果与选取规则

- `churn=0` 预期比 `churn=80` 稳定；若 RMSE 改善超过 20%，淘汰 churn=80。
- E=4 可能降低随机方差；若相对 E=1 改善小于 3%，选 E=1。
- 最终硬门槛仍是 `model/persistence < 1`，调参排名不能替代该门槛。
- 若所有组都不及 persistence，停止扩大 sampler 搜索空间。

## 主要产物

`checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/` 下的：

- `eval_val_h1_ch0_e1_s123_ckptEp2.npz`
- `eval_val_h1_ch0_e1_s123_ckptEp3.npz`
- `eval_val_h1_ch0_e1_s123_ckptEp4.npz`
- `eval_val_h1_ch80_e1_s123_ckptEp3.npz`
- `eval_val_h1_ch0_e4_s123_ckptEp3.npz`
- `eval_val_h1_ch0_e1_s123_ckptEp3_sigmax3.npz`

结果见 [RESULTS.md](./RESULTS.md)。
