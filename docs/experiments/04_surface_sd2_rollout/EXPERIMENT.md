# 实验 04：surface SD2 15 天 rollout

> 状态：已执行
> 数据：独立 test 集，154 个 rollout 窗口
> 结果文件：`eval_test_h15_ch0_e1_s123_ckptEp3.npz`

## 实验目的

用采样消融选出的 surface SD2 配置执行正式 15 天自回归评估，回答修复后模型是否
在任何预报时效稳定优于 persistence，以及误差如何随 lead day 累积。

## 实际采用的配置

| 项目 | 值 |
|---|---|
| checkpoint | `Ep3.pth` |
| rollout | 15 天 |
| sampler | 32-step Heun，`S_churn=0` |
| ensemble | 1 |
| seed | 123；按窗口起点派生独立 seed |
| 评估域 | 原生 staggered u/v 网格、各自 native mask |

## 对照实验

- persistence：把输入第 7 天原生 u/v 保持到所有未来日。
- zero field：预测全零流场。
- rho-oracle：量化 rho→native 映射引入的不可逆误差下界。
- 与 SD1 旧 test 使用相同窗口起点，并核验 persistence 数组一致。

## 记录指标

- 每个 lead day × u/v × layer 的 RMSE、MAE。
- 按有效点总数汇总的 pooled RMSE。
- model/persistence ratio 和相对 SD1 改善率。
- day 1/3/5/7/10/15 的 truth、prediction、error 图。

## 执行方法

设置 `ROLLOUT_DAYS=15`、`CHECKPOINT=Ep3.pth`、`SAMPLER_S_CHURN=0`、
`ENSEMBLE_SIZE=1` 后运行：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u pre_evaluate.py
```

评估结束后核验窗口起点、persistence 数组、checkpoint 元数据和输出 tag。

## 预期结果与通过标准

- 所有 lead day 的 `model/persistence < 1`。
- day-1 ratio 目标不高于 0.8，并且长期误差不出现异常尖刺式发散。
- 若 day-1 已失败，则 15 天结果用于诊断和完整记录，不进入 full3d。

实际结果见 [RESULTS.md](./RESULTS.md)。
