# PRE 实验索引与记录规范

> 更新日期：2026-09-01

## 实验索引

| 编号 | 实验问题 | 状态 | 实验设计 | 结果与分析 |
|---:|---|---|---|---|
| 01 | surface SD1 旧尺度基线 | 已完成，失败 | [EXPERIMENT](./01_surface_sd1_baseline/EXPERIMENT.md) | [RESULTS](./01_surface_sd1_baseline/RESULTS.md) |
| 02 | 修复 sigma_data 后，surface diffusion 是否过门槛 | 已完成，未过门槛 | [EXPERIMENT](./02_surface_sd2_retrain/EXPERIMENT.md) | [RESULTS](./02_surface_sd2_retrain/RESULTS.md) |
| 03 | checkpoint/sampler 参数是否是 day-1 失败主因 | 已完成，否 | [EXPERIMENT](./03_sampler_ablation/EXPERIMENT.md) | [RESULTS](./03_sampler_ablation/RESULTS.md) |
| 04 | surface SD2 diffusion 的 15-day rollout 表现 | 已完成，失败 | [EXPERIMENT](./04_surface_sd2_rollout/EXPERIMENT.md) | [RESULTS](./04_surface_sd2_rollout/RESULTS.md) |
| 05 | condition 是否有效、任务是否有可预测信号 | 部分完成，已有明确结论 | [EXPERIMENT](./05_condition_diagnostics/EXPERIMENT.md) | [RESULTS](./05_condition_diagnostics/RESULTS.md) |
| 06 | full3d 30 层训练与评估 | 未执行；等待数据/资源准入证据 | [EXPERIMENT](./06_full3d/EXPERIMENT.md) | [RESULTS](./06_full3d/RESULTS.md) |
| 07 | persistence-residual 是否能建立确定性基线 | 已完成；day-1 通过，15-day 未通过 | [EXPERIMENT](./07_residual_baseline/EXPERIMENT.md) | [RESULTS](./07_residual_baseline/RESULTS.md) |
| 08 | 静态 mask 输入是否改善 day-1/近岸预测 | 已完成；不保留 | [EXPERIMENT](./08_static_mask_ablation/EXPERIMENT.md) | [RESULTS](./08_static_mask_ablation/RESULTS.md) |
| 09 | 每步 remask feedback 是否改善 15-day rollout | 已完成；保持 rf0 | [EXPERIMENT](./09_remask_feedback_ablation/EXPERIMENT.md) | [RESULTS](./09_remask_feedback_ablation/RESULTS.md) |

## 证据链

```text
diffusion 基线失败
  ├─ 修复 sigma_data 后仍失败
  ├─ sampler/checkpoint 消融不能解释主要差距
  └─ 条件诊断证明：任务有信号，模型也确实读取 condition
       └─ persistence-residual 确定性基线
            ├─ day-1 优于 persistence
            ├─ 15-day overall 持平略差
            ├─ 长时效问题：方差塌缩 + 相关衰减 + 偏差漂移
            ├─ 静态 mask 输入无稳定增益
            └─ 每步 remask 只有中段增益，长段转差
                 └─ 当前方向：detached autoregressive multi-step
                      └─ 通过门槛后再做 MS10、代表层和 full3d probe/pilot
```

当前困难、下一步顺序与验收门槛见
[当前困难与下一步](../project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md)。

## 单次实验文档职责

每个实验目录只回答一个可证伪问题，并固定包含：

### `EXPERIMENT.md`

记录实验开始前定义的内容，以及任务是否执行：

- 科学问题、目标和假设；
- 对照组、控制变量、数据 split 和配置；
- 任务列表与执行状态；
- 指标、准入/停止条件和结果入口。

可以把状态从“未执行”更新为“执行中/已完成/失败”，但不在这里堆实际结果表或事后改写
原始假设。若执行偏离设计，只记录偏离事实并指向 `RESULTS.md`。

### `RESULTS.md`

只记录实际发生的实验事实：

- 实际环境、配置、checkpoint、日志和产物路径；
- 指标、图表、失败或中断；
- 对照分析、结论、适用边界和下一步影响；
- 实验中发现的代码问题，以及该问题是否影响结果可信度。

`RESULTS.md` 不记录“新增了哪些函数、修改了哪些文件、多少项单元测试通过”等代码修改
成果；这些内容统一写入 [项目 Changelog](../project/CHANGELOG.md)。当前可执行命令统一维护在
[运行手册](../operations/PRE_runbook.md)。

## 更新规则

1. 新实验执行前建立目录和 `EXPERIMENT.md`；`RESULTS.md` 可先标记“未执行”。
2. validation 用于选配置和做 A/B；test 只在配置冻结后运行，不能反向改实验设计。
3. 结果判断以原生 C-grid、native mask、物理单位 m/s 为准。
4. 方向文档引用 `RESULTS.md` 的结论，不复制整张结果表。
5. 同一科学问题不要通过“追加小节”塞进旧实验；建立新的实验编号。
