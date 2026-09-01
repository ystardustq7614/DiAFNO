# DiAFNO / PRE 文档索引

> 更新日期：2026-09-01

文档按用途分为架构、数据、运行、项目交接和实验五类。实验目录统一使用
`EXPERIMENT.md` 记录实验设计，`RESULTS.md` 记录实际结果与分析，避免把“准备做什么”
和“已经得到什么”写在同一份报告中。

## 目录结构

```text
docs/
├── README.md
├── architecture/     # 原始 DiAFNO、IAFNO 与迁移框架
├── data/             # PRE 数据字典、网格和数据审计
├── operations/       # 可执行运行手册
├── project/          # 当前状态、修改计划、changelog 与交接摘要
└── experiments/      # 每组实验各自一个目录
    └── <experiment>/
        ├── EXPERIMENT.md
        └── RESULTS.md
```

## 通用文档

| 类别 | 文档 | 用途 |
|---|---|---|
| 架构 | [DiAFNO 源码分析与 7→15 帧改造评估](./architecture/DiAFNO源码分析与7到15帧改造评估.md) | 原框架、迁移可行性、代码改动与整体执行状态 |
| 架构 | [IAFNO 网络架构与公式对应](./architecture/IAFNO网络架构与公式对应.md) | IAFNO/EDM 数据流和公式—代码对应 |
| 数据 | [PRE_ocean_data 数据说明](./data/PRE_ocean_data.md) | 变量、shape、网格、mask、时间和数据质量 |
| 运行 | [PRE 运行手册](./operations/PRE_runbook.md) | 预处理、训练、评估和复现命令 |
| 项目 | [项目汇报与交接总结](./project/PROJECT_HANDOFF_SUMMARY.md) | 当前结论、证据边界和后续工作 |
| 项目 | [PRE 模型代码修改计划（已归档）](./project/archive/CODE_MODIFICATION_PLAN_20260830.md) | 2026-08-30 计划，Phase 0-5 已全部执行完毕；Phase 6 新计划待另立 |
| 项目 | [项目 Changelog](./project/CHANGELOG.md) | 已完成变更与尚未实施计划的状态记录 |
| 实验 | [实验索引](./experiments/README.md) | 所有实验的状态和结果入口 |

## 实验文档约定

- `EXPERIMENT.md`：状态、目的、假设、对照组、控制变量、执行步骤、指标、预期结果和停止条件。
- `RESULTS.md`：实际运行配置、产物、数值结果、异常、分析、结论和后续决策。
- 未执行实验的 `RESULTS.md` 只记录“未执行”和阻塞原因，不填造预测数值。
- 结果判断以原生 C-grid、native mask、物理单位 m/s 的 test 指标为准；validation 用于选配置，不能冒充最终成绩。
