# DiAFNO / PRE 文档索引

> 更新日期：2026-09-01

文档按用途分为架构、数据、运行、项目交接和实验五类。活跃项目文档只保留两个入口：
交接概要回答“项目是什么、做到哪里”，当前困难与下一步回答“为什么卡住、接下来怎么做”。
代码实现记录归 Changelog，单次实验事实归各实验目录。

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
| 项目 | [项目交接概要](./project/PROJECT_HANDOFF_SUMMARY.md) | 面向新成员/agent 的快速项目全貌、当前进度和接手入口 |
| 项目 | [当前困难与下一步](./project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md) | 当前问题、工作顺序、准入门槛和事后回顾表；代码/实验尚未执行 |
| 项目 | [PRE 模型代码修改计划（已归档）](./project/archive/CODE_MODIFICATION_PLAN_20260830.md) | 2026-08-30 历史实施计划，仅用于追溯 |
| 项目 | [项目 Changelog](./project/CHANGELOG.md) | 已完成变更与尚未实施计划的状态记录 |
| 实验 | [实验索引](./experiments/README.md) | 所有实验的状态和结果入口 |

## 实验文档约定

- `EXPERIMENT.md`：单一科学问题、目标、假设、对照、任务、指标、判定规则和执行状态。
- `RESULTS.md`：实际运行配置、产物、数值结果、异常、分析、结论和适用边界。
- 未执行实验的 `RESULTS.md` 只记录“未执行”和阻塞原因，不填造预测数值。
- 结果判断以原生 C-grid、native mask、物理单位 m/s 的 test 指标为准；validation 用于选配置，不能冒充最终成绩。
- 代码文件/函数的新增修改及测试通过情况只写 Changelog，不写进实验结果；实验发现的代码问题及其对结果的影响可以写进 `RESULTS.md`。
- 完整规则见 [实验索引与记录规范](./experiments/README.md)。
