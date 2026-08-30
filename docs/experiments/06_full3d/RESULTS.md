# 实验 06 结果：full3d 30 层训练与评估

> 状态：未执行
> 结果：无 checkpoint、无训练曲线、无正式评估指标。

## 阻塞原因

前置 surface SD2 实验没有达到 `model/persistence < 1`：

- day-1 test ratio：2.201。
- 15-day overall ratio：1.640。

根据预先设定的 Go/No-Go 规则，full3d 暂停，不应把“代码支持 full3d”写成“full3d 已验证”。

## 结果分析

由于实验未启动，目前没有可分析的 full3d 数值结果。阻塞本身是一次按预设门槛作出的
实验决策，而不是 OOM、运行报错或训练失败；不能为该实验填写 surface 数值作为替代。

## 恢复条件

满足以下条件后再执行：

1. condition-only 确定性 surface 基线在 validation/test day-1 优于 persistence；
2. surface 15 天 rollout 稳定优于 persistence；
3. full3d 统计缓存和单 batch 显存/I/O 探针通过；
4. 明确记录实际 epoch、容量调整和评估窗口数。

后续真实产物应在本文件追加，`EXPERIMENT.md` 的设计与门槛保持不变。
