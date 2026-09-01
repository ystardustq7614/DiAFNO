# 实验 06 结果：full3d 30 层训练与评估

> 状态：**未执行**
> 结果：没有 full3d checkpoint、训练曲线、资源实测或正式评估指标。

## 当前证据与未执行原因

- surface 确定性基线已通过 day-1 门槛，但 test 15-day overall ratio 仍为 `1.018`；
- 当前可靠科学证据集中在 surface，30 层的尺度、增量和 persistence skill 尚未审计；
- 现有 full3d 成本估计来自尺寸外推，缺少实测峰值显存、I/O 和吞吐；
- detached multi-step 代码尚未实现，因此 K3 pilot 目前不能执行。

这些条件不足以支持直接启动 full3d 正式长训，但不再阻止只读数据画像、资源 probe 和
现有 single-step K1 smoke。

## 恢复与准入条件

1. 全 30 层数据连续性、mask、finite 值和归一化画像通过；
2. stats cache、单样本 I/O 和单 batch 峰值显存有完整记录；
3. K1 smoke 无 OOM、非有限 loss 或 AMP update 异常；
4. single-step pilot 在逐层指标上显示可预测信号；
5. multi-step 路径通过 surface 回归后，才允许 K3 pilot；
6. 正式 epoch、容量、GPU 数和评估窗口在启动前单独冻结。

## 结果记录规则

后续运行后在本文追加实际环境、配置、产物、逐层/分 band 指标、资源数据、异常和科学
分析。代码实现及测试结果写入项目 Changelog，不在本文记录。
