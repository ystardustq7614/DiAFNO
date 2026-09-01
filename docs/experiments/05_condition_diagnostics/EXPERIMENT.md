# 实验 05：条件通路与任务可预测性诊断

> 状态：部分执行
> 诊断目录：`checkpoints/PRE/diag_noGo_20260828/`

## 实验目的

区分以下可能原因：

1. 数据或 condition/target 时间配对没有可预测信号；
2. condition 根本没有进入网络；
3. condition 已进入，但扩散训练/采样没有形成可靠的条件预测器；
4. 失败主要集中在海岸和 mask 边界。

## 任务与执行状态

| 子实验 | 对照 | 目的 | 状态 |
|---|---|---|---|
| 14 通道共享 ridge/linear probe | persistence、zero | 验证历史条件本身是否含可利用信号 | 已完成 |
| condition 破坏实验 | 真实、另一窗口、全零、14 通道反转 | 验证 condition 是否进入并影响网络 | 已完成 |
| 空间相关 | diffusion prediction vs persistence | 判断是否保留真实大尺度结构 | 已完成 |
| 区域分层 | coastal band vs open ocean | 判断海岸/mask 是否主导失败 | 已完成 |
| 网络敏感度与采样轨迹 | 不同 sigma 区间 | 定位条件信息在哪个阶段丢失 | 未执行，当前延期 |

## 固定条件

- 已执行的 condition 对照使用相同 validation 156 个窗口和相同 seed。
- 指标链保持同一归一化、rho→native 映射和 mask 口径。
- 只改变条件内容，不改变加噪初值和 sampler 配置。

## 记录指标

- pooled native day-1 RMSE 及 model/persistence ratio。
- prediction/truth 空间相关系数。
- coastal/open-ocean RMSE。
- 条件被破坏后的性能下降顺序。
- 后续轨迹探针需记录每个 sigma 区间的 condition sensitivity。

## 执行方法

诊断脚本位于 `checkpoints/PRE/diag_noGo_20260828/`。已执行的日志由
`probe_linear.py` 和 `probe_sample_conds.py` 生成；汇总图和一致性断言运行：

```powershell
D:\CondaData\envs_dirs\diafno\python.exe scripts\analyze_checkpoint_results.py
```

尚未执行的脚本必须在生成日志后才能把状态改为完成；当前主线不依赖这两项，因此保持
“部分执行”，不以脚本存在代替实验完成。

## 预期判别

- 若 linear probe 优于 persistence：任务和时间配对存在可预测信号。
- 若真实 condition 优于错误/全零/反转 condition：条件通路没有断。
- 若以上两项成立但 diffusion 仍败于 persistence：主要问题转向训练目标和高噪声采样中的条件约束。
- 若 coastal 显著更差但 open ocean 也失败：mask 是放大因素，不是唯一根因。

结果和未完成诊断见 [RESULTS.md](./RESULTS.md)。
