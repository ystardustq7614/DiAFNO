# 汇报图表清单（presentation/figures/）

生成方式：`python presentation/make_figures.py`（repo 根目录，diafno 环境）。
所有数值图由本地 `checkpoints/PRE/` NPZ 复算，脚本内置 20 项与 RESULTS.md 的
逐值断言（不一致即 FAIL）；非 NPZ 数字集中在脚本 CONSTANTS 并注明出处。
重新生成只需重跑脚本，勿手工改图。

| 文件 | 对应大纲页 | 内容 | 数据来源 |
|---|---|---|---|
| `fig_p20_lead_ratio.png` | **P20 主图** | Surface 15-day 逐 lead ratio：单步/MS5/MS10 三实线 + SD2 diffusion 虚线参照 | 4 个 test h15 NPZ（RES Ep10、MS5 Ep4、MS10 Ep2、SD2 Ep3） |
| `fig_p19_overall_bars.png` | P19 | Day-1 与 15-day overall ratio 双组柱 × 4 模型 | 同上 |
| `fig_p13_condition_signal.png` | P13 | 6 柱条件诊断（probe/persistence/true/错配/zero/reversed） | exp 05 RESULTS 常量（156 val 窗口、同 seed） |
| `fig_p12_sampler_ablation.png` | P12 | sampler/checkpoint/ensemble 消融 6 柱 | SD2 目录 val h1 NPZ 复算 |
| `fig_p22_layers.png` | P22 | 三层单步 vs MS5（test h15 overall）；middle 优先采用 Ep4 正式 test，不存在时注明 Ep2 探索性 | 6 个 test h15 NPZ（surface/middle/bottom × RES/MS5） |
| `fig_p24_diagnostics.png` | P24 | 三缺陷诊断：ratio 回升 / corr 反超 / var_ratio 塌缩（val，MS5 Ep4 + MS10 Ep2） | 修复后 leadtime_diag NPZ ×2（**surface 单步 diag 为修复前坏档，未用**） |
| `fig_p07_forecast_maps.png` | P7 | 真值/预测/误差 三联拼版（v 分量，d1 与 d15；u 面板未归档） | MS10 test 归档 PNG 拼版 |
| `p06_field_mask_sanity.png` | P6 | 区域地图 + 陆地 mask 素材（复制自 `plots/`） | plots/01_field_mask_sanity.png |

未画（PPT 内用表格/文字呈现即可）：P11 SD1→SD2 对照表、P21 已关闭分支、
P23 full3d 资源、P25 决策页。架构图（P4）、方法流程图（P14/P18）、因果链（P14）
建议在 PPT 工具中重绘。

注意：当前 P22 的 middle MS5 柱为 **Ep2 探索性 test（0.830）**。若汇报前在
`middle_smoke_..._RES_MS5/` 补跑 Ep4 正式 test，生成唯一的
`eval_test_h15_*_ckptEp4*.npz` 后重跑本脚本即可自动更新该柱、标题数字与“Ep4 正式”标记；
脚本会核对 `split=test`、`preset=middle_smoke`、15-day、residual objective 和 `Ep4.pth`，
不会把现有 Ep4 validation NPZ 误用为 test。
