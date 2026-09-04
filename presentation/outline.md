# 组会汇报大纲（修订版 v2）

> 基准：需求确认版大纲（2026-09-04）。
> 本版修订：数字全部对照 `docs/experiments/*/RESULTS.md` 与本地 `checkpoints/PRE/` NPZ 核验，
> 修正 4 处不妥（见文末"修订记录"）；证据来源速查见附录 A。
> ⚠️ 标记 = 本地仓库无法核实、需对照原论文原文确认的数字。

整场主线：

```text
从原始 DiAFNO 出发
→ 完成 PRE 数据适配
→ direct diffusion 明确失败
→ 排除数据、condition、sampler 等原因
→ 转向 persistence-residual
→ multi-step 解决长期 rollout 退化
→ 讨论 diffusion 应以什么形式重新进入
```

核心结论可以概括为：

> 原始 DiAFNO 在周期、完整三维状态、统计评价的湍流任务中有效，但直接迁移到复杂海岸网格上的确定性海流点预测并不成立。通过 persistence-residual 和 detached multi-step，我们已在 surface 15-day RMSE 上稳定超过 persistence；下一步应围绕长期方差塌缩、代表层确认和 mask-aware residual diffusion 展开。

# 一、问题背景与结论先行｜0–4 分钟｜2 页

### 第 1 页：标题

**面向 PRE 海流场的 DiAFNO 迁移与适配：从条件扩散到确定性多步预测**

副标题：

> 7-day condition → 15-day autoregressive u/v forecasting

口头开场：

> 导师最初希望验证原始 DiAFNO 能否迁移到 PRE 海流数据。项目首先忠实跑通了 diffusion 路线，但它没有超过 persistence。后续工作围绕"为什么失败"和"怎样得到可靠预测"展开。

### 第 2 页：三条阶段结论

只放三个大结论：

1. Direct diffusion（SD2 修正尺度后）：day-1 ratio `2.201`，15-day overall ratio `1.640`，失败。
2. Detached multi-step：`1.018 → 0.871 → 0.838`，长期 crossover 消除。
3. 当前瓶颈：d15 方差塌缩、middle 正式确认、full3d 投入决策。

这一页先让听众知道最后得到什么，再进入过程。

# 二、原始 DiAFNO 为什么合理｜4–11 分钟｜3 页

### 第 3 页：原论文解决什么问题

介绍原论文：

- 输入一个完整三维速度场 `U_m`；
- 条件 diffusion 生成 `U_{m+1}`；
- 将预测重新作为下一步条件；
- 面向 HIT、decaying HIT 和 channel flow；
- 核心目标是长期湍流统计稳定。

### 第 4 页：原始 DiAFNO 架构

建议重绘一张简化图：

```text
U_m ───────────────┐
                   ├─ IAFNO denoiser ─ Heun sampling ─ U_{m+1}
noisy U_{m+1} ─────┘
```

说明两个组成：

- EDM：从噪声生成下一时刻流场；
- IAFNO：在频域捕获全局结构和能量分布。

### 第 5 页：为什么原论文能成功

重点讲四点：

- 周期、规则、结构化网格适合 FFT；
- fDNS 经过谱滤波，目标场相对平滑；
- 完整三维 `u/v/w` 接近 Markov state；
- 长期主要看 spectrum、RMS、PDF、Reynolds stress，而不是唯一轨迹逐点吻合。

补充一句：

> 原论文 Appendix C.3 明确说明：CF590 的长期 point-wise error 会累积并稳定在约 `0.14`；Appendix C.2 Table 10 的 400-step 不同随机种子结果为 `0.1374–0.1439`（基准 seed 123 为 `0.1400`）。因此它的核心成功是长期统计分布稳定，不是持续追踪同一条真实轨迹。

来源：[论文 v3（Appendix C.2/C.3）](https://arxiv.org/html/2512.12628v3)。

# 三、PRE 任务与仓库适配｜11–19 分钟｜4 页

### 第 6 页：PRE 数据是什么

展示：

- 1994–2022；
- 10,591 个连续日平均场；
- `400×441×30` sigma layers；
- staggered C-grid；
- surface/middle/bottom/full3d；
- 大量陆地 mask 和复杂海岸。

最好放一张区域地图、mask 或代表流场。

### 第 7 页：预测任务定义

画成时间窗口：

```text
过去 7 天 u/v                   未来 15 天
[t-6 ... t] → day 1 prediction → feedback → ... → day 15
   14 ch          2 ch
```

强调：

- 正式任务是确定性点预测；
- 正式指标为 native C-grid、物理单位 `m/s`；
- persistence 是必须战胜的基线；
- `ratio < 1` 才算有效。

### 第 8 页：原论文与 PRE 的关键差异

| 原论文 | PRE | 影响 |
|---|---|---|
| 周期结构化网格 | 曲线网格、海岸、mask | FFT 会跨边界全局混合 |
| 完整三维 `u/v/w` | surface/单层 `u/v` | 状态不闭合 |
| 固定流态和 forcing | 季节变化及缺失外部强迫 | 条件分布更宽 |
| 谱滤波低分辨率 | `400×441` 复杂空间结构 | 从噪声重建难度更高 |
| 长期统计评价 | 逐点 RMSE/MAE | stochastic sample 处于劣势 |
| Table 7：100 epochs 内最小 train/test loss | SD2 早停于 5 epochs | 训练预算不完全等价 |

这一页是整场报告的重要认知桥梁。

口径说明：`100 epochs` 是论文 Table 7 的模型比较窗口，不是公开代码的固定默认值；上游 `trainer.py` 当前写的是 `num_epochs = 150`。来源：[论文 v3](https://arxiv.org/html/2512.12628v3)、[上游 trainer.py](https://github.com/yuchi-richard-jiang/DiAFNO/blob/main/trainer.py)。

### 第 9 页：仓库整体管线

```text
raw PRE
  ↓
preprocess_align_uv.py
  ↓
pre_dataset.py
  ↓
IAFNO.py / diffusion.py / pre_models.py
  ↓
pre_trainer.py
  ↓
pre_rollout.py / pre_evaluate.py
  ↓
docs/experiments/01–11
```

只讲职责，不介绍函数细节。

# 四、Direct diffusion 实验与失败诊断｜19–30 分钟｜5 页

### 第 10 页：如何迁移原始 DiAFNO

说明适配内容：

- condition 从 3 通道上一时刻，改为 14 通道七天历史；
- target 改为下一天 surface `u/v`；
- 加入 ocean mask；
- 建立 15-day autoregressive rollout；
- 建立 persistence、zero 和 native-grid 评价。

### 第 11 页：SD1 → SD2

用对照表：

| 版本 | 问题 | 结果 |
|---|---|---|
| SD1 | `sigma_data` 使用 `[0,1]` 尺度 | day-1 约为 persistence 的 2.7–3.1 倍（u 2.72 / v 3.11） |
| SD2 | 按 `[-1,1]` 修正尺度 | 有改善，但仍明显失败（day-1 2.201 / 15-day 1.640） |

结论：

> 尺度错误是问题之一，但不是根因。

### 第 12 页：Sampler、checkpoint、ensemble 消融

突出：

- `churn=0` 优于 `churn=80`；
- Ep3 是最佳 diffusion checkpoint；
- `sigma_max=3` 无效（ratio 2.204）；
- E=4 仅改善 4.4%；
- 最好结果仍约为 persistence 的 1.91 倍。

结论：

> sampler 可以改变结果，但不能救回当前条件模型。

### 第 13 页：任务到底有没有信号

展示一张柱状图（实验 05，同 156 个 validation 窗口、同 seed）：

| 方法/条件 | Day-1 RMSE (m/s) |
|---|---:|
| Linear probe | `0.1177` |
| Persistence | `0.1293` |
| Diffusion + true condition | `0.2584` |
| 错配条件（另一窗口） | `0.3408` |
| Zero condition | `0.4775` |
| Reversed condition | `0.5655` |

解释：

- linear probe 证明任务可预测；
- condition 错配/破坏后单调变差，证明通路没有断；
- diffusion 使用了 condition，但没有学成可靠预测器。

### 第 14 页：当前对 diffusion 失败的机制判断

建议画成因果链：

```text
复杂 masked domain
+ 部分观测状态
+ 从噪声重建完整流场
+ denoising objective 与点预测指标错位
+ 无 persistence 锚点
        ↓
高噪声阶段 condition 无法稳定锚定真实流场
        ↓
预测趋向总体平均/zero-like field
        ↓
day-1 已失败，rollout 进一步积累
```

明确说：

> 不能证明 diffusion 普遍无效；能证明的是当前 direct-field conditional EDM 在 PRE 确定性任务上不成立。

# 五、为什么转向 Persistence-Residual｜30–38 分钟｜4 页

### 第 15 页：路线转向依据

提出任务结构：

\[
U_{t+1}\approx U_t+\Delta U_t
\]

所以不必从噪声重新生成整个流场，只需要预测相对 persistence 的修正。

### 第 16 页：确定性 residual 模型

\[
\hat U_{t+1}=U_t+f_\theta(U_{t-6:t})
\]

强调：

- 使用同一个 IAFNO backbone；
- zero-init residual head；
- 未训练模型严格等于 persistence；
- 精细结构直接通过 persistence 保留。

### 第 17 页：单步成功，但长期仍失败

展示 single-step 结果（surface，Ep10）：

- day-1 明显优于 persistence（ratio `0.833`）；
- 15-day overall ratio `1.018`；
- test 全量口径 crossover 约在 day 5，之后持续恶化（d7 1.014、d15 1.117）¹；
- correlation 下降、variance collapse、bias 漂移。

¹ 口径脚注：exp 07 test 全量（stride 7）crossover ≈ d5；exp 10 的 val 补测（stride 14、77 窗口）为 d13。两者 split/stride 不同，不矛盾；被追问时看备份页。

结论：

> backbone 和 condition 都具备预测能力，新的主要问题变成 exposure bias。

### 第 18 页：Detached multi-step 方法

画训练流程：

```text
condition
   ↓
预测 J-1 步（no_grad）
   ↓
构造模型自己的反馈窗口
   ↓
第 J 步预测
   ↓
只对第 J 步反向传播
```

补充：

- `J=1` 保留 50% anchor；
- MS5 从 single-step 初始化；
- MS10 从 MS5 初始化；
- 不做完整 BPTT，控制显存和训练稳定性。

# 六、主要结果与泛化｜38–46 分钟｜5 页

### 第 19 页：Surface 主结果

大数字展示（test，154 窗口，stride 7）：

| 模型 | Day-1 ratio | 15-day overall ratio |
|---|---:|---:|
| Single-step Ep10 | `0.833` | `1.018` |
| MS5 Ep4 | `0.843` | `0.871` |
| MS10 Ep2 | `0.833` | **`0.838`** |

当前正式模型：

```text
surface MS10 Ep2
model overall RMSE     0.1759 m/s
persistence            0.2098 m/s
relative reduction     16.2%
```

### 第 20 页：逐 lead 曲线

主图：

- 横轴 day 1–15；
- 纵轴 model/persistence ratio；
- single-step、MS5、MS10 三条曲线；
- `ratio=1` 水平线。

这是整场最重要的一张图。

### 第 21 页：已经关闭的消融分支

- 静态 mask 输入：不保留；
- feedback remask：不保留；
- 当前配置：
  - 14 dynamic channels；
  - no static mask；
  - `remask_feedback=False`。

强调项目不是不断堆功能，而是通过实验关闭无效分支。

### 第 22 页：垂向代表层

状态矩阵（口径统一：**test h15 overall ratio**）：

| 层 | 单步（test h15 overall） | MS5（test h15 overall） | 当前结论 |
|---|---:|---:|---|
| Surface | `1.018` | `0.871` / MS10 `0.838` | 正式成立 |
| Middle | `1.183` | Ep2 `0.830`（探索性） | 正式应选 Ep4（val h15 0.820），test 待确认 |
| Bottom | `0.930` | `0.813` | 全部门槛通过 |

middle 勘误主动说明，不隐藏（备份页有完整数字）。

### 第 23 页：Full3d 状态

展示：

- 峰值显存约 `22.6 GB`；
- 约 `2.3 h/epoch`；
- 50 epochs 约 5 天；
- 1-epoch pilot 健康；
- 但 60 个 layer×variable 都没有出现明确 day-1 技能；
- K3 按预注册门槛暂时阻塞。

结论：

> 当前障碍不是代码没跑通，而是计算预算与准入证据不足。

# 七、未解决问题与下一步决策｜46–50 分钟｜2 页

### 第 24 页：当前科学缺陷

集中展示三项：

1. `var_ratio≈0.3@d15`（MS10 0.337）：预测偏平滑；
2. d15 ratio 回升到约 `0.894`（全 rollout 最差 lead）；
3. u/v 出现不同形式的 bias 和 correlation 损失。

然后把原论文重新接回来：

> 原论文中 diffusion 的优势正是保持长期统计分布，因此它可能不是被永久放弃，而应以 residual/probabilistic branch 的形式重新进入。

候选方向：

\[
\hat U_{t+1}
=
U_t+\mu_\theta(c)+\epsilon_\phi(c)
\]

- deterministic MS 模型负责条件均值；
- diffusion 建模未解析残差和不确定性；
- 同时评价 RMSE、variance、spectrum、CRPS。

### 第 25 页：希望组内讨论的三个决策

1. 是否先补 middle Ep4 正式 test，完成代表层闭环？
2. full3d 是追加 single-step epochs，还是暂时冻结等待独立预算？
3. 下一分支优先级：
   - mask-aware diffusion diagnostic；
   - residual diffusion；
   - physical loss weighting/direct multi-horizon。

推荐立场：

> 先闭合 middle 和 diffusion 机制诊断；full3d 暂不直接进入昂贵 K3；后续 diffusion 以解决方差塌缩和概率预测为目标，而不是重新竞争单轨迹 day-1 RMSE。

# 备份页建议

准备 6–8 页，不主动讲：

1. 实验 01–11 完整索引；
2. `sigma_data` 修正推导；
3. EDM 训练与 Heun sampler 公式；
4. 原论文与 PRE 的完整参数对照；
5. checkpoint 选择规则及 `best.pth` 陷阱；
6. middle 勘误完整数字（门槛 0.05269；Ep4 0.05240 过 / Ep5 0.05238 过 / Ep1–3 未过；选型 Ep4 val h15 0.8202 < Ep5 0.8231）；
7. DDP+AMP detached feedback 修复；
8. 数据、checkpoint、NPZ 的本地/服务器证据范围；
9. crossover 口径对照（test 全量 ≈d5 vs val stride-14 补测 d13）。

## 时间控制原则

- 原论文与任务差异：7 分钟，不能略过；
- diffusion 失败证据：11 分钟，是导师最可能追问的部分；
- residual/multi-step 与结果：16 分钟，是主要贡献；
- 工程细节只讲一页；
- full3d 不展开代码，只讲资源和决策；
- 每页只保留一个明确结论。

---

## 修订记录（v1 需求确认版 → v2）

1. **P13 术语对齐**："Shuffled condition" 改为"错配条件（另一窗口）"——实验 05 原文口径是 condition 错配（swap），不是 shuffle；同时标注实验条件（同 156 窗口、同 seed），防止被追问。
2. **P17 crossover 口径**：原文"day 4–5 后 crossover"改为 test 全量口径（≈d5），并加脚注说明 exp 10 val 补测 d13 的差异（split/stride 不同）——这是最可能被追问的数字，备份页新增第 9 条。
3. **P22 口径统一**：表头明确为 test h15 overall ratio（原"Single-step/MS5"列混用 day-1 与 15-day 口径）；middle 行补充 Ep4 的 val 选型数字 0.820。
4. **P19 单步 day-1**："优于 persistence" 落成具体数字 `0.833`，与 MS10 对齐可比。
5. **原论文数字核实**：P5 的 `0.14` 限定为 CF590 长期 point-wise error 平台（Appendix C.3；Table 10 的 400-step seed 结果为 0.1374–0.1439）；P8 的 `100 epochs` 限定为 Table 7 的比较窗口，并注明上游训练模板默认 150，避免混淆。

## 待办素材（本文件夹后续内容）

- [x] P20 主图：`figures/fig_p20_lead_ratio.png`（3 确定性曲线 + diffusion 虚线，数值自检通过）
- [x] P5/P8：已按论文 Appendix C.2/C.3 与 Table 7 核实 `0.14`、`100 epochs` 的适用口径
- [x] P6：区域地图/mask 素材 → `figures/p06_field_mask_sanity.png`
- [x] P12/P13 柱状图 → `figures/fig_p12_sampler_ablation.png` / `fig_p13_condition_signal.png`
- [x] 增值图：P19 柱状（`fig_p19_overall_bars.png`）、P22 分层（`fig_p22_layers.png`）、
  P24 诊断三联（`fig_p24_diagnostics.png`）、P7 场拼版（`fig_p07_forecast_maps.png`）
- [ ] middle Ep4 正式 test（可选，汇报前闭环则 P22 更新为正式数字；属预注册纪律下的
  正式一次，需按 Runbook §4 冻结配置后执行；完成后重跑 `make_figures.py` 更新 P22 柱）

全部图表的对应关系与数据来源见 `figures/MANIFEST.md`。

## 附录 A：证据来源速查

| 汇报数字 | 来源 | 本地可复算 |
|---|---|---|
| CF590 长期 point-wise error ≈0.14；Table 7 的 100-epoch 比较窗口 | [论文 v3](https://arxiv.org/html/2512.12628v3) Appendix C.2/C.3、Table 7；[上游 trainer.py](https://github.com/yuchi-richard-jiang/DiAFNO/blob/main/trainer.py) | 论文侧核对（非本项目 NPZ） |
| SD1 day-1 2.72/3.11 | `docs/experiments/01_surface_sd1_baseline/RESULTS.md` | — |
| SD2 早停 5 epochs、scale 修复 | `docs/experiments/02_surface_sd2_retrain/RESULTS.md` | — |
| churn/sigma_max/E=4 消融 | `docs/experiments/03_sampler_ablation/RESULTS.md` | `checkpoints/PRE/surface_..._SD2/eval_*.npz` |
| 2.201 / 1.640 | `docs/experiments/04_surface_sd2_rollout/RESULTS.md` | `checkpoints/PRE/surface_..._SD2/` |
| linear probe 0.1177、条件破坏对照 | `docs/experiments/05_condition_diagnostics/RESULTS.md` | `checkpoints/PRE/diag_uv_predictability_20260901/` |
| single-step 0.833/1.018、crossover ≈d5 | `docs/experiments/07_residual_baseline/RESULTS.md` | `checkpoints/PRE/surface_..._RES/eval_test_h15_..._Ep10.npz` |
| MS5 0.871 / MS10 0.838、var_ratio、0.894@d15 | `docs/experiments/10_multistep_deterministic/RESULTS.md` | `checkpoints/PRE/surface_..._RES_MS5|MS10/eval_test_*.npz` |
| middle 1.183、Ep2 0.830（探索性）、勘误数字 | `docs/experiments/11_representative_layers/RESULTS.md` | `checkpoints/PRE/middle_..._RES|_MS5/eval_*.npz`（2026-09-04 已独立复算确认 Ep4 选型） |
| bottom 0.930/0.813 | `docs/experiments/11_representative_layers/RESULTS.md` | `checkpoints/PRE/bottom_..._RES|_MS5/` |
| full3d 22.6 GB、2.3 h/epoch、pilot 无信号 | `docs/experiments/06_full3d/RESULTS.md` | — |
