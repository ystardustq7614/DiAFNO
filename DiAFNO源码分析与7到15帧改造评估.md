---
title: DiAFNO 适配 PRE_ocean_data 的“前 7 天 → 后 15 天”预测评估
aliases:
  - DiAFNO PRE 海流预测源码分析
  - PRE_ocean_data 7 天到 15 天改造评估
tags:
  - DiAFNO
  - IAFNO
  - diffusion
  - PRE_ocean_data
  - ocean-current-forecasting
status: pre-implementation-plan
date: 2026-08-23
updated: 2026-08-24
---

# DiAFNO 适配 PRE_ocean_data 的“前 7 天 → 后 15 天”预测评估

> [!abstract] 结论先行
> 本文只讨论一个任务：使用 PRE_ocean_data 的连续 7 个日平均流场预测未来 15 天的 `u_eastward`、`v_northward`。不覆盖其他数据集，也不设计多数据集通用框架。
>
> 当前仓库是面向三维湍流数据的条件式单步扩散预测原型，只实现 `t → t+1`，没有完整的 autoregressive rollout。针对 PRE，本报告选择“7 天条件的单步模型 + 自回归滚动 15 次”作为第一版路线：输入条件通道为 `7 × 2 = 14`，每步目标通道为 2。仅修改 `InferenceWidth` 不足以完成任务，数据窗口、条件/目标通道、海陆 mask、归一化、空间网格和评估流程都需要适配。

> [!note] 2026-08-24 同步后状态
> 本报告首次完成后，分支已移除 `IAFNODiff` 构造和 padding 中的强制 CUDA 绑定，新增 `load_checkpoint`、`checkpoint_path` 与 `smoke_test.py`。这些变更解决了核心模型的 CPU/device 冒烟问题，但尚未实现 PRE 数据管线或 7→15 rollout。本文是实施前的任务收敛文档，不表示代码改造已经完成。

## 0. 分析范围与结论标记

本报告完整检查了以下文件：

- `IAFNO.py`：367 行；
- `diffusion.py`：289 行；
- `trainer.py`：261 行；
- `utilities3.py`：322 行；
- `README.md`：29 行。

同时检查了 `requirements-lock.txt`、`environment.yml` 和 `docs/PRE_ocean_data.md`。当前已有 CPU smoke test，但没有执行真实 PRE 数据训练；训练路径和数据路径仍是占位字符串。

下文使用三种标记：

- **源码确认**：代码可直接证明；
- **README 确认**：项目说明明确陈述；
- **PRE 文档确认**：由当前仓库的 `docs/PRE_ocean_data.md` 确认；
- **待确认**：现有源码和数据说明都不能决定，必须由实验要求或真实样例确认。

## 1. 文件作用与调用关系

### 1.1 总体调用图

```mermaid
flowchart TD
    CLI[python trainer.py] --> Load[np.load 与 TensorDataset]
    Load --> TrainLoader[train_loader]
    Load --> TestLoader[test_loader]

    Trainer[trainer.py 顶层训练脚本] --> Utils[utilities3.py]
    Trainer --> Diff[ElucidatedDiffusion]
    Trainer --> Backbone[IAFNODiff]

    Diff --> Backbone
    Backbone --> Patch[PatchEmbed]
    Backbone --> Blocks[反复调用 Block]
    Blocks --> AFNO[AFNO 三维频域混合]
    Blocks --> MLP[Mlp 局部通道混合]

    TrainLoader --> Forward[model target, condition]
    Forward --> Noise[目标加噪与 EDM 预条件]
    Noise --> Backbone
    Backbone --> Loss[加权去噪 MSE]

    TestLoader --> Sample[model.sample condition]
    Sample --> Denoise[随机噪声逐级去噪]
    Denoise --> Backbone
    Denoise --> Metric[LpLoss 相对 L2]
```

### 1.2 `trainer.py`

**作用：** 当前唯一的可执行训练脚本。它把配置、数据加载、数据切窗、归一化统计、模型构造、优化器、训练循环、测试循环和 checkpoint 保存全部放在模块顶层。

关键位置：

- 导入 `ElucidatedDiffusion` 与 `IAFNODiff`：`trainer.py:23-24`；
- 数据加载及切窗：`trainer.py:75-93`；
- 归一化统计与 `sigma_data` 计算：`trainer.py:95-130`；
- `DataLoader`：`trainer.py:134-140`；
- `IAFNODiff` 构造：`trainer.py:142-154`；
- `ElucidatedDiffusion` 包装：`trainer.py:158-164`；
- 训练与测试：`trainer.py:186-241`；
- 每轮保存权重与损失：`trainer.py:249-261`。

由于没有 `if __name__ == "__main__":`，导入 `trainer.py` 也会立刻尝试加载数据并开始训练。

### 1.3 `IAFNO.py`

**作用：** 定义扩散模型使用的三维 IAFNO 去噪骨干。

主要组件：

| 组件 | 位置 | 作用 |
| --- | --- | --- |
| `SinusoidalPosEmb` | `IAFNO.py:38-52` | 把扩散噪声等级编码成正余弦向量；不是物理时间编码。 |
| `RMSNorm` | `IAFNO.py:54-61` | 对通道维做归一化。 |
| `PatchEmbed` | `IAFNO.py:71-91` | 把空间块通过 `Conv3d` 投影成 token embedding。 |
| `Mlp` | `IAFNO.py:95-117` | 每个空间 token 上的通道 MLP。 |
| `Block` | `IAFNO.py:121-157` | `LayerNorm → AFNO → 残差 → MLP → 残差`。 |
| `AFNO` | `IAFNO.py:161-226` | 对三个 token 空间轴做 FFT、分块复数线性变换、稀疏收缩和逆 FFT。 |
| `IAFNODiff` | `IAFNO.py:230-367` | 条件拼接、扩散时间调制、patch 编解码、隐式迭代与输出重建。 |

`IAFNO.py:19` 仍使用 `from utilities3 import *`，但 padding 已改用 `x.new_zeros` 跟随输入的 dtype/device，不再依赖 `utilities3.py` 的全局 `device`。

### 1.4 `diffusion.py`

**作用：** 实现 EDM 风格的扩散训练与采样外壳，具体去噪函数由传入的 `net` 提供。

关键位置：

- EDM 预条件系数：`c_skip`、`c_out`、`c_in`、`c_noise`，`diffusion.py:100-110`；
- 条件去噪调用：`preconditioned_network_forward`，`diffusion.py:115-134`；
- 噪声日程：`sample_schedule`，`diffusion.py:141-151`；
- 三维 Euler/Heun 采样：`sample`，`diffusion.py:153-212`；
- 训练损失：`forward`，`diffusion.py:259-289`。

`sample_using_dpmpp` 位于 `diffusion.py:214-249`，但它创建的是四维 `[B, C, H, W]` 噪声，而且没有向骨干传入外部条件，与当前三维 `IAFNODiff` 接口不兼容；当前训练脚本没有调用它。

### 1.5 `utilities3.py`

**作用：** 通用数据读取、归一化、损失和参数计数工具集合。

| 组件 | 位置 | 当前是否被主流程使用 |
| --- | --- | --- |
| `load_checkpoint` | `utilities3.py:17-30` | 可加载纯模型 state dict，也兼容包含 optimizer/scheduler/scaler state 的字典。 |
| `MatReader` | `utilities3.py:32-83` | 未使用；支持旧版 MAT 和 HDF5 MAT。 |
| `UnitGaussianNormalizer` | `utilities3.py:86-121` | 未使用。 |
| `GaussianNormalizer` | `utilities3.py:124-147` | 未使用。 |
| `RangeNormalizer` | `utilities3.py:150-171` | 未使用。 |
| `LpLoss` | `utilities3.py:174-218` | 测试阶段使用；默认返回逐样本相对 L2 后取均值。 |
| `HsLoss` | `utilities3.py:221-285` | 未使用。 |
| `DenseNet` | `utilities3.py:288-314` | 未使用。 |
| `count_params` | `utilities3.py:318-322` | `trainer.py:182` 使用。 |

### 1.6 `README.md`

**作用：** 说明项目论文、原始应用、数据下载和引用信息。它没有给出安装命令、数据数组 schema、训练命令、checkpoint 恢复或推理示例。

README 明确把 DiAFNO 描述为 IAFNO 与 diffusion 的组合，用于三维湍流的 autoregressive prediction（`README.md:1-8`）；数据链接位于 `README.md:10-12`。

## 2. 当前训练入口

**源码确认：** 训练入口是 `trainer.py` 的模块顶层，通常只能通过以下形式启动：

```powershell
python trainer.py
```

但仓库当前不能直接运行：

- `np.load('your dataset')` 是占位路径，见 `trainer.py:77`；
- 归一化信息目录是占位字符串，见 `trainer.py:97`；
- checkpoint 保存目录是占位字符串，见 `trainer.py:249`；
- 没有 CLI 参数、配置文件解析或主函数；`checkpoint_path` 提供了基础权重加载，但当前 checkpoint 保存格式不含 epoch、optimizer、scheduler 或 scaler 状态，不能视为完整断点续训。

## 3. Dataset 与 DataLoader 在哪里定义

**源码确认：** 没有自定义 `Dataset` 类。虽然 `Dataset` 在 `trainer.py:10` 被导入，但没有实现或实例化。

当前数据管线全部位于 `trainer.py:77-140`：

1. `np.load` 读取完整 NPY 数组；
2. `data[0:trainset_num, ..., 0:3]` 只保留前 `trainset_num=20` 个 case 和最后一维前 3 个变量；
3. 双层循环构造时间窗口；
4. `torch.utils.data.TensorDataset(data_set[:, 0, ...], data_set[:, 1, ...])` 构造输入/目标对；
5. `random_split` 按样本随机划分 80% train、20% test；
6. 使用两个 `torch.utils.data.DataLoader`。

当前没有 validation dataset/loader。随机划分发生在高度重叠的时间窗口之后，因此相邻时刻、同一条轨迹的样本可能同时进入 train 和 test；对时空预测而言存在明显的信息泄漏风险。

## 4. 当前模型输入 tensor shape

需要区分“原始数据”“扩散模型接口”和“IAFNO 骨干输入”。

### 4.1 原始与 DataLoader shape

根据 `trainer.py:77-92` 的索引方式，源码预期原始数组为：

```text
[N_case, N_time, X, Y, Z, C_all]
```

当前切片后：

```text
data: [最多 20, N_time, X, Y, Z, 3]
```

当前默认 `InferenceWidth=1` 时：

```text
data_set: [N_window, 2, X, Y, Z, 3]
xx:       [B, X, Y, Z, 3]
yy:       [B, X, Y, Z, 3]
```

模型配置固定 `X=64, Y=65, Z=32`，所以实际期望为：

```text
xx, yy before rearrange: [B, 64, 65, 32, 3]
```

### 4.2 `ElucidatedDiffusion` 接口 shape

`trainer.py:198-203` 把训练样本转成 channel-first：

```text
xx: [B, 3, 64, 65, 32]  # 条件场，即当前帧
yy: [B, 3, 64, 65, 32]  # 目标场，即下一帧
loss = model(yy, xx)
```

因此 `ElucidatedDiffusion.forward(images, self_cond)` 中：

- `images` 是要加噪并复原的未来目标 `yy`；
- `self_cond` 是外部条件 `xx`。

### 4.3 `IAFNODiff` 实际接收 shape

扩散模块把加噪目标传给 `IAFNODiff.forward(x, time, x_self_cond)`：

```text
x:           [B, 3, 64, 65, 32]  # 加噪后的目标
x_self_cond: [B, 3, 64, 65, 32]  # 当前帧条件
time:        [B]                  # 扩散噪声等级的编码输入
```

`IAFNO.py:311-313` 沿通道维拼接后：

```text
[B, 6, 64, 65, 32]
```

这解释了 `IAFNODiff.__init__` 在 `self_condition=True` 时把传入的 `in_chans=3` 内部翻倍为 6（`IAFNO.py:252`）。这里的 `self_condition` 实际承担“外部前一帧条件”的作用；标准 diffusion self-conditioning 代码在 `diffusion.py:274-280` 已被注释掉。

## 5. 当前模型输出 tensor shape

`IAFNODiff.head` 在 `IAFNO.py:275` 输出每个 patch 的 `out_chans × patch_volume`，`IAFNO.py:349-366` 再还原空间网格并切掉 padding。

当前输出为：

```text
IAFNODiff output:             [B, 3, 64, 65, 32]
ElucidatedDiffusion.sample:   [B, 3, 64, 65, 32]
test rearrange 后的 pred:     [B, 64, 65, 32, 3]
```

`ElucidatedDiffusion.sample` 最终把 `[-1, 1]` 映射回 `[0, 1]`，见 `diffusion.py:211-212`。`trainer.py:233-234` 再使用训练集的 `y_min/y_max` 恢复物理量范围。

> [!warning] shape 检查不完整
> `diffusion.py:260-263` 只显式检查 `H`、`W` 和通道数，没有检查实际 `Z` 是否等于 `image_size_z`。不过错误的 `Z` 后续通常仍会在位置 embedding、patch 重建或条件拼接处失败。

## 6. 时间维度如何组织

### 6.1 物理时间

物理时间最初位于原始数组的第 2 维，即索引 1：

```text
[case, time, x, y, z, variable]
```

`trainer.py:86-90` 先抽取长度为 `InferenceWidth + 1` 的窗口，但 `trainer.py:92` 无条件只取 `data_set[:, 0, ...]` 和 `data_set[:, 1, ...]`。因此：

- 当前 `InferenceWidth=1` 时，得到 `t → t+1`；
- 把 `InferenceWidth` 改大只会让临时窗口变长，模型仍只看到第 0、1 帧；
- `InitialInterval` 只出现在文件名和日志语义中，未参与索引，见 `trainer.py:58,98`；
- 进入网络后没有独立的物理时间轴，也没有对时间做 FFT/attention；当前历史长度实质上是 1。

### 6.2 扩散时间

`IAFNODiff.forward` 的 `time` 是扩散噪声等级 `c_noise(sigma)`，由 `diffusion.py:123-126` 传入。它通过 `SinusoidalPosEmb` 和 scale-shift 调制卷积特征（`IAFNO.py:277-328`）。

它不是：

- 数据的日期/时刻；
- 输入帧序号；
- lead time；
- 预报第几帧。

## 7. IAFNO/FNO 在模型中的作用

仓库中没有名为 `FNO` 的独立类，实际实现是 AFNO，并通过重复共享 block 形成 IAFNO 风格骨干。

### 7.1 AFNO 的空间频域混合

`AFNO.forward` 位于 `IAFNO.py:178-226`：

1. 输入 token shape 为 `[B, Xp, Yp, Zp, embed_dim]`；
2. `torch.fft.rfftn(..., dim=(1, 2, 3))` 只沿三个 patch 后的空间轴做 FFT；
3. embedding 通道被分成 `num_blocks`；
4. 对复数频谱做两层分块线性变换与 ReLU；
5. `softshrink` 施加频谱稀疏化；
6. 逆 FFT 回到空间域并与输入残差相加。

因此 AFNO 的主要作用是低复杂度地建立全局空间耦合，帮助扩散去噪阶段恢复全局一致的三维结构。它不负责 diffusion noise schedule，也不直接组织历史时间。

### 7.2 IAFNO 的“隐式”重复

`IAFNODiff.forward_features` 位于 `IAFNO.py:292-307`。当前配置 `explicit_layer=4, implicit_layer=4` 时，代码反复调用同一组 4 个 `Block`，共 4 轮；同一个 block 的权重跨隐式轮次共享。外层使用系数 `1 / (implicit_layer × explicit_layer)` 做残差更新。

这里还有一个源码风险：`IAFNO.py:298` 使用位运算符 `&` 写条件，而不是逻辑 `and`。当前 `4/4` 配置进入 `else` 分支，结果明确；对其他尤其是奇数 `nlayer` 配置，表达式语义可能偏离注释意图。

### 7.3 空间 patch 与 padding

当前配置：

```text
真实网格 dim_f = [64, 65, 32]
模型网格 dim   = [64, 66, 32]
patch_size      = [2, 2, 2]
```

`IAFNO.py:335-345` 在 y 方向补 1 个零点，使各轴能被 patch size 整除；输出时再删除该点（`IAFNO.py:359-364`）。该实现每个不等轴只会补 1 个点，并不是通用任意长度 padding。

## 8. diffusion 模块在预测流程中的作用

### 8.1 训练

`ElucidatedDiffusion.forward` 的数据流为：

```text
真实未来场 yy（先归一化至 [0, 1]）
    → 映射至 [-1, 1]
    → 从 log-normal 分布采样 sigma
    → 加高斯噪声
    → 以当前场 xx 为条件调用 IAFNODiff
    → EDM 预条件组合得到 denoised
    → sigma 加权逐元素 MSE
```

对应代码：`diffusion.py:253-289`。训练的本质是学习条件分布中的去噪器，而不是直接用普通回归损失拟合 `xx → yy`。

### 8.2 推理

`ElucidatedDiffusion.sample(xx)` 从 `[B, C, H, W, Z]` 的随机高斯噪声开始，按噪声日程逐步降低 `sigma`。每一步都把同一个条件 `xx` 传给 IAFNO；除最后一步外还使用二阶 Heun 修正。对应 `diffusion.py:153-212`。

因此 diffusion 的作用是：

- 表达给定过去场后未来场的条件分布；
- 通过多次去噪生成一个未来样本；
- 保留随机性，可用于生成 ensemble。

它当前不负责：

- 构造历史 7 帧窗口；
- 生成物理 lead-time 编码；
- 把输出回灌形成 autoregressive rollout；
- 处理 PRE 曲线网格的坐标、海陆 mask 或变量。

## 9. `trainer.py` 的训练、验证、测试流程

### 9.1 训练前处理

1. 固定随机种子 123，选择 CUDA/CPU，见 `trainer.py:26-34`；
2. 载入 NPY 并截取前 20 个 case、前 3 个变量、前 200 个时间索引，见 `trainer.py:54,77-88`；
3. 构造相邻帧样本并随机 80/20 划分，见 `trainer.py:86-93`；
4. 仅从 train 输入帧计算每变量 min/max 和整体标准差 `sigma`，见 `trainer.py:108-130`；
5. 构造 IAFNO、EDM、Adam 和 CosineAnnealingLR，见 `trainer.py:142-171`；
6. `checkpoint_path` 非空时加载 checkpoint，见 `trainer.py:173-174`。

### 9.2 训练循环

`trainer.py:186-210`：

- 输入与目标都做 per-variable min-max 归一化；
- 调整成 `[B, C, X, Y, Z]`；
- 调用 `model(yy, xx)`；
- 使用 AMP 与 `GradScaler` 反向传播；
- 记录的是 EDM 的 sigma 加权去噪 MSE。

### 9.3 验证流程

**不存在独立验证流程。** 没有 validation split、validation loader、early stopping 或 best-checkpoint 选择。名为 `test_loader` 的数据在每个 epoch 都被评估，实际上承担了验证集的使用方式。

### 9.4 测试循环

`trainer.py:212-241`：

- 使用 `model.sample(xx)` 完成 32 步扩散采样；
- 计算归一化空间中的 `LpLoss`；
- 反归一化后再次计算 `LpLoss`；
- 每轮把 checkpoint 写入新文件。

注意：变量名 `mse_test` 和 `mse_real` 不准确。这里使用的是 `utilities3.py:202-217` 的相对 L2 loss，不是 MSE。

### 9.5 当前训练流程的源码级限制

以下均由源码直接可见：

- `scheduler` 被创建，但没有任何 `scheduler.step()`，学习率实际不会按 cosine schedule 更新；
- `count` 先取真实时间长度，随后被硬编码覆盖为 200，见 `trainer.py:82-83`；
- min-max 除法没有 epsilon，常量变量会导致除零；
- 归一化信息目录在保存前没有创建；
- `pred` 的 `rearrange` 显式指定 `bs=batch_size`，最后一个不足 batch 的测试批次可能不满足该约束；
- 每个 epoch 都保存完整权重，没有 best/last 策略；
- `random_split` 未传入独立 generator，且只设置了 `torch.manual_seed`，结果在当前进程通常可复现，但数据切分仍不具有按时间或 case 隔离的科学含义；
- 核心 IAFNO 的强制 `.cuda()` 与 padding 全局 device 依赖已经移除；`utilities3.py` 中遗留的 `.cuda()` 是 reader/normalizer 的可选便捷方法，不会在主流程中自动执行；
- `environment.yml` 已无绝对 Linux prefix，但它没有声明 `xarray`，也没有给出安装 `torch==2.4.1+cu124` 所需的 wheel 来源；它与 `requirements-lock.txt` 仍不是完全等价的可复现环境；
- `checkpoint_path` 虽把 optimizer/scheduler/scaler 传给加载器，但当前保存端只写 `model.state_dict()`，因此实际只能恢复模型权重，且不会恢复已完成 epoch。

## 10. 当前代码原本针对的数据与任务

### 10.1 README 确认

项目面向三维湍流的长期预测，覆盖：

- 强迫均匀各向同性湍流（forced HIT）；
- 衰减均匀各向同性湍流（decaying HIT）；
- `Re_tau ≈ 395` 与 `Re_tau ≈ 590` 的湍流槽道流。

见 `README.md:2-12`。

### 10.2 源码确认

当前实现固定使用 `64 × 65 × 32` 三维网格和最后一维前 3 个变量，训练相邻时刻的条件单步生成任务。源码没有变量名；结合三维湍流背景，前 3 个变量很可能是三分量速度，但这一语义仍需通过数据说明或真实数组确认，不能仅凭 `[..., 0:3]` 断言。

## 11. 当前代码是否已有 autoregressive prediction

**结论：当前提交的训练/测试代码没有实现完整 autoregressive prediction。**

证据：

- Dataset 只构造相邻帧 `t → t+1`，见 `trainer.py:86-93`；
- 测试只调用一次 `model.sample(xx)`，见 `trainer.py:229`；
- 预测 `pred` 没有追加到历史窗口，也没有再次作为下一步条件；
- `diffusion.sample` 的多次循环是在同一个输出场上的扩散去噪步骤，不是多个物理时间步。

README 的论文级描述确实宣称 autoregressive framework，但该仓库版本没有把 rollout 循环放进 `trainer.py` 或其他文件。当前模型可以作为 autoregressive rollout 的单步算子，但“可以被外部重复调用”不等于“当前代码已经实现”。

## 12. PRE_ocean_data 任务定义

### 12.1 固定范围

第一版只完成以下任务：

```text
数据集：PRE_ocean_data
时间分辨率：日平均，frame_stride = 1 天
输入：连续 7 天的 u_eastward、v_northward
输出：随后 15 天的 u_eastward、v_northward
预测方式：单步条件扩散模型，自回归滚动 15 次
空间范围：PRE 固定区域网格
```

选择 `u_eastward`、`v_northward`，而不是 ROMS 原始交错网格上的 `u`、`v`。前两者已经旋转到东西/南北方向并插值到相同的 rho 网格，shape 均为 `[10591, 30, 400, 441]`，便于作为两个对齐通道建模。

当前推荐的最小可行任务是**表层流速预测**：使用垂向索引 29（资料说明索引 0 为底层、29 为表层）。这样每帧是 `[2, 400, 441]`。如果导师要求预测全部 30 层，这将成为三维海流预测任务，输入规模、输出规模、mask 和模型空间轴都需重新评估，不能作为同一配置静默切换。

> [!warning] 实施前唯一需要确认的任务边界
> “预测 `u, v`”是否明确指表层 `u_eastward/v_northward`。本文后续 shape、显存与实施路线均以表层两通道为默认；全 30 层不在当前最小版本范围内。

### 12.2 选定 autoregressive 路线

每个训练样本使用过去 7 天预测下一天：

```text
history: [B, 7, 2, H, W]
condition after flatten: [B, 14, H, W]
target: [B, 2, H, W]
```

推理时重复以下操作 15 次：

1. 用当前 7 帧历史预测下一帧；
2. 移除最旧帧；
3. 把预测帧追加到历史末尾；
4. 保存该 lead time 的结果。

最终预测 shape 为 `[B, 15, 2, H, W]`。这条路线保留了原论文/README 的自回归思想，单次生成目标仍是两通道，比一次性生成 `15 × 2 = 30` 个目标通道更接近现有代码。代价是会累积 rollout 误差，因此指标必须分别报告第 1～15 天，而不能只给一个总体平均值。

第一版训练先采用单步 teacher forcing；只有在单步和 rollout 基线可运行后，再判断是否需要多步 loss、scheduled sampling 或 curriculum。当前阶段不预先加入这些复杂机制。

> [!important] 为什么不能只改一个参数
> `IAFNODiff` 当前假设加噪目标和条件各有 `in_chans` 个通道，再拼接为 `2 × in_chans`。PRE 任务的条件是 14 通道，目标只有 2 通道，因此必须把 `condition_channels`、`target_channels` 和 `out_channels` 分开。

### 12.3 数据窗口和时间切分

PRE 时间范围为 1994-01-01T12 至 2022-12-30T12，共 10591 个连续日平均时刻。数据窗口必须在时间切分之后分别生成，不能先生成高度重叠的窗口再 `random_split`，否则相邻日期会跨集合泄漏。

建议首轮实验使用连续年份切分：

| 集合 | 建议时间范围 | 用途 |
| --- | --- | --- |
| train | 1994-01-01 至 2016-12-31 | 训练模型、计算归一化统计量 |
| validation | 2017-01-01 至 2019-12-31 | 选 checkpoint 和超参数 |
| test | 2020-01-01 至 2022-12-30 | 最终 15 天 rollout 评估 |

这是实施建议，不是数据集官方划分。若导师已有规定，应替换边界，但仍必须保持按连续时间切分。

训练一个单步样本至少需要连续 8 帧；一次完整 15 天评估样本需要连续 22 帧。窗口只能在各自 split 内部构造，不得跨越边界。PRE 文档记录时间连续、无缺失日，但数据加载时仍应断言时间差为 1 天，防止文件遗漏或排序错误。

### 12.4 数据 shape 与加载约定

以表层任务为准，建议 Dataset 的外部接口保持清晰的时间轴：

```text
stored variable: [T, depth=30, H=400, W=441]
surface pair:    [T, variable=2, H=400, W=441]
history:         [7, 2, 400, 441]
next target:     [2, 400, 441]
rollout target:  [15, 2, 400, 441]
mask_rho:        [400, 441]
```

只有进入模型前才把 `[7, 2]` 展平为 14 个条件通道，评估和可视化时保留时间、变量轴。这样可以避免把“第几天”和“哪个速度分量”混在难以追踪的通道索引中。

处理后数据约 1.5 TB，不能整体载入内存。Dataset 应按索引读取或使用内存映射/分块存储，只取当前窗口、表层索引和两个目标变量。第一版不需要建设多数据集抽象层，只需一个专用 PRE Dataset。

### 12.5 海陆 mask、NaN 与归一化

`u_eastward`、`v_northward` 的陆地区域为 NaN，约占网格的 29.96%；`mask_rho` 中 1 表示海洋、0 表示陆地。不能直接把含 NaN 的数组送入 loss 或归一化器。

建议处理顺序：

1. 读取 `mask_rho`，并检查它与速度变量后两维完全一致；
2. 只使用 train 时间段、仅在海洋点上分别计算 `u_eastward` 和 `v_northward` 的统计量；
3. 第一版沿用扩散代码的数值假设，将两个变量分别归一化到 `[0, 1]`；
4. 归一化后把陆地点填为 0，但同时保留 mask，不能把填充值当成真实海流；
5. 训练 loss、验证指标和测试指标只在海洋点上计算；
6. 保存统计量并在 validation/test/inference 中复用，禁止用测试集重新拟合。

现有 `diffusion.py` 的 loss 不接收 mask，这属于必改接口。仅用 `nan_to_num` 消除 NaN 而不做 masked loss，会让大片陆地零值主导优化结果。

### 12.6 PRE 网格对 IAFNO 的影响

PRE rho 网格为 `400 × 441` 的区域曲线正交网格，约覆盖 112.315°E～115.678°E、20.896°N～23.028°N；两个方向网格尺度约 0.76 km 和 0.41 km。它不是普通等距笛卡尔周期网格。

如果先做表层任务，最小代码迁移可以暂时保留现有 `Conv3d/FFT3d` 骨干并增加 singleton 轴：

```text
[B, C, H, W] → [B, C, H, W, 1]
```

此时 z 方向 `patch_size` 必须设为 1。是否改写为真正的 `Conv2d/FFT2d`，应由首次显存、速度和正确性测试决定；当前不同时维护 2D、3D 两套骨干。

还需要注意三点：

- `400 × 441` 远大于原型的 `64 × 65 × 32`，不能假定原 batch size、embedding 维度和 patch 配置可直接运行；
- 441 可能不能被候选 patch size 整除，padding/cropping 必须显式记录并在输出时恢复原网格；
- AFNO 对空间轴做 FFT，但 PRE 两个区域边界都不是天然周期边界，海岸线也不连续。边界伪影和跨陆地频域混合需要通过可视化、mask 指标和基线比较验证。

`lon_rho`、`lat_rho`、`h` 和 `mask_rho` 都是可用静态场。第一版只把 mask 作为 loss/metric 必需信息；经纬度和水深是否加入条件通道，应在基础模型跑通后通过消融实验决定。

## 13. 预计代码修改范围（本轮不实施）

| 文件 | 是否必须改 | PRE 专用修改 |
| --- | --- | --- |
| `trainer.py` | 必须 | 替换占位 NPY 读取；先按连续时间切分，再构造 7 天条件/单步目标；加入 validation；实现 15 步 rollout；按 lead day 保存指标；补 `scheduler.step()` 和可靠的 checkpoint 选择。 |
| `IAFNO.py` | 必须 | 分离 14 个条件通道与 2 个目标/输出通道；适配 `400 × 441 × 1` 及 z 向 patch size 1；根据显存测试缩放网络。 |
| `diffusion.py` | 必须 | 采样初始噪声和输出 shape 绑定 2 个目标通道；适配新的条件接口；让训练 loss 接收并正确广播 ocean mask。 |
| `utilities3.py` | 需要 | 提供只基于 train 海洋点的双变量归一化和 masked RMSE/MAE；修正零范数安全性。 |
| `pre_dataset.py`（新） | 推荐 | 专门封装 PRE 的按需读取、表层选择、7/15 窗口和时间连续性检查。保持单一实现，不建立通用数据集继承体系。 |
| `README.md` | 实现后必须 | 只记录 PRE 任务定义、目录/变量约定、时间切分、训练/推理命令、checkpoint 和指标复现方式。 |

建议核心配置明确命名为：

```text
context_frames = 7
forecast_frames = 15
frame_stride = 1
input_variables = [u_eastward, v_northward]
target_variables = [u_eastward, v_northward]
depth_index = 29
condition_channels = 14
target_channels = 2
spatial_shape = [400, 441, 1]
ocean_mask = mask_rho
```

`InferenceWidth` 和 `InitialInterval` 当前语义与实现不一致。实现时应替换为上述明确配置，不能继续让一个参数同时暗示窗口宽度、通道数和预测跨度。

## 14. PRE 专用评估方案

最低限度应包含一个不训练的 persistence baseline：把输入第 7 天的流场重复 15 次。模型若不能稳定优于该基线，不应继续增加架构复杂度。

所有指标都只在 `mask_rho == 1` 的位置计算，并逐 lead day 报告：

- `u_eastward` 的 RMSE、MAE；
- `v_northward` 的 RMSE、MAE；
- 二维速度矢量误差 `sqrt((u_pred-u_true)^2 + (v_pred-v_true)^2)`；
- 可选的流速大小误差；
- 第 1、3、5、7、10、15 天的真值/预测/误差空间图。

扩散采样具有随机性。首次联调可固定随机种子保证可重复；正式结果至少记录采样步数、随机种子和 checkpoint。多成员 ensemble 属于后续实验，不是第一版跑通条件。

## 15. 开始改代码前的只读验收清单

服务器可用后，按以下顺序检查真实 PRE 文件：

1. 确认 `u_eastward`、`v_northward`、`mask_rho` 的真实文件名、路径、dtype 和 shape；
2. 抽查索引 29 确为海表层，并向导师确认任务是否只预测表层；
3. 验证时间轴有 10591 个日平均时刻、严格递增且无断日；
4. 验证两个速度变量都在 rho 网格上，NaN 分布与 `mask_rho == 0` 一致；
5. 统计 train 时段海洋点的范围、均值、标准差和异常值；
6. 用少量连续窗口验证 Dataset 输出的 history/target 日期与 shape；
7. 先跑 persistence baseline，再进行最小 batch 的前向、loss、反向和 15 步 rollout；
8. 根据显存实测决定全网格训练、空间 patch 训练或模型缩小方案。

> [!done] 当前结论
> 任务已经收敛为 PRE_ocean_data 的日平均表层 `u_eastward/v_northward` 预测，推荐实现路线为“7 天条件单步模型 + 15 步自回归 rollout”。本次只修订分析文档，没有修改任何 Python 代码，也没有宣称模型已在真实数据上运行。
