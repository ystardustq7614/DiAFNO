# DiAFNO

> 面向三维湍流自回归预测的傅里叶神经算子与扩散模型融合方法

本仓库是论文 [“Integrating Fourier Neural Operator with Diffusion Model for
Autoregressive Predictions of Three-dimensional Turbulence”](https://arxiv.org/abs/2512.12628)
的配套代码。论文作者为 Yuchi Jiang、Yunpeng Wang、Huiyu Yang 和 Jianchun Wang。

## 摘要

三维湍流的高精度自回归预测一直是机器学习方法面临的难题。扩散模型已在二维湍流
预测中展现出较高精度，但在三维湍流中的应用仍较有限。为实现可靠的三维湍流
自回归预测，本文提出 DiAFNO：将隐式自适应傅里叶神经算子（IAFNO）与扩散模型结合。
IAFNO 能有效捕获全局频率和结构特征，这对扩散模型去噪过程中的全局一致重建十分
关键。在此基础上，DiAFNO 利用扩散模型的条件生成能力构建自回归框架，以获得长期
稳定的三维湍流预测。

模型使用固定超参数，分别在多类三维湍流数据上独立训练和测试，包括 Taylor 雷诺数
$Re_{\lambda}\approx100$ 的受迫均匀各向同性湍流（HIT）、初始 Taylor 雷诺数
$Re_{\lambda}\approx100$ 的衰减 HIT，以及摩擦雷诺数 $Re_{\tau}\approx395$ 和
$Re_{\tau}\approx590$ 的湍流槽道流。后验测试表明，与阐明扩散模型（EDM）和采用
动态 Smagorinsky 模型（DSM）的传统大涡模拟（LES）相比，DiAFNO 在多数统计量上
具有更高的预测精度，包括速度谱、速度与涡量的均方根值以及 Reynolds 应力。虽然
DiAFNO 并非在每项统计量上都最优，但其整体表现明显优于 EDM 和 DSM。推理耗时对比
还表明，在不计训练成本时，训练完成的 DiAFNO 比 EDM 和采用 DSM 的 LES 具有更高的
推理效率。

## 数据集

原论文数据集可从
[IAFNO_fDNS_kaggle](https://www.kaggle.com/datasets/yuchirichardjiang/coarsened-fdns-data-iafno)
下载。

## 引用

arXiv 版本：

```bibtex
@misc{jiang2026integratingfourierneuraloperator,
      title={Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence},
      author={Yuchi Jiang and Yunpeng Wang and Huiyu Yang and Jianchun Wang},
      year={2026},
      eprint={2512.12628},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      url={https://arxiv.org/abs/2512.12628},
}
```

该论文已被《Acta Mechanica Sinica》接收，引用信息为：Acta Mech. Sin. 43,
360674 (2027)，DOI：10.1007/s10409-026-60674-x。期刊最终版本可检索后，请优先使用
最终版本的信息进行引用。

---

## PRE 海洋流场预测任务（本仓库分支 `adapt-weather-ocean`）

任务目标：输入连续 7 天的原始三维海流 `u/v` 场，预测下一天，并进一步自回归滚动
预测 15 天。完整运行步骤见 [PRE 运行手册](docs/operations/PRE_runbook.md)，全部文档
与实验入口见 [文档索引](docs/README.md)。

本文档与代码状态同步至 2026-09-05。preset 仍采用模块级配置；训练模式、preset、
恢复 checkpoint 等也可通过环境变量选择。

当前实验结论：

- surface 确定性 persistence-residual 主线已完成，当前正式最优为 MS10 `Ep2.pth`，
  test 15-day overall ratio 为 `0.838`；
- bottom MS5 `Ep5.pth` 通过全部预注册门槛，test overall ratio 为 `0.813`；
- middle 正式选型为 MS5 `Ep4.pth`，test overall ratio 为 `0.851`，test 门槛全部
  通过；validation gate 5 仅在 d15 边缘未过，已裁定接受并如实保留；
- full3d 已选择 Path B，冻结等待独立正式预算；K3 继续受逐层 day-1 信号门槛阻塞；
- 六个后续模型分支当前均不满足准入条件，因此没有待执行实验。详细证据见
  [当前困难与下一步](docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md)和
  [实验索引](docs/experiments/README.md)。

### 网格、掩膜与插值（不旋转）

| 网格 | u | v | 掩膜 |
|---|---|---|---|
| 原生 ROMS C-grid | `(T, 30, 400, 440)` | `(T, 30, 399, 441)` | `mask_u` `(400,440)`、`mask_v` `(399,441)` |
| 模型 rho 网格 | `u_rho.npy` `(10591, 30, 400, 441)` | `v_rho.npy`，shape 相同 | `mask_u_rho` / `mask_v_rho` `(400,441)` |

- 共定位到 rho 网格时保留原始网格 xi/eta 分量语义，**不旋转**到 east/north。
  `u_rho[r,c]` 是相邻 u 面 `u[r,c-1]` 与 `u[r,c]` 的 NaN-aware 均值，边界采用
  单侧值；`v_rho` 沿 eta 方向作同类处理；陆地区域保持 NaN。
- 双变量有效掩膜 `mask_u_rho` / `mask_v_rho` 使用与插值相同的 stencil 从
  `mask_u` / `mask_v` 推导：只要相邻面中至少一个有效，对应 rho 点即有效。
  `mask_uv.npy` 只是交集兼容文件。训练统计、masked diffusion loss 和 validation
  均使用**双变量掩膜**。预处理以给定掩膜为准：若任意日期/层的 `mask==1` 单元出现
  NaN（动态缺测），立即报错；若 `mask==0` 单元存在数值，则丢弃为 NaN 并计数
  （本数据集中有 45 个此类静态陆地边界 u 面）。
- 正式评估按固定规则把 rho 预测映射回原生网格：rho u 沿 xi 方向对相邻 rho 点
  求均值得到 `(400,440)`；rho v 沿 eta 方向处理得到 `(399,441)`。

### 归一化与裁剪

- 分变量使用各自**训练集海洋点**做 min-max 归一化到 `[0,1]`：u 使用
  `mask_u_rho`，v 使用 `mask_v_rho`。归一化后陆地填 0，loss 和指标始终应用掩膜。
- 默认**禁用**百分位裁剪（`clip_pct = None`），必须显式配置才会启用。stats cache
  记录裁剪策略、深度 preset、split 边界和 mask hash；cache 过期、split 改变或缺少
  `splits` 字段时都会重新计算。cache 中的标准差是 `[0,1]` 归一化后 u+v 拼接数据的
  pooled std（surface preset 为 0.08560，包含 u/v 均值差异项）。计算 pooling 前，
  数值按照与数据集归一化完全相同的方式裁剪到各变量范围。由于 `diffusion.py` 使用
  `images*2-1` 再归一化，EDM 应取 **`sigma_data = 2.0 * stats["sigma"]`**
  （0.17120）。共享换算位于 `pre_config.py` 的 `SIGMA_DATA_SCALE`、
  `sigma_data_from_stats` 和 `sigma_data_from_checkpoint`；训练与评估必须调用同一实现。
  新 checkpoint 保存 `config.{stats_sigma,sigma_data_scale,sigma_data}`；评估优先使用
  checkpoint 值，旧 checkpoint 缺少该值时会明确告警并退回 legacy stats-only scale。
- 正式指标使用**未裁剪的原生真值**：`NativeUVReader` 直接读取原始 `u.npy` / `v.npy`，
  不允许用归一化 target 反归一化后代替原始真值。

### 时间与数据切分

- 已通过权威 `ocean_time` 元数据验证连续日时间戳：共 10,591 个时间点，严格递增且
  间隔正好 24 小时。检查精度为 `datetime64[s]`，因此 23/25 小时间隔也会失败。
  时间缓存为 `aligned/ocean_time_seconds.npy`（精确 `datetime64[s]`）和
  `aligned/ocean_time.npy`（兼容用日期视图 `datetime64[D]`）。
- 连续切分为 train `[0,8401)`、val `[8401,9496)`、test `[9496,10591)`；滑动窗口
  不得跨越 split 边界。stats cache 会记录这些边界，并在边界变化时重新计算。

### 运行命令（仓库根目录，`diafno` 环境）

```bash
GPU_ID=3  # 先用 nvidia-smi 检查并替换
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/preprocess_align_uv.py  # 一次性 CUDA 共定位
CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_trainer.py   # 默认：安全的真实数据 smoke
DIAFNO_TRAIN_MODE=full CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_trainer.py
# 多 GPU 正式训练：每张 GPU 一个进程，batch_size 按单卡计算
DIAFNO_TRAIN_MODE=full CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc_per_node=4 pre_trainer.py
CUDA_VISIBLE_DEVICES="$GPU_ID" python pre_evaluate.py  # 15 步 rollout + persistence + figures
python smoke_test.py && python pre_smoke_test.py        # 最小回归测试
```

`scripts/preprocess_align_uv.py` 依赖 CUDA，在 `CUDA_VISIBLE_DEVICES` 过滤后使用逻辑
设备 `cuda:0`，并以覆盖模式打开 `u_rho.npy` / `v_rho.npy`。正式重跑前，先用
`scripts/profile_preprocess_align_uv.py` 在私有 scratch 目录对代表性分块做 benchmark；
准确命令见 [PRE 运行手册](docs/operations/PRE_runbook.md)。

- preset 定义在 `pre_config.py`，训练与评估必须选择一致的 preset。训练 checkpoint
  写入 `<checkpoint_dir>/PRE/<run_tag>/{Ep{n}.pth,best.pth,loss.dat}`；
  `run_tag_for()` 会在 legacy tag 后添加 `_SD2`，避免 fixed-scale 与 sd1 run 共用目录。
  全层训练使用 `DIAFNO_PRESET=full3d`。smoke 与 DDP 输出分别添加 `_SMOKE` 和
  `_DDP<n>`，避免与单卡正式 run 冲突。
- 评估输出与 checkpoint 放在同一目录，并按采样配置和 checkpoint stem 命名：
  `eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_ckpt{stem}[_tag].npz` 与
  `figures_<tag>/`。若目标产物已存在，程序会拒绝覆盖。NPZ 保存原生网格上的
  `rmse_model/mae_model/rmse_persistence/mae_persistence/rmse_zero/rmse_oracle/mae_*`，
  shape 为 `(ROLLOUT_DAYS, 2, Z)`；`surface_smoke` 的 `Z=1`，`full3d` 的 `Z=30`。
  同时保存 rollout_days、ensemble_size、S_churn、seed（按 window 使用
  `EVAL_SEED + start_day`，与 batch size 无关）、batch_size、sigma_data、checkpoint、
  epoch、preset、sampling_steps、stride、window start、norm stats 和网格映射规则等
  复现元数据。figure 命名为 `d{1,3,5,7,10,15}_s{layer}_{u|v}.png`，内容为
  truth/prediction/error。只保存原生网格正式指标，不保存 rho 网格补充数组。
- overall RMSE 定义为 `sqrt(sum(squared_error)/sum(valid_count))`，不得对逐层 RMSE
  直接做算术平均。终端摘要也通过 `pre_metrics.pooled_rmse` 按相同规则对每个 lead day
  和变量进行 pooling。
- `pre_metrics.py` 提供训练、评估和 smoke test 共用的指标实现：`rho_to_native`、
  `masked_error_sums`、`pooled_rmse`、`masked_rel_l2`、`oracle_native_error_sums`；
  测试中不得重新实现这些公式。
- `NativeUVReader.get()` 对所有 preset 均返回 sigma 轴在最后的统一布局：
  u `(days,H,W-1,Z)`、v `(days,H-1,W,Z)`；surface 的 `Z=1`，full3d 的 `Z=30`。
  评估阶段不再转置该布局。
- validation diffusion sampling 在固定 `VAL_SEED` 的 `torch.random.fork_rng()` 中运行，
  从而隔离 validation RNG，并在结束后恢复训练 RNG 状态。
- 复现种子：训练为 123，validation sampling 为 1234；评估按 window 的 start day
  使用 `EVAL_SEED + start_day`，并把 rollout_days、ensemble_size、S_churn、seed、
  sampling_steps、checkpoint 等采样配置写入评估元数据。
