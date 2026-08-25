# PRE_ocean_data 7→15 天海流预测运行手册

> 任务：用连续 7 天的日平均三维 u/v（方案 A：原始交错网格自对齐到 rho 网格，保留网格方向分量），
> 通过 DiAFNO 单步条件扩散模型预测第 8 天，再自回归滚动 15 次得到未来 1~15 天；
> 按预测天数 × u/v × 垂向层统计 masked RMSE/MAE，并与 persistence baseline 比较。
>
> 已确定决策（2026-08-25）：原始 u/v 用**方案 A** 自对齐；先**表层冒烟**（层 29），跑通后再上**全 30 层**。

## 0. 文件清单

| 文件 | 作用 |
|---|---|
| `scripts/preprocess_align_uv.py` | 方案 A 预处理：原始 u/v → rho 网格对齐，输出 `u_rho.npy`/`v_rho.npy`/`mask_uv.npy` |
| `pre_config.py` | 两套预设 `surface_smoke` / `full3d`（patch、embed_dim、batch 等），无副作用，可安全 import |
| `pre_dataset.py` | Dataset（滑窗/连续时间切分/逐变量 clip 归一化/陆地填 0）、归一化统计量计算与缓存 |
| `pre_trainer.py` | 单步 teacher-forcing 训练入口（masked loss、AMP、cosine scheduler、best/last checkpoint） |
| `pre_evaluate.py` | 15 步自回归 rollout + persistence baseline + masked RMSE/MAE 统计 |

数据产物（均在仓库外）：
- 对齐数据：`~/data_processed/PRE/aligned/{u_rho,v_rho}.npy`（各 209GB，float32，陆地 NaN）、`mask_uv.npy`
- 归一化缓存：`~/data_processed/PRE/norm/stats_d{29|all}_clip0.1.npz`（删除即重算）
- checkpoint：`~/checkpoints/PRE/<run_tag>/{Ep{n}.pth,best.pth,loss.dat}`、`eval_test.npz`

关键约定：
- **时间切分**（连续、不重叠）：train [0,8401) / val [8401,9496) / test [9496,10591)
- **条件通道顺序**：day-major 交错 —— ch0=u(d0), ch1=v(d0), ch2=u(d1), …, ch13=v(d6)；rollout 时去掉最旧 2 通道、追加新帧 2 通道
- **垂向索引**：层 0=海底，层 29=海面；sigma 层为地形追随坐标，mask 是二维的（30 层 NaN 位置一致，已实测验证）
- **归一化**：仅用 train 段海洋点；每变量先 clip 到 [p0.1, p99.9] 再 min-max 到 [0,1]；陆地在归一化后填 0；loss/指标全程只用海洋点（masked）
- mask 用 `mask_uv.npy`（= mask_rho ∧ 对齐后 u/v 均有数据），不要用 `mask_rho` 直接评估

## 1. 步骤 0：预处理（一次性，约 15~60 分钟）

```bash
cd ~/projects/DiAFNO
nohup python scripts/preprocess_align_uv.py > ~/data_processed/PRE/aligned/preprocess.log 2>&1 &
tail -f ~/data_processed/PRE/aligned/preprocess.log
```

完成后检查日志：
- `raw u/v NaN == (mask_u/mask_v==0)` 应为 True；
- `ocean-but-no-data pts` 应很少（个位数~几十）；若很大需排查；
- 末尾统计给出对齐后 u/v 的 min/max/mean/std 与 p0.1/p99.9 —— 关注 raw u 的 ~7 m/s 奇异点在平均后的幅度（clip 归一化已能吸收）。

## 2. 步骤 1：表层冒烟训练（GPU 5/6 空闲）

```bash
cd ~/projects/DiAFNO
CUDA_VISIBLE_DEVICES=5 python pre_trainer.py    # PRESET='surface_smoke'
```

- 首次运行会先计算归一化统计量（表层只读层 29，约 1 分钟），缓存到 `~/data_processed/PRE/norm/`。
- 预设：400×441×1、patch (4,3,1)（整除，无 padding）、14.7k token、B=4、10 epoch。
- 检查点：`train_loss` 应逐 epoch 下降；`val_masked_relL2`（物理单位、海洋点、扩散采样）应明显低于 1 且下降。
- 快速试跑可把 `pre_config.py` 中 `surface_smoke` 的 `max_train_windows` 改为 2000、`num_epochs` 改为 2。

## 3. 步骤 2：冒烟评估（rollout + persistence）

```bash
CUDA_VISIBLE_DEVICES=5 python pre_evaluate.py   # PRESET 与训练一致
```

- 默认 test 段、每 7 天一个起点（~154 个窗口）、每窗口 15 步 × 32 采样步 × Heun 2 次前向。
- 想先快速验证：`MAX_WINDOWS = 8`。
- 输出 `~/checkpoints/PRE/<run_tag>/eval_test.npz`（rmse/mae × model/pers，shape (15,2,Z)）并打印逐天汇总。
- **通过标准**：模型在各 lead day 的 masked RMSE 稳定低于 persistence（打印的 model/pers 比值 < 1）。

## 4. 步骤 3：全 30 层全量

1. `pre_trainer.py` 与 `pre_evaluate.py` 中 `PRESET = "full3d"`；
2. 首次运行会重算全 30 层归一化统计（3 遍流式扫描 ~530GB，约 15~30 分钟，一次性）；
3. 预设：patch (4,3,2) → 100×147×15 = 220.5k token、embed_dim 128、implicit 2、B=1。
   **若 OOM，按此顺序调**：`embed_dim` 128→96 → `implicit_layer` 2→1 → `batch_size` 已为 1；不要轻易改 patch（441 只能被 1/3/7/9 整除）；
4. 训练窗口逐样本读取 ~340MB（两个变量 8 天）；数据总量 418GB < 内存 943GB，首轮后主要靠页缓存；
5. 评估时 `BATCH_SIZE` 建议 1~2；`EVAL_STRIDE` 可加大到 14 先出粗结果。

## 5. 注意事项

- `pre_trainer.py` / `pre_evaluate.py` 都是**脚本**（模块顶层执行），不要 import 它们；共享配置一律从 `pre_config.py` 取。
- 断点续训：设置 `pre_trainer.py` 的 `checkpoint_path`；保存含 model/optimizer/scheduler/scaler/epoch，可真正续训。
- 扩散采样有随机性：脚本已固定 `torch.manual_seed(123)`；正式报告请记录 seed、sampling_steps 与 checkpoint。
- persistence baseline 已内建于 `pre_evaluate.py`，无需单独运行。
- 原始仓库的 `trainer.py` 保持未动；对 `IAFNO.py`/`diffusion.py` 的改动向后兼容（`cond_chans=None` 时行为同旧版 doubling；loss 不传 mask 时与原逻辑一致）。
