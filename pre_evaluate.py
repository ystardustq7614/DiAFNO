#!/usr/bin/env python3
"""模块职责：PRE_ocean_data 正式评估：自回归 rollout 对比持续性基线。

正式指标在"原生 staggered u/v 网格"上、对未 clip 的原始物理真值（raw
u.npy/v.npy）计算，并使用原生 mask_u/mask_v：

    rho u -> 原生 u：沿 xi 对相邻两个 rho 点求均值：(400, 440)
    rho v -> 原生 v：沿 eta 对相邻两个 rho 点求均值：(399, 441)
    （Plan A 共定位 stencil 的逆变换，无单侧补边、不旋转）

每个测试窗口（CONTEXT + ROLLOUT_DAYS 个连续日）的流水线：
    1. （可选）rho 网格上的 ensemble rollout：从当前 7 天条件预测次日，滑动窗口
       （丢最旧一天、追加预测）× ROLLOUT_DAYS；每个 ensemble member 独立跑一条
       轨迹，最后对 member 求均值；
    2. 把每个 rho 预测映射回原生 u/v 网格（rho_to_native）；
    3. 与第 8..8+ROLLOUT_DAYS-1 天的原始原生真值比较（未 clip，陆地=NaN），经
       单个 NativeUVReader 读取；
    4. 持续性基线 = 把第 7 天的"原生"物理 u/v 重复 ROLLOUT_DAYS 次（绝不是
       clip/归一化的条件输入）；
    5. 诊断基线：零流（全零原生预测）与 rho-oracle（数据集真实 rho 目标，
       反归一化并经同一 rho_to_native stencil 映射——单独度量转换的不可逆误差）；
    6. 逐 lead 天（1..ROLLOUT_DAYS）× 变量（u,v）× sigma 层的掩膜 RMSE/MAE。

总体 RMSE = sqrt(sum(squared_error) / sum(valid_count))——绝不是逐层 RMSE 的
算术平均；控制台汇总同样按池化口径（pooled_rmse）。

采样完全可配置（ROLLOUT_DAYS、ENSEMBLE_SIZE、SAMPLER_S_CHURN、
SAMPLER_SIGMA_MAX、EVAL_SEED），并在 CUDA AMP（autocast）下运行，与历史评估
路径完全一致。每个窗口用自己的起始日播种（EVAL_SEED + start），轨迹可复现且
与 batch 大小/装载分组无关。每个输出文件/目录都带采样配置 + 检查点文件名的
标签，已存在的输出一律拒绝、绝不覆盖。sigma_data 优先读检查点 config；legacy
检查点回退到旧 stats-only 尺度并显式告警。

输出：<ckpt_dir>/eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_rf{0|1}[_msk1]_ckpt{stem}[_sm{sigma_max}][_{tag}].npz
      <ckpt_dir>/figures_h{rd}_ch{churn}_e{es}_s{seed}_rf{0|1}[_msk1]_ckpt{stem}[_sm{sigma_max}][_{tag}]/d{...}_*.png

从仓库根目录运行：python pre_evaluate.py

不负责：不可作为库导入——本文件是 module-top-level 脚本，import 即加载检查点、
执行 rollout 并写盘（NPZ/图件/输出目录拒绝覆盖）；共享配置与模型重建规则在
pre_config.py。

检查点的 `config.objective` 决定重建哪个模型：
    "diffusion"            -> 条件 EDM + Heun 采样器（对早于 objective 字段的
                              检查点是 legacy 默认）
    "persistence_residual" -> 确定性 PersistenceResidualIAFNO；采样器参数
                              "不适用"（not applicable）并按此写入输出元数据；
                              ENSEMBLE_SIZE 被强制为 1（member 全同）。
检查点的 `config.static_mask_input`/`model_cond_chans`（实验 08 B 臂）决定
backbone 的条件通道数；静态掩膜检查点会把双变量 rho 掩膜的两个通道一次性
构建，并经 `ensemble_rollout(static_cond=...)` 转发给每一步 rollout——动态
14 通道窗口及其滑动保持纯净。不可能或矛盾的元数据直接拒绝
（pre_config.static_mask_from_checkpoint），该设置写入输出标签（msk1）与
NPZ 元数据。`REMASK_FEEDBACK` 可选地在每个预测重新进入下一条件窗口之前重施
海洋掩膜（陆地 -> 0）；该设置是输出标签（rf0/rf1）与元数据的一部分，最终
取值由 Phase-5 A/B 决定。终端进度遵循共享的 pre_config.ProgressReporter 约定
（交互进度条 + 可解析 PROGRESS 状态行）。
"""
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_models import PersistenceResidualIAFNO
from pre_config import (PRESETS, OUT_ROOT, CONTEXT, run_tag_for, sigma_data_from_stats,
                        sigma_data_from_checkpoint, RESIDUAL_TIME_SIGMA,
                        objective_from_checkpoint, ProgressReporter, format_progress,
                        install_progress_failure_hook, mark_progress_failed,
                        check_norm_fingerprint, static_mask_from_checkpoint)
from pre_dataset import (PREUVDataset, NativeUVReader, native_masks,
                         compute_or_load_stats, build_mask_tensor, mask_version)
from pre_metrics import (rho_to_native, masked_error_sums, pooled_rmse,
                         oracle_native_error_sums)
from pre_rollout import ensemble_rollout, ensemble_mean

torch.manual_seed(123)

# 逃出受保护 rollout 块的异常统一发标准 status=failed 行；stage 字段跟踪当前
# 脚本段，使失败报告能指出发生位置（setup -> data_model -> rollout ->
# postprocess 四段）
EVAL_STAGE = ["setup"]
install_progress_failure_hook("eval", stage=lambda: EVAL_STAGE[0])

# 评估配置

PRESET = "surface_smoke"            # 必须与被评估检查点的预设一致
CHECKPOINT = None                   # None -> <run_dir>/best.pth（run_dir 由 run_tag_for 得到）
SPLIT = "test"
ROLLOUT_DAYS = 15                   # 数据集视界、rollout 步数、指标视界、出图天数四者共用
ENSEMBLE_SIZE = 1                   # 末尾求均值的独立 rollout member 数
SAMPLER_S_CHURN = 0                 # 由表层 SD2 验证消融选定
SAMPLER_SIGMA_MAX = None            # None -> ElucidatedDiffusion 默认（80）
EVAL_SEED = 123                     # 逐窗口 rollout 种子（EVAL_SEED + 绝对起始日）
OUTPUT_TAG = None                   # 追加到输出目录/文件名的额外后缀
EVAL_STRIDE = 7                     # 每 N 天启动一个 rollout 窗口
MAX_WINDOWS = None                  # 快速检查时设小（如 8）
BATCH_SIZE = 4                      # rollout 的 batch；full3d 若 OOM 用 1
SAMPLING_STEPS = None               # None -> 取预设值
FIG_DAYS = (1, 3, 5, 7, 10, 15)     # 代表性 lead 天（被 ROLLOUT_DAYS 过滤）
REMASK_FEEDBACK = False             # True：在每个预测重新进入下一条件窗口之前
                                    # 重施海洋掩膜（陆地 -> 0）；默认 False = 历史
                                    # 无掩膜反馈；最终取值经 Phase-5 A/B 决定

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = os.path.join(OUT_ROOT, run_tag_for(PRESET))
ckpt_path = CHECKPOINT or os.path.join(run_dir, "best.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

# 一次 weights_only 加载：检查点的 config.objective 决定重建哪个模型类
#（缺失该字段的 legacy 检查点一律是 diffusion）
try:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
except Exception as e:
    raise RuntimeError(
        f"failed to load {ckpt_path} with weights_only=True ({type(e).__name__}: {e}); "
        f"only pass weights_only=False for a verified project checkpoint") from e
ckpt_epoch = ckpt.get("epoch", None)
ckpt_cfg = ckpt.get("config") or {}
OBJECTIVE = objective_from_checkpoint(ckpt)
if "preset" in ckpt_cfg and ckpt_cfg["preset"] != PRESET:
    raise RuntimeError(f"checkpoint preset={ckpt_cfg['preset']!r} vs PRESET={PRESET!r}; "
                       "evaluation must use the trained preset")
RESIDUAL_BASE = "last_day" if OBJECTIVE == "persistence_residual" else "n/a"
if OBJECTIVE == "persistence_residual" and ENSEMBLE_SIZE != 1:
    print(f"NOTE: deterministic objective ignores ENSEMBLE_SIZE={ENSEMBLE_SIZE}; "
          "members would be identical -> using 1")
    ENSEMBLE_SIZE = 1

# 重建与训练时完全一致的输入布局：静态掩膜臂（实验 08）经 static_cond 把双
# 变量 rho 掩膜的两个通道拼接到 backbone 条件；动态 14 通道窗口与 rollout 滑窗
# 均不变。矛盾元数据在此拒绝——任何计算开始之前。
CKPT_STATIC_MASK, MODEL_COND_CH = static_mask_from_checkpoint(ckpt_cfg, OBJECTIVE)

# 输出总是位于检查点旁，并由采样配置与检查点文件名共同打标，因此绝不覆盖既有
# 评估文件或图目录；已存在的输出路径直接拒绝。rf{0,1} 记录 remask_feedback 的
# A/B 取值，使两臂可以共存。
out_dir = os.path.dirname(os.path.abspath(ckpt_path))
ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
tag_parts = [f"h{ROLLOUT_DAYS}", f"ch{SAMPLER_S_CHURN}", f"e{ENSEMBLE_SIZE}",
             f"s{EVAL_SEED}", f"rf{int(bool(REMASK_FEEDBACK))}", f"ckpt{ckpt_stem}"]
if CKPT_STATIC_MASK:
    # 静态掩膜检查点不得与同一检查点文件名的普通 14 通道输出碰撞（它们是不同
    # 的模型）
    tag_parts.insert(5, "msk1")
if SAMPLER_SIGMA_MAX is not None:
    tag_parts.append(f"sm{SAMPLER_SIGMA_MAX:g}")
if OUTPUT_TAG:
    tag_parts.append(OUTPUT_TAG)
tag = "_".join(tag_parts)
out_path = os.path.join(out_dir, f"eval_{SPLIT}_{tag}.npz")
fig_dir = os.path.join(out_dir, f"figures_{tag}")
if os.path.exists(out_path):
    raise RuntimeError(f"{out_path} already exists; delete it or set OUTPUT_TAG "
                       f"to a new name before re-running this configuration")
if os.path.isdir(fig_dir) and any(os.scandir(fig_dir)):
    raise RuntimeError(f"{fig_dir} is not empty; delete it or set OUTPUT_TAG "
                       f"to a new name before re-running this configuration")
os.makedirs(fig_dir, exist_ok=True)

# 模型

stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)
# 反归一化与掩膜语义必须与检查点训练时的一致——拒绝，而不是报告静默错误的
# 物理数值
for fp_warning in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                         mask_version()):
    print(f"WARNING: {ckpt_path}: {fp_warning}")
EVAL_STAGE[0] = "data_model"

dm_backbone = IAFNODiff(
    dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
    num_blocks=1, in_chans=2, out_chans=2, cond_chans=MODEL_COND_CH,
    ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
    hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True,
).to(device)
if OBJECTIVE == "diffusion":
    # sigma_data 是普通属性（不在 state_dict 里）：先用当前尺度构造、加载权重，
    # 再从检查点解析权威值（新检查点有存储；legacy 检查点回退到旧 stats-only
    # 尺度并显式提示）。
    model = ElucidatedDiffusion(
        dm_backbone, channels=2,
        num_sample_steps=SAMPLING_STEPS or cfg["sampling_steps"],
        image_size_h=H, image_size_w=W, image_size_z=Z,
        sigma_data=sigma_data_from_stats(stats["sigma"]),
        S_churn=SAMPLER_S_CHURN,
    )
else:
    model = PersistenceResidualIAFNO(
        dm_backbone,
        time_sigma=float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA)))
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
if OBJECTIVE == "diffusion":
    sigma_data, sd_in_ckpt = sigma_data_from_checkpoint(ckpt, stats["sigma"])
    if not sd_in_ckpt:
        print(f"WARNING: {ckpt_path} has no config.sigma_data (legacy checkpoint); "
              f"using the OLD scale sigma_data = stats sigma = {sigma_data:.5f}")
    model.sigma_data = sigma_data
    if SAMPLER_SIGMA_MAX is not None:
        model.sigma_max = SAMPLER_SIGMA_MAX
    model.eval()
    print(f"loaded {ckpt_path} (epoch={ckpt_epoch}) objective=diffusion  "
          f"sigma_data={sigma_data:.5f}  S_churn={SAMPLER_S_CHURN}  "
          f"sigma_max={model.sigma_max}")
else:
    model.eval()
    print(f"loaded {ckpt_path} (epoch={ckpt_epoch}) objective={OBJECTIVE}  "
          f"residual_base={model.residual_base}  "
          f"time_sigma={model.time_sigma:g}  "
          f"static_mask_input={CKPT_STATIC_MASK}  "
          f"(sampler parameters not applicable; remask_feedback={REMASK_FEEDBACK})")

# 数据

eval_ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]},
                       context=CONTEXT, horizon=ROLLOUT_DAYS,
                       depth_index=cfg["depth_index"], stride=EVAL_STRIDE,
                       max_windows=MAX_WINDOWS)
eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=2, pin_memory=True)
print(f"{SPLIT} rollout windows: {len(eval_ds)} (stride {EVAL_STRIDE}, "
      f"horizon {ROLLOUT_DAYS}, ensemble {ENSEMBLE_SIZE})")

mask_u, mask_v = native_masks()                       # 原生 staggered 网格掩膜
reader = NativeUVReader(cfg["depth_index"])           # 单一 reader，统一布局

# rollout 反馈回路用的 rho 网格双变量海洋掩膜（1 = 海洋、0 = 陆地）；仅在启用
# 再掩膜时才物化
ocean_mask = (build_mask_tensor(device, cfg["depth_index"])
              if REMASK_FEEDBACK else None)
if REMASK_FEEDBACK:
    print("remask_feedback=ON: every rollout prediction is re-masked "
          "(land -> 0) before re-entering the next condition window")

# 静态掩膜检查点接收与 trainer 经 static_cond 所喂相同的双变量 rho 掩膜两个
# 通道；一次性构建，对每个 batch 广播
static_cond = (build_mask_tensor(device, cfg["depth_index"])
               if CKPT_STATIC_MASK else None)
if static_cond is not None:
    print(f"static_mask_input=ON (checkpoint metadata): model_cond_chans="
          f"{MODEL_COND_CH}; the two rho mask channels are forwarded via "
          "static_cond at every rollout step")

# rollout 与指标累计

def unnormalize(x):
    """功能：与 trainer 相同的反归一化：[0,1] 归一化值 -> 物理 m/s（逐变量
    clip 范围，y_lo/y_hi 为 (1,2,1,1,1) 广播形状）。"""
    return x * (y_hi - y_lo) + y_lo


# 原生网格误差累计器（四套）：model / persistence / zero / oracle，各含 se/ae；
# shape (ROLLOUT_DAYS, 2, Z)，轴 0=lead 天、轴 1=u/v、轴 2=sigma 层；float64
# 累计跨窗口求和，避免大样本下的精度损失。land 格点贡献恒为 0（masked）。
se_m = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_m = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_p = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_p = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_z = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_z = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_o = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_o = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
n_count = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)   # 有效格点计数：每层 = 原生掩膜面积 × 窗口数

window_starts = []
fig_capture = None

# reporter 以真实窗口数（不是 loader batch 数）计数：total 是数据集窗口数，每次
# update 按该 batch 的真实窗口数推进（末 batch 可能不满），sample/s 以 lead 天
# 样本为单位
eval_rep = ProgressReporter(
    "eval", total=len(eval_ds), unit="window",
    samples_per_unit=ROLLOUT_DAYS,
    context={"split": SPLIT, "objective": OBJECTIVE})
EVAL_STAGE[0] = "rollout"
eval_t0 = time.perf_counter()
w_done = 0
try:
    with torch.no_grad():
        for bi, (cond, target, starts) in enumerate(eval_loader):
            cond = cond.to(device)                      # (B,14,H,W,Z)，[0,1] 归一化条件窗口
            target_np = target.numpy()                  # (B,L,2,H,W,Z)，[0,1] 归一化 rho 真值目标

            # --- rho 网格上的 ensemble rollout（最后对 member 求均值）。
            #     每个窗口用它自己的起始日播种（EVAL_SEED + start），轨迹可复现
            #     且与 batch 大小/装载分组无关（见 pre_rollout.ensemble_rollout）。
            starts_np = np.asarray(starts)
            window_starts.extend(int(s) for s in starts_np)
            preds = ensemble_rollout(model, cond, ROLLOUT_DAYS, ENSEMBLE_SIZE,
                                     num_sample_steps=SAMPLING_STEPS or cfg["sampling_steps"],
                                     seeds=[EVAL_SEED + int(s) for s in starts_np],
                                     clamp=True,
                                     remask_feedback=REMASK_FEEDBACK,
                                     ocean_mask=ocean_mask,
                                     static_cond=static_cond)
            rho_pred = unnormalize(ensemble_mean(preds)).cpu().numpy()  # 反归一化 -> (B,L,2,H,W,Z)

            # --- 固定的 rho -> 原生网格重采样（不旋转）
            u_pred, v_pred = rho_to_native(rho_pred)    # u:(B,L,H,W-1,Z) 沿 xi 均值，v:(B,L,H-1,W,Z) 沿 eta 均值

            # --- 未 clip 的原生真值：日索引 [s+7, s+7+ROLLOUT_DAYS)，陆地=NaN
            tu_parts, tv_parts = [], []
            for s in starts_np:
                u_s, v_s = reader.get(int(s) + CONTEXT, ROLLOUT_DAYS)
                tu_parts.append(u_s)
                tv_parts.append(v_s)
            tu_t = np.stack(tu_parts)                   # 堆叠各窗口真值：u:(B,L,H,W-1,Z)
            tv_t = np.stack(tv_parts)                   # v:(B,L,H-1,W,Z)（sigma 轴已移到末位）

            se_u, ae_u = masked_error_sums(u_pred, tu_t, mask_u)   # 模型
            se_v, ae_v = masked_error_sums(v_pred, tv_t, mask_v)
            se_m[:, 0, :] += se_u
            ae_m[:, 0, :] += ae_u
            se_m[:, 1, :] += se_v
            ae_m[:, 1, :] += ae_v

            # --- 零流基线：全零原生预测
            se_u, ae_u = masked_error_sums(np.zeros_like(tu_t), tu_t, mask_u)
            se_v, ae_v = masked_error_sums(np.zeros_like(tv_t), tv_t, mask_v)
            se_z[:, 0, :] += se_u
            ae_z[:, 0, :] += ae_u
            se_z[:, 1, :] += se_v
            ae_z[:, 1, :] += ae_v

            # --- rho-oracle：数据集的真实 rho 目标经同一反归一化 + rho_to_native
            #     路径 -> 只度量转换的不可逆误差
            se_o_b, ae_o_b = oracle_native_error_sums(target_np, stats["lo"], stats["hi"],
                                                      tu_t, tv_t, mask_u, mask_v)
            se_o += se_o_b
            ae_o += ae_o_b

            # --- 持续性基线：把第 7 天的"原生"物理 u/v 重复 ROLLOUT_DAYS 次
            pu_parts, pv_parts = [], []
            for s in starts_np:
                u_s, v_s = reader.get(int(s) + CONTEXT - 1, 1)
                pu_parts.append(u_s)
                pv_parts.append(v_s)
            pu_t = np.broadcast_to(np.stack(pu_parts),    # broadcast_to 返回只读 view：u (B,1,H,W-1,Z)
                                   (len(starts_np), ROLLOUT_DAYS, H, W - 1, Z))
            pv_t = np.broadcast_to(np.stack(pv_parts),    # v (B,1,H-1,W,Z) -> (B,L,H-1,W,Z)，沿 lead 轴广播
                                   (len(starts_np), ROLLOUT_DAYS, H - 1, W, Z))
            se_u, ae_u = masked_error_sums(pu_t, tu_t, mask_u)
            se_v, ae_v = masked_error_sums(pv_t, tv_t, mask_v)
            se_p[:, 0, :] += se_u
            ae_p[:, 0, :] += ae_u
            se_p[:, 1, :] += se_v
            ae_p[:, 1, :] += ae_v

            if bi == 0:
                # 仅捕获第一个窗口的图件数据：truth/pred 为 (L,H,W-1,Z)（u）或
                # (L,H-1,W,Z)（v），配各自原生掩膜
                fig_capture = {
                    "u": (tu_t[0], u_pred[0], mask_u),
                    "v": (tv_t[0], v_pred[0], mask_v),
                }

            # 进度后缀用的滚动 day-1 池化原生 RMSE（模型 vs 持续性）——此刻四套
            # lead/变量/层累计器都已更新完毕
            w_done += cond.shape[0]
            cnt_run = np.empty((2, Z), np.float64)
            cnt_run[0, :] = mask_u.sum() * w_done
            cnt_run[1, :] = mask_v.sum() * w_done
            d1_m = pooled_rmse(se_m[0], cnt_run)
            d1_p = pooled_rmse(se_p[0], cnt_run)

            if (bi + 1) % 10 == 0 or bi + 1 == len(eval_loader):
                eval_rep.note(f"  [{bi + 1}/{len(eval_loader)}] windows done")
            # 按该 batch 的真实窗口数推进（末 loader batch 可能小于 BATCH_SIZE，
            # 因此 batch 数 != 窗口数）
            eval_rep.update(cond.shape[0],
                            d1_rmse=f"{d1_m:.4f}", d1_pers=f"{d1_p:.4f}",
                            ratio=(f"{d1_m / d1_p:.3f}" if d1_p > 0 else "n/a"))
except BaseException as exc:
    mark_progress_failed()          # 失败兜底 hook 不得重复这条
    print(format_progress("eval", "failed", stage=EVAL_STAGE[0],
                          error=f"{type(exc).__name__}: {exc}"), flush=True)
    raise
eval_rep.close()
EVAL_STAGE[0] = "postprocess"
print(f"evaluation loop finished: {w_done} windows in "
      f"{time.perf_counter() - eval_t0:.1f}s")

n_w = len(eval_ds)
n_count[:, 0, :] = mask_u.sum() * n_w
n_count[:, 1, :] = mask_v.sum() * n_w

# 分母为 0 的 (lead,变量,层) 槽位安全置 0（where=n_count > 0），避免 0/0 NaN
rmse_m = np.sqrt(np.divide(se_m, n_count, out=np.zeros_like(se_m), where=n_count > 0))
mae_m = np.divide(ae_m, n_count, out=np.zeros_like(ae_m), where=n_count > 0)
rmse_p = np.sqrt(np.divide(se_p, n_count, out=np.zeros_like(se_p), where=n_count > 0))
mae_p = np.divide(ae_p, n_count, out=np.zeros_like(ae_p), where=n_count > 0)
rmse_z = np.sqrt(np.divide(se_z, n_count, out=np.zeros_like(se_z), where=n_count > 0))
mae_z = np.divide(ae_z, n_count, out=np.zeros_like(ae_z), where=n_count > 0)
rmse_o = np.sqrt(np.divide(se_o, n_count, out=np.zeros_like(se_o), where=n_count > 0))
mae_o = np.divide(ae_o, n_count, out=np.zeros_like(ae_o), where=n_count > 0)

# overall = sqrt(total_se / total_valid_count)，绝不是逐层 RMSE 的平均
overall_m = pooled_rmse(se_m, n_count)
overall_p = pooled_rmse(se_p, n_count)
overall_z = pooled_rmse(se_z, n_count)
overall_o = pooled_rmse(se_o, n_count)

# 保存指标 + 可复现性元数据

# 目标相关字段：确定性基线没有采样器，因此采样相关元数据显式记录为"不适用"
#（not applicable），而不是静默沿用 diffusion 路径的取值
if OBJECTIVE == "diffusion":
    sigma_data_out = np.array([sigma_data])
    sampling_steps_out = np.array([SAMPLING_STEPS or cfg["sampling_steps"]])
    sigma_max_out = np.array([model.sigma_max])
    sampler_name = "edm_heun"
    sampler_note = ("S_churn / sigma_max / sampling_steps / ensemble_size / seed "
                    "apply to the stochastic EDM sampler")
    time_sigma_out = np.array([np.nan])
else:
    sigma_data_out = np.array([np.nan])
    sampling_steps_out = np.array([-1])
    sigma_max_out = np.array([np.nan])
    sampler_name = "deterministic"
    sampler_note = ("objective=persistence_residual: S_churn / sigma_max / "
                    "sampling_steps / ensemble_size / seed are NOT applicable "
                    "(single deterministic forward per rollout step)")
    time_sigma_out = np.array([float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA))])

out_path = os.path.join(out_dir, f"eval_{SPLIT}_{tag}.npz")
# NPZ payload：键名是评估产物的稳定契约（presentation/make_figures.py 与
# scripts/analyze_checkpoint_results.py 依赖它们），绝不能改名
np.savez(out_path,
         rmse_model=rmse_m, mae_model=mae_m,
         rmse_persistence=rmse_p, mae_persistence=mae_p,
         rmse_zero=rmse_z, mae_zero=mae_z,
         rmse_oracle=rmse_o, mae_oracle=mae_o,
         valid_count=n_count,
         n_windows=np.array([n_w]), stride=np.array([EVAL_STRIDE]),
         batch_size=np.array([BATCH_SIZE]),
         rollout_days=np.array([ROLLOUT_DAYS]),
         ensemble_size=np.array([ENSEMBLE_SIZE]),
         objective=np.str_(OBJECTIVE),
         residual_base=np.str_(RESIDUAL_BASE),
         static_mask_input=np.array([bool(CKPT_STATIC_MASK)]),
         model_cond_chans=np.array([MODEL_COND_CH]),
         remask_feedback=np.array([bool(REMASK_FEEDBACK)]),
         sampler=np.str_(sampler_name), sampler_note=np.str_(sampler_note),
         time_sigma=time_sigma_out,
         S_churn=np.array([SAMPLER_S_CHURN]),
         sigma_max=sigma_max_out,
         seed=np.array([EVAL_SEED]),
         seed_scheme=np.str_("per-window: EVAL_SEED + window start day (independent "
                             "of batch size / loader batching)"),
         sigma_data=sigma_data_out,
         sampling_steps=sampling_steps_out,
         checkpoint_path=np.str_(os.path.abspath(ckpt_path)),
         checkpoint_epoch=np.array([-1 if ckpt_epoch is None else ckpt_epoch]),
         preset=np.str_(PRESET), split=np.str_(SPLIT),
         window_start_indices=np.array(window_starts),
         norm_lo=stats["lo"], norm_hi=stats["hi"], norm_sigma=np.array([stats["sigma"]]),
         grid_mapping_rule=np.str_(
             "rho u -> native u: mean of adjacent rho points along xi -> (400, 440); "
             "rho v -> native v: mean of adjacent rho points along eta -> (399, 441); "
             "no rotation; formal metrics on native grids with native mask_u/mask_v"))
print(f"saved {out_path}")

# 终端汇总

var_names = ["u", "v"]
print(f"\n=== objective={OBJECTIVE}  residual_base={RESIDUAL_BASE}  "
      f"remask_feedback={REMASK_FEEDBACK}  sampler={sampler_name} ===")
print("=== NATIVE-grid masked RMSE (m/s), pooled over u/v/layers, per lead day ===")
print("lead |  model  |  pers  | model/pers |  zero  | oracle")
for l in range(ROLLOUT_DAYS):
    rm = pooled_rmse(se_m[l], n_count[l])
    rp = pooled_rmse(se_p[l], n_count[l])
    rz = pooled_rmse(se_z[l], n_count[l])
    ro = pooled_rmse(se_o[l], n_count[l])
    print(f" {l + 1:>2}  | {rm:.4f} | {rp:.4f} | {rm / rp:.3f} | {rz:.4f} | {ro:.4f}")

print("\n=== day-1 and overall comparison table (native-grid pooled RMSE, m/s) ===")
print("mode | d1 RMSE | pers d1 | ratio | overall RMSE | pers overall | ratio")
for name, se in (("model", se_m), ("zero", se_z), ("oracle", se_o)):
    r1 = pooled_rmse(se[0], n_count[0])
    rp1 = pooled_rmse(se_p[0], n_count[0])
    ro_all = pooled_rmse(se, n_count)
    rp_all = pooled_rmse(se_p, n_count)
    print(f"{name:>6} | {r1:.4f} | {rp1:.4f} | {r1 / rp1:.3f} | "
          f"{ro_all:.4f} | {rp_all:.4f} | {ro_all / rp_all:.3f}")

print("\n=== native per-variable pooled RMSE at lead days 1/5/10/15 ===")
for k in range(2):
    line = f"{var_names[k]}: "
    for l in (0, 4, 9, 14):
        if l >= ROLLOUT_DAYS:
            break
        rm = pooled_rmse(se_m[l, k], n_count[l, k])
        rp = pooled_rmse(se_p[l, k], n_count[l, k])
        line += f"d{l + 1} {rm:.4f} (pers {rp:.4f})  "
    print(line)

print(f"\noverall native RMSE (sqrt(sum_se/sum_n)): model {overall_m:.4f} m/s "
      f"| persistence {overall_p:.4f} m/s | zero {overall_z:.4f} m/s "
      f"| rho-oracle {overall_o:.4f} m/s")

# 代表性图件

layers = [Z - 1] if Z == 1 else [0, Z // 2, Z - 1]
expected_figs = []
for day in (d for d in FIG_DAYS if d <= ROLLOUT_DAYS):
    for layer in layers:
        for var, (truth, pred, mask) in (("u", fig_capture["u"]), ("v", fig_capture["v"])):
            t = np.ma.masked_where(~mask, truth[day - 1, :, :, layer])
            p = np.ma.masked_where(~mask, pred[day - 1, :, :, layer])
            err = np.where(mask, pred[day - 1, :, :, layer] - truth[day - 1, :, :, layer], np.nan)
            e = np.ma.masked_invalid(err)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, data, title, cmap in (
                    (axes[0], t, f"truth d{day} s{layer} {var} [m/s]", "RdBu_r"),
                    (axes[1], p, f"prediction d{day} s{layer} {var} [m/s]", "RdBu_r"),
                    (axes[2], e, f"error (pred-truth) d{day} s{layer} {var} [m/s]", "RdBu_r")):
                im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
                fig.colorbar(im, ax=ax, shrink=0.85)
                ax.set_title(title)
            fig.tight_layout()
            fp = os.path.join(fig_dir, f"d{day:02d}_s{layer:02d}_{var}.png")
            fig.savefig(fp)
            plt.close(fig)
            expected_figs.append(fp)

# 每个选定的 lead/层必须在盘上同时有 u、v 两个面板——此处回归会静默丢掉 u
# 面板并让长运行泄漏未关闭的图
missing_figs = [os.path.basename(fp) for fp in expected_figs
                if not os.path.isfile(fp)]
n_u = sum(1 for fp in expected_figs if fp.endswith("_u.png"))
n_v = sum(1 for fp in expected_figs if fp.endswith("_v.png"))
if missing_figs or n_u != n_v:
    raise RuntimeError(
        f"figure save check failed: {len(missing_figs)}/{len(expected_figs)} "
        f"missing {missing_figs[:5]}, u={n_u} v={n_v} (must be paired)")
print(f"figures saved to {fig_dir} ({n_u} u + {n_v} v = {len(expected_figs)} "
      f"png files, u/v paired per lead/layer)")

# 脚本级结束：只有此刻运行才算 completed——NPZ、汇总与全部图件都已落盘
#（rollout 循环本身只报了 phase_done）
print(format_progress("eval", "completed", objective=OBJECTIVE,
                      windows=len(eval_ds), remask_feedback=REMASK_FEEDBACK,
                      elapsed_s=f"{time.perf_counter() - eval_t0:.1f}"), flush=True)
