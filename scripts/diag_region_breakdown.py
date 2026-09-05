"""模块职责：persistence-residual 双臂的 coastal / 开阔海域分区对比。

对 CHECKPOINTS 列表中的每个 checkpoint 复现正式 validation day-1 协议
（SPLIT="val"、ROLLOUT_DAYS=1、确定性采样；checkpoint 装载规则与
pre_evaluate.py 相同），报告按 coastal / 开阔海域拆分、逐区域池化的
原生网格 RMSE（model 与 persistence 各自累计）。coastal = 距陆地
COASTAL_BUFFER 格以内的有效格点（对每个原生掩膜补集做 binary_dilation），
开阔海域 = 其余格点。

不负责：不重训模型、不修改任何训练/评估产物；数据集与 checkpoint 只读。
唯一写盘是每个 checkpoint 同目录的 region_diag_ckpt<stem>.npz（已存在则
拒绝运行）；对比表只打到 stdout。脚本为 module top-level（同
pre_evaluate.py），只能从仓库根目录运行，禁止作为模块 import（顶层即
执行全部诊断）。

关键约束：
- region_sums 丢弃尾部 Z 轴必须用 [..., 0]：[:, :, 0] 切的是 W 轴，
  会把陆地 NaN 涂满每行、污染误差和（详见该函数 docstring）；
- 双臂重建由 checkpoint config 驱动：static_mask_input=True 时
  cond_chans 加入 STATIC_MASK_CHANNELS=2 个静态掩膜通道（经 static_cond
  传入 sample，动态滑窗仍是纯 14 通道条件）；
- 归一化/掩膜指纹漂移只打 WARNING（legacy checkpoint 无法验证时）。

依赖关系：pre_config（CONTEXT / RESIDUAL_TIME_SIGMA /
STATIC_MASK_CHANNELS / PRESETS / check_norm_fingerprint）；pre_dataset
（NativeUVReader / PREUVDataset / build_mask_tensor /
compute_or_load_stats / mask_version / native_masks）；
pre_metrics.rho_to_native；pre_models.PersistenceResidualIAFNO；
IAFNO.IAFNODiff；scipy.ndimage（陆地掩膜膨胀）。

脚本 —— 从仓库根目录运行：
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/diag_region_breakdown.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy import ndimage

from IAFNO import IAFNODiff
from pre_config import (CONTEXT, RESIDUAL_TIME_SIGMA, STATIC_MASK_CHANNELS,
                        PRESETS, check_norm_fingerprint)
from pre_dataset import (NativeUVReader, PREUVDataset, build_mask_tensor,
                         compute_or_load_stats, mask_version, native_masks)
from pre_metrics import rho_to_native
from pre_models import PersistenceResidualIAFNO

PRESET = "surface_smoke"
CHECKPOINTS = [
    ("/data2/user/zyq/checkpoints/PRE/"
     "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/Ep10.pth",
     "A: 14-ch (no static mask input)"),
    ("/data2/user/zyq/checkpoints/PRE/"
     "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MSK/Ep10.pth",
     "B: 14-ch + 2 static mask channels"),
]
SPLIT = "val"
EVAL_STRIDE = 7
BATCH_SIZE = 4
COASTAL_BUFFER = 5        # 距陆地的格数阈值：格点在该范围内算 coastal

# 与正式评估一致的模块级种子（top-level 执行即生效）
torch.manual_seed(123)

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)
mask_u, mask_v = native_masks()
reader = NativeUVReader(cfg["depth_index"])


def region_masks(mask2d):
    """(H, W) 原生掩膜 -> (coastal, offshore) 两个布尔格点掩膜。

    coastal = 距陆地 COASTAL_BUFFER 格以内的有效格点：陆地取掩膜补集，
    用 scipy.ndimage 默认十字结构元 binary_dilation 迭代 COASTAL_BUFFER
    次（与陆地的 L1 距离 <= COASTAL_BUFFER）；offshore = 其余有效格点。
    口径与 scripts/diag_uv_predictability.py 一致，只是作用在原生网格。
    """
    valid = np.asarray(mask2d, bool)
    land = ~valid
    near_land = ndimage.binary_dilation(land, iterations=COASTAL_BUFFER)
    coastal = valid & near_land
    offshore = valid & ~near_land
    return coastal, offshore


regions = {"coastal": {}, "offshore": {}}
regions["coastal"]["u"], regions["offshore"]["u"] = region_masks(mask_u)
regions["coastal"]["v"], regions["offshore"]["v"] = region_masks(mask_v)


def region_sums(pred, truth, cell_mask):
    """在给定布尔格点掩膜 (H, W) 内累计带符号/平方误差和。

    pred/truth：(B, 1, H, W, Z=1) 原生网格物理场（m/s）。丢弃尾部 Z 轴
    （以及调用方已切掉的 lead 轴）必须用 [..., 0]，绝不能用 [:, :, 0]：
    后者切的是 W 轴，会把陆地 NaN 涂满每一行、污染全部误差统计。

    返回 (e.sum(), (e**2).sum(), B*cell_mask.sum())：误差和 / 平方和 /
    有效格点计数（掩膜外格点贡献恒为 0）。
    """
    pred = np.asarray(pred, np.float64)[..., 0]       # 形状 (B, H, W)：尾部 Z=1 轴已丢弃
    truth = np.asarray(truth, np.float64)[..., 0]     # 形状 (B, H, W)：同上
    e = np.where(cell_mask[None], pred - truth, 0.0)
    return e.sum(), (e ** 2).sum(), int(pred.shape[0]) * int(cell_mask.sum())


summary = []
for ckpt_path, label in CHECKPOINTS:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_cfg = ckpt.get("config") or {}
    if ckpt_cfg.get("objective") != "persistence_residual":
        raise RuntimeError(f"{ckpt_path}: not a persistence_residual checkpoint")
    static_mask = bool(ckpt_cfg.get("static_mask_input", False))
    model_cond_ch = 2 * CONTEXT + (STATIC_MASK_CHANNELS if static_mask else 0)
    for fp in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                     mask_version()):
        print(f"WARNING: {ckpt_path}: {fp}")
    dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"],
                   embed_dim=cfg["embed_dim"], num_blocks=1, in_chans=2,
                   out_chans=2, cond_chans=model_cond_ch,
                   ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
                   hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True,
                   ).to(device)
    model = PersistenceResidualIAFNO(
        dm, time_sigma=float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA)))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    static_cond = (build_mask_tensor(device, cfg["depth_index"])
                   if static_mask else None)

    ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]},
                      context=CONTEXT, horizon=1, depth_index=cfg["depth_index"],
                      stride=EVAL_STRIDE, max_windows=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=2, pin_memory=True)
    n_windows = len(ds)

    # 累计器：region -> var -> [n, se_m, se_signed_m, se_p] 四元素 float 数组；
    # RMSE 在循环外由 sqrt(se/n) 合成，绝不先取分段 RMSE 再平均
    acc = {r: {v: np.zeros(4) for v in ("u", "v")} for r in regions}
    t0 = time.perf_counter()
    starts_all = []
    with torch.no_grad():
        for bi, (cond, target, starts) in enumerate(loader):
            cond = cond.to(device)
            starts_np = np.asarray(starts)
            starts_all.extend(int(s) for s in starts_np)
            with torch.amp.autocast(device_type="cuda" if cond.is_cuda else "cpu"):
                if static_cond is None:
                    pred = model.sample(cond, num_sample_steps=1, clamp=True)
                else:
                    pred = model.sample(cond, num_sample_steps=1, clamp=True,
                                        static_cond=static_cond)
            # 反归一化回物理 m/s（rho 网格），再映射到原生 staggered 网格
            rho_pred = (pred.float() * (y_hi - y_lo) + y_lo).cpu().numpy()
            u_pred, v_pred = rho_to_native(rho_pred[:, None])  # 补 L=1 维，凑成 (B, L, 2, H, W, Z)
            tu, tv = [], []
            pu, pv = [], []
            for s in starts_np:
                u_t, v_t = reader.get(int(s) + CONTEXT, 1)
                tu.append(u_t)
                tv.append(v_t)
                u_p, v_p = reader.get(int(s) + CONTEXT - 1, 1)
                pu.append(u_p)
                pv.append(v_p)
            tu_t, tv_t = np.stack(tu), np.stack(tv)
            # 持续性基线：窗口末天（s+CONTEXT-1）作为 day-1 的同一持续性源；
            # broadcast_to 返回只读 view，不复制数据
            pu_t = np.broadcast_to(np.stack(pu), (len(starts_np), 1, H, W - 1, Z))
            pv_t = np.broadcast_to(np.stack(pv), (len(starts_np), 1, H - 1, W, Z))
            for r in regions:
                s, se, n = region_sums(u_pred[:, 0], tu_t[:, 0],
                                       regions[r]["u"])
                acc[r]["u"][0] += n
                acc[r]["u"][1] += se
                acc[r]["u"][2] += s
                s, se, n = region_sums(v_pred[:, 0], tv_t[:, 0],
                                       regions[r]["v"])
                acc[r]["v"][0] += n
                acc[r]["v"][1] += se
                acc[r]["v"][2] += s
                s, se, _ = region_sums(pu_t[:, 0], tu_t[:, 0], regions[r]["u"])
                acc[r]["u"][3] += se
                s, se, _ = region_sums(pv_t[:, 0], tv_t[:, 0], regions[r]["v"])
                acc[r]["v"][3] += se
            if (bi + 1) % 10 == 0 or bi + 1 == len(loader):
                print(f"[{label}] [{bi + 1}/{len(loader)}] windows "
                      f"{min((bi + 1) * BATCH_SIZE, n_windows)}/{n_windows} "
                      f"elapsed_s={time.perf_counter() - t0:.0f}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    npz_path = os.path.join(out_dir, f"region_diag_ckpt{stem}.npz")
    if os.path.exists(npz_path):
        raise RuntimeError(f"{npz_path} already exists")
    # 输出 NPZ：{coastal|offshore}_{u|v} 各存四元素 [n, se_m, se_signed_m, se_p]，
    # 另加 coastal_buffer / n_windows / window_start_indices 溯源字段；已存在则拒绝
    np.savez(npz_path,
             coastal_buffer=np.int64(COASTAL_BUFFER),
             n_windows=np.int64(n_windows),
             window_start_indices=np.array(starts_all, np.int64),
             **{f"{r}_{v}": acc[r][v] for r in regions for v in ("u", "v")})

    print(f"\n=== {label}  ({SPLIT} day-1, {n_windows} windows, "
          f"coastal = within {COASTAL_BUFFER} cells of land) ===")
    print("region   | var |   n    | model  | pers   | ratio")
    row = {"label": label, "ckpt": ckpt_path}
    for r in regions:
        for v in ("u", "v"):
            n, se_m, _, se_p = acc[r][v]
            rm = float(np.sqrt(se_m / n))
            rp = float(np.sqrt(se_p / n))
            print(f"{r:8s} | {v} | {int(n):6d} | {rm:.4f} | {rp:.4f} | {rm / rp:.3f}")
            row[f"{r}_{v}_m"] = rm
            row[f"{r}_{v}_p"] = rp
        rm_all = float(np.sqrt(sum(acc[r][v][1] for v in ("u", "v"))
                               / sum(acc[r][v][0] for v in ("u", "v"))))
        rp_all = float(np.sqrt(sum(acc[r][v][3] for v in ("u", "v"))
                               / sum(acc[r][v][0] for v in ("u", "v"))))
        print(f"{r:8s} | all |        | {rm_all:.4f} | {rp_all:.4f} | "
              f"{rm_all / rp_all:.3f}")
        row[f"{r}_all_m"] = rm_all
        row[f"{r}_all_p"] = rp_all
    summary.append(row)
    print(f"saved {npz_path}", flush=True)

print("\n=== A/B region comparison (model/persistence ratio, day-1 val) ===")
print("arm | coastal_u | coastal_v | offshore_u | offshore_v")
for row in summary:
    print(f"{row['label']} | {row['coastal_u_m'] / row['coastal_u_p']:.3f} | "
          f"{row['coastal_v_m'] / row['coastal_v_p']:.3f} | "
          f"{row['offshore_u_m'] / row['offshore_u_p']:.3f} | "
          f"{row['offshore_v_m'] / row['offshore_v_p']:.3f}")
print("PROGRESS phase=diag status=completed")
