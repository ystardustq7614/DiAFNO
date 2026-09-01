"""Region breakdown (coastal vs open-ocean) for the persistence-residual arms.

Replays the official validation day-1 protocol (same checkpoint load rules as
pre_evaluate.py, SPLIT="val", ROLLOUT_DAYS=1, deterministic sample) for a list
of checkpoints and reports pooled native-grid RMSE split into coastal and
open-ocean cells. Coastal = valid cells within COASTAL_BUFFER cells of land
(binary dilation of the complement of each native mask), open-ocean = the
rest. Model and persistence are reported for each region.

Script (module top-level, like pre_evaluate.py) — run from repo root:
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/diag_region_breakdown.py

Outputs one region_diag_ckpt<stem>.npz next to each checkpoint (refused if it
already exists); the comparison table goes to stdout.
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
COASTAL_BUFFER = 5        # cells to land within a cell counts as coastal

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
    """(H, W) native mask -> (coastal, offshore) boolean cell masks."""
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
    """Signed/squared error sums over the given boolean cell mask (H, W).

    pred/truth: (B, 1, H, W, Z=1) native grids — the trailing Z axis (and the
    lead axis, already sliced by the caller) is dropped with [..., 0], NEVER
    with [:, :, 0] (which would slice the W axis and smear land NaN across
    each row).
    """
    pred = np.asarray(pred, np.float64)[..., 0]       # (B, H, W)
    truth = np.asarray(truth, np.float64)[..., 0]     # (B, H, W)
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

    # accumulators: region -> var -> [n, se_m, se_signed_m, se_p]
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
            rho_pred = (pred.float() * (y_hi - y_lo) + y_lo).cpu().numpy()
            u_pred, v_pred = rho_to_native(rho_pred[:, None])  # add L=1 dim
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
