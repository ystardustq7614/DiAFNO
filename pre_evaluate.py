#!/usr/bin/env python3
"""PRE_ocean_data evaluation: 15-day autoregressive rollout vs persistence baseline.

For each test window (22 consecutive days: 7 context + 15 horizon):
    1. rollout: predict next day from current 7-day condition, then shift the window
       (drop oldest day, append prediction) and repeat 15 times;
    2. persistence: repeat the last context day (day 7) 15 times.

Metrics (physical units, m/s), ocean points only (mask_uv):
    masked RMSE and MAE, broken down by lead day (1..15) x variable (u,v) x sigma layer.
    Mask is horizontal (NaN pattern is identical across the 30 layers, verified).

Run (from repo root):  python pre_evaluate.py
Output: <run_dir>/eval_<split>.npz + printed summary table.
"""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast

from utilities3 import load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_config import PRESETS, OUT_ROOT, CONTEXT, HORIZON, run_tag_for
from pre_dataset import PREUVDataset, load_mask, compute_or_load_stats

torch.manual_seed(123)
np.random.seed(123)

########## eval config ##########

PRESET = "surface_smoke"            # must match the trained checkpoint's preset
CHECKPOINT = None                   # None -> <run_dir>/best.pth
SPLIT = "test"
EVAL_STRIDE = 7                     # start a rollout window every N days (~154 test windows)
MAX_WINDOWS = None                  # set small (e.g. 8) for a quick check
BATCH_SIZE = 4                      # rollout batch; use 1 for full3d if OOM
SAMPLING_STEPS = None               # None -> preset value

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = os.path.join(OUT_ROOT, run_tag_for(PRESET))
ckpt_path = CHECKPOINT or os.path.join(run_dir, "best.pth")

########## model ##########

stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)

dm_backbone = IAFNODiff(
    dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
    num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
    ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
    hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True,
).to(device)
model = ElucidatedDiffusion(
    dm_backbone, channels=2,
    num_sample_steps=SAMPLING_STEPS or cfg["sampling_steps"],
    image_size_h=H, image_size_w=W, image_size_z=Z,
    sigma_data=float(stats["sigma"]),
)
load_checkpoint(ckpt_path, model, map_location=device)
model.eval()
print(f"loaded {ckpt_path}")

########## data ##########

eval_ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]},
                       context=CONTEXT, horizon=HORIZON,
                       depth_index=cfg["depth_index"], stride=EVAL_STRIDE,
                       max_windows=MAX_WINDOWS)
eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=2, pin_memory=True)
print(f"{SPLIT} rollout windows: {len(eval_ds)} (stride {EVAL_STRIDE})")

mask2d = torch.tensor(load_mask(), device=device, dtype=torch.float32)  # (H,W)
mask_b = mask2d.reshape(1, 1, H, W, 1)                                   # broadcast
n_ocean = float(mask2d.sum())

########## rollout + metrics ##########

def unnormalize(x):
    return x * (y_hi - y_lo) + y_lo


# accumulators: (HORIZON, 2, Z)
se_m = np.zeros((HORIZON, 2, Z), np.float64)
ae_m = np.zeros((HORIZON, 2, Z), np.float64)
se_p = np.zeros((HORIZON, 2, Z), np.float64)
ae_p = np.zeros((HORIZON, 2, Z), np.float64)
n_count = np.zeros((HORIZON, 2, Z), np.float64)

with torch.no_grad():
    for bi, (cond, target, starts) in enumerate(eval_loader):
        cond = cond.to(device)                      # (B,14,H,W,Z) normalized
        tgt = unnormalize(target.to(device))        # (B,15,2,H,W,Z) physical

        # --- model rollout
        cur = cond
        preds = []
        for step in range(HORIZON):
            with autocast():
                p = model.sample(cur)               # (B,2,H,W,Z) in [0,1]
            p = p.float()
            preds.append(p)
            cur = torch.cat([cur[:, 2:], p], dim=1)  # shift window by one day
        pred = unnormalize(torch.stack(preds, dim=1))  # (B,15,2,H,W,Z) physical

        # --- persistence: repeat last context day (channels -2,-1 = u,v of day 7)
        pers = unnormalize(cond[:, -2:]).unsqueeze(1).expand(-1, HORIZON, -1, -1, -1, -1)

        for name, pr, se, ae in (("model", pred, se_m, ae_m), ("pers", pers, se_p, ae_p)):
            err = (pr - tgt) * mask_b               # (B,15,2,H,W,Z)
            se += err.pow(2).sum(dim=(0, 3, 4)).double().cpu().numpy()   # (15,2,Z)
            ae += err.abs().sum(dim=(0, 3, 4)).double().cpu().numpy()
        n_count += tgt.shape[0] * n_ocean

        if (bi + 1) % 10 == 0 or bi + 1 == len(eval_loader):
            print(f"  [{bi + 1}/{len(eval_loader)}] windows done", flush=True)

rmse_m = np.sqrt(se_m / n_count)
mae_m = ae_m / n_count
rmse_p = np.sqrt(se_p / n_count)
mae_p = ae_p / n_count

out_path = os.path.join(run_dir, f"eval_{SPLIT}.npz")
np.savez(out_path, rmse_model=rmse_m, mae_model=mae_m,
         rmse_pers=rmse_p, mae_pers=mae_p,
         n_windows=np.array([len(eval_ds)]), stride=np.array([EVAL_STRIDE]))
print(f"saved {out_path}")

########## summary ##########

var_names = ["u", "v"]
print("\n=== masked RMSE (m/s), mean over u/v/layers, per lead day ===")
print("lead |  model |  pers  | model/pers")
for l in range(HORIZON):
    m_, p_ = rmse_m[l].mean(), rmse_p[l].mean()
    print(f" {l + 1:>2}  | {m_:.4f} | {p_:.4f} | {m_ / p_:.3f}")

print("\n=== per-variable RMSE at lead days 1/5/10/15 (mean over layers) ===")
for k in range(2):
    line = f"{var_names[k]}: "
    for l in (0, 4, 9, 14):
        line += f"d{l + 1} {rmse_m[l, k].mean():.4f} (pers {rmse_p[l, k].mean():.4f})  "
    print(line)

print(f"\noverall model RMSE {rmse_m.mean():.4f} m/s | persistence {rmse_p.mean():.4f} m/s")
