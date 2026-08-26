#!/usr/bin/env python3
"""PRE_ocean_data evaluation: 15-day autoregressive rollout vs persistence baseline.

FORMAL metrics are computed on the NATIVE staggered u/v grids against the
UNCLIPPED raw physical truth (raw u.npy/v.npy), using the native mask_u/mask_v:

    rho u -> native u:  average the two adjacent rho points along xi: (400, 440)
    rho v -> native v:  average the two adjacent rho points along eta: (399, 441)
    (inverse of the Plan A colocation stencil, one-sided-free: no rotation)

Pipeline per test window (22 consecutive days: 7 context + 15 horizon):
    1. rollout on the rho grid: predict next day from the current 7-day
       condition, shift the window (drop oldest day, append prediction) x15;
    2. map each rho prediction back to the native u/v grids (rho_to_native);
    3. compare with the raw native truth of days 8..22 (unclipped, land=NaN)
       read via a single NativeUVReader (unified (days, H, W, Z) layout);
    4. persistence baseline = repeat the day-7 NATIVE physical u/v 15 times
       (never the clipped/normalized condition input);
    5. masked RMSE/MAE per lead day (1..15) x variable (u,v) x sigma layer,
       valid cells counted per native mask.

Overall RMSE = sqrt(sum(squared_error) / sum(valid_count)) — NOT the arithmetic
mean of per-layer RMSEs; the console summary pools the same way (pooled_rmse).

Output: <run_dir>/eval_<split>.npz (native-grid metrics + full reproducibility metadata)
        <run_dir>/figures/d{1,3,5,7,10,15}_s{layer}_{u|v}.png  (truth/pred/error)

Run (from repo root):  python pre_evaluate.py
"""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities3 import load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_config import PRESETS, OUT_ROOT, CONTEXT, HORIZON, run_tag_for
from pre_dataset import (PREUVDataset, NativeUVReader, native_masks,
                         compute_or_load_stats)
from pre_metrics import rho_to_native, masked_error_sums, pooled_rmse

torch.manual_seed(123)

########## eval config ##########

PRESET = "surface_smoke"            # must match the trained checkpoint's preset
CHECKPOINT = None                   # None -> <run_dir>/best.pth
SPLIT = "test"
EVAL_STRIDE = 7                     # start a rollout window every N days
MAX_WINDOWS = None                  # set small (e.g. 8) for a quick check
BATCH_SIZE = 4                      # rollout batch; use 1 for full3d if OOM
SAMPLING_STEPS = None               # None -> preset value
FIG_DAYS = (1, 3, 5, 7, 10, 15)     # representative lead days for figures

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = os.path.join(OUT_ROOT, run_tag_for(PRESET))
ckpt_path = CHECKPOINT or os.path.join(run_dir, "best.pth")
fig_dir = os.path.join(run_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

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
ckpt = load_checkpoint(ckpt_path, model, map_location=device)
ckpt_epoch = ckpt.get("epoch", None)
model.eval()
print(f"loaded {ckpt_path} (epoch={ckpt_epoch})")

########## data ##########

eval_ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]},
                       context=CONTEXT, horizon=HORIZON,
                       depth_index=cfg["depth_index"], stride=EVAL_STRIDE,
                       max_windows=MAX_WINDOWS)
eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=2, pin_memory=True)
print(f"{SPLIT} rollout windows: {len(eval_ds)} (stride {EVAL_STRIDE})")

mask_u, mask_v = native_masks()                       # native staggered grids
reader = NativeUVReader(cfg["depth_index"])           # single reader, unified layout

########## rollout + metrics ##########

def unnormalize(x):
    return x * (y_hi - y_lo) + y_lo


# native accumulators: (HORIZON, 2, Z)
se_m = np.zeros((HORIZON, 2, Z), np.float64)
ae_m = np.zeros((HORIZON, 2, Z), np.float64)
se_p = np.zeros((HORIZON, 2, Z), np.float64)
ae_p = np.zeros((HORIZON, 2, Z), np.float64)
n_count = np.zeros((HORIZON, 2, Z), np.float64)

window_starts = []
fig_capture = None


with torch.no_grad():
    for bi, (cond, target, starts) in enumerate(eval_loader):
        cond = cond.to(device)                      # (B,14,H,W,Z) normalized

        # --- model rollout on the rho grid
        cur = cond
        preds = []
        for _ in range(HORIZON):
            with autocast():
                p = model.sample(cur)               # (B,2,H,W,Z) in [0,1]
            preds.append(p.float())
            cur = torch.cat([cur[:, 2:], p], dim=1)  # shift window by one day
        rho_pred = unnormalize(torch.stack(preds, dim=1)).cpu().numpy()  # (B,15,2,H,W,Z)

        # --- fixed rho -> native resampling (no rotation)
        u_pred, v_pred = rho_to_native(rho_pred)    # (B,15,H,W-1,Z), (B,15,H-1,W,Z)

        starts_np = np.asarray(starts)
        window_starts.extend(int(s) for s in starts_np)

        # --- UNCLIPPED native truth: days [s+7, s+22), one get() per start,
        #     unpacked into (u, v) (unified (days, H, W, Z) layout)
        tu_parts, tv_parts = [], []
        for s in starts_np:
            u_s, v_s = reader.get(int(s) + CONTEXT, HORIZON)
            tu_parts.append(u_s)
            tv_parts.append(v_s)
        tu_t = np.stack(tu_parts)                   # (B,15,H,W-1,Z)
        tv_t = np.stack(tv_parts)                   # (B,15,H-1,W,Z)

        se_u, ae_u = masked_error_sums(u_pred, tu_t, mask_u)   # (15,Z)
        se_v, ae_v = masked_error_sums(v_pred, tv_t, mask_v)
        se_m[:, 0, :] += se_u
        ae_m[:, 0, :] += ae_u
        se_m[:, 1, :] += se_v
        ae_m[:, 1, :] += ae_v

        # --- persistence: repeat the day-7 NATIVE physical u/v 15 times
        pu_parts, pv_parts = [], []
        for s in starts_np:
            u_s, v_s = reader.get(int(s) + CONTEXT - 1, 1)
            pu_parts.append(u_s)
            pv_parts.append(v_s)
        pu_t = np.broadcast_to(np.stack(pu_parts),    # (B,1,H,W-1,Z)
                               (len(starts_np), HORIZON, H, W - 1, Z))
        pv_t = np.broadcast_to(np.stack(pv_parts),    # (B,1,H-1,W,Z)
                               (len(starts_np), HORIZON, H - 1, W, Z))
        se_u, ae_u = masked_error_sums(pu_t, tu_t, mask_u)
        se_v, ae_v = masked_error_sums(pv_t, tv_t, mask_v)
        se_p[:, 0, :] += se_u
        ae_p[:, 0, :] += ae_u
        se_p[:, 1, :] += se_v
        ae_p[:, 1, :] += ae_v

        if bi == 0:
            fig_capture = {
                "u": (tu_t[0], u_pred[0], mask_u),
                "v": (tv_t[0], v_pred[0], mask_v),
            }

        if (bi + 1) % 10 == 0 or bi + 1 == len(eval_loader):
            print(f"  [{bi + 1}/{len(eval_loader)}] windows done", flush=True)

n_w = len(eval_ds)
n_count[:, 0, :] = mask_u.sum() * n_w
n_count[:, 1, :] = mask_v.sum() * n_w

rmse_m = np.sqrt(np.divide(se_m, n_count, out=np.zeros_like(se_m), where=n_count > 0))
mae_m = np.divide(ae_m, n_count, out=np.zeros_like(ae_m), where=n_count > 0)
rmse_p = np.sqrt(np.divide(se_p, n_count, out=np.zeros_like(se_p), where=n_count > 0))
mae_p = np.divide(ae_p, n_count, out=np.zeros_like(ae_p), where=n_count > 0)

# overall = sqrt(total_se / total_valid_count), never an average of layer RMSEs
overall_m = pooled_rmse(se_m, n_count)
overall_p = pooled_rmse(se_p, n_count)

########## save metrics + reproducibility metadata ##########

out_path = os.path.join(run_dir, f"eval_{SPLIT}.npz")
np.savez(out_path,
         rmse_model=rmse_m, mae_model=mae_m,
         rmse_persistence=rmse_p, mae_persistence=mae_p,
         valid_count=n_count,
         n_windows=np.array([n_w]), stride=np.array([EVAL_STRIDE]),
         seed=np.array([123]), sampling_steps=np.array([SAMPLING_STEPS or cfg["sampling_steps"]]),
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

########## summary ##########

var_names = ["u", "v"]
print("\n=== NATIVE-grid masked RMSE (m/s), pooled over u/v/layers, per lead day ===")
print("lead |  model |  pers  | model/pers")
for l in range(HORIZON):
    rm = pooled_rmse(se_m[l], n_count[l])
    rp = pooled_rmse(se_p[l], n_count[l])
    print(f" {l + 1:>2}  | {rm:.4f} | {rp:.4f} | {rm / rp:.3f}")

print("\n=== native per-variable pooled RMSE at lead days 1/5/10/15 ===")
for k in range(2):
    line = f"{var_names[k]}: "
    for l in (0, 4, 9, 14):
        rm = pooled_rmse(se_m[l, k], n_count[l, k])
        rp = pooled_rmse(se_p[l, k], n_count[l, k])
        line += f"d{l + 1} {rm:.4f} (pers {rp:.4f})  "
    print(line)

print(f"\noverall native RMSE (sqrt(sum_se/sum_n)): model {overall_m:.4f} m/s "
      f"| persistence {overall_p:.4f} m/s")

########## representative figures ##########

layers = [Z - 1] if Z == 1 else [0, Z // 2, Z - 1]
for day in FIG_DAYS:
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
print(f"figures saved to {fig_dir}")