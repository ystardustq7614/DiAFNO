"""Probe E: condition-assembly integrity + data statistics (CPU, mmap, read-only).

Verifies:
 1. cond channel order from PREUVDataset == manual day-major interleave of
    normalized u_rho/v_rho (ch 2k = u day start+k, ch 2k+1 = v day start+k)
 2. per-channel stats of cond over val windows (min/max/mean/std, land-zero
    fraction, NaN check)
 3. pooled normalized mean/std of u,v over train ocean points (subsampled days)
    -> predicts the RMSE of constant-mean-field predictors
 4. my rho->native masked-RMSE pipeline reproduces the formal persistence
    (0.1294) and zero (0.2620) on the SAME val windows as the formal eval
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/data2/user/zyq/projects/DiAFNO")
from pre_dataset import (PREUVDataset, load_masks, NativeUVReader, native_masks,
                         compute_or_load_stats, ALIGNED_DIR, CONTEXT)
from pre_metrics import rho_to_native, masked_error_sums, pooled_rmse
from pre_config import PRESETS

cfg = PRESETS["surface_smoke"]
stats = compute_or_load_stats(depth_index=cfg["depth_index"], verbose=False)
lo, hi = stats["lo"], stats["hi"]
print("stats lo:", lo, "hi:", hi, "sigma:", stats["sigma"])

# ---- 1. cond assembly check ------------------------------------------------
ds = PREUVDataset("val", stats, context=CONTEXT, horizon=1,
                  depth_index=cfg["depth_index"], stride=7)
print("\nval windows:", len(ds), "first starts:", ds.starts[:4])
start = int(ds.starts[0])
cond, target, s0 = ds[0]
print("cond shape:", tuple(cond.shape), "target shape:", tuple(target.shape), "start:", s0)

u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")
mu, mv = load_masks()
def norm(a, j):
    l, h = float(lo[j]), float(hi[j])
    a = np.clip(a, l, h)
    a = (a - l) / (h - l)
    return np.nan_to_num(a, nan=0.0).astype(np.float32)

ok = True
for k in range(CONTEXT):
    u_ref = norm(np.asarray(u[s0 + k, cfg["depth_index"]]), 0)[..., None]
    v_ref = norm(np.asarray(v[s0 + k, cfg["depth_index"]]), 1)[..., None]
    du = np.abs(cond[2 * k].numpy() - u_ref).max()
    dv = np.abs(cond[2 * k + 1].numpy() - v_ref).max()
    tu = np.abs(target[0, 0].numpy() - norm(np.asarray(u[s0 + 7, cfg["depth_index"]]), 0)[..., None]).max()
    tv = np.abs(target[0, 1].numpy() - norm(np.asarray(v[s0 + 7, cfg["depth_index"]]), 1)[..., None]).max()
    ok &= (du == 0 and dv == 0)
    print(f"  day {k}: max|cond_u - manual_u|={du:.2e}  max|cond_v - manual_v|={dv:.2e}")
print(f"  target vs day+7: max|dU|={tu:.2e} max|dV|={tv:.2e}   assembly_ok={ok}")
print("  cond NaN count:", int(torch.isnan(cond).sum()), " target NaN count:", int(torch.isnan(target).sum()))
ocean = (mu & mv)
land_only_u = (~mu) & (cond[0].numpy()[..., 0] != 0)
print("  nonzero on u-land cells (should be ~0):", int(land_only_u.sum()))

# ---- 2. per-channel stats over 24 val windows -------------------------------
print("\nper-channel cond stats (24 val windows):")
idx = np.linspace(0, len(ds) - 1, 24).astype(int)
acc = []
for i in idx:
    c, t, s = ds[int(i)]
    acc.append(c.numpy())
C = np.stack(acc)                      # (24, 14, H, W, 1)
for ch in range(14):
    c = C[:, ch]
    kind = "u" if ch % 2 == 0 else "v"
    print(f"  ch{ch:2d} ({kind} day{ch//2}): min={c.min():.4f} max={c.max():.4f} "
          f"mean={c.mean():.4f} std={c.std():.4f} zero_frac={(c==0).mean():.3f}")

# ---- 3. pooled normalized data stats over train (subsampled days) -----------
print("\ntrain-split pooled stats (every 40th day, ocean points):")
days = np.arange(0, 8401, 40)
us, vs, n = [], [], 0
for d in days:
    au = np.asarray(u[d, cfg["depth_index"]])[mu]
    av = np.asarray(v[d, cfg["depth_index"]])[mv]
    au = np.nan_to_num((np.clip(au, lo[0], hi[0]) - lo[0]) / (hi[0] - lo[0]))
    av = np.nan_to_num((np.clip(av, lo[1], hi[1]) - lo[1]) / (hi[1] - lo[1]))
    us.append(au); vs.append(av)
us = np.concatenate(us); vs = np.concatenate(vs)
su = float(us.std()); sv = float(vs.std())
mu_u, mu_v = float(us.mean()), float(vs.mean())
print(f"  u: mean={mu_u:.4f} std={su:.4f}   v: mean={mu_v:.4f} std={sv:.4f}")
print(f"  pooled sigma check vs cache ({float(np.asarray(stats['sigma'])):.5f}): "
      f"{float(np.sqrt(((us-us.mean())**2).sum()+((vs-vs.mean())**2).sum())/(us.size+vs.size)):.5f}")
print(f"  [0,1]-space value of physical 0: u={(0-lo[0])/(hi[0]-lo[0]):.4f} v={(0-lo[1])/(hi[1]-lo[1]):.4f}")
print(f"  physical value of [0,1]-space 0.5: u={(0.5*(hi[0]-lo[0])+lo[0]):.4f} v={(0.5*(hi[1]-lo[1])+lo[1]):.4f}")
print(f"  => RMSE of a constant 'normalized-0.5' predictor ~ "
      f"{float(np.sqrt(((us-0.5)**2).mean()*(hi[0]-lo[0])**2 + ((vs-0.5)**2).mean()*(hi[1]-lo[1])**2)):.4f} (rough pooled)")
print(f"  => RMSE of a constant 'train-mean' predictor ~ "
      f"{float(np.sqrt(((us-mu_u)**2).mean()*(hi[0]-lo[0])**2 + ((vs-mu_v)**2).mean()*(hi[1]-lo[1])**2)):.4f} (rough pooled)")

# ---- 4. persistence / zero calibration on the formal val windows ------------
print("\ncalibration on 156 stride-7 val windows (rho-copy persistence):")
mask_u, mask_v = native_masks()
reader = NativeUVReader(cfg["depth_index"])
se_p = np.zeros((1, 2, 1)); se_z = np.zeros((1, 2, 1)); cnt = np.zeros((1, 2, 1))
cnt[0, 0, 0] = mask_u.sum() * len(ds); cnt[0, 1, 0] = mask_v.sum() * len(ds)
y_lo = lo.reshape(1, 1, 2, 1, 1, 1); y_hi = hi.reshape(1, 1, 2, 1, 1, 1)
for bi in range(len(ds)):
    c, t, s = ds[bi]
    s = int(s)
    last_u = c[-2].numpy()                    # day-7 u (rho, normalized) (400,441,1)
    last_v = c[-1].numpy()
    rho_pred = np.stack([last_u, last_v], axis=0)[None]  # (1,1,2,H,W,1)
    phys = rho_pred * (y_hi - y_lo) + y_lo
    u_nat, v_nat = rho_to_native(phys)
    tu, tv = reader.get(s + CONTEXT, 1)
    tu = tu[None]; tv = tv[None]              # (1,L,H,W,Z) layout
    se_u, _ = masked_error_sums(u_nat, tu, mask_u)
    se_v, _ = masked_error_sums(v_nat, tv, mask_v)
    se_p[0, 0, 0] += se_u.sum(); se_p[0, 1, 0] += se_v.sum()
    se_u, _ = masked_error_sums(np.zeros_like(tu), tu, mask_u)
    se_v, _ = masked_error_sums(np.zeros_like(tv), tv, mask_v)
    se_z[0, 0, 0] += se_u.sum(); se_z[0, 1, 0] += se_v.sum()
print(f"  persistence pooled RMSE: {pooled_rmse(se_p, cnt):.4f}  (formal: 0.1294)")
print(f"  zero      pooled RMSE: {pooled_rmse(se_z, cnt):.4f}  (formal: 0.2620)")
