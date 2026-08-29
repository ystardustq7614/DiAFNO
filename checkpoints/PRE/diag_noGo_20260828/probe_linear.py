"""Probe B2 (linear): can the 14-ch condition linearly predict the next-day
target at all? Spatially-shared ridge map  cond(14) -> (u01, v01) per pixel.

Fit:  ~1000 train windows x 300 random ocean pixels (mask_uv), standardized
      features + intercept, ridge closed form, small lambda sweep.
Eval: all 156 stride-7 val windows, pooled NATIVE masked RMSE (the exact
      formal metric path) -> directly comparable with
      model 0.2584 / zero 0.2620 / persistence 0.1294.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/data2/user/zyq/projects/DiAFNO")
from pre_config import PRESETS, CONTEXT
from pre_dataset import (PREUVDataset, compute_or_load_stats, NativeUVReader,
                         native_masks, load_masks)
from pre_metrics import rho_to_native, masked_error_sums, pooled_rmse

PRESET = "surface_smoke"
TRAIN_STRIDE = 8          # ~1049 train windows
PIX_PER_WIN = 300
LAMBDAS = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]

cfg = PRESETS[PRESET]
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
lo = np.asarray(stats["lo"], np.float64); hi = np.asarray(stats["hi"], np.float64)
mu, mv = load_masks()
muv = mu & mv
yy, xx = np.where(muv)
print(f"ocean pixels (u&v): {len(yy)}")

ds_tr = PREUVDataset("train", stats, context=CONTEXT, horizon=1,
                     depth_index=cfg["depth_index"], stride=TRAIN_STRIDE)
print(f"train windows: {len(ds_tr)} (stride {TRAIN_STRIDE})")
rng = np.random.default_rng(0)
sel_pix = rng.choice(len(yy), size=PIX_PER_WIN, replace=False)

Xs, Ys = [], []
for i in range(len(ds_tr)):
    cond, tgt, s = ds_tr[i]
    c = cond.numpy()[:, :, :, 0]                       # (14,H,W)
    t = tgt[0].numpy()[:, :, :, 0]                     # (2,H,W)
    py, px = yy[sel_pix], xx[sel_pix]
    Xs.append(c[:, py, px].T)                          # (npix,14)
    Ys.append(t[:, py, px].T)                          # (npix,2)
X = np.concatenate(Xs).astype(np.float64)
Y = np.concatenate(Ys).astype(np.float64)
print(f"design matrix: {X.shape} -> {Y.shape}")

fmu, fsd = X.mean(0), X.std(0) + 1e-12
Xz = (X - fmu) / fsd
Xz1 = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
G = Xz1.T @ Xz1
B = Xz1.T @ Y
evals, evecs = np.linalg.eigh(G)

def ridge_beta(lmb):
    inv = evecs @ np.diag(1.0 / (evals + lmb)) @ evecs.T
    return inv @ B

ds_va = PREUVDataset("val", stats, context=CONTEXT, horizon=1,
                     depth_index=cfg["depth_index"], stride=7)
print(f"val windows: {len(ds_va)}")
mask_u_nat, mask_v_nat = native_masks()
reader = NativeUVReader(cfg["depth_index"])
y_lo = lo.reshape(1, 1, 2, 1, 1, 1); y_hi = hi.reshape(1, 1, 2, 1, 1, 1)

def eval_windows(pred_fn, tag):
    se = np.zeros((1, 2, 1)); se_p = np.zeros((1, 2, 1)); se_z = np.zeros((1, 2, 1))
    cnt = np.zeros((1, 2, 1))
    cnt[0, 0, 0] = mask_u_nat.sum() * len(ds_va); cnt[0, 1, 0] = mask_v_nat.sum() * len(ds_va)
    for i in range(len(ds_va)):
        cond, tgt, s = ds_va[i]
        s = int(s)
        c = cond.numpy()[:, :, :, 0]
        pred = pred_fn(c)                              # (2,H,W) normalized
        phys = pred[None, None] * (y_hi - y_lo) + y_lo # (1,1,2,H,W,1)
        u_nat, v_nat = rho_to_native(phys)
        tu, tv = reader.get(s + CONTEXT, 1); tu = tu[None]; tv = tv[None]
        seu, _ = masked_error_sums(u_nat, tu, mask_u_nat)
        sev, _ = masked_error_sums(v_nat, tv, mask_v_nat)
        se[0, 0, 0] += seu.sum(); se[0, 1, 0] += sev.sum()
        pers = np.stack([c[-2], c[-1]], axis=0)[..., None][None, None]
        phys = pers * (y_hi - y_lo) + y_lo
        u_nat, v_nat = rho_to_native(phys)
        seu, _ = masked_error_sums(u_nat, tu, mask_u_nat)
        sev, _ = masked_error_sums(v_nat, tv, mask_v_nat)
        se_p[0, 0, 0] += seu.sum(); se_p[0, 1, 0] += sev.sum()
        seu, _ = masked_error_sums(np.zeros_like(tu), tu, mask_u_nat)
        sev, _ = masked_error_sums(np.zeros_like(tv), tv, mask_v_nat)
        se_z[0, 0, 0] += seu.sum(); se_z[0, 1, 0] += sev.sum()
    print(f"  {tag:28s}: pooled native RMSE = {pooled_rmse(se, cnt):.4f} m/s")
    print(f"  {'(pers anchor)':28s}: {pooled_rmse(se_p, cnt):.4f}   "
          f"{'(zero anchor)':15s}: {pooled_rmse(se_z, cnt):.4f}")
    return pooled_rmse(se, cnt)

print("\n=== ridge: 14-dim cond -> next-day target (spatially-shared linear) ===")
for lmb in LAMBDAS:
    beta = ridge_beta(lmb)
    def pred_fn(c, beta=beta):
        f = c.reshape(14, -1).T                        # (HW,14)
        fz = (f - fmu) / fsd
        fz1 = np.concatenate([fz, np.ones((len(fz), 1))], axis=1)
        out = (fz1 @ beta).T.reshape(2, 400, 441, 1)
        return out
    eval_windows(pred_fn, f"ridge lambda={lmb:g}")

# persistence IS in the linear hypothesis class: check its weight vector
beta = ridge_beta(0.0)
names = [f"{'u' if k%2==0 else 'v'} d{k//2}" for k in range(14)]
print("\nbeta (lambda=0) per feature, target u / target v:")
for j, nm in enumerate(names):
    print(f"  {nm:6s}: {beta[j,0]:+.4f} / {beta[j,1]:+.4f}")
