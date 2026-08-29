#!/usr/bin/env python3
"""Corrected diagnostic: physical units, pooled-per-cell denominators.
Pred/tgt/day7 are unnormalized to physical m/s; climatology is physical.
"""
import sys
sys.path.insert(0, "/data2/user/zyq/projects/DiAFNO_lr3e4")
import numpy as np
import torch
from pre_config import PRESETS, CONTEXT, sigma_data_from_stats, sigma_data_from_checkpoint
from pre_dataset import PREUVDataset, build_mask_tensor, compute_or_load_stats, ALIGNED_DIR, load_masks
from pre_metrics import masked_rel_l2
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from utilities3 import load_checkpoint

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/data2/user/zyq/checkpoints/PRE_lr3e4/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/best.pth"
N_WIN = 12
device = torch.device("cuda")
cfg = PRESETS["surface_smoke"]
H, W, Z = 400, 441, 1

stats = compute_or_load_stats(depth_index=29, verbose=False)
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)
unnorm = lambda x: x * (y_hi - y_lo) + y_lo

dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
               num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
               ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
               hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True).to(device)
model = ElucidatedDiffusion(dm, channels=2, num_sample_steps=32, image_size_h=H,
                            image_size_w=W, image_size_z=Z,
                            sigma_data=sigma_data_from_stats(stats["sigma"]), S_churn=0)
ck = load_checkpoint(CKPT, model, map_location=device)
model.sigma_data, _ = sigma_data_from_checkpoint(ck, stats["sigma"])
model.eval()

ds = PREUVDataset("val", {"lo": stats["lo"], "hi": stats["hi"]}, context=7, horizon=1,
                  depth_index=29, stride=7)
conds, tgts, starts = [], [], []
for i in range(N_WIN):
    c, t, s = ds[i]
    conds.append(c[None]); tgts.append(t[0][None]); starts.append(s)
cond = torch.cat(conds).to(device)
tgt = unnorm(torch.cat(tgts).to(device))          # physical
day7 = unnorm(cond[:, -2:].clone())               # physical
mask = build_mask_tensor(device, 29)
m2 = mask[0].bool()
B = N_WIN


def sample(cond_batch, seed):
    torch.manual_seed(seed)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        return unnorm(model.sample(cond_batch, num_sample_steps=32, clamp=True).float())


preds = torch.cat([sample(cond[i:i + 1], 123 + starts[i]) for i in range(N_WIN)])

mu_r, mv_r = load_masks()
u_np = np.load(f"{ALIGNED_DIR}/u_rho.npy", mmap_mode="r")
v_np = np.load(f"{ALIGNED_DIR}/v_rho.npy", mmap_mode="r")
su = np.zeros((H, W)); sv = np.zeros((H, W))
for ts in range(0, 8401, 200):
    te = min(ts + 200, 8401)
    au = np.where(mu_r[None], u_np[ts:te, 29], np.nan)
    av = np.where(mv_r[None], v_np[ts:te, 29], np.nan)
    su += np.nansum(au, axis=0); sv += np.nansum(av, axis=0)
clim = torch.from_numpy(np.stack([su / 8401.0, sv / 8401.0]).astype(np.float32))[..., None].to(device)
climB = clim.expand(B, 2, H, W, Z)


def _flat(x, k):
    mk = m2[k].expand(x.shape[0], -1, -1, -1)
    return x[:, k][mk].reshape(-1)


def mstats(x):
    out = []
    for k in range(2):
        v = _flat(x, k)
        out.append((float(v.mean()), float(v.std())))
    return out


def rmse(a, b):
    """pooled RMSE over ocean cells of all windows (per-variable then pooled)."""
    per = []
    for k in range(2):
        d = _flat(a, k) - _flat(b, k)
        per.append(float(torch.sqrt((d ** 2).mean())))
    pu = _flat(a, 0) - _flat(b, 0); pv = _flat(a, 1) - _flat(b, 1)
    pooled = float(torch.sqrt(((pu ** 2).sum() + (pv ** 2).sum()) / (pu.numel() + pv.numel())))
    return per, pooled


def corr(a, b):
    out = []
    for k in range(2):
        va = _flat(a, k); vb = _flat(b, k)
        va = va - va.mean(); vb = vb - vb.mean()
        out.append(float((va * vb).sum() / (va.norm() * vb.norm())))
    return out


for name, x in (("pred", preds), ("tgt ", tgt), ("day7", day7), ("clim", climB)):
    s = mstats(x)
    print(f"{name}: u mean/std=({s[0][0]:+.4f},{s[0][1]:.4f})  v mean/std=({s[1][0]:+.4f},{s[1][1]:.4f})  [m/s]")

for name, a, b in (("pred  vs tgt ", preds, tgt), ("day7  vs tgt (persistence)", day7, tgt),
                   ("clim  vs tgt (climatology) ", climB, tgt), ("pred  vs clim", preds, climB),
                   ("pred  vs day7", preds, day7)):
    per, pooled = rmse(a, b)
    print(f"RMSE {name}: u={per[0]:.4f}  v={per[1]:.4f}  pooled={pooled:.4f}")

print(f"\ncorr(pred, tgt)  u/v: {[f'{c:.3f}' for c in corr(preds, tgt)]}")
print(f"corr(day7, tgt)  u/v: {[f'{c:.3f}' for c in corr(day7, tgt)]}")
print(f"corr(pred, day7) u/v: {[f'{c:.3f}' for c in corr(preds, day7)]}")
print(f"corr(pred, clim) u/v: {[f'{c:.3f}' for c in corr(preds, climB)]}")

# anomaly skill: does pred track the day8-vs-day7 change at all?
anom = tgt - day7
print(f"\nanomaly (tgt-day7) std u/v: {[f'{s:.4f}' for s in [float(_flat(anom,k).std()) for k in range(2)]]}")
print(f"corr(pred-day7, tgt-day7) u/v: {[f'{c:.3f}' for c in corr(preds - day7, anom)]}")

with torch.no_grad():
    print(f"\ntrainer-style relL2 (physical, per-window mean): "
          f"pred={masked_rel_l2(preds, tgt, mask):.3f}  day7={masked_rel_l2(day7, tgt, mask):.3f}  "
          f"zero={masked_rel_l2(torch.zeros_like(tgt), tgt, mask):.3f}")

pa = sample(cond[0:1], 777)
pb = sample(cond[1:2], 777)
print(f"\nE corr(pred(condA), pred(condB)) same seed u/v: {[f'{c:.3f}' for c in corr(pa, pb)]}")
per, _ = rmse(pa, pb)
print(f"E RMSE(predA, predB) same seed: u={per[0]:.4f} v={per[1]:.4f} pooled={rmse(pa,pb)[1]:.4f}")

mem = torch.cat([sample(cond[0:1], s) for s in (123, 7, 42, 2024)])
mm = mem.mean(0, keepdim=True).expand(4, 2, H, W, Z)
mstd = [float(_flat(mem - mm, k).std()) for k in range(2)]
print(f"F member spread (std around member-mean) u/v: {[f'{s:.4f}' for s in mstd]}")
print(f"F RMSE(mean-of-4, tgtA): {rmse(mm[:1] if mm.shape[0]==4 else mm, tgt[0:1])[1]:.4f}"
      f"  single-seed: {rmse(preds[0:1], tgt[0:1])[1]:.4f}")
