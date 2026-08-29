"""Probe B3+: net-level condition sensitivity + low-noise reconstruction.

Loads Ep3 exactly like pre_evaluate.py (same IAFNODiff/ElucidatedDiffusion
construction, sigma_data from checkpoint config), then for a few val windows
and several noise levels sigma compares the EDM denoiser output
D = c_skip*x_noisy + c_out*F_theta(c_in*x_noisy, c_noise, cond) across
condition variants:
  (a) true cond          (b) all-zero cond
  (c) cond of ANOTHER val window (wrong content, right distribution)
  (d) channel-order-reversed cond (day order destroyed)
Reports masked physical RMSE of D vs the true target (ocean cells, m/s).
If (a)<<(b)/(c)/(d) the net uses the condition; if all similar it does not.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/data2/user/zyq/projects/DiAFNO")
from pre_config import PRESETS, CONTEXT
from pre_dataset import PREUVDataset, compute_or_load_stats, build_mask_tensor
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff

PRESET = "surface_smoke"
CKPT = "/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/Ep3.pth"
DEV = torch.device("cuda")
WIN_IDX = [0, 40, 80, 120]          # spread over the 156 stride-7 val windows
SIGMAS = [0.02, 0.05, 0.1, 0.5, 2.0]  # x sigma_data ([-1,1] image space)

cfg = PRESETS[PRESET]
H, W, Z = 400, 441, 1
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=DEV).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=DEV).reshape(1, 2, 1, 1, 1)
sigma_data = float(torch.tensor(stats["sigma"])) * 2.0

dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
               num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
               ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
               hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True).to(DEV)
model = ElucidatedDiffusion(dm, channels=2, num_sample_steps=cfg["sampling_steps"],
                            image_size_h=H, image_size_w=W, image_size_z=Z,
                            sigma_data=sigma_data, S_churn=0)
ck = torch.load(CKPT, map_location=DEV, weights_only=True)
model.load_state_dict(ck["model_state_dict"])
model.sigma_data = float(ck["config"]["sigma_data"]); sigma_data = model.sigma_data
model.eval()
print(f"loaded {CKPT} sigma_data={sigma_data:.5f}")

ds = PREUVDataset("val", {"lo": stats["lo"], "hi": stats["hi"]}, context=CONTEXT,
                  horizon=1, depth_index=cfg["depth_index"], stride=7)
mask_t = build_mask_tensor(DEV, cfg["depth_index"])            # (1,2,H,W,Z)
mask_np = mask_t[0, :, :, :, 0].cpu().numpy().astype(bool)     # (2,H,W)

def phys_rmse(pred_m1, tgt01):
    """pred in [-1,1], tgt in [0,1] -> per-var masked RMSE (m/s)."""
    p01 = (pred_m1.clamp(-1, 1) + 1) * 0.5
    err = (p01 - tgt01) * (y_hi.reshape(2, 1, 1, 1) - y_lo.reshape(2, 1, 1, 1))
    err = err.squeeze(-1)                                      # (B,2,H,W)
    out = []
    for v in range(2):
        m = torch.from_numpy(mask_np[v]).to(DEV)
        e = err[:, v][:, m]
        out.append(float(e.pow(2).mean().sqrt()))
    return out

conds, tgts, starts = [], [], []
for i in WIN_IDX:
    c, t, s = ds[i]
    conds.append(c.to(DEV)); tgts.append(t[0].to(DEV)); starts.append(int(s))
cond_true = torch.stack(conds)                                 # (N,14,H,W,Z)
tgt01 = torch.stack(tgts)                                      # (N,2,H,W,Z)
ym1 = tgt01 * 2 - 1
N = len(WIN_IDX)

cond_zero = torch.zeros_like(cond_true)
cond_other = cond_true.roll(1, dims=0)                         # wrong window content
cond_rev = cond_true.flip(1)                                   # day order reversed
variants = {"a_true": cond_true, "b_zero": cond_zero,
            "c_other": cond_other, "d_rev": cond_rev}

print(f"\nwindows {starts}  N={N}  sigma_data={sigma_data:.4f}")
hdr = "sigma(xsd) | sigma_abs | " + " | ".join(f"{k:8s}" for k in variants)
print(hdr); print("-" * len(hdr))
torch.manual_seed(1000)
noise = torch.randn_like(ym1)
rows = []
for f in SIGMAS:
    sig = f * sigma_data
    x_noisy = ym1 + sig * noise
    line = [f"{f:9.2f} | {sig:9.4f}"]
    rec = {"sigma": f}
    for k, cv in variants.items():
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            D = model.preconditioned_network_forward(x_noisy, sig, cv)
        rmse = phys_rmse(D.float(), tgt01)
        rec[k] = rmse
        line.append(f"  {rmse[0]:.3f}/{rmse[1]:.3f}")
    rows.append(rec)
    print(" | ".join(line))

print("\n(columns: u RMSE / v RMSE, m/s, ocean-masked)")
print("persistence-style reference: day7-copy RMSE computed below")
for i in range(N):
    c = cond_true[i:i+1]; t = tgt01[i:i+1]
    pers = torch.stack([c[0, -2], c[0, -1]])[None]
    r = phys_rmse((pers * 2 - 1), t)
    print(f"  window {starts[i]}: persistence(day7 copy) u={r[0]:.3f} v={r[1]:.3f}")
