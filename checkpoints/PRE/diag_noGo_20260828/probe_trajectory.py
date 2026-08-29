"""Probe B3b: where does the sampling chain lose the conditional information?

The net denoises beautifully near the manifold (see probe_net_sensitivity):
D(y+0.34n, cond_true) has RMSE 0.044 m/s vs y. Yet the full 32-step Heun
sampler from sigma_max=80 produces ~0.258 m/s (mean field). This probe:

 1. large-sigma single-step: D(y + sigma*n, cond) for sigma up to 80
    (true / zero cond) — does the net stay conditional far off-manifold?
 2. basin-recovery: x0 = train-mean CONSTANT field (+ sigma*n) — does D snap
    to the conditional target or stay at the mean field?  Also x0 = day-7
    field (+ sigma*n) — does D move toward day-8 truth or copy day 7?
 3. full Heun trajectory log (formal config: 32 steps, churn=0, clamp=True,
    per-window seed 123+start): per-step sigma and ocean-masked RMSE of the
    Heun-averaged D vs the true target; final sample RMSE.
"""
import os, sys
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
WIN_IDX = [0, 40, 80, 120]

cfg = PRESETS[PRESET]
H, W, Z = 400, 441, 1
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=DEV).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=DEV).reshape(1, 2, 1, 1, 1)
sigma_data = float(2 * stats["sigma"])

dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
               num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
               ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
               hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True).to(DEV)
model = ElucidatedDiffusion(dm, channels=2, num_sample_steps=cfg["sampling_steps"],
                            image_size_h=H, image_size_w=W, image_size_z=Z,
                            sigma_data=sigma_data, S_churn=0)
ck = torch.load(CKPT, map_location=DEV, weights_only=True)
model.load_state_dict(ck["model_state_dict"])
model.sigma_data = float(ck["config"]["sigma_data"])
model.eval()

ds = PREUVDataset("val", {"lo": stats["lo"], "hi": stats["hi"]}, context=CONTEXT,
                  horizon=1, depth_index=cfg["depth_index"], stride=7)
mask_t = build_mask_tensor(DEV, cfg["depth_index"])
mask_b = (mask_t[0, :, :, :, 0] > 0)                    # (2,H,W) on DEV

conds, tgts, starts = [], [], []
for i in WIN_IDX:
    c, t, s = ds[i]
    conds.append(c.to(DEV)); tgts.append(t[0].to(DEV)); starts.append(int(s))
cond = torch.stack(conds)
tgt01 = torch.stack(tgts)
ym1 = tgt01 * 2 - 1
N = len(WIN_IDX)

# train-mean constant field in [-1,1] (from probe_cond_sanity: u .4849, v .6132)
mean01 = torch.tensor([0.4849, 0.6132], device=DEV).reshape(1, 2, 1, 1, 1)
mean_field = (mean01 * 2 - 1).expand(N, 2, H, W, Z).contiguous()
day7 = torch.stack([cond[:, -2], cond[:, -1]], dim=1)   # (N,2,H,W,Z) [-1,1] after *2-1
day7 = day7 * 2 - 1

def d_rmse(D):
    """ocean-masked physical RMSE of D([-1,1]) vs target, pooled u+v."""
    p01 = (D.clamp(-1, 1) + 1) * 0.5
    err = (p01 - tgt01) * (y_hi.reshape(1, 2, 1, 1, 1) - y_lo.reshape(1, 2, 1, 1, 1))
    err = err[..., 0]                                        # (N,2,H,W)
    m = mask_b.unsqueeze(0).expand_as(err)
    e = err[m]
    return float(e.pow(2).mean().sqrt())

@torch.no_grad()
def D_of(x_noisy, sig, c):
    return model.preconditioned_network_forward(x_noisy, sig, c).float()

print(f"sigma_data={sigma_data:.5f}  windows={starts}\n")

# ---------- 1. large-sigma single-step from the TRUE target ----------
print("=== 1. D(y + sigma*n, cond): true vs zero cond, sigma up to 80 ===")
print(f"{'sigma':>8} | {'true':>7} | {'zero':>7} | {'other':>7}")
torch.manual_seed(2000)
n1 = torch.randn_like(ym1)
for sig in [0.34, 0.86, 1.7, 3.4, 8.6, 17.1, 34.2, 80.0]:
    x = ym1 + sig * n1
    r = [d_rmse(D_of(x, sig, c)) for c in (cond, torch.zeros_like(cond), cond.roll(1, 0))]
    print(f"{sig:8.2f} | {r[0]:7.4f} | {r[1]:7.4f} | {r[2]:7.4f}")

# ---------- 2. basin recovery: start from mean field / day-7 ----------
print("\n=== 2. D(x0 + sigma*n, cond_true): can D escape the wrong basin? ===")
print("x0 = train-mean constant field (RMSE vs target at sigma=0: "
      f"{d_rmse(mean_field):.4f})")
print("x0 = day-7 field               (RMSE vs target at sigma=0: "
      f"{d_rmse(day7):.4f})")
print(f"{'sigma':>8} | {'from-mean':>9} | {'from-day7':>9} | {'from-truth':>10}")
torch.manual_seed(3000)
n2 = torch.randn_like(ym1)
for sig in [0.086, 0.17, 0.34, 0.86, 1.7, 3.4]:
    rows = []
    for x0 in (mean_field, day7, ym1):
        D = D_of(x0 + sig * n2, sig, cond)
        rows.append(d_rmse(D))
    print(f"{sig:8.3f} | {rows[0]:9.4f} | {rows[1]:9.4f} | {rows[2]:10.4f}")

# ---------- 3. full Heun trajectory (formal sampler config) ----------
print("\n=== 3. Heun trajectory, 32 steps, churn=0, clamp=True ===")
@torch.no_grad()
def heun_trace(x_seed_noise, c, y_target_m1, w_start):
    sigmas = model.sample_schedule(model.num_sample_steps)
    pairs = list(zip(sigmas[:-1], sigmas[1:]))
    x = x_seed_noise.clone()
    trace = []
    for step, (s0, s1) in enumerate(pairs):
        s0 = float(s0); s1 = float(s1)
        D = model.preconditioned_network_forward(x, s0, c, clamp=True).float()
        den = (x - D) / s0
        x_next = x + (s1 - s0) * den
        if s1 != 0:
            D2 = model.preconditioned_network_forward(x_next, s1, c, clamp=True).float()
            den2 = (x_next - D2) / s1
            x_next = x + 0.5 * (s1 - s0) * (den + den2)
        # per-window RMSE of D vs target (mean over windows for the log)
        p01 = (D.clamp(-1, 1) + 1) * 0.5
        t01 = (y_target_m1 + 1) * 0.5
        err = (p01 - t01) * (y_hi.reshape(1, 2, 1, 1, 1) - y_lo.reshape(1, 2, 1, 1, 1))
        err = err[..., 0]
        m = mask_b.unsqueeze(0).expand_as(err)
        e = err[m]
        x01 = (x.clamp(-1, 1) + 1) * 0.5
        errx = (x01 - t01) * (y_hi.reshape(1, 2, 1, 1, 1) - y_lo.reshape(1, 2, 1, 1, 1))
        errx = errx[..., 0]
        ex = errx[m]
        trace.append((s0, float(e.pow(2).mean().sqrt()), float(ex.pow(2).mean().sqrt())))
        x = x_next
    return x, trace

for w in range(N):
    torch.manual_seed(123 + starts[w])
    x0 = float(sigmas0 := model.sample_schedule(model.num_sample_steps)[0]) * torch.randn(
        1, 2, H, W, Z, device=DEV)
    x_fin, trace = heun_trace(x0, cond[w:w+1], ym1[w:w+1], starts[w])
    xs = [t[0] for t in trace]; rs = [t[1] for t in trace]; rx = [t[2] for t in trace]
    sel = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 31]
    print(f"window {starts[w]}: sigma:RMSE(D)/RMSE(x) along the trajectory:")
    print("   " + " ".join(f"{int(xs[i])}:{rs[i]:.3f}/{rx[i]:.3f}" for i in sel))
    # final sample RMSE (clamp + unnormalize like sample())
    fin = ((x_fin.clamp(-1, 1) + 1) * 0.5)
    err = (fin - tgt01[w:w+1]) * (y_hi.reshape(1, 2, 1, 1, 1) - y_lo.reshape(1, 2, 1, 1, 1))
    err = err[..., 0]
    m = mask_b.unsqueeze(0).expand_as(err)
    e = err[m]
    print(f"   FINAL sample RMSE = {float(e.pow(2).mean().sqrt()):.4f} m/s")
