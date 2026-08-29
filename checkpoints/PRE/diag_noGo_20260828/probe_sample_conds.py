"""Probe B1 (decisive) + B4 (statistics): full formal-style day-1 sampling with
four condition variants on the same windows/seeds as the formal val eval.

Variants:
  a_true : the dataset condition (formal path)
  b_zero : condition set to all zeros
  c_other: condition of a DIFFERENT window (fixed rng shuffle, wrong content)
  d_rev  : channel order reversed (day order destroyed, content kept)

Sampling config == formal val h1 eval: 32 Heun steps, S_churn=0, clamp=True,
per-window seed = 123 + start_day, autocast, rho rollout -> rho_to_native ->
native masked RMSE. Also captures fields for mean-field statistics.
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, "/data2/user/zyq/projects/DiAFNO")
from pre_config import PRESETS, CONTEXT
from pre_dataset import (PREUVDataset, compute_or_load_stats, NativeUVReader,
                         native_masks, load_masks)
from pre_metrics import rho_to_native, masked_error_sums, pooled_rmse
from pre_rollout import ensemble_rollout
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff

ap = argparse.ArgumentParser()
ap.add_argument("--max-windows", type=int, default=None)
ap.add_argument("--window-stride", type=int, default=1)
ap.add_argument("--conds", type=str, default="a,b,c,d")
ap.add_argument("--out", type=str, default="/tmp/opencode/diag/results/probe_sample_conds.npz")
args = ap.parse_args()

PRESET = "surface_smoke"
CKPT = "/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/Ep3.pth"
DEV = torch.device("cuda")
EVAL_SEED = 123

cfg = PRESETS[PRESET]
H, W, Z = 400, 441, 1
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo_np = np.asarray(stats["lo"], np.float32).reshape(1, 1, 2, 1, 1, 1)
y_hi_np = np.asarray(stats["hi"], np.float32).reshape(1, 1, 2, 1, 1, 1)
rng_scale = (y_hi_np - y_lo_np).reshape(2, 1, 1)      # (2,1,1) physical range per var

dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
               num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
               ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
               hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True).to(DEV)
model = ElucidatedDiffusion(dm, channels=2, num_sample_steps=cfg["sampling_steps"],
                            image_size_h=H, image_size_w=W, image_size_z=Z,
                            sigma_data=2 * float(stats["sigma"]), S_churn=0)
ck = torch.load(CKPT, map_location=DEV, weights_only=True)
model.load_state_dict(ck["model_state_dict"])
model.sigma_data = float(ck["config"]["sigma_data"])
model.eval()

ds = PREUVDataset("val", {"lo": stats["lo"], "hi": stats["hi"]}, context=CONTEXT,
                  horizon=1, depth_index=cfg["depth_index"], stride=7)
n_all = len(ds)
sel = list(range(0, n_all, args.window_stride))
if args.max_windows:
    sel = sel[:args.max_windows]
print(f"windows: {len(sel)}/{n_all} (window-stride {args.window_stride})")

mask_u_nat, mask_v_nat = native_masks()
reader = NativeUVReader(cfg["depth_index"])
mu_rho, mv_rho = load_masks()

variants = [v for v in args.conds.split(",") if v]
se = {v: np.zeros((1, 2, 1)) for v in variants}
se_p = np.zeros((1, 2, 1)); se_z = np.zeros((1, 2, 1))
cnt = np.zeros((1, 2, 1))
cnt[0, 0, 0] = mask_u_nat.sum() * len(sel); cnt[0, 1, 0] = mask_v_nat.sum() * len(sel)

rng = np.random.default_rng(0)
perm = rng.permutation(n_all)                    # for c_other

cap = {v: [] for v in variants}
cap_t, cap_pers, cap_start = [], [], []

t0 = time.time()
for k, i in enumerate(sel):
    cond, tgt, s = ds[i]
    s = int(s)
    cond = cond.unsqueeze(0).to(DEV)             # (1,14,H,W,Z)
    cv = {}
    if "a" in variants: cv["a"] = cond
    if "b" in variants: cv["b"] = torch.zeros_like(cond)
    if "c" in variants: cv["c"] = ds[int(perm[i])][0].unsqueeze(0).to(DEV)
    if "d" in variants: cv["d"] = cond.flip(1)
    seeds = [EVAL_SEED + s]
    for v, c in cv.items():
        preds = ensemble_rollout(model, c, 1, 1, num_sample_steps=cfg["sampling_steps"],
                                 seeds=seeds, clamp=True)
        rho = preds[:, 0].float().cpu().numpy()                  # (1,2,H,W,Z)
        phys = rho * (y_hi_np - y_lo_np) + y_lo_np
        u_nat, v_nat = rho_to_native(phys)
        tu, tv = reader.get(s + CONTEXT, 1)
        tu = tu[None]; tv = tv[None]
        seu, _ = masked_error_sums(u_nat, tu, mask_u_nat)
        sev, _ = masked_error_sums(v_nat, tv, mask_v_nat)
        se[v][0, 0, 0] += seu.sum(); se[v][0, 1, 0] += sev.sum()
        cap[v].append(rho[0, 0].astype(np.float16))  # (2,H,W,Z)
    # persistence (day-7 rho copy) and zero on the same native truth
    rho_pers = np.stack([cond[0, -2].cpu().numpy(), cond[0, -1].cpu().numpy()])[None]
    phys = rho_pers * (y_hi_np - y_lo_np) + y_lo_np
    u_nat, v_nat = rho_to_native(phys)
    seu, _ = masked_error_sums(u_nat, tu, mask_u_nat)
    sev, _ = masked_error_sums(v_nat, tv, mask_v_nat)
    se_p[0, 0, 0] += seu.sum(); se_p[0, 1, 0] += sev.sum()
    seu, _ = masked_error_sums(np.zeros_like(tu), tu, mask_u_nat)
    sev, _ = masked_error_sums(np.zeros_like(tv), tv, mask_v_nat)
    se_z[0, 0, 0] += seu.sum(); se_z[0, 1, 0] += sev.sum()
    cap_t.append(tgt[0].numpy().astype(np.float16))              # (2,H,W,Z)
    cap_pers.append(rho_pers[0].astype(np.float16))
    cap_start.append(s)
    if (k + 1) % 10 == 0 or k + 1 == len(sel):
        el = time.time() - t0
        print(f"  [{k+1}/{len(sel)}] {el/(k+1):.1f}s/win  elapsed {el:.0f}s", flush=True)

print("\n=== pooled native masked RMSE (m/s), day-1, same windows ===")
res = {"starts": np.array(cap_start), "variants": np.array(variants, dtype=object)}
for v in variants:
    r = pooled_rmse(se[v], cnt)
    print(f"  {v:8s}: {r:.4f}")
    res[f"rmse_{v}"] = np.array([r])
rp = pooled_rmse(se_p, cnt); rz = pooled_rmse(se_z, cnt)
print(f"  pers    : {rp:.4f}\n  zero    : {rz:.4f}")
res["rmse_pers"] = np.array([rp]); res["rmse_zero"] = np.array([rz])

# ---------------- B4 statistics (rho grid, ocean cells) ----------------
def ocean(a, m):
    return a[..., 0][m].astype(np.float32)

print("\n=== field statistics (rho grid, per captured window) ===")
for v in variants + ["pers"]:
    fields = cap_pers if v == "pers" else cap[v]
    stds, cor_t, cor_p = [], [], []
    for j in range(len(sel)):
        f32 = fields[j].astype(np.float32); t32 = cap_t[j].astype(np.float32)
        p32 = cap_pers[j].astype(np.float32)
        for c, m in ((0, mu_rho), (1, mv_rho)):
            pv = ocean(f32[c], m); tv = ocean(t32[c], m); pe = ocean(p32[c], m)
            stds.append(pv.std())
            if pv.std() > 1e-6:
                cor_t.append(np.corrcoef(pv, tv)[0, 1])
                cor_p.append(np.corrcoef(pv, pe)[0, 1])
    print(f"  {v:8s}: mean spatial std(pred)={np.mean(stds):.4f} m/s | "
          f"corr(pred,truth)={np.nanmean(cor_t):.3f} | corr(pred,persistence)={np.nanmean(cor_p):.3f}")

# error split open-ocean vs coastal band (8-pixel erosion of the rho masks)
from scipy.ndimage import binary_erosion
inner = {0: binary_erosion(mu_rho, iterations=8), 1: binary_erosion(mv_rho, iterations=8)}
masks = {0: mu_rho, 1: mv_rho}
print("\n=== error split (rho grid, m/s) ===")
for v in variants:
    errs_in, errs_co = [], []
    for j in range(len(sel)):
        f32 = cap[v][j].astype(np.float32); t32 = cap_t[j].astype(np.float32)
        for c in (0, 1):
            e = (f32[c][..., 0] - t32[c][..., 0]) * float(rng_scale[c, 0, 0])
            m, inn = masks[c], inner[c]
            errs_in.append(e[m & inn]); errs_co.append(e[m & ~inn])
    ein = float(np.sqrt((np.concatenate(errs_in).astype(np.float64) ** 2).mean()))
    eco = float(np.sqrt((np.concatenate(errs_co).astype(np.float64) ** 2).mean()))
    print(f"  {v:8s}: RMSE open-ocean={ein:.4f}  coastal-band={eco:.4f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, **res)
print(f"saved {args.out}")
