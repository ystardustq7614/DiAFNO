"""Lead-time error diagnosis for the persistence-residual baseline (Ep10).

Replays the official test rollout protocol (same checkpoint / split / seed /
remask semantics as pre_evaluate.py) on a strided subset of windows and
accumulates per-lead-day native-grid statistics the evaluation NPZ does not
store: signed bias, pooled variance ratio (pred/truth; <1 = blur) and
per-window spatial pattern correlation, separately for the model and the
persistence baseline.

Script (module top-level, like pre_evaluate.py) — run from repo root:
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/diag_leadtime_residual.py

Outputs next to the checkpoint (refused if they already exist):
    leadtime_diag_ckpt<stem>.npz / leadtime_diag_ckpt<stem>.png
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from IAFNO import IAFNODiff
from pre_config import (CONTEXT, RESIDUAL_TIME_SIGMA, PRESETS,
                        check_norm_fingerprint)
from pre_dataset import (NativeUVReader, PREUVDataset, build_mask_tensor,
                         compute_or_load_stats, mask_version, native_masks)
from pre_metrics import rho_to_native
from pre_models import PersistenceResidualIAFNO
from pre_rollout import ensemble_mean, ensemble_rollout

PRESET = "surface_smoke"
CHECKPOINT = ("/data2/user/zyq/checkpoints/PRE/"
              "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/Ep10.pth")
SPLIT = "test"
ROLLOUT_DAYS = 15
EVAL_STRIDE = 14
BATCH_SIZE = 4
EVAL_SEED = 123
REMASK_FEEDBACK = False

torch.manual_seed(123)

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
ckpt_cfg = ckpt.get("config") or {}
if ckpt_cfg.get("objective") != "persistence_residual":
    raise RuntimeError(f"expected persistence_residual checkpoint, got "
                       f"{ckpt_cfg.get('objective')!r}")
stats = compute_or_load_stats(depth_index=cfg["depth_index"])
for fp_warning in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                         mask_version()):
    print(f"WARNING: {fp_warning}")

dm = IAFNODiff(dim=(H, W, Z), patch_size=cfg["patch_size"],
               embed_dim=cfg["embed_dim"], num_blocks=1, in_chans=2, out_chans=2,
               cond_chans=2 * CONTEXT, ex_layer=cfg["explicit_layer"],
               nlayer=cfg["implicit_layer"], hidden_size_factor=4,
               dim_f=(H, W, Z), self_condition=True).to(device)
model = PersistenceResidualIAFNO(
    dm, time_sigma=float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA)))
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
print(f"loaded {CHECKPOINT} (epoch={ckpt.get('epoch')}) objective="
      f"persistence_residual", flush=True)

y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)

ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]}, context=CONTEXT,
                  horizon=ROLLOUT_DAYS, depth_index=cfg["depth_index"],
                  stride=EVAL_STRIDE, max_windows=None)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=2, pin_memory=True)
print(f"{SPLIT} diagnosis windows: {len(ds)} (stride {EVAL_STRIDE}, "
      f"horizon {ROLLOUT_DAYS}, deterministic)", flush=True)

mask_u, mask_v = native_masks()
reader = NativeUVReader(cfg["depth_index"])
ocean_mask = (build_mask_tensor(device, cfg["depth_index"])
              if REMASK_FEEDBACK else None)

L = ROLLOUT_DAYS
FIELDS = ("n", "se", "se_signed", "sp", "sp2", "st", "st2")
acc = {f"{name}{var}": {k: np.zeros(L) for k in FIELDS}
       for name in ("m", "p") for var in ("u", "v")}
corr = {f"{name}{var}": [[] for _ in range(L)]
        for name in ("m", "p") for var in ("u", "v")}
window_starts = []


def accumulate(pred, truth, mask, store, corrs):
    """Accumulate masked per-lead-day moments; per-window correlation list.

    pred/truth: (B, L, H, W, Z=1) physical native grids; mask: (H, W) native.
    """
    m = np.asarray(mask, bool)[:, :, None]              # (H, W, 1)
    B = pred.shape[0]
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    e = np.where(m[None, None], pred - truth, 0.0)
    p = np.where(m[None, None], pred, 0.0)
    t = np.where(m[None, None], truth, 0.0)
    store["n"] += B * float(m.sum())
    store["se"] += (e ** 2).sum(axis=(0, 2, 3)).reshape(-1)
    store["se_signed"] += e.sum(axis=(0, 2, 3)).reshape(-1)
    store["sp"] += p.sum(axis=(0, 2, 3)).reshape(-1)
    store["sp2"] += (p ** 2).sum(axis=(0, 2, 3)).reshape(-1)
    store["st"] += t.sum(axis=(0, 2, 3)).reshape(-1)
    store["st2"] += (t ** 2).sum(axis=(0, 2, 3)).reshape(-1)
    mflat = m[:, :, 0]
    for b in range(B):
        for l in range(L):
            pv = pred[b, l, :, :, 0][mflat]
            tv = truth[b, l, :, :, 0][mflat]
            if pv.std() > 1e-12 and tv.std() > 1e-12:
                corrs[l].append(float(np.corrcoef(pv, tv)[0, 1]))


def finalize(store, corrs):
    n = store["n"]
    rmse = np.sqrt(store["se"] / n)
    bias = store["se_signed"] / n
    var_p = store["sp2"] / n - (store["sp"] / n) ** 2
    var_t = store["st2"] / n - (store["st"] / n) ** 2
    var_ratio = var_p / np.maximum(var_t, 1e-12)
    corr_mean = np.array([np.mean(c) if c else np.nan for c in corrs])
    corr_med = np.array([np.median(c) if c else np.nan for c in corrs])
    return dict(rmse=rmse, bias=bias, var_ratio=var_ratio,
                corr_mean=corr_mean, corr_med=corr_med, n=n)


t0 = time.perf_counter()
with torch.no_grad():
    for bi, (cond, target, starts) in enumerate(loader):
        cond = cond.to(device)                          # (B,14,H,W,Z) normalized
        starts_np = np.asarray(starts)
        window_starts.extend(int(s) for s in starts_np)

        preds = ensemble_rollout(model, cond, ROLLOUT_DAYS, 1,
                                 num_sample_steps=cfg["sampling_steps"],
                                 seeds=[EVAL_SEED + int(s) for s in starts_np],
                                 clamp=True,
                                 remask_feedback=REMASK_FEEDBACK,
                                 ocean_mask=ocean_mask)
        rho_pred = (ensemble_mean(preds) * (y_hi - y_lo) + y_lo).cpu().numpy()
        u_pred, v_pred = rho_to_native(rho_pred)

        tu, tv = [], []
        for s in starts_np:
            u_s, v_s = reader.get(int(s) + CONTEXT, ROLLOUT_DAYS)
            tu.append(u_s)
            tv.append(v_s)
        tu_t = np.stack(tu)
        tv_t = np.stack(tv)

        pu, pv = [], []
        for s in starts_np:
            u_s, v_s = reader.get(int(s) + CONTEXT - 1, 1)
            pu.append(u_s)
            pv.append(v_s)
        pu_t = np.broadcast_to(np.stack(pu),
                               (len(starts_np), ROLLOUT_DAYS, H, W - 1, Z))
        pv_t = np.broadcast_to(np.stack(pv),
                               (len(starts_np), ROLLOUT_DAYS, H - 1, W, Z))

        accumulate(u_pred, tu_t, mask_u, acc["mu"], corr["mu"])
        accumulate(v_pred, tv_t, mask_v, acc["mv"], corr["mv"])
        accumulate(pu_t, tu_t, mask_u, acc["pu"], corr["pu"])
        accumulate(pv_t, tv_t, mask_v, acc["pv"], corr["pv"])
        print(f"[{bi + 1}/{len(loader)}] windows_done="
              f"{min((bi + 1) * BATCH_SIZE, len(ds))}/{len(ds)} "
              f"elapsed_s={time.perf_counter() - t0:.0f}", flush=True)

res = {f"{name}{var}": finalize(acc[f"{name}{var}"], corr[f"{name}{var}"])
       for name in ("m", "p") for var in ("u", "v")}

out_dir = os.path.dirname(os.path.abspath(CHECKPOINT))
stem = os.path.splitext(os.path.basename(CHECKPOINT))[0]
npz_path = os.path.join(out_dir, f"leadtime_diag_ckpt{stem}.npz")
png_path = os.path.join(out_dir, f"leadtime_diag_ckpt{stem}.png")
for p in (npz_path, png_path):
    if os.path.exists(p):
        raise RuntimeError(f"{p} already exists; delete it or change CHECKPOINT")

for var in ("u", "v"):
    mu, pu = res[f"m{var}"], res[f"p{var}"]
    print(f"\n=== {var}: per-lead-day native stats over {len(ds)} windows ===")
    print("lead | rmse_m | rmse_p | ratio | bias_m | bias_p | var_ratio_m | "
          "corr_m | corr_p")
    for l in range(L):
        print(f" {l + 1:2d}  | {mu['rmse'][l]:.4f} | {pu['rmse'][l]:.4f} | "
              f"{mu['rmse'][l] / pu['rmse'][l]:.3f} | {mu['bias'][l]:+.4f} | "
              f"{pu['bias'][l]:+.4f} | {mu['var_ratio'][l]:.3f} | "
              f"{mu['corr_mean'][l]:.3f} | {pu['corr_mean'][l]:.3f}")

m_all = np.sqrt(res["mu"]["rmse"] ** 2 + res["mv"]["rmse"] ** 2)
p_all = np.sqrt(res["pu"]["rmse"] ** 2 + res["pv"]["rmse"] ** 2)
ratio = m_all / p_all
cross = next((l + 1 for l in range(L) if ratio[l] > 1.0), None)
print(f"\npooled ratio per lead: {[round(r, 3) for r in ratio]}")
print(f"crossover day (pooled ratio first > 1): {cross}")
print(f"var_ratio_m @ d1/d7/d15 (u): "
      f"{[round(res['mu']['var_ratio'][l], 3) for l in (0, 6, 14)]}")
print(f"corr_mean_m @ d1/d7/d15 (u): "
      f"{[round(res['mu']['corr_mean'][l], 3) for l in (0, 6, 14)]}")

np.savez(npz_path,
         lead=np.arange(1, L + 1),
         **{f"{k}_{var}": res[f"{name}{var}"][k]
            for var in ("u", "v")
            for name in ("m", "p")
            for k in res[f"{name}{var}"]},
         checkpoint_path=np.str_(CHECKPOINT), split=np.str_(SPLIT),
         stride=np.int64(EVAL_STRIDE), n_windows=np.int64(len(ds)),
         rollout_days=np.int64(ROLLOUT_DAYS), eval_seed=np.int64(EVAL_SEED),
         remask_feedback=np.bool_(REMASK_FEEDBACK),
         objective=np.str_("persistence_residual"),
         window_start_indices=np.array(window_starts, np.int64))
print(f"\nsaved {npz_path}", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
lead = np.arange(1, L + 1)
ax = axes[0, 0]
ax.plot(lead, m_all, "o-", label="model")
ax.plot(lead, p_all, "s--", label="persistence")
ax2 = ax.twinx()
ax2.plot(lead, ratio, "^:", color="gray", alpha=0.7, label="model/pers")
ax2.axhline(1.0, color="gray", lw=0.5)
if cross is not None:
    ax2.axvline(cross, color="red", lw=0.8, alpha=0.6)
ax2.set_ylabel("model/pers ratio")
ax.set_title("Pooled native RMSE (u+v) and ratio")
ax.set_xlabel("lead day")
ax.legend(loc="upper left")
ax2.legend(loc="lower right")
for ax, key, title in ((axes[0, 1], "bias", "Signed bias (m/s)"),
                       (axes[1, 0], "var_ratio", "Variance ratio (pred/truth)"),
                       (axes[1, 1], "corr_mean", "Per-window spatial corr")):
    ax.plot(lead, res["mu"][key], "o-", label=f"model u")
    ax.plot(lead, res["mv"][key], "o-", label=f"model v")
    ax.plot(lead, res["pu"][key], "s--", alpha=0.6, label=f"pers u")
    ax.plot(lead, res["pv"][key], "s--", alpha=0.6, label=f"pers v")
    if key == "var_ratio":
        ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("lead day")
    ax.legend(fontsize=8)
fig.suptitle(f"Lead-time diagnosis — persistence_residual {stem} "
             f"({SPLIT}, {len(ds)} windows, stride {EVAL_STRIDE})")
fig.tight_layout()
fig.savefig(png_path, dpi=140)
print(f"saved {png_path}")
print("PROGRESS phase=diag status=completed")
