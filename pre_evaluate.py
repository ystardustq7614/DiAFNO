#!/usr/bin/env python3
"""PRE_ocean_data evaluation: autoregressive rollout vs persistence baseline.

FORMAL metrics are computed on the NATIVE staggered u/v grids against the
UNCLIPPED raw physical truth (raw u.npy/v.npy), using the native mask_u/mask_v:

    rho u -> native u:  average the two adjacent rho points along xi: (400, 440)
    rho v -> native v:  average the two adjacent rho points along eta: (399, 441)
    (inverse of the Plan A colocation stencil, one-sided-free: no rotation)

Pipeline per test window (CONTEXT + ROLLOUT_DAYS consecutive days):
    1. (optional) ensemble rollout on the rho grid: predict next day from the
       current 7-day condition, shift the window (drop oldest day, append
       prediction) x ROLLOUT_DAYS; each ensemble member runs an independent
       trajectory and predictions are averaged over members at the end;
    2. map each rho prediction back to the native u/v grids (rho_to_native);
    3. compare with the raw native truth of days 8..8+ROLLOUT_DAYS-1
       (unclipped, land=NaN) read via a single NativeUVReader;
    4. persistence baseline = repeat the day-7 NATIVE physical u/v ROLLOUT_DAYS
       times (never the clipped/normalized condition input);
    5. diagnostic baselines: zero-current (all-zero native prediction) and
       rho-oracle (the dataset's real rho target, denormalized and mapped with
       the same rho_to_native stencil — measures the conversion's irreversible
       error alone);
    6. masked RMSE/MAE per lead day (1..ROLLOUT_DAYS) x variable (u,v) x layer.

Overall RMSE = sqrt(sum(squared_error) / sum(valid_count)) — NOT the arithmetic
mean of per-layer RMSEs; the console summary pools the same way (pooled_rmse).

Sampling is fully configurable (ROLLOUT_DAYS, ENSEMBLE_SIZE, SAMPLER_S_CHURN,
SAMPLER_SIGMA_MAX, EVAL_SEED) and runs under CUDA AMP (autocast) exactly like the historical
evaluation path. Each window is seeded by its OWN start day (EVAL_SEED + start),
so trajectories are reproducible AND independent of batch size / batching.
Every output file/dir carries a tag with the sampling config + checkpoint stem
and existing outputs are REFUSED, never overwritten. sigma_data is read from
the checkpoint's config when present; legacy checkpoints fall back to the old
stats-only scale with an explicit warning.

Output: <ckpt_dir>/eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_rf{0|1}_ckpt{stem}[_sm{sigma_max}][_{tag}].npz
        <ckpt_dir>/figures_h{rd}_ch{churn}_e{es}_s{seed}_rf{0|1}_ckpt{stem}[_sm{sigma_max}][_{tag}]/d{...}_*.png

Run (from repo root):  python pre_evaluate.py

The checkpoint's `config.objective` decides which model is rebuilt:
    "diffusion"            -> conditional EDM + Heun sampler (legacy default for
                              checkpoints that predate the objective field)
    "persistence_residual" -> deterministic PersistenceResidualIAFNO; sampler
                              parameters are NOT applicable and are recorded as
                              such in the output metadata; ENSEMBLE_SIZE is
                              forced to 1 (members would be identical).
`REMASK_FEEDBACK` optionally re-applies the ocean mask (land -> 0) to every
prediction before it re-enters the next condition window; the setting is part
of the output tag (rf0/rf1) and metadata, and its final value is decided by the
Phase-5 A/B. Terminal progress follows the shared pre_config.ProgressReporter
convention (interactive bar + parseable PROGRESS status lines).
"""
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_models import PersistenceResidualIAFNO
from pre_config import (PRESETS, OUT_ROOT, CONTEXT, run_tag_for, sigma_data_from_stats,
                        sigma_data_from_checkpoint, RESIDUAL_TIME_SIGMA,
                        objective_from_checkpoint, ProgressReporter, format_progress,
                        install_progress_failure_hook, mark_progress_failed,
                        check_norm_fingerprint)
from pre_dataset import (PREUVDataset, NativeUVReader, native_masks,
                         compute_or_load_stats, build_mask_tensor, mask_version)
from pre_metrics import (rho_to_native, masked_error_sums, pooled_rmse,
                         oracle_native_error_sums)
from pre_rollout import ensemble_rollout, ensemble_mean

torch.manual_seed(123)

# Standard status=failed line for exceptions that escape the guarded rollout
# block; the stage field tracks the current script section so failures report
# WHERE they happened (setup -> data_model -> rollout -> postprocess)
EVAL_STAGE = ["setup"]
install_progress_failure_hook("eval", stage=lambda: EVAL_STAGE[0])

########## eval config ##########

PRESET = "surface_smoke"            # must match the trained checkpoint's preset
CHECKPOINT = None                   # None -> <run_dir>/best.pth (run_dir from run_tag_for)
SPLIT = "test"
ROLLOUT_DAYS = 15                   # dataset horizon, rollout steps, metric horizon, fig days
ENSEMBLE_SIZE = 1                   # independent rollout members averaged at the end
SAMPLER_S_CHURN = 0                 # selected by the surface SD2 validation ablation
SAMPLER_SIGMA_MAX = None            # None -> ElucidatedDiffusion default (80)
EVAL_SEED = 123                     # per-window rollout seed (EVAL_SEED + absolute start_day)
OUTPUT_TAG = None                   # extra suffix appended to output dirs/files
EVAL_STRIDE = 7                     # start a rollout window every N days
MAX_WINDOWS = None                  # set small (e.g. 8) for a quick check
BATCH_SIZE = 4                      # rollout batch; use 1 for full3d if OOM
SAMPLING_STEPS = None               # None -> preset value
FIG_DAYS = (1, 3, 5, 7, 10, 15)     # representative lead days (filtered by ROLLOUT_DAYS)
REMASK_FEEDBACK = False             # True: re-apply the ocean mask (land -> 0) to
                                    # every prediction BEFORE it re-enters the next
                                    # condition window (default False = historical
                                    # unmasked feedback; final value via Phase-5 A/B)

cfg = PRESETS[PRESET]
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = os.path.join(OUT_ROOT, run_tag_for(PRESET))
ckpt_path = CHECKPOINT or os.path.join(run_dir, "best.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

# one weights_only load: the checkpoint's config.objective decides which model
# class is rebuilt (legacy checkpoints without the field are always diffusion)
try:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
except Exception as e:
    raise RuntimeError(
        f"failed to load {ckpt_path} with weights_only=True ({type(e).__name__}: {e}); "
        f"only pass weights_only=False for a verified project checkpoint") from e
ckpt_epoch = ckpt.get("epoch", None)
ckpt_cfg = ckpt.get("config") or {}
OBJECTIVE = objective_from_checkpoint(ckpt)
if "preset" in ckpt_cfg and ckpt_cfg["preset"] != PRESET:
    raise RuntimeError(f"checkpoint preset={ckpt_cfg['preset']!r} vs PRESET={PRESET!r}; "
                       "evaluation must use the trained preset")
RESIDUAL_BASE = "last_day" if OBJECTIVE == "persistence_residual" else "n/a"
if OBJECTIVE == "persistence_residual" and ENSEMBLE_SIZE != 1:
    print(f"NOTE: deterministic objective ignores ENSEMBLE_SIZE={ENSEMBLE_SIZE}; "
          "members would be identical -> using 1")
    ENSEMBLE_SIZE = 1

# outputs always live next to the checkpoint and are tagged by the sampling
# config AND the checkpoint file stem, so no existing eval file or figure dir
# is ever overwritten; an already-existing output path is refused. rf{0,1}
# records the remask_feedback A/B setting so both arms can coexist.
out_dir = os.path.dirname(os.path.abspath(ckpt_path))
ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
tag_parts = [f"h{ROLLOUT_DAYS}", f"ch{SAMPLER_S_CHURN}", f"e{ENSEMBLE_SIZE}",
             f"s{EVAL_SEED}", f"rf{int(bool(REMASK_FEEDBACK))}", f"ckpt{ckpt_stem}"]
if SAMPLER_SIGMA_MAX is not None:
    tag_parts.append(f"sm{SAMPLER_SIGMA_MAX:g}")
if OUTPUT_TAG:
    tag_parts.append(OUTPUT_TAG)
tag = "_".join(tag_parts)
out_path = os.path.join(out_dir, f"eval_{SPLIT}_{tag}.npz")
fig_dir = os.path.join(out_dir, f"figures_{tag}")
if os.path.exists(out_path):
    raise RuntimeError(f"{out_path} already exists; delete it or set OUTPUT_TAG "
                       f"to a new name before re-running this configuration")
if os.path.isdir(fig_dir) and any(os.scandir(fig_dir)):
    raise RuntimeError(f"{fig_dir} is not empty; delete it or set OUTPUT_TAG "
                       f"to a new name before re-running this configuration")
os.makedirs(fig_dir, exist_ok=True)

########## model ##########

stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)
# the denormalization and mask semantics must be the ones the checkpoint was
# trained with — refuse rather than report silently wrong physical numbers
for fp_warning in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                         mask_version()):
    print(f"WARNING: {ckpt_path}: {fp_warning}")
EVAL_STAGE[0] = "data_model"

dm_backbone = IAFNODiff(
    dim=(H, W, Z), patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
    num_blocks=1, in_chans=2, out_chans=2, cond_chans=2 * CONTEXT,
    ex_layer=cfg["explicit_layer"], nlayer=cfg["implicit_layer"],
    hidden_size_factor=4, dim_f=(H, W, Z), self_condition=True,
).to(device)
if OBJECTIVE == "diffusion":
    # sigma_data is a plain attribute (not in the state_dict): construct with the
    # current-scale value, load weights, then resolve the authoritative value from
    # the checkpoint (new checkpoints store it; legacy ones fall back to the old
    # stats-only scale with an explicit notice).
    model = ElucidatedDiffusion(
        dm_backbone, channels=2,
        num_sample_steps=SAMPLING_STEPS or cfg["sampling_steps"],
        image_size_h=H, image_size_w=W, image_size_z=Z,
        sigma_data=sigma_data_from_stats(stats["sigma"]),
        S_churn=SAMPLER_S_CHURN,
    )
else:
    model = PersistenceResidualIAFNO(
        dm_backbone,
        time_sigma=float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA)))
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
if OBJECTIVE == "diffusion":
    sigma_data, sd_in_ckpt = sigma_data_from_checkpoint(ckpt, stats["sigma"])
    if not sd_in_ckpt:
        print(f"WARNING: {ckpt_path} has no config.sigma_data (legacy checkpoint); "
              f"using the OLD scale sigma_data = stats sigma = {sigma_data:.5f}")
    model.sigma_data = sigma_data
    if SAMPLER_SIGMA_MAX is not None:
        model.sigma_max = SAMPLER_SIGMA_MAX
    model.eval()
    print(f"loaded {ckpt_path} (epoch={ckpt_epoch}) objective=diffusion  "
          f"sigma_data={sigma_data:.5f}  S_churn={SAMPLER_S_CHURN}  "
          f"sigma_max={model.sigma_max}")
else:
    model.eval()
    print(f"loaded {ckpt_path} (epoch={ckpt_epoch}) objective={OBJECTIVE}  "
          f"residual_base={model.residual_base}  "
          f"time_sigma={model.time_sigma:g}  "
          f"(sampler parameters not applicable; remask_feedback={REMASK_FEEDBACK})")

########## data ##########

eval_ds = PREUVDataset(SPLIT, {"lo": stats["lo"], "hi": stats["hi"]},
                       context=CONTEXT, horizon=ROLLOUT_DAYS,
                       depth_index=cfg["depth_index"], stride=EVAL_STRIDE,
                       max_windows=MAX_WINDOWS)
eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=2, pin_memory=True)
print(f"{SPLIT} rollout windows: {len(eval_ds)} (stride {EVAL_STRIDE}, "
      f"horizon {ROLLOUT_DAYS}, ensemble {ENSEMBLE_SIZE})")

mask_u, mask_v = native_masks()                       # native staggered grids
reader = NativeUVReader(cfg["depth_index"])           # single reader, unified layout

# rho-grid bivariate ocean mask (1 = ocean) for the rollout feedback loop;
# only materialized when remasking is enabled
ocean_mask = (build_mask_tensor(device, cfg["depth_index"])
              if REMASK_FEEDBACK else None)
if REMASK_FEEDBACK:
    print("remask_feedback=ON: every rollout prediction is re-masked "
          "(land -> 0) before re-entering the next condition window")

########## rollout + metrics ##########

def unnormalize(x):
    return x * (y_hi - y_lo) + y_lo


# native accumulators: (ROLLOUT_DAYS, 2, Z) — model / persistence / zero / oracle
se_m = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_m = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_p = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_p = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_z = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_z = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
se_o = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
ae_o = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)
n_count = np.zeros((ROLLOUT_DAYS, 2, Z), np.float64)

window_starts = []
fig_capture = None

# the reporter counts actual WINDOWS (not loader batches): total is the
# dataset window count, each update advances by the batch's true window count
# (the final batch may be partial), and sample/s counts lead-day samples
eval_rep = ProgressReporter(
    "eval", total=len(eval_ds), unit="window",
    samples_per_unit=ROLLOUT_DAYS,
    context={"split": SPLIT, "objective": OBJECTIVE})
EVAL_STAGE[0] = "rollout"
eval_t0 = time.perf_counter()
w_done = 0
try:
    with torch.no_grad():
        for bi, (cond, target, starts) in enumerate(eval_loader):
            cond = cond.to(device)                      # (B,14,H,W,Z) normalized
            target_np = target.numpy()                  # (B,L,2,H,W,Z) normalized rho targets

            # --- ensemble rollout on the rho grid (members averaged at the end).
            #     Each window is seeded by ITS OWN start day (EVAL_SEED + start),
            #     so trajectories are reproducible AND independent of the batch
            #     size / loader batching (see pre_rollout.ensemble_rollout).
            starts_np = np.asarray(starts)
            window_starts.extend(int(s) for s in starts_np)
            preds = ensemble_rollout(model, cond, ROLLOUT_DAYS, ENSEMBLE_SIZE,
                                     num_sample_steps=SAMPLING_STEPS or cfg["sampling_steps"],
                                     seeds=[EVAL_SEED + int(s) for s in starts_np],
                                     clamp=True,
                                     remask_feedback=REMASK_FEEDBACK,
                                     ocean_mask=ocean_mask)
            rho_pred = unnormalize(ensemble_mean(preds)).cpu().numpy()  # (B,L,2,H,W,Z)

            # --- fixed rho -> native resampling (no rotation)
            u_pred, v_pred = rho_to_native(rho_pred)    # (B,L,H,W-1,Z), (B,L,H-1,W,Z)

            # --- UNCLIPPED native truth: days [s+7, s+7+ROLLOUT_DAYS)
            tu_parts, tv_parts = [], []
            for s in starts_np:
                u_s, v_s = reader.get(int(s) + CONTEXT, ROLLOUT_DAYS)
                tu_parts.append(u_s)
                tv_parts.append(v_s)
            tu_t = np.stack(tu_parts)                   # (B,L,H,W-1,Z)
            tv_t = np.stack(tv_parts)                   # (B,L,H-1,W,Z)

            se_u, ae_u = masked_error_sums(u_pred, tu_t, mask_u)   # model
            se_v, ae_v = masked_error_sums(v_pred, tv_t, mask_v)
            se_m[:, 0, :] += se_u
            ae_m[:, 0, :] += ae_u
            se_m[:, 1, :] += se_v
            ae_m[:, 1, :] += ae_v

            # --- zero-current baseline: all-zero native prediction
            se_u, ae_u = masked_error_sums(np.zeros_like(tu_t), tu_t, mask_u)
            se_v, ae_v = masked_error_sums(np.zeros_like(tv_t), tv_t, mask_v)
            se_z[:, 0, :] += se_u
            ae_z[:, 0, :] += ae_u
            se_z[:, 1, :] += se_v
            ae_z[:, 1, :] += ae_v

            # --- rho-oracle: the dataset's real rho target through the same
            #     denormalize + rho_to_native path -> conversion irreversibility
            se_o_b, ae_o_b = oracle_native_error_sums(target_np, stats["lo"], stats["hi"],
                                                      tu_t, tv_t, mask_u, mask_v)
            se_o += se_o_b
            ae_o += ae_o_b

            # --- persistence: repeat the day-7 NATIVE physical u/v ROLLOUT_DAYS times
            pu_parts, pv_parts = [], []
            for s in starts_np:
                u_s, v_s = reader.get(int(s) + CONTEXT - 1, 1)
                pu_parts.append(u_s)
                pv_parts.append(v_s)
            pu_t = np.broadcast_to(np.stack(pu_parts),    # (B,1,H,W-1,Z)
                                   (len(starts_np), ROLLOUT_DAYS, H, W - 1, Z))
            pv_t = np.broadcast_to(np.stack(pv_parts),    # (B,1,H-1,W,Z)
                                   (len(starts_np), ROLLOUT_DAYS, H - 1, W, Z))
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

            # running day-1 pooled native RMSE (model vs persistence) for the
            # progress postfix — all four lead-day/variable/layer accumulators
            # are already updated at this point
            w_done += cond.shape[0]
            cnt_run = np.empty((2, Z), np.float64)
            cnt_run[0, :] = mask_u.sum() * w_done
            cnt_run[1, :] = mask_v.sum() * w_done
            d1_m = pooled_rmse(se_m[0], cnt_run)
            d1_p = pooled_rmse(se_p[0], cnt_run)

            if (bi + 1) % 10 == 0 or bi + 1 == len(eval_loader):
                eval_rep.note(f"  [{bi + 1}/{len(eval_loader)}] windows done")
            # advance by the batch's TRUE window count (the final loader batch
            # may be smaller than BATCH_SIZE, so batches != windows)
            eval_rep.update(cond.shape[0],
                            d1_rmse=f"{d1_m:.4f}", d1_pers=f"{d1_p:.4f}",
                            ratio=(f"{d1_m / d1_p:.3f}" if d1_p > 0 else "n/a"))
except BaseException as exc:
    mark_progress_failed()          # the failure hook must not duplicate this
    print(format_progress("eval", "failed", stage=EVAL_STAGE[0],
                          error=f"{type(exc).__name__}: {exc}"), flush=True)
    raise
eval_rep.close()
EVAL_STAGE[0] = "postprocess"
print(f"evaluation loop finished: {w_done} windows in "
      f"{time.perf_counter() - eval_t0:.1f}s")

n_w = len(eval_ds)
n_count[:, 0, :] = mask_u.sum() * n_w
n_count[:, 1, :] = mask_v.sum() * n_w

rmse_m = np.sqrt(np.divide(se_m, n_count, out=np.zeros_like(se_m), where=n_count > 0))
mae_m = np.divide(ae_m, n_count, out=np.zeros_like(ae_m), where=n_count > 0)
rmse_p = np.sqrt(np.divide(se_p, n_count, out=np.zeros_like(se_p), where=n_count > 0))
mae_p = np.divide(ae_p, n_count, out=np.zeros_like(ae_p), where=n_count > 0)
rmse_z = np.sqrt(np.divide(se_z, n_count, out=np.zeros_like(se_z), where=n_count > 0))
mae_z = np.divide(ae_z, n_count, out=np.zeros_like(ae_z), where=n_count > 0)
rmse_o = np.sqrt(np.divide(se_o, n_count, out=np.zeros_like(se_o), where=n_count > 0))
mae_o = np.divide(ae_o, n_count, out=np.zeros_like(ae_o), where=n_count > 0)

# overall = sqrt(total_se / total_valid_count), never an average of layer RMSEs
overall_m = pooled_rmse(se_m, n_count)
overall_p = pooled_rmse(se_p, n_count)
overall_z = pooled_rmse(se_z, n_count)
overall_o = pooled_rmse(se_o, n_count)

########## save metrics + reproducibility metadata ##########

# objective-dependent fields: the deterministic baseline has no sampler, so
# the sampler-related metadata is explicitly recorded as NOT applicable
# instead of being silently reused from the diffusion path
if OBJECTIVE == "diffusion":
    sigma_data_out = np.array([sigma_data])
    sampling_steps_out = np.array([SAMPLING_STEPS or cfg["sampling_steps"]])
    sigma_max_out = np.array([model.sigma_max])
    sampler_name = "edm_heun"
    sampler_note = ("S_churn / sigma_max / sampling_steps / ensemble_size / seed "
                    "apply to the stochastic EDM sampler")
    time_sigma_out = np.array([np.nan])
else:
    sigma_data_out = np.array([np.nan])
    sampling_steps_out = np.array([-1])
    sigma_max_out = np.array([np.nan])
    sampler_name = "deterministic"
    sampler_note = ("objective=persistence_residual: S_churn / sigma_max / "
                    "sampling_steps / ensemble_size / seed are NOT applicable "
                    "(single deterministic forward per rollout step)")
    time_sigma_out = np.array([float(ckpt_cfg.get("time_sigma", RESIDUAL_TIME_SIGMA))])

out_path = os.path.join(out_dir, f"eval_{SPLIT}_{tag}.npz")
np.savez(out_path,
         rmse_model=rmse_m, mae_model=mae_m,
         rmse_persistence=rmse_p, mae_persistence=mae_p,
         rmse_zero=rmse_z, mae_zero=mae_z,
         rmse_oracle=rmse_o, mae_oracle=mae_o,
         valid_count=n_count,
         n_windows=np.array([n_w]), stride=np.array([EVAL_STRIDE]),
         batch_size=np.array([BATCH_SIZE]),
         rollout_days=np.array([ROLLOUT_DAYS]),
         ensemble_size=np.array([ENSEMBLE_SIZE]),
         objective=np.str_(OBJECTIVE),
         residual_base=np.str_(RESIDUAL_BASE),
         remask_feedback=np.array([bool(REMASK_FEEDBACK)]),
         sampler=np.str_(sampler_name), sampler_note=np.str_(sampler_note),
         time_sigma=time_sigma_out,
         S_churn=np.array([SAMPLER_S_CHURN]),
         sigma_max=sigma_max_out,
         seed=np.array([EVAL_SEED]),
         seed_scheme=np.str_("per-window: EVAL_SEED + window start day (independent "
                             "of batch size / loader batching)"),
         sigma_data=sigma_data_out,
         sampling_steps=sampling_steps_out,
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
print(f"\n=== objective={OBJECTIVE}  residual_base={RESIDUAL_BASE}  "
      f"remask_feedback={REMASK_FEEDBACK}  sampler={sampler_name} ===")
print("=== NATIVE-grid masked RMSE (m/s), pooled over u/v/layers, per lead day ===")
print("lead |  model  |  pers  | model/pers |  zero  | oracle")
for l in range(ROLLOUT_DAYS):
    rm = pooled_rmse(se_m[l], n_count[l])
    rp = pooled_rmse(se_p[l], n_count[l])
    rz = pooled_rmse(se_z[l], n_count[l])
    ro = pooled_rmse(se_o[l], n_count[l])
    print(f" {l + 1:>2}  | {rm:.4f} | {rp:.4f} | {rm / rp:.3f} | {rz:.4f} | {ro:.4f}")

print("\n=== day-1 and overall comparison table (native-grid pooled RMSE, m/s) ===")
print("mode | d1 RMSE | pers d1 | ratio | overall RMSE | pers overall | ratio")
for name, se in (("model", se_m), ("zero", se_z), ("oracle", se_o)):
    r1 = pooled_rmse(se[0], n_count[0])
    rp1 = pooled_rmse(se_p[0], n_count[0])
    ro_all = pooled_rmse(se, n_count)
    rp_all = pooled_rmse(se_p, n_count)
    print(f"{name:>6} | {r1:.4f} | {rp1:.4f} | {r1 / rp1:.3f} | "
          f"{ro_all:.4f} | {rp_all:.4f} | {ro_all / rp_all:.3f}")

print("\n=== native per-variable pooled RMSE at lead days 1/5/10/15 ===")
for k in range(2):
    line = f"{var_names[k]}: "
    for l in (0, 4, 9, 14):
        if l >= ROLLOUT_DAYS:
            break
        rm = pooled_rmse(se_m[l, k], n_count[l, k])
        rp = pooled_rmse(se_p[l, k], n_count[l, k])
        line += f"d{l + 1} {rm:.4f} (pers {rp:.4f})  "
    print(line)

print(f"\noverall native RMSE (sqrt(sum_se/sum_n)): model {overall_m:.4f} m/s "
      f"| persistence {overall_p:.4f} m/s | zero {overall_z:.4f} m/s "
      f"| rho-oracle {overall_o:.4f} m/s")

########## representative figures ##########

layers = [Z - 1] if Z == 1 else [0, Z // 2, Z - 1]
for day in (d for d in FIG_DAYS if d <= ROLLOUT_DAYS):
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

# script-level end: only NOW is the run completed — the NPZ, the summary and
# all figures are on disk (the rollout loop itself merely reported phase_done)
print(format_progress("eval", "completed", objective=OBJECTIVE,
                      windows=len(eval_ds), remask_feedback=REMASK_FEEDBACK,
                      elapsed_s=f"{time.perf_counter() - eval_t0:.1f}"), flush=True)
