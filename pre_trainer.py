#!/usr/bin/env python3
"""PRE_ocean_data trainer: 7-day condition -> next-day u/v, single-step conditional diffusion.

Task (see docs/PRE_runbook.md):
    cond   = 7 consecutive days of collocated raw u/v on the rho grid -> 14 channels
    target = day 8 u/v                                                ->  2 channels
    15-day forecasts are produced by autoregressive rollout in pre_evaluate.py.

Two presets (module-level PRESET):
    'surface_smoke' : surface layer only (depth_index=29), grid 400x441x1, patch (4,3,1)
    'full3d'        : all 30 sigma layers, grid 400x441x30, patch (4,3,2)
Both patch choices divide the grid exactly, so no padding is triggered in IAFNO.

Run (from repo root):  python pre_trainer.py
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from utilities3 import count_params, load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_config import (PRESETS, OUT_ROOT, CONTEXT, TARGET_CH, run_tag_for,
                        SIGMA_DATA_SCALE, sigma_data_from_stats,
                        sigma_data_from_checkpoint, resume_sigma_decision)
from pre_dataset import PREUVDataset, build_mask_tensor, compute_or_load_stats
from pre_metrics import masked_rel_l2

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########## PRESETS ##########

PRESET = "surface_smoke"   # 'surface_smoke' | 'full3d'
cfg = PRESETS[PRESET]

# per-preset epoch overrides for short retrains (None -> the preset's num_epochs;
# OTHER presets are never affected by this override)
EPOCH_OVERRIDES = {"surface_smoke": None}
VAL_SEED = 1234            # fixed seed for validation diffusion sampling

# Resume policy when the checkpoint's sigma_data differs from the current
# (SD2) scale — the default REFUSES, because silently mixing scales would
# corrupt the EDM preconditioning AND produce contradictory metadata
# (dir=*_SD2, scale=2.0, actual sigma_data=old):
#   "error"   : raise RuntimeError (safe default)
#   "migrate" : explicit scale migration — KEEP the current (SD2) scale and
#               continue in the SD2 run dir
#   "adopt"   : explicit legacy continuation — strictly use the checkpoint's
#               (old) scale, write outputs into a DEDICATED subdirectory of the
#               checkpoint's own dir, and record the ACTUAL scale (1.0) in
#               config (a resumed run can never be mistaken for SD2, and never
#               overwrites the original experiment's Ep{n}.pth / loss.dat)
RESUME_SIGMA_POLICY = "error"

# subdirectory used by "adopt" continuations (next to the resumed checkpoint)
LEGACY_RESUME_DIR = "legacy_resume"

########## fixed task constants ##########

COND_CH = 2 * CONTEXT              # 14, day-major interleaved (see pre_dataset.py)
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

hidden_size_factor = 4
num_blocks = 1                     # AFNO channel blocks
checkpoint_path = "/data2/user/zyq/checkpoints/PRE_lr3e4/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2/Ep3.pth"  # resume to 10 epochs (handover §5.1)

run_tag = run_tag_for(PRESET)
run_dir = os.path.join(OUT_ROOT, run_tag)   # redirected to the checkpoint's own
                                            # directory under "adopt"

########## data ##########

stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)

train_dataset = PREUVDataset("train", stats, context=CONTEXT, horizon=1,
                             depth_index=cfg["depth_index"], stride=cfg["train_stride"],
                             max_windows=cfg["max_train_windows"])
val_dataset = PREUVDataset("val", stats, context=CONTEXT, horizon=1,
                           depth_index=cfg["depth_index"], stride=1)
print(f"train windows: {len(train_dataset)}   val windows: {len(val_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                           shuffle=True, num_workers=cfg["num_workers"],
                                           pin_memory=True, drop_last=True)
# validation: fixed number of windows uniformly spread over the WHOLE val period
# (deterministic linspace, no RNG), so checkpoints across epochs are comparable.
val_idx = np.linspace(0, len(val_dataset) - 1, cfg["val_windows"]).astype(int)
val_subset = torch.utils.data.Subset(val_dataset, val_idx.tolist())
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfg["batch_size"],
                                         shuffle=False, num_workers=cfg["num_workers"],
                                         pin_memory=True, drop_last=False)
print(f"val subset: {len(val_subset)} windows at indices {val_idx[0]}..{val_idx[-1]}")

mask = build_mask_tensor(device, cfg["depth_index"])   # (1,2,H,W,Z) bivariate

########## model ##########

dm_backbone = IAFNODiff(
    dim=(H, W, Z),
    patch_size=cfg["patch_size"],
    embed_dim=cfg["embed_dim"],
    num_blocks=num_blocks,
    in_chans=TARGET_CH,
    out_chans=TARGET_CH,
    cond_chans=COND_CH,
    ex_layer=cfg["explicit_layer"],
    nlayer=cfg["implicit_layer"],
    hidden_size_factor=hidden_size_factor,
    dim_f=(H, W, Z),
    self_condition=True,
).to(device)

model = ElucidatedDiffusion(
    dm_backbone,
    channels=TARGET_CH,
    num_sample_steps=cfg["sampling_steps"],
    image_size_h=H,
    image_size_w=W,
    image_size_z=Z,
    sigma_data=sigma_data_from_stats(stats["sigma"]),   # [-1,1] image-space scale
)

optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
n_epochs = EPOCH_OVERRIDES.get(PRESET) or cfg["num_epochs"]
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(train_loader))
scaler = GradScaler(device.type)   # torch.amp.GradScaler (new AMP API)

########## resume (history + best_val must survive) ##########

hist = {"train": [], "val_rel": [], "time": []}
best_val = float("inf")
start_epoch = 0
sigma_scale = SIGMA_DATA_SCALE      # actual stats_sigma -> sigma_data multiplier
if checkpoint_path is not None:
    ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                           map_location=device)
    sd_ckpt, sd_in_ckpt = sigma_data_from_checkpoint(ckpt, stats["sigma"])
    if not sd_in_ckpt:
        print(f"WARNING: {checkpoint_path} has no config.sigma_data (legacy "
              f"checkpoint); its sigma_data is the old stats-only scale {sd_ckpt:.5f}")
    model.sigma_data, adopted = resume_sigma_decision(
        sd_ckpt, model.sigma_data, RESUME_SIGMA_POLICY)
    if adopted:
        # legacy continuation: write into a DEDICATED subdir next to the
        # checkpoint — the original experiment's Ep{n}.pth / loss.dat are
        # NEVER touched, and the run can never be mistaken for an SD2 run
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if os.path.basename(ckpt_dir) == LEGACY_RESUME_DIR:
            run_dir = ckpt_dir        # resuming a previous legacy continuation
        else:
            run_dir = os.path.join(ckpt_dir, LEGACY_RESUME_DIR)
        sigma_scale = model.sigma_data / float(stats["sigma"])
        print(f"adopted checkpoint scale: sigma_data={model.sigma_data:.5f} "
              f"(stats_sigma x {sigma_scale:.3f}); outputs -> {run_dir}")
    elif abs(sd_ckpt - model.sigma_data) <= 1e-6:
        print(f"checkpoint sigma_data {sd_ckpt:.5f} matches the current "
              f"(SD2) scale")
    start_epoch = ckpt.get("epoch", -1) + 1

loss_file = os.path.join(run_dir, "loss.dat")
# history is READ from the ORIGINAL experiment when adopting (the continuation
# dir is fresh), so the written loss.dat always contains the FULL history;
# for every other mode the history source is the output dir itself.
hist_src = loss_file
if checkpoint_path is not None and adopted:
    hist_src = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                            "loss.dat")

if checkpoint_path is not None:
    best_val = ckpt.get("best_val")
    if best_val is None:
        # older checkpoint without best_val: recompute from loss.dat history
        if os.path.exists(hist_src):
            arr = np.loadtxt(hist_src).reshape(-1, 3)
            best_val = float(arr[:start_epoch, 2].min())
            print(f"recomputed best_val={best_val:.5f} from {hist_src}")
        else:
            best_val = float("inf")
            print("WARNING: checkpoint has no best_val and loss.dat is missing; "
                  "starting best_val from inf")
    print(f"resumed from {checkpoint_path} (epoch {start_epoch}, "
          f"best_val={best_val:.5f})")
    if os.path.exists(hist_src):
        arr = np.loadtxt(hist_src).reshape(-1, 3)
        n_old = min(start_epoch, len(arr))
        hist["time"] = list(arr[:n_old, 0])
        hist["train"] = list(arr[:n_old, 1])
        hist["val_rel"] = list(arr[:n_old, 2])
        print(f"restored {n_old} epochs of history from {hist_src}")

os.makedirs(run_dir, exist_ok=True)

print("Model Total Params:", count_params(model))
print(f"preset={PRESET} grid=({H},{W},{Z}) patch={cfg['patch_size']} cond_ch={COND_CH} "
      f"target_ch={TARGET_CH} stats_sigma={stats['sigma']:.5f} "
      f"sigma_data={model.sigma_data:.5f} (scale {sigma_scale:.3f}x) "
      f"epochs={n_epochs}  run_dir={run_dir}")

########## pre-flight checks (refuse BEFORE wasting a training run) ##########

# every epoch this run will write must be free: a collision would silently
# rewrite history (e.g. resuming a checkpoint that predates later epochs of
# the same experiment). Checked once, up front.
for ep in range(start_epoch, n_epochs):
    ep_out = os.path.join(run_dir, f"Ep{ep + 1}.pth")
    if os.path.exists(ep_out):
        raise RuntimeError(
            f"{ep_out} already exists; refusing to overwrite. Delete it or "
            f"resume from a checkpoint that leaves epoch {ep + 1} free")

# loss.dat must never be truncated: refuse up front if even a FULL run cannot
# outgrow the existing history (early-stop truncation is still caught by the
# per-epoch guard before any checkpoint is saved).
if os.path.exists(loss_file):
    n_existing = len(np.loadtxt(loss_file).reshape(-1, 3))
    n_written = len(hist["train"]) + (n_epochs - start_epoch)
    if n_existing > n_written:
        raise RuntimeError(
            f"{loss_file} has {n_existing} rows but at most {n_written} epochs "
            f"of history will be written — refusing to truncate")


########## helpers ##########

def unnormalize(x):
    """(B,2,H,W,Z) [0,1] -> physical m/s (per-channel clip range)."""
    return x * (y_hi - y_lo) + y_lo


########## training loop ##########

worse_epochs = 0   # consecutive epochs with val_masked_relL2 strictly above best
for ep in range(start_epoch, n_epochs):
    model.train()
    t1 = time.time()
    t_batch = time.time()
    train_loss = 0.0
    n_batch = 0
    succ_updates = 0
    skipped_updates = 0
    for bi, (cond, target, _) in enumerate(train_loader):
        xx = cond.to(device, non_blocking=True)          # (B,14,H,W,Z) in [0,1]
        yy = target[:, 0].to(device, non_blocking=True)  # (B,2,H,W,Z)  in [0,1]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            loss = model(yy, xx, mask=mask)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite training loss {float(loss)} at epoch {ep + 1} batch {bi} "
                f"(grad_scale={scaler.get_scale():.4e}, n_batch={n_batch}); aborting")
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            # inf/nan gradient detected: this step was SKIPPED (no update)
            skipped_updates += 1
        else:
            succ_updates += 1
            scheduler.step()   # only after a real optimizer update
        train_loss += float(loss)
        n_batch += 1
        if (bi + 1) % 100 == 0:
            dt_b = time.time() - t_batch
            print(f"  [ep {ep + 1}] batch {bi + 1}/{len(train_loader)}  "
                  f"avg_loss {train_loss / n_batch:.5f}  {dt_b / 100:.2f}s/batch  "
                  f"scale {scaler.get_scale():.4e}", flush=True)
            t_batch = time.time()
    train_loss /= max(n_batch, 1)

    # validation: full diffusion sampling on the fixed uniform val windows.
    # fork_rng isolates the CPU (and, on CUDA, the current device) RNG so the
    # fixed VAL_SEED cannot perturb the training RNG stream; the training
    # CPU/CUDA RNG state is restored on exit from the context.
    model.eval()
    val_rel, nb = 0.0, 0
    # torch.device("cuda").index is None and fork_rng(devices=[None]) crashes;
    # current_device() is the actual ordinal of the device the model is on.
    rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(VAL_SEED)
        for cond, target, _ in val_loader:
            xx = cond.to(device, non_blocking=True)
            yy = target[:, 0].to(device, non_blocking=True)
            with autocast(device_type=device.type):
                pred = model.sample(xx)
            val_rel += masked_rel_l2(unnormalize(pred.float()), unnormalize(yy), mask)
            nb += 1
    val_rel /= max(nb, 1)

    dt = time.time() - t1
    hist["train"].append(train_loss)
    hist["val_rel"].append(val_rel)
    hist["time"].append(dt)
    print(f"epoch {ep + 1}/{n_epochs}  {dt:.1f}s  "
          f"train_loss {train_loss:.5f}  val_masked_relL2 {val_rel:.5f}  "
          f"updates {succ_updates} (skipped {skipped_updates})  "
          f"grad_scale {scaler.get_scale():.4e}  "
          f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

    # checkpoint order: decide is_best FIRST, update best_val, then build ONE
    # state dict that both Ep{n}.pth and best.pth share — so a new-best epoch
    # never writes a best.pth with a stale best_val.
    is_best = val_rel < best_val
    if is_best:
        best_val = val_rel
        worse_epochs = 0
    else:
        worse_epochs += 1
    state = {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "config": {
            "preset": PRESET, **cfg, "context": CONTEXT,
            "stats_sigma": float(stats["sigma"]),
            "sigma_data_scale": sigma_scale,
            "sigma_data": model.sigma_data,
        },
    }
    # loss.dat must never be truncated: checked BEFORE any checkpoint is saved
    # this epoch (an early stop can still shrink the history below a
    # pre-existing file; the up-front check covers a full run).
    if os.path.exists(loss_file):
        n_existing = len(np.loadtxt(loss_file).reshape(-1, 3))
        if n_existing > len(hist["train"]):
            raise RuntimeError(
                f"{loss_file} has {n_existing} rows but only {len(hist['train'])} "
                f"epochs of history will be written — refusing to truncate")
    ckpt_out = os.path.join(run_dir, f"Ep{ep + 1}.pth")
    torch.save(state, ckpt_out)
    if is_best:
        torch.save(state, os.path.join(run_dir, "best.pth"))
    # loss.dat always contains the FULL history (restored on resume), so a
    # resumed run never silently overwrites previous epochs.
    np.savetxt(loss_file,
               np.dstack((hist["time"], hist["train"], hist["val_rel"])).squeeze(),
               fmt="%16.7f")

    if worse_epochs >= 8 and ep >= 1:
        print(f"early stop: val_masked_relL2 worsened for {worse_epochs} consecutive "
              f"epochs (best {best_val:.5f})")
        break

print(f"done. best val_masked_relL2 = {best_val:.5f}; checkpoints in {run_dir}")