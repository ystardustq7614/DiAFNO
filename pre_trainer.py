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
from torch.cuda.amp import autocast, GradScaler

from utilities3 import count_params, load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_config import PRESETS, OUT_ROOT, CONTEXT, TARGET_CH, run_tag_for
from pre_dataset import PREUVDataset, build_mask_tensor, compute_or_load_stats

torch.manual_seed(123)
np.random.seed(123)

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########## PRESETS ##########

PRESET = "surface_smoke"   # 'surface_smoke' | 'full3d'
cfg = PRESETS[PRESET]

########## fixed task constants ##########

COND_CH = 2 * CONTEXT              # 14, day-major interleaved (see pre_dataset.py)
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

hidden_size_factor = 4
num_blocks = 1                     # AFNO channel blocks
checkpoint_path = None             # set to a .pth to resume model weights

run_tag = run_tag_for(PRESET)
run_dir = os.path.join(OUT_ROOT, run_tag)
os.makedirs(run_dir, exist_ok=True)

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
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["batch_size"],
                                         shuffle=False, num_workers=cfg["num_workers"],
                                         pin_memory=True, drop_last=True)

mask = build_mask_tensor(device, cfg["depth_index"])   # (1,1,H,W,Z)

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
).to(device).to(DTYPE)

model = ElucidatedDiffusion(
    dm_backbone,
    channels=TARGET_CH,
    num_sample_steps=cfg["sampling_steps"],
    image_size_h=H,
    image_size_w=W,
    image_size_z=Z,
    sigma_data=stats["sigma"],
)

optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"] * len(train_loader))
scaler = GradScaler()

start_epoch = 0
if checkpoint_path is not None:
    ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, map_location=device)
    start_epoch = ckpt.get("epoch", -1) + 1
    print(f"resumed from {checkpoint_path} (epoch {start_epoch})")

print("Model Total Params:", count_params(model))
print(f"preset={PRESET} grid=({H},{W},{Z}) patch={cfg['patch_size']} cond_ch={COND_CH} "
      f"target_ch={TARGET_CH} sigma_data={stats['sigma']:.5f}")


########## helpers ##########

def unnormalize(x):
    """(B,2,H,W,Z) [0,1] -> physical m/s (per-channel clip range)."""
    return x * (y_hi - y_lo) + y_lo


def masked_rel_l2(pred, tgt):
    """Relative L2 over ocean points only, mean over batch. pred/tgt physical units."""
    diff2 = ((pred - tgt) * mask).sum(dim=(1, 2, 3, 4))
    tgt2 = (tgt * tgt * mask).sum(dim=(1, 2, 3, 4))
    return (diff2.sqrt() / tgt2.sqrt().clamp(min=1e-12)).mean().item()


########## training loop ##########

hist = {"train": [], "val_rel": [], "time": []}
best_val = float("inf")

for ep in range(start_epoch, cfg["num_epochs"]):
    model.train()
    t1 = time.time()
    train_loss = 0.0
    for cond, target, _ in train_loader:
        xx = cond.to(device, non_blocking=True)          # (B,14,H,W,Z) in [0,1]
        yy = target[:, 0].to(device, non_blocking=True)  # (B,2,H,W,Z)  in [0,1]

        optimizer.zero_grad()
        with autocast():
            loss = model(yy, xx, mask=mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation: full diffusion sampling on a few val batches, physical masked rel-L2
    model.eval()
    val_rel, nb = 0.0, 0
    with torch.no_grad():
        for cond, target, _ in val_loader:
            if nb >= cfg["val_batches"]:
                break
            xx = cond.to(device, non_blocking=True)
            yy = target[:, 0].to(device, non_blocking=True)
            with autocast():
                pred = model.sample(xx)
            val_rel += masked_rel_l2(unnormalize(pred.float()), unnormalize(yy))
            nb += 1
    val_rel /= max(nb, 1)

    dt = time.time() - t1
    hist["train"].append(train_loss)
    hist["val_rel"].append(val_rel)
    hist["time"].append(dt)
    print(f"epoch {ep + 1}/{cfg['num_epochs']}  {dt:.1f}s  "
          f"train_loss {train_loss:.5f}  val_masked_relL2 {val_rel:.5f}", flush=True)

    state = {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {"preset": PRESET, **cfg, "context": CONTEXT},
    }
    torch.save(state, os.path.join(run_dir, f"Ep{ep + 1}.pth"))
    if val_rel < best_val:
        best_val = val_rel
        torch.save(state, os.path.join(run_dir, "best.pth"))
    np.savetxt(os.path.join(run_dir, "loss.dat"),
               np.dstack((hist["time"], hist["train"], hist["val_rel"])).squeeze(),
               fmt="%16.7f")

print(f"done. best val_masked_relL2 = {best_val:.5f}; checkpoints in {run_dir}")
