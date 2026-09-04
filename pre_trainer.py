#!/usr/bin/env python3
"""PRE_ocean_data trainer: 7-day condition -> next-day u/v (conditional EDM or
deterministic persistence-residual; optional detached multi-step training).

Task (see docs/operations/PRE_runbook.md):
    cond   = 7 consecutive days of collocated raw u/v on the rho grid -> 14 channels
    target = day 8 u/v                                                ->  2 channels
    15-day forecasts are produced by autoregressive rollout in pre_evaluate.py.

Four presets (DIAFNO_PRESET):
    'surface_smoke' : surface layer only (depth_index=29), grid 400x441x1, patch (4,3,1)
    'middle_smoke'  : middle sigma layer (depth_index=14), otherwise identical to surface_smoke
    'bottom_smoke'  : bottom sigma layer (depth_index=0),  otherwise identical to surface_smoke
    'full3d'        : all 30 sigma layers, grid 400x441x30, patch (4,3,2)
All patch choices divide the grid exactly, so no padding is triggered in IAFNO.

Training objective (DIAFNO_OBJECTIVE):
    'diffusion'            : conditional EDM (legacy default)
    'persistence_residual' : deterministic PersistenceResidualIAFNO baseline
                             (last-day persistence + zero-init residual head,
                             masked-MSE objective, run tag suffix _RES)

Detached multi-step (DIAFNO_TRAIN_HORIZON=K, doc
docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md §5; run tag suffix _MS{K}):
    persistence_residual only, no static mask. For batch i the training lead
    J = lead_for_batch(i, K) (fixed schedule 1,2,1,3,1,4,1,5,... for K=5; 50%
    day-1 anchor); the model's OWN predictions are rolled forward J-1 steps
    under torch.no_grad() (clamp [0,1], rf0, same sliding window as the formal
    rollout) and only the J-th step is backpropagated. K=1 is the exact
    historical single-step path. MS runs default to lr 1e-4 / 5 epochs
    (pre_config.MS_DEFAULTS) and support weights-only initialization from a
    finished run via DIAFNO_INIT_CHECKPOINT (fresh optimizer/scheduler/
    history; mutually exclusive with DIAFNO_CHECKPOINT).

Run from repo root (safe default is a short smoke run):
    python pre_trainer.py
    DIAFNO_TRAIN_MODE=full python pre_trainer.py
    DIAFNO_TRAIN_MODE=full torchrun --standalone --nproc_per_node=4 pre_trainer.py
"""
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from utilities3 import count_params, load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_models import PersistenceResidualIAFNO, masked_mse_loss
from pre_config import (OUT_ROOT, CONTEXT, TARGET_CH, training_config,
                        training_run_tag, static_mask_input,
                        SIGMA_DATA_SCALE, sigma_data_from_stats,
                        sigma_data_from_checkpoint, resume_sigma_decision,
                        DEFAULT_OBJECTIVE, MASK_SCHEME, RESIDUAL_TIME_SIGMA,
                        STATIC_MASK_CHANNELS,
                        train_horizon, init_checkpoint,
                        lead_for_batch, lead_schedule_str,
                        check_multistep_config, restore_worse_epochs,
                        validate_objective, ensure_objective_compatible,
                        check_norm_fingerprint, check_residual_time_sigma,
                        ProgressReporter, format_progress,
                        install_progress_failure_hook, mark_progress_failed)
from pre_dataset import (PREUVDataset, build_mask_tensor, compute_or_load_stats,
                         mask_version)
from pre_metrics import masked_rel_l2
from pre_rollout import detached_feedback_window

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DISTRIBUTED = WORLD_SIZE > 1
if DISTRIBUTED:
    if not torch.cuda.is_available():
        raise RuntimeError("multi-process training requires CUDA/NCCL; launch one process on CPU")
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl")
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", LOCAL_RANK) if DISTRIBUTED else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN = RANK == 0


def log(*args, **kwargs):
    if IS_MAIN:
        print(*args, **kwargs)


# All ranks must initialize identical weights; rank-specific RNG streams are
# selected only after DDP has synchronized the model below.
torch.manual_seed(123)
log(f"Using device: {device}  world_size={WORLD_SIZE} rank={RANK}")

# Standard status=failed line for exceptions that escape the guarded training
# block (initialization / data / model / pre-flight failures have no live
# reporter). The guarded block's own handler deduplicates via
# mark_progress_failed(); non-main DDP ranks print plain tracebacks.
if IS_MAIN:
    install_progress_failure_hook("train")

# honest scope labeling for the rank-0 progress lines: under DDP both the
# train and validation loaders are RANK-SHARDED, so their step/batch totals
# are per-rank, never global (global sample/s is labeled sample_per_s)
PROGRESS_SCOPE = f"rank{RANK}_shard_of_{WORLD_SIZE}" if DISTRIBUTED else "whole_split"

########## PRESETS ##########

PRESET = os.environ.get("DIAFNO_PRESET", "surface_smoke")
TRAIN_MODE = os.environ.get("DIAFNO_TRAIN_MODE", "smoke").lower()
OBJECTIVE = validate_objective(os.environ.get("DIAFNO_OBJECTIVE", DEFAULT_OBJECTIVE))
# Phase-5 mask-input A/B (arm B): static mask channels are implemented for the
# deterministic objective only; the diffusion path keeps its exact historical
# layout (refuse rather than silently change the EDM input shape).
STATIC_MASK = static_mask_input()
if STATIC_MASK and OBJECTIVE != "persistence_residual":
    raise RuntimeError(
        "DIAFNO_STATIC_MASK=1 is only supported with "
        "DIAFNO_OBJECTIVE=persistence_residual (the diffusion path keeps its "
        "historical 14-channel layout)")

# Detached multi-step (work package 2): K=1 is the exact historical single-step
# teacher-forcing path; K>1 mirrors the formal deterministic rollout with the
# model's own detached feedback (doc §5). Only the deterministic objective,
# without the (rejected in experiments 08/09) mask arms, is allowed.
TRAIN_HORIZON = train_horizon()
LEAD_SCHEDULE = lead_schedule_str(TRAIN_HORIZON)
if TRAIN_HORIZON > 1 and (OBJECTIVE != "persistence_residual" or STATIC_MASK):
    raise RuntimeError(
        f"DIAFNO_TRAIN_HORIZON={TRAIN_HORIZON} (detached multi-step) is only "
        "supported with DIAFNO_OBJECTIVE=persistence_residual and "
        "DIAFNO_STATIC_MASK unset: the feedback must mirror the formal "
        "deterministic rollout (rf0, no static mask channels)")
cfg = training_config(PRESET, TRAIN_MODE, WORLD_SIZE, train_horizon=TRAIN_HORIZON)

# Optional per-preset overrides for one-off runs.  Normal defaults live only in
# pre_config.py so the scheduler horizon and documented preset cannot drift.
EPOCH_OVERRIDES = {}
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
# backbone condition channels: the dynamic window plus (arm B) the two static
# mask channels forwarded separately via static_cond
MODEL_COND_CH = COND_CH + (STATIC_MASK_CHANNELS if STATIC_MASK else 0)
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

hidden_size_factor = 4
num_blocks = 1                     # AFNO channel blocks
checkpoint_path = os.environ.get("DIAFNO_CHECKPOINT") or None
if checkpoint_path is not None:
    checkpoint_path = os.path.expanduser(checkpoint_path)

# weights-only init (fresh optimizer/scheduler/history) is mutually exclusive
# with a full resume, and scoped to the deterministic objective it was planned
# for (doc §6 WP3: initialize MS5 from the finished experiment-07 Ep10 weights)
INIT_CHECKPOINT = init_checkpoint()
if INIT_CHECKPOINT is not None:
    if checkpoint_path is not None:
        raise RuntimeError(
            "DIAFNO_INIT_CHECKPOINT (weights-only init) and DIAFNO_CHECKPOINT "
            "(full resume) are mutually exclusive; remove one of them")
    if OBJECTIVE != "persistence_residual":
        raise RuntimeError(
            "DIAFNO_INIT_CHECKPOINT is only supported with "
            "DIAFNO_OBJECTIVE=persistence_residual")

run_tag = training_run_tag(PRESET, cfg, TRAIN_MODE, WORLD_SIZE, OBJECTIVE,
                           static_mask=STATIC_MASK, train_horizon=TRAIN_HORIZON)
run_dir = os.path.join(OUT_ROOT, run_tag)   # redirected to the checkpoint's own
                                            # directory under "adopt"

########## data ##########

# A missing/stale stats cache is expensive to build and unsafe to write from
# several ranks concurrently. Rank 0 creates it, then the others load it.
if DISTRIBUTED:
    stats = compute_or_load_stats(depth_index=cfg["depth_index"]) if IS_MAIN else None
    dist.barrier()
    if not IS_MAIN:
        stats = compute_or_load_stats(depth_index=cfg["depth_index"], verbose=False)
else:
    stats = compute_or_load_stats(depth_index=cfg["depth_index"])
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)

# multi-step: TRAIN windows cover leads 1..K (target[:, J-1] selects the
# training lead); validation windows stay single-step because the per-epoch
# val_masked_relL2 is only a training-health signal (formal selection =
# pre_evaluate.py validation 15-day deterministic protocol).
train_dataset = PREUVDataset("train", stats, context=CONTEXT, horizon=TRAIN_HORIZON,
                             depth_index=cfg["depth_index"], stride=cfg["train_stride"],
                             max_windows=cfg["max_train_windows"])
val_dataset = PREUVDataset("val", stats, context=CONTEXT, horizon=1,
                           depth_index=cfg["depth_index"], stride=1)
log(f"train windows: {len(train_dataset)}   val windows: {len(val_dataset)}")

train_sampler = DistributedSampler(
    train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True,
    seed=123, drop_last=True) if DISTRIBUTED else None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg["batch_size"],
    shuffle=train_sampler is None, sampler=train_sampler,
    num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", drop_last=True)
if len(train_loader) == 0:
    raise RuntimeError(
        f"training loader is empty: windows={len(train_dataset)}, world_size={WORLD_SIZE}, "
        f"per_device_batch={cfg['batch_size']}")
# validation: fixed number of windows uniformly spread over the WHOLE val period
# (deterministic linspace, no RNG), so checkpoints across epochs are comparable.
val_idx = np.linspace(0, len(val_dataset) - 1, cfg["val_windows"]).astype(int)
rank_val_idx = val_idx[RANK::WORLD_SIZE]
val_subset = torch.utils.data.Subset(val_dataset, rank_val_idx.tolist())
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfg["batch_size"],
                                         shuffle=False, num_workers=cfg["num_workers"],
                                         pin_memory=device.type == "cuda", drop_last=False)
log(f"val subset: {len(val_idx)} windows at indices {val_idx[0]}..{val_idx[-1]} "
    f"({WORLD_SIZE} rank shard(s))")

mask = build_mask_tensor(device, cfg["depth_index"])   # (1,2,H,W,Z) bivariate

########## model ##########

dm_backbone = IAFNODiff(
    dim=(H, W, Z),
    patch_size=cfg["patch_size"],
    embed_dim=cfg["embed_dim"],
    num_blocks=num_blocks,
    in_chans=TARGET_CH,
    out_chans=TARGET_CH,
    cond_chans=MODEL_COND_CH,
    ex_layer=cfg["explicit_layer"],
    nlayer=cfg["implicit_layer"],
    hidden_size_factor=hidden_size_factor,
    dim_f=(H, W, Z),
    self_condition=True,
).to(device)

if OBJECTIVE == "diffusion":
    model = ElucidatedDiffusion(
        dm_backbone,
        channels=TARGET_CH,
        num_sample_steps=cfg["sampling_steps"],
        image_size_h=H,
        image_size_w=W,
        image_size_z=Z,
        sigma_data=sigma_data_from_stats(stats["sigma"]),   # [-1,1] image-space scale
    )
else:
    # deterministic persistence-residual baseline: prediction = last-day
    # persistence + residual; the zero-initialized head makes the UNTRAINED
    # model exactly persistence (verified below before any training happens).
    model = PersistenceResidualIAFNO(dm_backbone, time_sigma=RESIDUAL_TIME_SIGMA)
    with torch.no_grad():
        probe = torch.rand(1, COND_CH, H, W, Z, device=device)
        probe_static = mask if STATIC_MASK else None
        ident = model(probe, static_cond=probe_static)
    if not torch.equal(ident, probe[:, -TARGET_CH:]):
        raise RuntimeError(
            "zero-initialized persistence-residual model does not reduce to "
            "last-day persistence; refusing to train")
    log("zero-init check passed: untrained residual model == last-day persistence"
        + (" (with static mask input)" if STATIC_MASK else ""))

optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
n_epochs = EPOCH_OVERRIDES.get(PRESET) or cfg["num_epochs"]
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(train_loader))
scaler = GradScaler(device.type)   # torch.amp.GradScaler (new AMP API)

########## resume (history + best_val must survive) ##########

hist = {"train": [], "val_rel": [], "time": []}
best_val = float("inf")
start_epoch = 0
worse_epochs = 0   # consecutive epochs with val_masked_relL2 strictly above best
sigma_scale = SIGMA_DATA_SCALE      # actual stats_sigma -> sigma_data multiplier
adopted = False                     # legacy "adopt" continuation (diffusion only)
if checkpoint_path is not None:
    ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                           map_location=device)
    ckpt_cfg = ckpt.get("config") or {}
    if "preset" in ckpt_cfg and ckpt_cfg["preset"] != PRESET:
        raise RuntimeError(
            f"checkpoint preset={ckpt_cfg['preset']!r} vs current {PRESET!r}")
    if "train_mode" in ckpt_cfg and ckpt_cfg["train_mode"] != TRAIN_MODE:
        raise RuntimeError(
            f"checkpoint train_mode={ckpt_cfg['train_mode']!r} cannot resume in "
            f"{TRAIN_MODE!r}; smoke checkpoints are pipeline gates, not full-run starts")
    if "world_size" in ckpt_cfg and int(ckpt_cfg["world_size"]) != WORLD_SIZE:
        raise RuntimeError(
            f"checkpoint world_size={ckpt_cfg['world_size']} vs current {WORLD_SIZE}; "
            "resume with the original GPU count so optimizer/scheduler semantics stay fixed")
    if DISTRIBUTED and "world_size" not in ckpt_cfg:
        raise RuntimeError(
            "checkpoint predates DDP world-size metadata; resume it on one GPU or "
            "start a fresh multi-GPU run")
    # never load one model class into the other, and never resume across a
    # structural change (legacy checkpoints without these fields predate the
    # objective split and can only be diffusion runs — guarded below)
    ckpt_objective = ensure_objective_compatible(ckpt, OBJECTIVE)
    # multi-step semantics must survive resume unchanged (doc §6 WP2 item 6)
    check_multistep_config(ckpt_cfg, TRAIN_HORIZON, LEAD_SCHEDULE)
    for key, current in (("cond_chans", COND_CH), ("target_ch", TARGET_CH),
                         ("mask_scheme", MASK_SCHEME),
                         ("static_mask_input", STATIC_MASK)):
        if key in ckpt_cfg and ckpt_cfg[key] != current:
            raise RuntimeError(
                f"checkpoint {key}={ckpt_cfg[key]!r} vs current {current!r}; "
                "refusing to resume across a structural change")
    if OBJECTIVE == "persistence_residual" and "residual_base" in ckpt_cfg \
            and ckpt_cfg["residual_base"] != "last_day":
        raise RuntimeError(
            f"checkpoint residual_base={ckpt_cfg['residual_base']!r} is not "
            "supported (only 'last_day')")
    # semantic fingerprint: the recorded normalization range and mask version
    # must match the current stats/masks, otherwise a resumed run would
    # silently train with DIFFERENT data semantics than the checkpoint used
    # (legacy checkpoints predate the recorded fields and can only be warned)
    for fp_warning in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                             mask_version()):
        log(f"WARNING: {checkpoint_path}: {fp_warning}")
    if OBJECTIVE == "persistence_residual":
        check_residual_time_sigma(ckpt_cfg, model.time_sigma)
        if "stats_sigma" in ckpt_cfg and \
                abs(float(ckpt_cfg["stats_sigma"]) - float(stats["sigma"])) > 1e-6:
            raise RuntimeError(
                f"checkpoint stats_sigma={float(ckpt_cfg['stats_sigma']):.6f} vs "
                f"current {float(stats['sigma']):.6f}; the residual objective has "
                "no sigma migration policy — refusing to resume")
    if OBJECTIVE == "diffusion":
        # sigma_data preconditioning only exists for the EDM objective
        sd_ckpt, sd_in_ckpt = sigma_data_from_checkpoint(ckpt, stats["sigma"])
        if not sd_in_ckpt:
            log(f"WARNING: {checkpoint_path} has no config.sigma_data (legacy "
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
            log(f"adopted checkpoint scale: sigma_data={model.sigma_data:.5f} "
                f"(stats_sigma x {sigma_scale:.3f}); outputs -> {run_dir}")
        elif abs(sd_ckpt - model.sigma_data) <= 1e-6:
            log(f"checkpoint sigma_data {sd_ckpt:.5f} matches the current "
                f"(SD2) scale")
    else:
        log(f"residual checkpoint objective={ckpt_objective!r}; sigma_data "
            "policy not applicable to the deterministic objective")
    start_epoch = ckpt.get("epoch", -1) + 1
    # the early-stop streak must survive a resume: a pre-existing worsening
    # count still leads to the same 2-consecutive-epochs stop (legacy
    # checkpoints without the field keep the historical default 0)
    worse_epochs = restore_worse_epochs(ckpt)
    if worse_epochs:
        log(f"restored early-stop counter: {worse_epochs} consecutive "
            f"worsening epoch(s) from {checkpoint_path}")

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
            log(f"recomputed best_val={best_val:.5f} from {hist_src}")
        else:
            best_val = float("inf")
            log("WARNING: checkpoint has no best_val and loss.dat is missing; "
                "starting best_val from inf")
    log(f"resumed from {checkpoint_path} (epoch {start_epoch}, "
        f"best_val={best_val:.5f})")
    if os.path.exists(hist_src):
        arr = np.loadtxt(hist_src).reshape(-1, 3)
        n_old = min(start_epoch, len(arr))
        hist["time"] = list(arr[:n_old, 0])
        hist["train"] = list(arr[:n_old, 1])
        hist["val_rel"] = list(arr[:n_old, 2])
        log(f"restored {n_old} epochs of history from {hist_src}")

# weights-only init (mutually exclusive with resume above): load ONLY the model
# weights from a finished run — the source optimizer/scheduler/scaler/epoch/
# history are deliberately NOT restored (the source cosine schedule is over;
# doc §6 WP3). Everything here stays fresh: hist/best_val/start_epoch keep
# their initial values and the run writes into its OWN _MS{K} directory.
if INIT_CHECKPOINT is not None:
    init = torch.load(INIT_CHECKPOINT, map_location=device, weights_only=True)
    init_cfg = init.get("config") or {}
    ensure_objective_compatible(init, OBJECTIVE)
    if "preset" in init_cfg and init_cfg["preset"] != PRESET:
        raise RuntimeError(
            f"init checkpoint preset={init_cfg['preset']!r} vs current {PRESET!r}; "
            "weights-only init must stay within the same architecture preset")
    if "static_mask_input" in init_cfg and init_cfg["static_mask_input"] != STATIC_MASK:
        raise RuntimeError(
            f"init checkpoint static_mask_input={init_cfg['static_mask_input']!r} "
            f"vs current {STATIC_MASK!r}; refusing to init across a structural change")
    for fp_warning in check_norm_fingerprint(init_cfg, stats["lo"], stats["hi"],
                                             mask_version()):
        log(f"WARNING: {INIT_CHECKPOINT}: {fp_warning}")
    if OBJECTIVE == "persistence_residual":
        check_residual_time_sigma(init_cfg, model.time_sigma)
        if "stats_sigma" in init_cfg and \
                abs(float(init_cfg["stats_sigma"]) - float(stats["sigma"])) > 1e-6:
            raise RuntimeError(
                f"init checkpoint stats_sigma={float(init_cfg['stats_sigma']):.6f} "
                f"vs current {float(stats['sigma']):.6f}; the residual objective "
                "has no sigma migration policy — refusing to init")
    model.load_state_dict(init["model_state_dict"])
    log(f"weights-only init from {INIT_CHECKPOINT} "
        f"(source epoch {init.get('epoch')}); optimizer/scheduler/scaler/history "
        "are FRESH", flush=True)

os.makedirs(run_dir, exist_ok=True)

log("Model Total Params:", count_params(model))
if OBJECTIVE == "diffusion":
    scale_info = (f"stats_sigma={stats['sigma']:.5f} "
                  f"sigma_data={model.sigma_data:.5f} (scale {sigma_scale:.3f}x)")
else:
    scale_info = (f"stats_sigma={stats['sigma']:.5f} objective=persistence_residual "
                  f"(residual_base={model.residual_base}, time_sigma={model.time_sigma:g}; "
                  "sigma_data not applicable)")
log(f"preset={PRESET} mode={TRAIN_MODE} objective={OBJECTIVE} grid=({H},{W},{Z}) "
    f"patch={cfg['patch_size']} cond_ch={COND_CH} model_cond_ch={MODEL_COND_CH} "
    f"static_mask_input={STATIC_MASK} target_ch={TARGET_CH} "
    f"mask_scheme={MASK_SCHEME} {scale_info} epochs={n_epochs} "
    f"train_horizon={TRAIN_HORIZON} lead_schedule={LEAD_SCHEDULE} "
    f"init_checkpoint={INIT_CHECKPOINT} "
    f"world_size={WORLD_SIZE} per_device_batch={cfg['batch_size']} "
    f"effective_batch={cfg['batch_size'] * WORLD_SIZE} run_dir={run_dir}")

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

train_model = DistributedDataParallel(
    model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
) if DISTRIBUTED else model
torch.manual_seed(123 + RANK)


########## helpers ##########

def unnormalize(x):
    """(B,2,H,W,Z) [0,1] -> physical m/s (per-channel clip range)."""
    return x * (y_hi - y_lo) + y_lo


########## training loop ##########

if start_epoch >= n_epochs:
    raise RuntimeError(
        f"checkpoint already completed epoch {start_epoch}, but this run has only "
        f"{n_epochs} epoch(s)")

# rank-0 run lifecycle for monitoring agents: PROGRESS status=start now,
# status=completed after the smoke gate, status=failed on ANY exception
# (non-finite, OOM, config refusal) raised from the guarded block below.
run_t0 = time.perf_counter()
if IS_MAIN:
    log(format_progress("train", "start", objective=OBJECTIVE, preset=PRESET,
                        mode=TRAIN_MODE, world=WORLD_SIZE, epochs=n_epochs,
                        steps_per_epoch=len(train_loader),
                        train_horizon=TRAIN_HORIZON, lead_schedule=LEAD_SCHEDULE,
                        run_dir=run_dir), flush=True)

last_updates = last_skipped = 0
last_train_loss = last_val_rel = float("nan")
try:
    for ep in range(start_epoch, n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        train_model.train()
        t1 = time.time()
        t_batch = time.time()
        train_loss_sum = 0.0
        n_batch = 0
        succ_updates = 0
        skipped_updates = 0
        max_lead_seen = 0   # highest training lead J actually executed (smoke gate)
        # rank-0 interactive bar + periodic agent-readable status lines; other
        # ranks stay silent (never duplicate DDP progress). scope labels the
        # per-rank shard honestly; sample_per_s is the GLOBAL throughput.
        train_rep = ProgressReporter(
            "train", total=len(train_loader), unit="step",
            samples_per_unit=cfg["batch_size"] * WORLD_SIZE,   # GLOBAL sample/s
            desc=f"train ep{ep + 1}/{n_epochs}",
            context={"epoch": f"{ep + 1}/{n_epochs}", "scope": PROGRESS_SCOPE}
        ) if IS_MAIN else None
        for bi, (cond, target, _) in enumerate(train_loader):
            xx = cond.to(device, non_blocking=True)          # (B,14,H,W,Z) in [0,1]
            yy = target[:, 0].to(device, non_blocking=True)  # (B,2,H,W,Z)  in [0,1]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                if OBJECTIVE == "diffusion":
                    loss = train_model(yy, xx, mask=mask)
                elif TRAIN_HORIZON == 1:
                    # historical single-step path, kept bitwise identical
                    pred = train_model(xx, static_cond=mask if STATIC_MASK else None)
                    loss = masked_mse_loss(pred, yy, mask)
                else:
                    # detached multi-step (doc §5): the lead J follows the fixed
                    # batch schedule; J-1 detached self-feedback steps align the
                    # training input distribution with the 15-day rollout, then
                    # ONLY the J-th prediction carries gradients. J is a pure
                    # function of the batch index, so every DDP rank executes
                    # the same number of forwards per step (collective-safe).
                    lead = lead_for_batch(bi, TRAIN_HORIZON)
                    max_lead_seen = max(max_lead_seen, lead)
                    if lead > 1:
                        # feedback inference MUST run OUTSIDE the autocast
                        # weight cache: a forward under autocast+no_grad caches
                        # fp16 copies of the Linear-family weights as detached
                        # tensors, and the final grad forward inside the SAME
                        # autocast context would then reuse them, DISCONNECTING
                        # those params from the loss graph (their DDP hooks
                        # never fire -> "Expected to have finished reduction"
                        # on the next iteration). A nested disabled-autocast
                        # frame runs the feedback in fp32 without touching the
                        # cache; the final forward re-casts under grad and
                        # stays connected.
                        with autocast(device_type=device.type, enabled=False):
                            cur = detached_feedback_window(train_model, xx, lead)
                        pred = train_model(cur, static_cond=mask if STATIC_MASK else None)
                        loss = masked_mse_loss(
                            pred, target[:, lead - 1].to(device, non_blocking=True), mask)
                    else:
                        pred = train_model(xx, static_cond=mask if STATIC_MASK else None)
                        loss = masked_mse_loss(pred, yy, mask)
            finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
            if DISTRIBUTED:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError(
                    f"non-finite training loss {loss.detach().item()} at epoch {ep + 1} "
                    f"batch {bi} rank {RANK} "
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
            train_loss_sum += loss.detach().item()
            n_batch += 1
            if train_rep is not None:
                train_rep.update(
                    1, loss=f"{train_loss_sum / n_batch:.5f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    updates=succ_updates, skipped=skipped_updates)
            if (bi + 1) % 100 == 0:
                dt_b = time.time() - t_batch
                if train_rep is not None:
                    train_rep.note(
                        f"  [ep {ep + 1}] batch {bi + 1}/{len(train_loader)}  "
                        f"avg_loss {train_loss_sum / n_batch:.5f}  "
                        f"{dt_b / 100:.2f}s/batch  "
                        f"scale {scaler.get_scale():.4e}")
                t_batch = time.time()
        if train_rep is not None:
            train_rep.close(updates=succ_updates, skipped=skipped_updates)

        loss_stats = torch.tensor([train_loss_sum, n_batch], dtype=torch.float64,
                                  device=device)
        if DISTRIBUTED:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            min_updates = torch.tensor(succ_updates, device=device)
            total_skipped = torch.tensor(skipped_updates, device=device)
            dist.all_reduce(min_updates, op=dist.ReduceOp.MIN)
            dist.all_reduce(total_skipped, op=dist.ReduceOp.SUM)
            succ_updates = int(min_updates.item())
            skipped_updates = int(total_skipped.item())
        train_loss = float((loss_stats[0] / loss_stats[1].clamp(min=1)).item())

        # Validation windows are sharded across ranks and reduced, so every GPU
        # contributes and no rank idles long enough for a collective timeout.
        # fork_rng isolates the CPU (and, on CUDA, the current device) RNG so the
        # fixed rank seed cannot perturb the training RNG stream.
        train_model.eval()
        val_rel_sum, n_val = 0.0, 0
        # val windows are rank-sharded under DDP: this reporter's batch total
        # covers THIS rank's shard only (scope field says so explicitly)
        val_rep = ProgressReporter(
            "val", total=len(val_loader), unit="batch",
            desc=f"val ep{ep + 1}/{n_epochs}",
            context={"epoch": f"{ep + 1}/{n_epochs}", "scope": PROGRESS_SCOPE}
        ) if IS_MAIN else None
        # torch.device("cuda").index is None and fork_rng(devices=[None]) crashes;
        # current_device() is the actual ordinal of the device the model is on.
        rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(VAL_SEED + RANK)
            for cond, target, _ in val_loader:
                xx = cond.to(device, non_blocking=True)
                yy = target[:, 0].to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    pred = model.sample(xx, static_cond=mask if STATIC_MASK else None)
                batch_n = xx.shape[0]
                val_rel_sum += (masked_rel_l2(unnormalize(pred.float()), unnormalize(yy),
                                              mask) * batch_n)
                n_val += batch_n
                if val_rep is not None:
                    val_rep.update(1, rel_l2=f"{val_rel_sum / max(n_val, 1):.5f}")
        if val_rep is not None:
            val_rep.close()
        val_stats = torch.tensor([val_rel_sum, n_val], dtype=torch.float64, device=device)
        if DISTRIBUTED:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        if val_stats[1].item() == 0:
            raise RuntimeError("validation subset is empty across all ranks")
        val_rel = float((val_stats[0] / val_stats[1]).item())
        if not np.isfinite(val_rel):
            raise RuntimeError(f"non-finite validation metric {val_rel} at epoch {ep + 1}")

        dt = time.time() - t1
        hist["train"].append(train_loss)
        hist["val_rel"].append(val_rel)
        hist["time"].append(dt)
        log(f"epoch {ep + 1}/{n_epochs}  {dt:.1f}s  "
            f"train_loss {train_loss:.5f}  val_masked_relL2 {val_rel:.5f}  "
            f"updates/rank {succ_updates} (skipped across ranks {skipped_updates})  "
            f"max_lead {max_lead_seen if TRAIN_HORIZON > 1 else 1}  "
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
        if IS_MAIN:
            state = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val": best_val,
                "worse_epochs": worse_epochs,
                "config": {
                    "preset": PRESET, **cfg, "context": CONTEXT,
                    "train_mode": TRAIN_MODE,
                    "world_size": WORLD_SIZE,
                    "per_device_batch_size": cfg["batch_size"],
                    "effective_batch_size": cfg["batch_size"] * WORLD_SIZE,
                    "objective": OBJECTIVE,
                    "cond_chans": COND_CH,
                    "model_cond_chans": MODEL_COND_CH,
                    "static_mask_input": STATIC_MASK,
                    "target_ch": TARGET_CH,
                    "mask_scheme": MASK_SCHEME,
                    "train_horizon": TRAIN_HORIZON,
                    "lead_schedule": LEAD_SCHEDULE,
                    "feedback_detach": True,
                    "init_checkpoint": INIT_CHECKPOINT,
                    "init_weights_only": INIT_CHECKPOINT is not None,
                    "stats_sigma": float(stats["sigma"]),
                    "norm_lo": [float(x) for x in stats["lo"]],
                    "norm_hi": [float(x) for x in stats["hi"]],
                    "mask_version": mask_version(),
                },
            }
            if OBJECTIVE == "diffusion":
                state["config"]["sigma_data_scale"] = sigma_scale
                state["config"]["sigma_data"] = model.sigma_data
            else:
                state["config"]["residual_base"] = model.residual_base
                state["config"]["time_sigma"] = model.time_sigma
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
        if DISTRIBUTED:
            dist.barrier()

        last_updates, last_skipped = succ_updates, skipped_updates
        last_train_loss, last_val_rel = train_loss, val_rel

        if worse_epochs >= 2 and ep >= 1:
            log(f"early stop: val_masked_relL2 worsened for {worse_epochs} consecutive "
                f"epochs (best {best_val:.5f})")
            break

    if TRAIN_MODE == "smoke":
        if (not np.isfinite(last_train_loss) or not np.isfinite(last_val_rel)
                or last_updates < 1 or last_skipped != 0):
            raise RuntimeError(
                f"SMOKE FAIL: train_loss={last_train_loss}, val_rel={last_val_rel}, "
                f"updates/rank={last_updates}, skipped={last_skipped}")
        if TRAIN_HORIZON > 1 and max_lead_seen <= 1:
            # the smoke batches MUST actually exercise the detached feedback
            # path: a schedule/windowing regression that silently collapses to
            # J=1 would otherwise pass the gate while testing nothing (doc §6 WP3)
            raise RuntimeError(
                f"SMOKE FAIL: DIAFNO_TRAIN_HORIZON={TRAIN_HORIZON} but no J>1 "
                f"batch was executed in {n_batch} smoke batches "
                f"(max_lead_seen={max_lead_seen}); the lead schedule or the "
                "window alignment is broken")
        if IS_MAIN:
            required = (os.path.join(run_dir, "Ep1.pth"),
                        os.path.join(run_dir, "best.pth"), loss_file)
            missing = [path for path in required if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(f"SMOKE FAIL: missing outputs {missing}")
        log(f"SMOKE PASS: finite train/val, {last_updates} updates/rank, no AMP skips, "
            f"checkpoint outputs complete in {run_dir}")
except BaseException as exc:
    mark_progress_failed()          # the fallback hook must not duplicate this
    if IS_MAIN:
        print(format_progress("train", "failed", stage="run",
                              error=f"{type(exc).__name__}: {exc}",
                              elapsed_s=f"{time.perf_counter() - run_t0:.1f}"),
              flush=True)
    raise

if IS_MAIN:
    log(format_progress("train", "completed", objective=OBJECTIVE,
                        epochs_done=len(hist["train"]),
                        best_val=f"{best_val:.5f}",
                        elapsed_s=f"{time.perf_counter() - run_t0:.1f}",
                        run_dir=run_dir), flush=True)

log(f"done. best val_masked_relL2 = {best_val:.5f}; checkpoints in {run_dir}")
if DISTRIBUTED:
    dist.destroy_process_group()
