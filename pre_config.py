#!/usr/bin/env python3
"""Shared configuration for the PRE_ocean_data forecast task (imported by
pre_trainer.py and pre_evaluate.py — keep this module side-effect free)."""

OUT_ROOT = "/data2/user/zyq/checkpoints/PRE_lr3e4"

PRESETS = {
    # smoke test: verify the whole pipeline on the surface layer.
    "surface_smoke": dict(
        depth_index=29,            # 29 = sea surface (0 = bottom)
        patch_size=(4, 3, 1),      # 400/4=100, 441/3=147, 1/1=1 -> 14,700 tokens
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,            # window subsampling on train split
        max_train_windows=None,    # set e.g. 2000 for a faster dry run
        sampling_steps=32,
        val_windows=24,            # uniform val windows per epoch (whole val period)
        lr=3e-4,
    ),
    # full 3D: 30 sigma layers, 400/4 x 441/3 x 30/2 = 100x147x15 = 220,500 tokens.
    # memory-tight on a 24GB card: start with batch_size=1; if OOM, reduce embed_dim
    # or implicit_layer before touching anything else.
    "full3d": dict(
        depth_index=None,
        patch_size=(4, 3, 2),
        embed_dim=128,
        implicit_layer=2,
        explicit_layer=4,
        batch_size=1,
        num_workers=2,
        num_epochs=50,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=16,
        lr=1e-3,
    ),
}

CONTEXT = 7        # condition days
HORIZON = 15       # rollout days
TARGET_CH = 2      # u, v

# EDM sigma_data lives in the image space that ElucidatedDiffusion actually
# uses: diffusion.py normalizes training images with `images * 2 - 1`, i.e.
# the data distribution seen by the EDM is [-1, 1], whose std is TWICE the
# std of the [0, 1]-normalized stats cache. stats["sigma"] keeps storing the
# [0, 1]-space value; training and evaluation MUST both go through
# sigma_data_from_stats() / sigma_data_from_checkpoint() below.
SIGMA_DATA_SCALE = 2.0


def sigma_data_from_stats(stats_sigma):
    """[0,1]-space pooled sigma -> EDM sigma_data in the [-1,1] image space."""
    return SIGMA_DATA_SCALE * float(stats_sigma)


def sigma_data_from_checkpoint(checkpoint, stats_sigma):
    """Resolve the EDM sigma_data for a checkpoint.

    Priority: the checkpoint's own config["sigma_data"] (written by the
    fixed-scale trainer). Legacy checkpoints (no config / no sigma_data field)
    fall back to the OLD scale `stats["sigma"]` (NOT the doubled value) and
    report used_checkpoint=False so the caller can print an explicit notice.
    Returns (sigma_data: float, used_checkpoint_value: bool).
    """
    cfg = (checkpoint or {}).get("config") or {}
    if "sigma_data" in cfg:
        return float(cfg["sigma_data"]), True
    return float(stats_sigma), False


def resume_sigma_decision(sd_ckpt, sd_current, policy):
    """Decide which sigma_data a resume run must use.

    sd_ckpt: the checkpoint's sigma_data (resolved via
        sigma_data_from_checkpoint). sd_current: the current run's sigma_data
        (sigma_data_from_stats). policy is one of:
        "error"   (default): mismatch -> RuntimeError; never mix scales silently.
        "migrate"           : explicit scale migration — keep sd_current.
        "adopt"             : explicit legacy continuation — use sd_ckpt.
    Returns (sigma_data: float, adopted: bool). Matching scales always return
    (sd_current, False) regardless of policy.
    """
    sd_ckpt = float(sd_ckpt)
    sd_current = float(sd_current)
    mismatch = abs(sd_ckpt - sd_current) > 1e-6
    if not mismatch:
        return sd_current, False
    if policy == "error":
        raise RuntimeError(
            f"resume scale mismatch: checkpoint sigma_data={sd_ckpt:.5f} vs "
            f"current sigma_data={sd_current:.5f}; refusing to continue. Set "
            f"RESUME_SIGMA_POLICY='migrate' to keep the current scale (explicit "
            f"scale migration) or 'adopt' to continue in the checkpoint's old "
            f"scale (outputs written back into the checkpoint's directory)")
    if policy == "migrate":
        return sd_current, False
    if policy == "adopt":
        return sd_ckpt, True
    raise ValueError(f"unknown RESUME_SIGMA_POLICY {policy!r} "
                     f"(expected 'error', 'migrate' or 'adopt')")


def run_tag_for(preset, sd2=True):
    """Checkpoint/output dir tag. sd2=True appends the fixed-scale suffix so a
    re-trained run NEVER shares a directory with the legacy (sd1) runs."""
    cfg = PRESETS[preset]
    tag = (f"{preset}_BS{cfg['batch_size']}_EMD{cfg['embed_dim']}"
           f"_I{cfg['implicit_layer']}_E{cfg['explicit_layer']}"
           f"_S{cfg['sampling_steps']}_C{CONTEXT}")
    if sd2:
        tag += "_SD2"
    return tag
