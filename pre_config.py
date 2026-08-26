#!/usr/bin/env python3
"""Shared configuration for the PRE_ocean_data forecast task (imported by
pre_trainer.py and pre_evaluate.py — keep this module side-effect free)."""

OUT_ROOT = "/data2/user/zyq/checkpoints/PRE"

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
        lr=1e-3,
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


def run_tag_for(preset):
    cfg = PRESETS[preset]
    return (f"{preset}_BS{cfg['batch_size']}_EMD{cfg['embed_dim']}"
            f"_I{cfg['implicit_layer']}_E{cfg['explicit_layer']}"
            f"_S{cfg['sampling_steps']}_C{CONTEXT}")
