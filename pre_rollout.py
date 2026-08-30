#!/usr/bin/env python3
"""Side-effect-free autoregressive ensemble rollout for PRE evaluation.

The conditional EDM maps a 14-channel (2*CONTEXT) normalized [0,1] condition
window to the next-day u/v in [0,1]. A multi-day forecast is an autoregressive
rollout: sample day 8 from days 1..7, then shift the window (drop the oldest
day, append the prediction) and repeat.

The model is DUCK-TYPED: only `model.sample(cur, num_sample_steps=...,
clamp=...)` is called, so both the conditional EDM (`ElucidatedDiffusion`) and
the deterministic baseline (`PersistenceResidualIAFNO.sample`, which ignores
`num_sample_steps` and consumes no RNG) work unchanged. For a deterministic
model the per-window seeds and the ensemble size are inert: trajectories are
bitwise reproducible regardless of seeding and all members are identical.

Ensemble: each of the E members gets its own copy of the condition and runs a
COMPLETELY independent rollout — a member's predictions update only its own
condition window. Members meet only at the end, where per-day predictions are
averaged (ensemble_mean) before the point-prediction RMSE is computed.

`remask_feedback` (default False = historical behavior): when enabled, EVERY
prediction is multiplied by the ocean mask (land -> 0) BEFORE it is stored and
before it enters the next condition window. Metrics exclude land anyway, so
this only changes what the model is fed back (the historical path fed back
unmasked land fill values). The mask must be broadcastable to the prediction
shape (e.g. (1, 2, H, W, Z) or (2, H, W, Z)), 1 = ocean; the final on/off
decision belongs to the Phase-5 A/B and is recorded in eval metadata.

Sampling runs under torch.amp.autocast (device_type follows the tensors), which
reproduces the historical evaluation path (AMP on CUDA; a no-op for FP32-only
models on CPU).

Seeding is strictly per-window: pass `seeds` (one int per batch row) so each
window's trajectory is determined by its OWN seed and is therefore independent
of the batch size, the loader batching, and the other windows in the batch.
A scalar `seed` is kept for the single-window / legacy path (one seed applied
once for the whole batch).

No side effects at import time; no dependency on pre_dataset.py, pre_metrics.py
or the model.
"""
import torch


def _sample(model, cur, num_sample_steps, clamp):
    """model.sample under autocast — matches the historical evaluation path
    (CUDA AMP; CPU autocast is a no-op for models without autocast-sensitive
    ops but is applied uniformly for identical semantics)."""
    device_type = "cuda" if cur.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type):
        return model.sample(cur, num_sample_steps=num_sample_steps, clamp=clamp)


def _rollout_one(model, cond, horizon, num_sample_steps, clamp,
                 remask_feedback=False, ocean_mask=None):
    """Rollout one batch of windows (already expanded) under one seed.

    cond: (B, C, H, W, Z); returns (B, horizon, 2, H, W, Z) float32 preds.
    With remask_feedback=True, each prediction is remasked (land -> 0) before
    it is stored AND before it is appended to the next condition window.
    """
    if remask_feedback:
        assert ocean_mask is not None, "remask_feedback=True requires ocean_mask"
    cur = cond
    preds = []
    for _ in range(int(horizon)):
        p = _sample(model, cur, num_sample_steps, clamp).float()
        if remask_feedback:
            p = p * ocean_mask
        preds.append(p)
        cur = torch.cat([cur[:, 2:], p], dim=1)     # drop oldest day, append own prediction
    return torch.stack(preds, dim=1)                # (B, L, 2, H, W, Z)


def expand_ensemble(cond, ensemble_size):
    """Repeat each condition `ensemble_size` times -> (B*E, C, H, W, Z).

    Each ensemble member is an independent copy of the same condition window;
    E=1 returns a (fresh) copy of the input so the RNG stream consumed by the
    rollout is identical to the plain single-trajectory loop.
    """
    assert cond.dim() == 5, cond.shape
    assert int(ensemble_size) >= 1
    if ensemble_size == 1:
        return cond.clone()
    return cond.repeat_interleave(int(ensemble_size), dim=0)


def ensemble_rollout(model, cond, horizon, ensemble_size=1, num_sample_steps=None,
                     seed=None, seeds=None, clamp=True,
                     remask_feedback=False, ocean_mask=None):
    """Autoregressive rollout with `ensemble_size` fully independent members.

    Args:
        model: object with sample(cur, num_sample_steps=None, clamp=True)
            -> (B*, 2, H, W, Z) next-day prediction in [0,1]
            (e.g. ElucidatedDiffusion or PersistenceResidualIAFNO).
        cond: (B, 2*CONTEXT, H, W, Z) normalized [0,1] condition windows.
        horizon: number of rollout steps (lead days), >= 1.
        ensemble_size: number of independent members, >= 1 (inert for a
            deterministic model: members are identical).
        num_sample_steps: sampler steps (None -> the model's default; ignored
            by deterministic models).
        seed: scalar RNG seed applied ONCE before rolling out the whole batch
            (legacy path; members share one RNG stream but consume independent
            draws). Mutually exclusive with `seeds`.
        seeds: per-window seeds (len == B); each window is rolled out on its
            own RNG stream seeded by seeds[w], so the trajectory of window w
            does NOT depend on the batch size or batching of other windows.
        clamp: passed through to model.sample.
        remask_feedback: when True, every prediction is multiplied by
            `ocean_mask` (land -> 0) before storage and before it is fed back
            into the next condition window (default False = historical
            unmasked feedback).
        ocean_mask: broadcastable ocean mask (1 = valid ocean, 0 = land), e.g.
            (1, 2, H, W, Z); required iff remask_feedback is True.

    Returns (B, E, horizon, 2, H, W, Z) float32 normalized predictions, one
    slice per member. E=1 reproduces the plain sequential rollout exactly
    (same RNG consumption, same values for the same seed).
    """
    assert cond.dim() == 5, cond.shape
    assert cond.shape[1] % 2 == 0, "condition must be day-major interleaved"
    B = cond.shape[0]
    E = int(ensemble_size)
    assert E >= 1
    assert not (seed is not None and seeds is not None), "pass either seed or seeds"
    if remask_feedback:
        assert ocean_mask is not None, "remask_feedback=True requires ocean_mask"

    if seeds is not None:
        seeds = [int(s) for s in seeds]
        assert len(seeds) == B, (len(seeds), B)
        outs = []
        for w in range(B):
            torch.manual_seed(seeds[w])                       # window-scoped RNG
            outs.append(_rollout_one(model, expand_ensemble(cond[w:w + 1], E),
                                     horizon, num_sample_steps, clamp,
                                     remask_feedback, ocean_mask))
        return torch.stack(outs, dim=0)                       # (B, E, L, 2, H, W, Z)

    if seed is not None:
        torch.manual_seed(seed)
    out = _rollout_one(model, expand_ensemble(cond, E), horizon,
                       num_sample_steps, clamp, remask_feedback, ocean_mask)
    return out.view(B, E, out.shape[1], *out.shape[2:])       # (B, E, L, 2, H, W, Z)


def ensemble_mean(preds):
    """(B, E, L, C, H, W, Z) -> (B, L, C, H, W, Z) point prediction (member mean)."""
    assert preds.dim() == 7, preds.shape
    return preds.mean(dim=1)