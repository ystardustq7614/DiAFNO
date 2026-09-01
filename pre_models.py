#!/usr/bin/env python3
"""Deterministic persistence-residual model for the PRE forecast task (imported
by pre_trainer.py and pre_evaluate.py — keep this module side-effect free).

`PersistenceResidualIAFNO` wraps the SAME IAFNODiff backbone used by the
conditional EDM as a condition-only deterministic regressor:

    prediction = base + residual,     base = condition[:, -target_ch:]

`base` is the last-day persistence (the final target_ch channels of the
day-major u/v condition). The backbone's patch head is zero-initialized, so an
UNTRAINED model is EXACTLY persistence — the first optimizer step only moves
the head, and deeper layers start receiving gradients from the second step.

Data flow note (do NOT "fix" this): IAFNODiff.forward(x, time, x_self_cond)
concatenates `x_self_cond` with `x`, and in the diffusion path the "self_cond"
slot actually CARRIES THE 14-CHANNEL CONDITION (`ElucidatedDiffusion.sample`
passes the condition there). The wrapper mirrors that layout exactly:
    residual = net(x=zeros(target_ch), time=const c_noise, x_self_cond=cond)
so the patch-embed sees cond (14 ch) + zeros (2 ch) = in_chans (16 ch),
identical to the diffusion path. `time` is the constant EDM c_noise embedding
of `time_sigma` (0.25 * log(time_sigma)); the deterministic model has no noise
schedule, so any fixed constant is valid and is recorded in checkpoints.

`sample()` is rollout-compatible by duck typing: `pre_rollout` calls
`model.sample(cur, num_sample_steps=..., clamp=...)`; the deterministic
implementation ignores `num_sample_steps` and clamps to [0, 1] like the EDM
sampler's unnormalized output.

`masked_mse_loss` mirrors the masked semantics of
`ElucidatedDiffusion.forward`: per-sample mean over VALID elements only
(broadcastable mask, 1 = ocean), then a mean over the batch.
"""
import math

import torch
import torch.nn as nn


def masked_mse_loss(pred, target, mask):
    """Masked MSE: per-sample mean over valid elements, then batch mean.

    pred/target: (B, C, H, W, Z); mask broadcastable to that shape with
    1 = valid (ocean) and 0 = land. Land cells contribute nothing and the
    denominator is each sample's own valid-element count (same convention as
    diffusion.ElucidatedDiffusion.forward with a mask).
    """
    mse = (pred - target) ** 2
    m = mask.expand_as(mse)
    per_sample = (mse * m).sum(dim=(1, 2, 3, 4)) / m.sum(dim=(1, 2, 3, 4)).clamp(min=1.)
    return per_sample.mean()


class PersistenceResidualIAFNO(nn.Module):
    """Thin deterministic wrapper: last-day persistence + zero-init IAFNO residual.

    `net` must be an IAFNODiff with self_condition=True and in_chans equal to
    target channels + external condition channels, exactly as built for the
    conditional EDM. The state dict is `net.*` under this wrapper, matching the
    ElucidatedDiffusion layout.
    """

    def __init__(self, net, time_sigma=0.002):
        super().__init__()
        if not net.self_condition:
            raise ValueError(
                "PersistenceResidualIAFNO requires an IAFNODiff with "
                "self_condition=True (the condition enters through the "
                "x_self_cond slot)")
        self.net = net
        self.target_ch = int(net.out_chans)
        self.cond_chans = int(net.in_chans) - self.target_ch
        if self.cond_chans <= 0:
            raise ValueError(
                f"IAFNODiff in_chans={net.in_chans} leaves no condition "
                f"channels beyond out_chans={net.out_chans}")
        self.time_sigma = float(time_sigma)
        self.residual_base = "last_day"
        # zero-init the residual head: untrained forward() is EXACTLY `base`
        nn.init.zeros_(net.head.weight)

    def forward(self, cond, static_cond=None):
        """(B, cond_ch, H, W, Z) normalized condition -> (B, target_ch, H, W, Z).

        With `static_cond` (Phase-5 mask-input A/B, e.g. (1, 2, H, W, Z)
        bivariate rho masks broadcast over the batch) the backbone's
        x_self_cond slot receives `cat([cond, static_cond], dim=1)` — the
        DYNAMIC window must stay pure so `base = cond[:, -target_ch:]` is
        always the last-day persistence. Without `static_cond` the behaviour
        is bitwise identical to the historical 14-channel path.
        """
        if static_cond is not None:
            if static_cond.dim() != 5 or \
                    static_cond.shape[0] not in (1, cond.shape[0]):
                raise AssertionError(
                    f"static_cond shape {tuple(static_cond.shape)} is not "
                    f"broadcastable to batch {cond.shape[0]}")
            if static_cond.shape[2:] != cond.shape[2:]:
                raise AssertionError(
                    f"static_cond spatial shape {tuple(static_cond.shape[2:])} "
                    f"!= condition {tuple(cond.shape[2:])}")
            if static_cond.shape[0] == 1 and cond.shape[0] > 1:
                static_cond = static_cond.expand(cond.shape[0], -1, -1, -1, -1)
            x_self_cond = torch.cat([cond, static_cond], dim=1)
        else:
            x_self_cond = cond
        if x_self_cond.shape[1] != self.cond_chans:
            raise AssertionError(
                f"condition channels {x_self_cond.shape[1]} != expected "
                f"{self.cond_chans}")
        base = cond[:, -self.target_ch:]
        batch = cond.shape[0]
        time = torch.full((batch,), 0.25 * math.log(self.time_sigma),
                          device=cond.device)
        residual = self.net(torch.zeros_like(base), time, x_self_cond)
        return base + residual

    def sample(self, cond, batch_size=None, num_sample_steps=None, clamp=True,
               static_cond=None):
        """Deterministic prediction; `num_sample_steps` is accepted and ignored
        (rollout compatibility with the EDM sampler's call signature). With
        clamp=True the normalized prediction is clamped to [0, 1], matching the
        EDM sampler output range."""
        pred = self.forward(cond, static_cond=static_cond)
        if clamp:
            pred = pred.clamp(0., 1.)
        return pred
