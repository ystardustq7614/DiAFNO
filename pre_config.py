#!/usr/bin/env python3
"""Shared configuration for the PRE_ocean_data forecast task (imported by
pre_trainer.py and pre_evaluate.py — keep this module side-effect free).

Also hosts the two small runtime helpers shared by trainer and evaluation:
the training-objective configuration (diffusion vs deterministic
persistence-residual) and the rank-0 terminal progress reporting
(`ProgressReporter`: interactive tqdm bar + parseable PROGRESS key=value
status lines; no monitoring service, no new dependency beyond tqdm).
"""
import re
import sys
import threading
import time

from tqdm import tqdm

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
    # Work package 5 representative layers (experiment 11): MIDDLE (sigma
    # index 14) and BOTTOM (sigma index 0). Architecture, patch, budget and
    # protocol are IDENTICAL to surface_smoke — the ONLY difference is the
    # probed depth index (never translate a sigma index into a fixed meter
    # depth). Run tags: middle_smoke_*/bottom_smoke_*.
    "middle_smoke": dict(
        depth_index=14,            # middle representative sigma layer
        patch_size=(4, 3, 1),
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=24,
        lr=1e-3,
    ),
    "bottom_smoke": dict(
        depth_index=0,             # bottom representative sigma layer (seabed)
        patch_size=(4, 3, 1),
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=24,
        lr=1e-3,
    ),
}

# Pipeline smoke runs keep the production architecture/grid but execute only a
# handful of real optimizer steps per rank.  This catches data, memory, AMP,
# sampling and checkpoint failures without creating a second toy model.
SMOKE_BATCHES_PER_RANK = 4

CONTEXT = 7        # condition days
HORIZON = 15       # rollout days
TARGET_CH = 2      # u, v

########## training objective ##########

# "diffusion"            : conditional EDM (legacy default; legacy checkpoints
#                          predate the objective field and always resolve here)
# "persistence_residual" : deterministic PersistenceResidualIAFNO baseline
#                          (prediction = last-day persistence + zero-init residual)
OBJECTIVES = ("diffusion", "persistence_residual")
DEFAULT_OBJECTIVE = "diffusion"

# static mask input scheme recorded in checkpoints; only the bivariate rho
# mask exists today (mask_u_rho + mask_v_rho as two channels)
MASK_SCHEME = "bivariate_rho"

# constant sigma behind the deterministic model's c_noise time embedding
# (time = 0.25 * log(time_sigma); no noise schedule exists, any fixed constant
# is valid — the value is recorded in checkpoints for exact rebuilds)
RESIDUAL_TIME_SIGMA = 0.002

# Phase-5 mask-input A/B (arm B): append the two bivariate rho mask channels
# (mask_u_rho / mask_v_rho) to the backbone's condition. The DYNAMIC window
# stays 14-channel everywhere (dataset, rollout sliding window, persistence
# base); the masks are forwarded to the model separately via pre_rollout's
# `static_cond`. Enabled per-run with DIAFNO_STATIC_MASK=1 and recorded in
# checkpoint config as `static_mask_input` (run tag suffix "_MSK").
STATIC_MASK_ENV = "DIAFNO_STATIC_MASK"
STATIC_MASK_CHANNELS = 2


def static_mask_input(env=None):
    """Read the DIAFNO_STATIC_MASK flag ("1"/"true"/"yes" -> True)."""
    import os
    value = (env if env is not None else os.environ.get(STATIC_MASK_ENV, ""))
    return str(value).strip().lower() in ("1", "true", "yes", "on")


########## detached multi-step training (work package 2; doc §5) ##########

# detached autoregressive multi-step horizon K ("MS{K}"): the trainer rolls the
# model's OWN (no_grad, clamped) predictions forward for J-1 steps and
# backpropagates only the J-th step (doc §5 pseudocode). K=1 is the exact
# historical single-step teacher-forcing path. Only the deterministic
# persistence_residual objective with static_mask_input=False is allowed.
TRAIN_HORIZON_ENV = "DIAFNO_TRAIN_HORIZON"

# weights-only initialization source (e.g. experiment-07 Ep10): model weights
# are loaded, optimizer/scheduler/scaler/epoch/history are NOT (the source
# cosine schedule is finished; MS runs start a fresh optimizer at a lower LR).
# Mutually exclusive with DIAFNO_CHECKPOINT (full resume).
INIT_CHECKPOINT_ENV = "DIAFNO_INIT_CHECKPOINT"

# defaults applied ONLY when train_horizon > 1 (doc §6 WP3 frozen config:
# fresh optimizer, LR 1e-4, at most 5 epochs; smoke mode still overrides the
# epoch count). Recorded in checkpoint config like every other hyperparameter.
MS_DEFAULTS = dict(lr=1e-4, num_epochs=5)


def train_horizon(env=None):
    """Read DIAFNO_TRAIN_HORIZON ("MS{K}", int >= 1; default 1 = single-step)."""
    import os
    value = (env if env is not None else os.environ.get(TRAIN_HORIZON_ENV, ""))
    value = str(value).strip()
    if not value:
        return 1
    try:
        horizon = int(value)
    except ValueError:
        raise ValueError(f"{TRAIN_HORIZON_ENV}={value!r} is not an integer")
    if horizon < 1:
        raise ValueError(f"{TRAIN_HORIZON_ENV}={horizon} must be >= 1")
    return horizon


def init_checkpoint(env=None):
    """Read DIAFNO_INIT_CHECKPOINT (weights-only init source; None by default)."""
    import os
    value = (env if env is not None else os.environ.get(INIT_CHECKPOINT_ENV, ""))
    value = str(value).strip()
    return os.path.expanduser(value) if value else None


def lead_for_batch(batch_index, train_horizon):
    """Training lead J for a batch index (doc §5.1 fixed schedule).

    K=1  -> always 1 (the historical single-step path, schedule inert).
    K>1  -> even batch indices keep the day-1 anchor (50% of batches); odd
            indices cycle 2..K, i.e. MS5 produces 1,2,1,3,1,4,1,5,1,2,...

    Pure function of (batch_index, K): no RNG, no global state, so every DDP
    rank derives the SAME J for the same step (all ranks have equal batch
    counts because both the sampler and the loader use drop_last=True).
    """
    k = int(train_horizon)
    if k <= 1:
        return 1
    bi = int(batch_index)
    if bi % 2 == 0:
        return 1
    return 2 + ((bi // 2) % (k - 1))


def lead_schedule_str(train_horizon):
    """Canonical one-period schedule string for logs/checkpoint metadata
    (e.g. MS5 -> "1,2,1,3,1,4,1,5"; K=1 -> "1")."""
    k = int(train_horizon)
    if k <= 1:
        return "1"
    return ",".join(str(lead_for_batch(i, k)) for i in range(2 * (k - 1)))


def check_multistep_config(ckpt_cfg, train_horizon_now, schedule_now):
    """Resume guard for the multi-step semantics recorded in a checkpoint.

    A checkpoint that was trained with (or without) detached multi-step
    feedback must never be resumed under different semantics:
      - train_horizon must match exactly; legacy checkpoints without the field
        are only compatible with K=1 (they predate multi-step entirely);
      - when BOTH sides record a lead schedule, the canonical strings must
        match (a schedule change would silently alter the training
        distribution of a resumed run).
    """
    ckpt_cfg = ckpt_cfg or {}
    recorded = ckpt_cfg.get("train_horizon")
    if recorded is None:
        if int(train_horizon_now) != 1:
            raise RuntimeError(
                "checkpoint has no config.train_horizon (pre-multi-step) but "
                f"DIAFNO_TRAIN_HORIZON={train_horizon_now}; multi-step training "
                "cannot resume a single-step run — use DIAFNO_INIT_CHECKPOINT "
                "(weights-only init) instead")
        return
    if int(recorded) != int(train_horizon_now):
        raise RuntimeError(
            f"checkpoint train_horizon={int(recorded)} vs current "
            f"{int(train_horizon_now)}; refusing to resume across a "
            "multi-step horizon change")
    recorded_schedule = ckpt_cfg.get("lead_schedule")
    if recorded_schedule is not None and \
            str(recorded_schedule) != str(schedule_now):
        raise RuntimeError(
            f"checkpoint lead_schedule={recorded_schedule!r} vs current "
            f"{schedule_now!r}; refusing to resume across a schedule change")


def validate_objective(objective):
    """Normalize/validate a training-objective name."""
    objective = str(objective).lower()
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    return objective


def objective_from_checkpoint(checkpoint, default=DEFAULT_OBJECTIVE):
    """Objective recorded in a checkpoint (dict with optional "config").

    Legacy checkpoints predate the objective field and are always diffusion.
    """
    cfg = (checkpoint or {}).get("config") or {}
    return validate_objective(cfg.get("objective", default))


def ensure_objective_compatible(checkpoint, objective):
    """Refuse to load a checkpoint into a model of the OTHER objective class
    (resume or evaluation rebuild). Returns the checkpoint's objective."""
    ckpt_obj = objective_from_checkpoint(checkpoint)
    if ckpt_obj != objective:
        raise RuntimeError(
            f"checkpoint objective={ckpt_obj!r} is incompatible with the "
            f"requested {objective!r}; refusing to load a different model class")
    return ckpt_obj


def check_norm_fingerprint(ckpt_cfg, lo, hi, mask_version_now, tol=1e-6):
    """Verify a checkpoint's data-semantics fingerprint (normalization range +
    ocean mask version) against the CURRENT stats/masks before resuming or
    evaluating with it: a mismatch means the checkpoint was trained on
    different normalization semantics than the run would now use, which must
    never happen silently.

    Raises RuntimeError on any mismatch; RETURNS a list of WARNING strings for
    legacy checkpoints that predate the recorded fields (cannot be verified —
    the caller must log them). `lo`/`hi` are the current per-variable stats
    (any sequence of floats); `mask_version_now` is pre_dataset.mask_version().
    """
    ckpt_cfg = ckpt_cfg or {}
    warnings = []
    lo_now = [float(x) for x in lo]
    hi_now = [float(x) for x in hi]
    if "norm_lo" in ckpt_cfg and "norm_hi" in ckpt_cfg:
        lo_ck = [float(x) for x in ckpt_cfg["norm_lo"]]
        hi_ck = [float(x) for x in ckpt_cfg["norm_hi"]]
        same = (len(lo_ck) == len(lo_now) and len(hi_ck) == len(hi_now)
                and all(abs(a - b) <= tol for a, b in zip(lo_ck, lo_now))
                and all(abs(a - b) <= tol for a, b in zip(hi_ck, hi_now)))
        if not same:
            raise RuntimeError(
                f"checkpoint normalization fingerprint mismatch: "
                f"config.norm_lo/norm_hi={lo_ck}/{hi_ck} vs current {lo_now}/{hi_now}; "
                "the stats cache changed since this checkpoint was trained — "
                "refusing to continue/evaluate with different normalization semantics")
    else:
        warnings.append("checkpoint has no config.norm_lo/norm_hi (legacy); "
                        "normalization range could NOT be verified")
    if "mask_version" in ckpt_cfg:
        if str(ckpt_cfg["mask_version"]) != str(mask_version_now):
            raise RuntimeError(
                f"checkpoint mask_version={ckpt_cfg['mask_version']!r} vs current "
                f"{str(mask_version_now)!r}; the ocean masks changed since this "
                "checkpoint was trained — refusing")
    else:
        warnings.append("checkpoint has no config.mask_version (legacy); "
                        "mask fingerprint could NOT be verified")
    return warnings


def check_residual_time_sigma(ckpt_cfg, time_sigma):
    """The deterministic model's constant time embedding is part of its
    semantics: refuse to resume from a persistence-residual checkpoint recorded
    with a different (or unrecorded) time_sigma. (Evaluation ADOPTS the
    checkpoint's own value, so this guard is for training resume only.)"""
    ckpt_cfg = ckpt_cfg or {}
    if "time_sigma" not in ckpt_cfg:
        raise RuntimeError(
            "persistence-residual checkpoint has no config.time_sigma; the "
            "constant time embedding cannot be verified — refusing to resume")
    if abs(float(ckpt_cfg["time_sigma"]) - float(time_sigma)) > 1e-9:
        raise RuntimeError(
            f"checkpoint time_sigma={float(ckpt_cfg['time_sigma'])!r} vs current "
            f"{float(time_sigma)!r}; the residual time embedding changed — "
            "refusing to resume")

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


def training_config(preset, mode="full", world_size=1, train_horizon=1):
    """Return an isolated mutable config for a smoke or full training run.

    ``batch_size`` remains the per-device batch size.  A smoke run contains
    exactly ``SMOKE_BATCHES_PER_RANK`` full batches on every DDP rank, uses one
    epoch and short sampling, while preserving the selected preset's model
    architecture and physical grid.

    A detached multi-step run (train_horizon > 1) uses the frozen MS_DEFAULTS
    (lr/num_epochs) instead of the preset's single-step values; smoke mode
    still reduces the epoch count afterwards.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {tuple(PRESETS)}")
    if mode not in ("smoke", "full"):
        raise ValueError(f"unknown training mode {mode!r}; expected 'smoke' or 'full'")
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    train_horizon = int(train_horizon)
    if train_horizon < 1:
        raise ValueError(f"train_horizon must be >= 1, got {train_horizon}")

    cfg = dict(PRESETS[preset])
    if train_horizon > 1:
        cfg.update(MS_DEFAULTS)
    if mode == "smoke":
        cfg.update(
            num_epochs=1,
            max_train_windows=(SMOKE_BATCHES_PER_RANK * world_size
                               * cfg["batch_size"]),
            sampling_steps=4,
            val_windows=max(2, world_size),
        )
    return cfg


def run_tag_for(preset, sd2=True, config=None, objective=DEFAULT_OBJECTIVE,
                static_mask=False, train_horizon=1):
    """Checkpoint/output dir tag. sd2=True appends the fixed-scale suffix so a
    re-trained run NEVER shares a directory with the legacy (sd1) runs.
    objective="persistence_residual" additionally appends "_RES" so the
    deterministic baseline never shares a directory with a diffusion run.
    static_mask=True appends "_MSK" so the Phase-5 mask-input arm never shares
    a directory with the 14-channel baseline. train_horizon > 1 appends
    "_MS{K}" so a detached multi-step run never shares a directory with a
    single-step run (K=1 keeps the historical tag unchanged)."""
    cfg = PRESETS[preset] if config is None else config
    tag = (f"{preset}_BS{cfg['batch_size']}_EMD{cfg['embed_dim']}"
           f"_I{cfg['implicit_layer']}_E{cfg['explicit_layer']}"
           f"_S{cfg['sampling_steps']}_C{CONTEXT}")
    if sd2:
        tag += "_SD2"
    if validate_objective(objective) == "persistence_residual":
        tag += "_RES"
    if static_mask:
        tag += "_MSK"
    if int(train_horizon) > 1:
        tag += f"_MS{int(train_horizon)}"
    return tag


def training_run_tag(preset, config, mode="full", world_size=1,
                     objective=DEFAULT_OBJECTIVE, static_mask=False,
                     train_horizon=1):
    """Run tag with smoke/DDP isolation; single-GPU full tags stay legacy-compatible."""
    tag = run_tag_for(preset, config=config, objective=objective,
                      static_mask=static_mask, train_horizon=train_horizon)
    if mode == "smoke":
        tag += "_SMOKE"
    if int(world_size) > 1:
        tag += f"_DDP{int(world_size)}"
    return tag


########## rank-0 terminal progress (trainer + evaluation) ##########

# minimum spacing between periodic PROGRESS status lines while a phase runs
# (start/close/failed lines are ALWAYS emitted, even inside this interval)
PROGRESS_INTERVAL_S = 30.0


def _progress_value(value):
    """key=value tokens must contain NO whitespace at all: every run of
    spaces/newlines/tabs becomes a single underscore, so even a multi-line
    exception message keeps the status line single-line and parseable."""
    return re.sub(r"\s+", "_", str(value))


def format_progress(phase, status, **fields):
    """One parseable status line, e.g.
    'PROGRESS phase=train epoch=1/4 step=120/2101 elapsed_s=91.2 eta_s=1506.4
    step_per_s=1.31 sample_per_s=5.24 loss=0.0187 lr=1e-4 status=running'.
    `phase` is always the first field and `status` always the last, so both
    simple split() and per-token key=value parsing stay stable."""
    parts = ["PROGRESS", f"phase={_progress_value(phase)}"]
    for key, value in fields.items():
        parts.append(f"{key}={_progress_value(value)}")
    parts.append(f"status={_progress_value(status)}")
    return " ".join(parts)


class ProgressReporter:
    """Rank-0 progress for ONE phase (a training epoch, a validation pass, an
    evaluation run). Not a logging framework: a thin tqdm wrapper plus the
    agent-readable PROGRESS lines required for non-interactive runs.

    Interactive TTY (stream.isatty()): a single-line tqdm bar (desc, count/
    total, rate, ETA) plus a postfix string built from each update()'s fields.

    Non-interactive (pipe/file redirect/monitoring agent): NO bar and no
    carriage-return animation; instead a complete, newline-terminated and
    immediately flushed `PROGRESS key=value` line is emitted at start, at
    close, on failure, and at least every `interval` seconds while running.
    The periodic line is TIME-DRIVEN, not update-driven: a daemon heartbeat
    thread emits it even when a single batch/rollout step blocks the caller
    for longer than `interval` (no mid-batch silence).

    Status vocabulary (stable parsing contract):
      start       phase started (both modes, always)
      running     periodic heartbeat / update-driven progress line
      phase_done  THIS reporter's phase finished (intermediate: one training
                  epoch, the eval rollout loop) — NOT the end of the script
      failed      the run aborted (any exception)
    `status=completed` is reserved for the script-level end and is emitted by
    the entrypoints themselves (pre_trainer/pre_evaluate), never by a
    per-phase reporter, so a monitor can never mistake an epoch boundary for
    the end of the run.

    Both modes always emit the start/close/failed lines (so even a sub-30s
    smoke run produces status=start and a terminal status). `enabled=False`
    (non-rank-0 DDP ranks) makes everything a silent no-op. `stream` and
    `clock` are injectable for tests; `context` fields are merged into every
    emitted line (e.g. epoch=k/n, split=test, scope=rank0_shard_of_4).
    """

    def __init__(self, phase, total, stream=None, clock=None,
                 interval=PROGRESS_INTERVAL_S, interactive=None, enabled=True,
                 desc=None, unit="step", samples_per_unit=None, context=None):
        self.phase = phase
        self.total = int(total)
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.perf_counter
        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.samples_per_unit = None if samples_per_unit is None else int(samples_per_unit)
        if interactive is None:
            interactive = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.interactive = bool(interactive)
        self.desc = desc or phase
        self.unit = unit
        self.context = dict(context or {})
        self.done = 0
        self._t0 = self.clock()
        self._last_emit = None
        self._bar = None
        self._closed = False
        self._last_fields = {}          # most recent update()'s metric fields
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._hb_thread = None
        if not self.enabled:
            return
        with self._lock:
            self._emit_nolock("start")
        if self.interactive:
            self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit,
                             dynamic_ncols=True, leave=False, file=self.stream)
        else:
            self._last_emit = self._t0
            # time-driven heartbeat: emits status=running when `interval`
            # elapses without an update() call (daemon; stopped by close())
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True,
                name=f"progress-heartbeat-{phase}")
            self._hb_thread.start()

    # ------------------------------------------------------------------ core

    def _heartbeat_loop(self):
        """Daemon: emit a running line whenever `interval` passed since the
        last emission, even if the caller is blocked inside one batch."""
        poll = min(max(self.interval / 4.0, 0.05), 5.0)
        while not self._stop_evt.wait(poll):
            with self._lock:
                if self._closed or self.interactive:
                    return
                now = self.clock()
                if self._last_emit is None or now - self._last_emit >= self.interval:
                    self._last_emit = now
                    self._emit_nolock("running")

    def _throughput_fields(self, now, status_fields):
        """Progress + elapsed/ETA/rate fields shared by every emitted line.

        The most recent update()'s metric fields (loss, lr, d1_rmse, ...) are
        merged in so close/failed lines carry the last known metrics; fields
        passed to the emitting call itself win over them."""
        fields = dict(self.context)
        fields[self.unit] = f"{self.done}/{self.total}"
        elapsed = max(now - self._t0, 0.0)
        fields["elapsed_s"] = f"{elapsed:.1f}"
        rate = self.done / elapsed if elapsed > 0 and self.done > 0 else None
        if rate is not None:
            fields[f"{self.unit}_per_s"] = f"{rate:.3f}"
            if self.samples_per_unit:
                fields["sample_per_s"] = f"{rate * self.samples_per_unit:.3f}"
            if self.total > self.done:
                fields["eta_s"] = f"{(self.total - self.done) / rate:.1f}"
        fields.update(self._last_fields)
        fields.update(status_fields)
        return fields

    def _emit_nolock(self, status, **fields):
        """Format + write one status line. Caller must hold `self._lock`."""
        line = format_progress(self.phase, status,
                               **self._throughput_fields(self.clock(), fields))
        if self._bar is not None:
            self._bar.write(line, file=self.stream)
        else:
            print(line, file=self.stream, flush=True)

    # ------------------------------------------------------------------ API

    def update(self, n=1, **fields):
        """Advance the counter and refresh the postfix / periodic status line.

        `fields` (e.g. loss, lr, updates, d1_rmse) appear in the interactive
        postfix immediately and in periodic PROGRESS lines verbatim. `n` must
        count the reporter's OWN unit faithfully (e.g. actual windows in a
        possibly-partial final batch), not the number of calls.
        """
        if not self.enabled or self._closed:
            return
        with self._lock:
            self.done += int(n)
            now = self.clock()
            if fields:
                self._last_fields = dict(fields)
            if self._bar is not None:
                self._bar.update(int(n))
                if fields:
                    postfix = "  ".join(f"{k}={_progress_value(v)}"
                                        for k, v in fields.items())
                    self._bar.set_postfix_str(postfix)
            if not self.interactive and (self._last_emit is None
                                         or now - self._last_emit >= self.interval):
                self._last_emit = now
                self._emit_nolock("running", **fields)

    def note(self, message):
        """Print a regular line without corrupting an active bar (tqdm.write);
        a plain flush-immediate print in non-interactive mode."""
        if not self.enabled or self._closed:
            return
        with self._lock:
            if self._bar is not None:
                self._bar.write(str(message), file=self.stream)
            else:
                print(str(message), file=self.stream, flush=True)

    def close(self, status="phase_done", **fields):
        """Emit the terminal status line for this phase (default
        `phase_done` — the script-level `completed` is emitted by the
        entrypoints themselves) and stop the heartbeat thread. Idempotent."""
        if not self.enabled or self._closed:
            return
        with self._lock:
            self._closed = True
            self._stop_evt.set()
            self._emit_nolock(status, **fields)
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    def fail(self, **fields):
        """Emit status=failed (fields typically error=..., stage=...) and close."""
        self.close(status="failed", **fields)


########## standard failure reporting for UNGUARDED script sections ##########

# one failure per process: guarded-block handlers set this via
# mark_progress_failed() before emitting their own line, so the fallback hook
# below never duplicates it
_PROGRESS_FAILURE_REPORTED = False


def mark_progress_failed():
    """Record that a standard `status=failed` PROGRESS line was already emitted
    for this process (see install_progress_failure_hook)."""
    global _PROGRESS_FAILURE_REPORTED
    _PROGRESS_FAILURE_REPORTED = True


def reset_progress_failure_state():
    """Clear the dedup flag (test seam; no production caller)."""
    global _PROGRESS_FAILURE_REPORTED
    _PROGRESS_FAILURE_REPORTED = False


def install_progress_failure_hook(phase, stage=None, stream=None, fallback=None):
    """Install a `sys.excepthook` fallback that emits ONE standard
    `PROGRESS ... status=failed` line for exceptions that escape a script's
    guarded blocks — initialization, data/model setup, pre-flight refusals and
    post-processing failures have no live reporter of their own.

    phase: the PROGRESS phase field (e.g. "train" / "eval").
    stage: static stage name, or a zero-arg callable read AT FAILURE TIME so a
           script can track its current section (e.g. setup -> rollout ->
           postprocess) via a mutable variable.
    stream/fallback: injectable for tests (fallback defaults to the original
           sys.__excepthook__, which still prints the full traceback).

    Deduplicated: silent if mark_progress_failed() was already called (the
    guarded block reported its own failure). Call only on the reporting rank
    (rank 0 / single-process entrypoints). Returns the installed hook.
    """
    def _hook(exc_type, exc, tb):
        if not _PROGRESS_FAILURE_REPORTED:
            mark_progress_failed()
            stage_value = stage() if callable(stage) else (stage or "setup")
            print(format_progress(phase, "failed", stage=stage_value,
                                  error=f"{exc_type.__name__}: {exc}"),
                  file=stream if stream is not None else sys.stdout, flush=True)
        (fallback if fallback is not None else sys.__excepthook__)(
            exc_type, exc, tb)
    sys.excepthook = _hook
    return _hook
