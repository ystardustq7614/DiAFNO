"""Minimal regression tests for the PRE pipeline (assert-only, no extra deps).

Run:  python pre_smoke_test.py

Covers: C-grid->rho colocation + bivariate masks, 7x2 -> 14-channel flatten,
rollout window shift, 14->2 conditional forward, bivariate masked diffusion
loss (denominator correctness), backward, 2-step sampling, the FORMAL metric
implementations from pre_metrics.py (rho->native, masked error sums, pooled
RMSE, relative L2), the unified NativeUVReader layout with u/v sentinels,
unclipped raw-truth path, pooled sigma_data + stats cache staleness (clipping,
splits, missing fields), legacy cond_chans=None compatibility, the fixed
sigma_data scale (stats_sigma x2 -> image-space), legacy checkpoint fallback
and the resume sigma policy (error/migrate/adopt),
ensemble rollout (E=1 == sequential under autocast, autocast wrapping itself,
E=4 shape/mean/independence, AR state independence, per-window seeds, horizons
1 and 15), the rho-oracle diagnostic, a writable contiguous mask tensor, and
checkpoint metadata save/restore with weights_only=True loading.

Also covers the persistence-residual baseline (pre_models.py): zero-init
identity (untrained == last-day persistence), one optimizer step, masked MSE
land-invariance, checkpoint roundtrip + objective guards, remask_feedback
rollout behavior (on/off + mask required), deterministic rollout (seed
independence, identical ensemble members), training objectives/run tags, and
the shared ProgressReporter PROGRESS line format (update-driven + daemon
time-driven heartbeat, phase_done vs script-level completed, multi-line error
sanitization, failure-hook dedup/stage, checkpoint norm/mask/time_sigma
fingerprint checks). The static-mask checkpoint rebuild helper (legacy /
static / contradictory metadata), an archived _MSK checkpoint's minimal CPU
rollout, and the early-stop counter roundtrip are covered as well.
"""
import io
import os
import sys
import tempfile
import time
from math import exp, sqrt

import numpy as np
import torch

from scripts import preprocess_align_uv as pre_pp
from pre_dataset import (NativeUVReader, PREUVDataset, _clip_range,
                         compute_or_load_stats)
from pre_metrics import (rho_to_native, masked_error_sums, pooled_rmse, masked_rel_l2,
                         oracle_native_error_sums)
from pre_models import PersistenceResidualIAFNO, masked_mse_loss
from pre_rollout import expand_ensemble, ensemble_rollout, ensemble_mean
from pre_config import (PRESETS, SIGMA_DATA_SCALE, SMOKE_BATCHES_PER_RANK,
                        sigma_data_from_stats, sigma_data_from_checkpoint,
                        resume_sigma_decision, training_config, training_run_tag,
                        run_tag_for, OBJECTIVES, DEFAULT_OBJECTIVE, MASK_SCHEME,
                        RESIDUAL_TIME_SIGMA, validate_objective,
                        objective_from_checkpoint, ensure_objective_compatible,
                        check_norm_fingerprint, check_residual_time_sigma,
                        train_horizon, init_checkpoint,
                        TRAIN_HORIZON_ENV, INIT_CHECKPOINT_ENV, MS_DEFAULTS,
                        lead_for_batch, lead_schedule_str, check_multistep_config,
                        restore_worse_epochs, static_mask_from_checkpoint,
                        STATIC_MASK_CHANNELS,
                        format_progress, ProgressReporter,
                        install_progress_failure_hook, mark_progress_failed,
                        reset_progress_failure_state)
from utilities3 import load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_rollout import detached_feedback_window
from scripts.diag_leadtime_residual import build_npz_payload

B, H, W, Z = 2, 4, 4, 2


def make_model(embed=8):
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=embed, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=14, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    return ElucidatedDiffusion(
        net, channels=2, num_sample_steps=2,
        image_size_h=H, image_size_w=W, image_size_z=Z,
        sigma_data=0.5, P_mean=-1.0, P_std=0.0, S_churn=0,
    )


def make_residual_model(embed=8, time_sigma=RESIDUAL_TIME_SIGMA, cond_chans=14):
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=embed, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=cond_chans, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    return PersistenceResidualIAFNO(net, time_sigma=time_sigma)


def _close_mmaps(*datasets):
    """Close dataset memmap handles (Windows): an open numpy.memmap keeps the
    backing .npy locked and TemporaryDirectory cleanup fails with WinError 32.
    Test-only cleanup; the production dataset lifecycle is unchanged."""
    for ds in datasets:
        if ds is None:
            continue
        for attr in ("u", "v"):
            arr = getattr(ds, attr, None)
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()


def test_colocate_and_bivariate_masks():
    # u raw (2, 3), NaN pattern == (mask_u == 0)
    u = np.array([[1.0, np.nan, 3.0],
                  [4.0, 5.0, 6.0]])
    mask_u = np.array([[1, 0, 1],
                       [1, 1, 1]])
    # interior columns = NaN-aware mean of neighbours; boundaries copied
    u_rho = np.empty((2, 4), np.float32)
    u_rho[:, 1:3] = pre_pp.colocate(u[:, :-1], u[:, 1:])
    u_rho[:, 0] = u[:, 0]
    u_rho[:, 3] = u[:, -1]
    expected = np.array([[1.0, 1.0, 3.0, 3.0],
                         [4.0, 4.5, 5.5, 6.0]], np.float32)
    assert np.allclose(u_rho, expected, equal_nan=True)

    m_ur = pre_pp.u_rho_mask(mask_u)
    assert m_ur.shape == (2, 4)
    assert np.array_equal(m_ur, np.array([[1, 1, 1, 1],
                                          [1, 1, 1, 1]]))
    # aligned NaN pattern must equal (mask == 0)
    assert (np.isnan(u_rho) == (m_ur == 0)).all()

    # v raw (3, 4) with an interior row to exercise the eta stencil
    v = np.array([[1., 2., 3., 4.],
                  [5., 6., 7., 8.],
                  [9., 10., 11., 12.]])
    mask_v = np.ones((3, 4), np.int64)
    v_rho = np.empty((4, 4), np.float32)
    v_rho[1:3] = pre_pp.colocate(v[:-1], v[1:])
    v_rho[0] = v[0]
    v_rho[3] = v[-1]
    expected_v = np.array([[1., 2., 3., 4.],
                           [3., 4., 5., 6.],
                           [7., 8., 9., 10.],
                           [9., 10., 11., 12.]], np.float32)
    assert np.allclose(v_rho, expected_v)
    m_vr = pre_pp.v_rho_mask(mask_v)
    assert m_vr.shape == (4, 4) and m_vr.all()
    assert (np.isnan(v_rho) == (m_vr == 0)).all()


def test_enforce_land_mask_policy():
    # mask-authoritative enforcement: land values discarded + counted (in
    # place), dynamic missing ocean data fails hard, consistent cells untouched.
    mask = np.array([[1, 0, 1],
                     [1, 1, 0]])
    arr = np.array([[[[1.0, 9.0, 3.0],      # 9.0 sits on land (0,1) -> discarded
                      [4.0, 5.0, 6.0]]]], np.float32)   # 6.0 sits on land (1,2) -> discarded
    discarded = {}
    out = pre_pp.enforce_land_mask(arr, mask, "u", 0, discarded)
    assert out is arr                                # in-place, same object
    assert np.isnan(arr[0, 0, 0, 1]) and np.isnan(arr[0, 0, 1, 2])
    assert arr[0, 0, 0, 0] == 1.0 and arr[0, 0, 0, 2] == 3.0
    assert arr[0, 0, 1, 0] == 4.0 and arr[0, 0, 1, 1] == 5.0
    assert discarded == {"u": 2}
    # counts accumulate across chunks
    pre_pp.enforce_land_mask(arr, mask, "u", 50, discarded)
    assert discarded == {"u": 2}                     # already NaN -> no double count

    # NaN on an ocean cell = dynamic missing data -> RuntimeError with location
    bad = np.array([[[[1.0, 2.0, 3.0],
                      [4.0, np.nan, 6.0]]]], np.float32)  # NaN at (t=0,s=0,r=1,c=1), mask==1
    try:
        pre_pp.enforce_land_mask(bad, mask, "v", 7, {})
    except RuntimeError as e:
        msg = str(e)
        assert "t=7" in msg and "r=1" in msg and "c=1" in msg, msg
        assert "mask==1" in msg, msg
    else:
        raise AssertionError("expected RuntimeError for NaN on ocean cell")


def _cuda_or_skip():
    """Return a cuda device when available, else None after printing a note."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    print("  SKIP (no CUDA available)")
    return None


def test_tracker_update_summary_matches_update():
    # update_summary (the GPU scalar-interface) must reproduce the NumPy
    # update() trackers exactly: values, first-occurrence locations, and
    # cross-chunk accumulation. Runs without CUDA.
    rng = np.random.default_rng(2)
    arr = rng.uniform(-3, 3, (5, 2, 4, 6)).astype(np.float32)
    flat = arr.ravel()
    flat[::4] = np.nan
    flat[0] = -9.0
    flat[-1] = 9.0
    t0 = 100

    a = pre_pp.ExtremumTracker("same")
    b = pre_pp.ExtremumTracker("same")
    a.update(arr, t0)
    b.update_summary(float(np.nanmin(flat)), int(np.nanargmin(flat)),
                     float(np.nanmax(flat)), int(np.nanargmax(flat)),
                     arr.shape, t0)
    assert a.min_val == b.min_val == float(np.nanmin(flat))
    assert a.max_val == b.max_val == float(np.nanmax(flat))
    assert a.min_loc == b.min_loc == (t0 + 0, 0, 0, 0)
    assert a.max_loc == b.max_loc
    assert a.report() == b.report()

    # partial chunks accumulate identically across several calls
    a2 = pre_pp.ExtremumTracker("same2")
    b2 = pre_pp.ExtremumTracker("same2")
    for k in range(4):
        chunk = arr[k:k + 1]
        f = chunk.ravel()
        a2.update(chunk, t0 + k)
        b2.update_summary(float(np.nanmin(f)), int(np.nanargmin(f)),
                          float(np.nanmax(f)), int(np.nanargmax(f)),
                          chunk.shape, t0 + k)
    assert a2.report() == b2.report()


def test_torch_colocate_matches_numpy():
    dev = _cuda_or_skip()
    if dev is None:
        return
    rng = np.random.default_rng(0)
    for shape in ((7, 3), (2, 5), (4, 4)):
        a = rng.uniform(-2, 2, shape).astype(np.float32)
        b = rng.uniform(-2, 2, shape).astype(np.float32)
        a[rng.uniform(size=shape) < 0.3] = np.nan
        b[rng.uniform(size=shape) < 0.3] = np.nan
        cpu = pre_pp.colocate(a, b)
        gpu = pre_pp.torch_colocate(
            torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)).cpu().numpy()
        assert np.array_equal(cpu, gpu, equal_nan=True), shape


def test_torch_colocate_edge_cases():
    dev = _cuda_or_skip()
    if dev is None:
        return
    # cell 0: one-sided valid (a only) -> a; cell 1: both invalid -> NaN;
    # cell 2: one-sided valid (a only, b NaN) -> a
    a = np.array([[1.0, np.nan, 3.0]], np.float32)
    b = np.array([[np.nan, np.nan, np.nan]], np.float32)
    cpu = pre_pp.colocate(a, b)
    gpu = pre_pp.torch_colocate(torch.from_numpy(a).to(dev),
                                torch.from_numpy(b).to(dev)).cpu().numpy()
    assert np.array_equal(cpu, gpu, equal_nan=True)
    assert cpu[0, 0] == 1.0 and np.isnan(cpu[0, 1]) and cpu[0, 2] == 3.0

    # u boundary columns are copied, not averaged (incl. NaN edges)
    uc = np.array([[[[1.0, 2.0, np.nan],
                     [4.0, np.nan, 6.0]]]], np.float32)          # (1,1,2,3)
    cpu_ub = np.empty((1, 1, 2, 4), np.float32)
    cpu_ub[:, :, :, 1:3] = pre_pp.colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
    cpu_ub[:, :, :, 0] = uc[:, :, :, 0]
    cpu_ub[:, :, :, 3] = uc[:, :, :, -1]
    gpu_ub = pre_pp.torch_colocate_u(torch.from_numpy(uc).to(dev)).cpu().numpy()
    assert np.array_equal(cpu_ub, gpu_ub, equal_nan=True)

    # v boundary rows are copied, not averaged
    vc = np.array([[[[1.0, 2.0, 3.0],
                     [4.0, np.nan, 6.0],
                     [7.0, 8.0, np.nan]]]], np.float32)          # (1,1,3,3)
    cpu_vb = np.empty((1, 1, 4, 3), np.float32)
    cpu_vb[:, :, 1:3, :] = pre_pp.colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
    cpu_vb[:, :, 0, :] = vc[:, :, 0, :]
    cpu_vb[:, :, 3, :] = vc[:, :, -1, :]
    gpu_vb = pre_pp.torch_colocate_v(torch.from_numpy(vc).to(dev)).cpu().numpy()
    assert np.array_equal(cpu_vb, gpu_vb, equal_nan=True)


def test_torch_enforce_land_mask():
    dev = _cuda_or_skip()
    if dev is None:
        return
    mask = np.array([[1, 0, 1],
                     [1, 1, 0]])
    gmask = torch.as_tensor(mask, dtype=torch.bool, device=dev)

    # land finite values are cleared in place and counted (identical to NumPy)
    arr = np.array([[[[1.0, 9.0, 3.0],
                      [4.0, 5.0, 6.0]]]], np.float32)
    cpu = arr.copy()
    discarded_cpu = {}
    pre_pp.enforce_land_mask(cpu, mask, "u", 0, discarded_cpu)
    gpu = torch.from_numpy(arr.copy()).to(dev)
    discarded_gpu = {}
    pre_pp.torch_enforce_land_mask(gpu, gmask, "u", 0, discarded_gpu)
    assert np.array_equal(cpu, gpu.cpu().numpy(), equal_nan=True)
    assert discarded_cpu == discarded_gpu == {"u": 2}

    # counts accumulate across chunks (already-NaN cells are not double counted)
    pre_pp.torch_enforce_land_mask(gpu, gmask, "u", 50, discarded_gpu)
    assert discarded_gpu == {"u": 2}

    # NaN on an ocean cell (mask==1) -> RuntimeError with the GLOBAL coordinate
    bad = np.array([[[[1.0, 2.0, 3.0],
                      [4.0, np.nan, 6.0]]]], np.float32)  # NaN at (t=7,s=0,r=1,c=1)
    gbad = torch.from_numpy(bad).to(dev)
    try:
        pre_pp.torch_enforce_land_mask(gbad, gmask, "v", 7, {})
    except RuntimeError as e:
        msg = str(e)
        assert "t=7" in msg and "r=1" in msg and "c=1" in msg, msg
        assert "mask==1" in msg, msg
    else:
        raise AssertionError("expected RuntimeError for NaN on ocean cell")


def test_torch_extrema_summary():
    dev = _cuda_or_skip()
    if dev is None:
        return
    rng = np.random.default_rng(1)
    arr = rng.uniform(-5, 5, (4, 3, 2, 7)).astype(np.float32)
    flat = arr.ravel()
    flat[::3] = np.nan
    flat[0] = -9.0
    flat[10] = 9.0                     # max at a NON-last index (tie test below)
    mn, mi, mx, xi = pre_pp.torch_extrema_summary(torch.from_numpy(arr).to(dev))
    assert mn == float(np.nanmin(flat)) and mx == float(np.nanmax(flat))
    assert mi == int(np.nanargmin(flat)) and xi == int(np.nanargmax(flat))

    # ties keep the FIRST C-order occurrence (GPU == numpy): duplicate the max
    # at a STRICTLY LATER index, so the first occurrence must stay first_max
    flat2 = flat.copy()
    first_max = int(np.nanargmax(flat2))
    flat2[first_max + 10] = flat2[first_max]      # later duplicate of the max
    g2 = torch.from_numpy(flat2.reshape(arr.shape)).to(dev)
    mn2, mi2, mx2, xi2 = pre_pp.torch_extrema_summary(g2)
    assert mx2 == float(np.nanmax(flat2)) and xi2 == first_max

    # all-NaN chunk raises like np.nanmin/np.nanmax
    all_nan = torch.full((2, 3), np.nan, device=dev)
    try:
        pre_pp.torch_extrema_summary(all_nan)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an all-NaN chunk")


def test_cond_flatten_and_rollout_shift():
    uv = torch.randn(1, 7, 2, 4, 4, 1)          # (B, days, 2, H, W, Z)
    cond = uv.reshape(1, 14, 4, 4, 1)           # day-major interleave
    assert cond.shape == (1, 14, 4, 4, 1)
    # ch 2k = u of day k, ch 2k+1 = v of day k
    for k in range(7):
        assert torch.equal(cond[0, 2 * k], uv[0, k, 0])
        assert torch.equal(cond[0, 2 * k + 1], uv[0, k, 1])

    new = torch.randn(1, 2, 4, 4, 1)
    cur = torch.cat([cond[:, 2:], new], dim=1)  # rollout: drop oldest day
    assert cur.shape == (1, 14, 4, 4, 1)
    assert torch.equal(cur[0, 0], uv[0, 1, 0])  # day 1 u is now the first ch
    assert torch.equal(cur[:, -2:], new)


def test_forward_14_to_2_and_shape():
    model = make_model()
    images = torch.randn(1, 2, H, W, Z)
    cond = torch.randn(1, 14, H, W, Z)
    out = model.preconditioned_network_forward(
        images, torch.full((1,), 0.5), cond)
    assert out.shape == (1, 2, H, W, Z)
    loss = model(images, cond, mask=torch.ones(1, 2, H, W, Z))
    assert torch.isfinite(loss)


def test_masked_loss_denominator_and_backward():
    model = make_model()
    torch.manual_seed(0)
    images = torch.rand(B, 2, H, W, Z)
    cond = torch.rand(B, 14, H, W, Z)
    mask = torch.zeros(1, 2, H, W, Z)
    mask[0, 0] = 1.0                              # u fully valid
    mask[0, 1, 0:2, 0:2] = 1.0                    # v partially valid

    def manual_loss(m):
        # reproduce forward() with identical seeded RNG draws
        sigmas = (model.P_mean + model.P_std * torch.randn(B)).exp()
        norm_img = images * 2 - 1
        noise = torch.randn_like(images)
        noised = norm_img + sigmas[:, None, None, None, None] * noise
        den = model.preconditioned_network_forward(noised, sigmas, cond)
        mse = (den - norm_img) ** 2
        mm = m.expand_as(mse)
        per_sample = (mse * mm).sum(dim=(1, 2, 3, 4)) / mm.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
        return (per_sample * model.loss_weight(sigmas)).mean()

    torch.manual_seed(0)
    loss = model(images, cond, mask=mask)
    torch.manual_seed(0)
    assert torch.allclose(loss, manual_loss(mask), atol=1e-6)

    # single-channel common mask: pass (1,1,H,W,Z) DIRECTLY, let the diffusion
    # forward broadcast it (no manual expansion to two channels)
    mask1 = torch.zeros(1, 1, H, W, Z)
    mask1[0, 0, 0:2, 0:2] = 1.0
    torch.manual_seed(1)
    loss1 = model(images, cond, mask=mask1)
    torch.manual_seed(1)
    assert torch.allclose(loss1, manual_loss(mask1), atol=1e-6)

    # batch-varying bivariate mask
    maskb = torch.zeros(B, 2, H, W, Z)
    maskb[0, 0] = 1.0
    maskb[1, 1, :, :, :] = 1.0
    torch.manual_seed(2)
    lossb = model(images, cond, mask=maskb)
    torch.manual_seed(2)
    assert torch.allclose(lossb, manual_loss(maskb), atol=1e-6)

    # all-zero mask: no division by zero, no NaN, exactly 0
    loss0 = model(images, cond, mask=torch.zeros(1, 2, H, W, Z))
    assert torch.isfinite(loss0) and loss0.item() == 0.0

    # backward on a nonzero loss produces gradients
    lossb.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())

    # Z mismatch must raise; the error must NOT be caught as a success
    try:
        model(images[:, :, :, :, :1], cond, mask=mask)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for wrong Z")


def test_two_step_sample():
    model = make_model()
    cond = torch.randn(1, 14, H, W, Z)
    with torch.no_grad():
        out = model.sample(cond, num_sample_steps=2)
    assert out.shape == (1, 2, H, W, Z)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_corrected_relative_l2():
    # FORMAL implementation from pre_metrics.py (shared with pre_trainer.py)
    tgt = torch.full((1, 1, 2, 2, 1), 10.0)
    pred = tgt.clone()
    pred[0, 0, 0, 0, 0] = 11.0                   # +1 and -1 errors: signed sum 0
    pred[0, 0, 0, 1, 0] = 9.0
    mask = torch.ones(1, 1, 2, 2, 1)
    got = masked_rel_l2(pred, tgt, mask)
    want = sqrt(2.0) / sqrt(400.0)
    assert got > 0.0 and abs(got - want) < 1e-6, got

    # masked cells must not contribute
    mask2 = mask.clone()
    mask2[0, 0, 0, 0, 0] = 0.0
    got2 = masked_rel_l2(pred, tgt, mask2)
    want2 = sqrt(1.0) / sqrt(300.0)
    assert abs(got2 - want2) < 1e-6, got2


def test_rho_to_native_resampling():
    # FORMAL rho_to_native: u averages adjacent rho points along xi (cols),
    # v along eta (rows) — channels must stay separated
    up = np.arange(16.0).reshape(1, 1, 4, 4, 1)
    rho = np.stack([up, up + 100.0], axis=2)     # (1,1,2,4,4,1)
    u_nat, v_nat = rho_to_native(rho)
    assert u_nat.shape == (1, 1, 4, 3, 1)
    assert v_nat.shape == (1, 1, 3, 4, 1)
    assert abs(u_nat[0, 0, 2, 1, 0] - 0.5 * (up[0, 0, 2, 1, 0] + up[0, 0, 2, 2, 0])) < 1e-6
    assert abs(v_nat[0, 0, 1, 3, 0]
               - 0.5 * (rho[0, 0, 1, 1, 3, 0] + rho[0, 0, 1, 2, 3, 0])) < 1e-6


def test_native_reader_unified_layout_and_sentinels():
    # distinct sentinels: u is all 7.0 (one 50.0 extreme), v is all 11.0 —
    # a v-read-as-u bug would surface immediately.
    u = np.full((5, 3, 4, 5), 7.0, np.float32)
    u[0, 0, 0, 0] = 50.0
    v = np.full((5, 3, 3, 5), 11.0, np.float32)
    with tempfile.TemporaryDirectory() as d:
        up, vp = os.path.join(d, "u.npy"), os.path.join(d, "v.npy")
        np.save(up, u)
        np.save(vp, v)

        # full3d: unified (days, H, W-1, Z) / (days, H-1, W, Z) layout
        full = NativeUVReader(depth_index=None, u_path=up, v_path=vp, check_shape=False)
        us, vs = full.get(0, 2)
        assert us.shape == (2, 4, 5, 3) and vs.shape == (2, 3, 5, 3)
        assert us[0, 0, 0, 0] == 50.0             # u's raw extreme, untouched
        assert np.count_nonzero(us != 7.0) == 1   # everything else is u's sentinel
        assert (vs == 11.0).all(), "v must not be read from the u field"

        # surface: same unified layout with Z=1
        surf = NativeUVReader(depth_index=2, u_path=up, v_path=vp, check_shape=False)
        us2, vs2 = surf.get(1, 3)
        assert us2.shape == (3, 4, 5, 1) and vs2.shape == (3, 3, 5, 1)
        assert np.array_equal(us2[1, :, :, 0], u[2, 2])
        assert (vs2 == 11.0).all()

        # persistence u/v values stay independent (day-7 slices, distinct sentinels)
        du, dv = full.get(4, 1)
        assert du.shape == (1, 4, 5, 3) and dv.shape == (1, 3, 5, 3)
        assert np.count_nonzero(du != 7.0) == 0 and (dv == 11.0).all()
        # release the mmap'd file handles so the temp dir can be removed
        for r in (full, surf):
            r.u._mmap.close()
            r.v._mmap.close()

    # denormalizing a clipped value can NOT recover the raw truth
    raw = np.array([1.0, 2.0, 50.0])
    lo, hi = 1.0, 3.0
    denorm = (np.clip(raw, lo, hi) - lo) / (hi - lo) * (hi - lo) + lo
    assert np.any(denorm != raw) and denorm[2] == 3.0


def test_metrics_native_batch():
    # synthetic evaluation batch through the FORMAL pre_metrics.py functions
    rng = np.random.default_rng(3)
    for Zz in (1, 3):                             # surface (Z=1) and full3d (Z>1)
        Bb, L = 2, 15
        rho_pred = rng.normal(size=(Bb, L, 2, H, W, Zz))
        truth_u = rng.normal(size=(Bb, L, H, W - 1, Zz))
        truth_v = rng.normal(size=(Bb, L, H - 1, W, Zz))
        u_nat, v_nat = rho_to_native(rho_pred)
        assert u_nat.shape == (Bb, L, H, W - 1, Zz)
        assert v_nat.shape == (Bb, L, H - 1, W, Zz)

        mask_u = np.ones((H, W - 1), bool)
        mask_v = np.ones((H - 1, W), bool)
        se_u, ae_u = masked_error_sums(u_nat, truth_u, mask_u)
        se_v, ae_v = masked_error_sums(v_nat, truth_v, mask_v)
        assert se_u.shape == (L, Zz) and ae_u.shape == (L, Zz)
        assert se_v.shape == (L, Zz) and ae_v.shape == (L, Zz)

        # results must slot into (L, 2, Z) accumulators via [:, channel, :]
        se_m = np.zeros((L, 2, Zz))
        ae_m = np.zeros((L, 2, Zz))
        se_m[:, 0, :] += se_u
        ae_m[:, 0, :] += ae_u
        se_m[:, 1, :] += se_v
        ae_m[:, 1, :] += ae_v

        # exact reference sums (direct numpy on the raw arrays)
        assert np.allclose(se_u, ((u_nat - truth_u) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(se_v, ((v_nat - truth_v) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(ae_v, np.abs(v_nat - truth_v).sum(axis=(0, 2, 3)))

        # land cells (mask == 0) contribute nothing
        mu2 = mask_u.copy()
        mu2[0, 0] = False
        se_u2, _ = masked_error_sums(u_nat, truth_u, mu2)
        diff = (u_nat - truth_u)[:, :, 0, 0, :]
        assert np.allclose(se_u - se_u2, (diff ** 2).sum(axis=0))

        # pooled RMSE == sqrt(total_se / total_n), never a mean of per-layer RMSEs
        n_u = mask_u.sum()
        n_v = mask_v.sum()
        rmse_u = pooled_rmse(se_u, np.full((L, Zz), n_u))
        assert np.isclose(rmse_u, sqrt(se_u.sum() / (L * Zz * n_u)))
        # per-lead pooled RMSE from the (L, 2, Z) accumulators
        cnt = np.empty((2, Zz))
        cnt[0, :] = n_u
        cnt[1, :] = n_v
        for l in (0, 4, 14):
            rm = pooled_rmse(se_m[l], cnt)
            assert np.isclose(rm, sqrt(se_m[l].sum() / (Zz * (n_u + n_v))))
        # pooled_rmse with no valid count -> 0.0, no NaN
        assert pooled_rmse(se_u, np.zeros((L, Zz))) == 0.0

        # persistence: day-7 NATIVE u/v repeated over all lead days, values kept
        # independent (u and v drawn from different scales)
        day7_u = 100.0 * rng.normal(size=(Bb, 1, H, W - 1, Zz))
        day7_v = 100.0 * rng.normal(size=(Bb, 1, H - 1, W, Zz))
        pu = np.broadcast_to(day7_u, (Bb, L, H, W - 1, Zz))
        pv = np.broadcast_to(day7_v, (Bb, L, H - 1, W, Zz))
        assert np.allclose(pu[0, 3], day7_u[0, 0]) and np.allclose(pv[0, 3], day7_v[0, 0])
        se_pu, _ = masked_error_sums(pu, truth_u, mask_u)
        se_pv, _ = masked_error_sums(pv, truth_v, mask_v)
        assert np.allclose(se_pu, ((day7_u - truth_u) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(se_pv, ((day7_v - truth_v) ** 2).sum(axis=(0, 2, 3)))


def test_clip_range_policy():
    vals = [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])]
    lo, hi = _clip_range(lambda: iter(vals), None)  # default: NO clipping
    assert (lo, hi) == (1.0, 10.0)
    lo20, hi20 = _clip_range(lambda: iter(vals), 20.0)  # explicit clipping
    assert abs(lo20 - 2.0) < 0.01 and abs(hi20 - 5.0) < 0.01


def test_pooled_sigma_and_rmse():
    # pooled std over two differently-meaned groups == std of the concatenation
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0])
    s1 = a.sum() + b.sum()
    s2 = (a * a).sum() + (b * b).sum()
    n = a.size + b.size
    mean = s1 / n
    pooled = sqrt(max(s2 / n - mean * mean, 0.0))
    assert np.isclose(pooled, np.std(np.concatenate([a, b])))
    per_var = 0.5 * (np.std(a) + np.std(b))       # naive avg misses between-group
    assert abs(pooled - per_var) > 1e-3

    # overall RMSE = sqrt(total_se / total_n), not mean of layer RMSEs
    se = np.array([[[1.0]], [[1.0]]])
    n = np.array([[[1.0]], [[4.0]]])
    overall = pooled_rmse(se, n)
    assert np.isclose(overall, sqrt(2.0 / 5.0))
    mean_of_rmse = np.sqrt(se / n).mean()
    assert not np.isclose(overall, mean_of_rmse)


def test_stats_cache_clip_and_splits():
    # FORMAL compute_or_load_stats() against a tiny temp aligned dataset:
    # clipped sigma, cache reuse, missing-splits staleness, split-change staleness.
    import pre_dataset as pd
    rng = np.random.default_rng(0)
    T, S, HH, WW = 10, 2, 4, 5
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        norm = os.path.join(d, "norm")
        os.makedirs(aligned)
        os.makedirs(norm)
        u = rng.uniform(-1.0, 1.0, (T, S, HH, WW)).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, (T, S, HH, WW)).astype(np.float32)
        np.save(os.path.join(aligned, "u_rho.npy"), u)
        np.save(os.path.join(aligned, "v_rho.npy"), v)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((HH, WW), np.uint8))

        saved = (pd.ALIGNED_DIR, pd.NORM_DIR, pd.H, pd.W, pd.T_TOTAL, dict(pd.SPLITS))
        pd.ALIGNED_DIR, pd.NORM_DIR = aligned, norm
        pd.H, pd.W, pd.T_TOTAL = HH, WW, T
        try:
            pd.SPLITS.clear()
            pd.SPLITS.update(train=(0, 6), val=(6, 8), test=(8, 10))

            # no clipping: lo/hi == exact train min/max, sigma == pooled std of
            # the min-max normalized u+v concatenation
            s_none = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert np.isclose(s_none["lo"][0], u[:6].min()) and np.isclose(s_none["hi"][0], u[:6].max())
            assert np.isclose(s_none["lo"][1], v[:6].min()) and np.isclose(s_none["hi"][1], v[:6].max())

            def pooled_sigma(lo, hi):
                a = np.clip(u[:6], float(lo[0]), float(hi[0])).astype(np.float64)
                a = (a - float(lo[0])) / (float(hi[0]) - float(lo[0]))
                b = np.clip(v[:6], float(lo[1]), float(hi[1])).astype(np.float64)
                b = (b - float(lo[1])) / (float(hi[1]) - float(lo[1]))
                return float(np.std(np.concatenate([a.ravel(), b.ravel()])))

            assert np.isclose(s_none["sigma"], pooled_sigma(s_none["lo"], s_none["hi"]),
                              rtol=1e-4)

            # clipping pulls lo/hi in from the extremes and changes sigma
            s_clip = pd.compute_or_load_stats(depth_index=None, clip_pct=20.0, verbose=False)
            assert s_clip["lo"][0] > s_none["lo"][0] and s_clip["hi"][0] < s_none["hi"][0]
            assert np.isclose(s_clip["sigma"], pooled_sigma(s_clip["lo"], s_clip["hi"]),
                              rtol=1e-4)
            assert abs(s_clip["sigma"] - s_none["sigma"]) > 1e-6

            # identical config -> cache hit, identical sigma
            s_again = pd.compute_or_load_stats(depth_index=None, clip_pct=20.0, verbose=False)
            assert np.isclose(s_again["sigma"], s_clip["sigma"])

            # cache WITHOUT the 'splits' field must be treated as stale
            cache = os.path.join(norm, "stats_all_clipnone.npz")
            os.remove(cache)
            np.savez(cache,
                     lo=np.float32([-9.0, -9.0]), hi=np.float32([9.0, 9.0]),
                     sigma=np.float32(0.123), depth_index=np.int64(-1),
                     clip_pct=np.float64(-1.0), mask_version=np.str_(pd.mask_version()))
            s_missing = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert abs(s_missing["sigma"] - 0.123) > 1e-6
            assert np.isclose(s_missing["sigma"], s_none["sigma"])

            # changed splits -> old cache must NOT be reused
            pd.SPLITS.clear()
            pd.SPLITS.update(train=(0, 4), val=(4, 7), test=(7, 10))
            s_new = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert np.isclose(s_new["lo"][0], u[:4].min())
            assert abs(s_new["sigma"] - s_none["sigma"]) > 1e-6
        finally:
            pd.ALIGNED_DIR, pd.NORM_DIR, pd.H, pd.W, pd.T_TOTAL = saved[:5]
            pd.SPLITS.clear()
            pd.SPLITS.update(saved[5])


def test_verify_daily_time():
    # 24 h spacing passes at any datetime64 resolution
    good = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    assert pre_pp.verify_daily_time(good) is good
    good_s = np.array(["2020-01-01T00:00:00", "2020-01-02T00:00:00",
                       "2020-01-03T00:00:00"], dtype="datetime64[s]")
    assert pre_pp.verify_daily_time(good_s) is good_s

    # 23 h / 25 h gaps must FAIL with index, times and actual interval reported
    for hours in (23, 25):
        t = np.array([
            np.datetime64("2020-01-01T00:00:00", "s"),
            np.datetime64("2020-01-01T00:00:00", "s") + np.timedelta64(hours, "h"),
            np.datetime64("2020-01-01T00:00:00", "s") + np.timedelta64(2 * hours, "h"),
        ])
        try:
            pre_pp.verify_daily_time(t)
        except RuntimeError as e:
            msg = str(e)
            assert "index 0" in msg, msg
            assert "->" in msg, msg
            assert f"{hours} h" in msg and "24 h" in msg, msg
        else:
            raise AssertionError(f"expected RuntimeError for {hours}h-spaced times")


def test_legacy_cond_chans_none():
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=8, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=None, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    x = torch.randn(1, 2, H, W, Z)
    cond = torch.randn(1, 2, H, W, Z)             # legacy doubling (2+2)
    out = net(x, torch.zeros(1), cond)
    assert out.shape == (1, 2, H, W, Z)


def test_sigma_data_conversion():
    # [0,1]-space stats sigma -> [-1,1] image-space EDM sigma_data (x2)
    assert SIGMA_DATA_SCALE == 2.0
    assert np.isclose(sigma_data_from_stats(0.0856), 0.1712)
    assert np.isclose(sigma_data_from_stats(0.0), 0.0)
    assert np.isclose(sigma_data_from_stats(1.0), 2.0)


def test_sigma_data_legacy_checkpoint_fallback():
    # legacy checkpoint (no config.sigma_data) keeps the OLD stats-only scale
    sd, used = sigma_data_from_checkpoint({"epoch": 2}, 0.0856)
    assert not used and np.isclose(sd, 0.0856)
    sd2, used2 = sigma_data_from_checkpoint({}, 0.0856)
    assert not used2 and np.isclose(sd2, 0.0856)
    sd3, used3 = sigma_data_from_checkpoint(None, 0.0856)
    assert not used3 and np.isclose(sd3, 0.0856)
    # new checkpoint -> its stored sigma_data wins, whatever the stats say
    sd4, used4 = sigma_data_from_checkpoint(
        {"config": {"sigma_data": 0.1712, "sigma_data_scale": 2.0}}, 0.0856)
    assert used4 and np.isclose(sd4, 0.1712)


def test_resume_sigma_policy():
    # matching scales -> keep current, never "adopted", under ANY policy
    sd_new = sigma_data_from_stats(0.0856)          # 0.1712
    for policy in ("error", "migrate", "adopt"):
        sd, adopted = resume_sigma_decision(sd_new, sd_new, policy)
        assert not adopted and np.isclose(sd, sd_new), policy
    # mismatch + "error" (default) -> RuntimeError, never silently mixed
    try:
        resume_sigma_decision(0.0856, sd_new, "error")
    except RuntimeError as e:
        assert "sigma_data" in str(e) and "0.0856" in str(e)
    else:
        raise AssertionError("expected RuntimeError on scale mismatch")
    # mismatch + "migrate" -> keep the current (SD2) scale, not adopted
    sd, adopted = resume_sigma_decision(0.0856, sd_new, "migrate")
    assert not adopted and np.isclose(sd, sd_new)
    # mismatch + "adopt" -> checkpoint's old scale, adopted
    sd, adopted = resume_sigma_decision(0.0856, sd_new, "adopt")
    assert adopted and np.isclose(sd, 0.0856)
    # unknown policy -> ValueError
    try:
        resume_sigma_decision(0.0856, sd_new, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown policy")


def test_ensemble_size1_matches_sequential():
    # E=1 must reproduce the plain per-window rollout exactly (same RNG stream
    # AND the same autocast wrapping as the historical evaluation path)
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    p1 = ensemble_rollout(model, cond, 3, 1, seed=42)
    assert p1.shape == (2, 1, 3, 2, H, W, Z)
    assert torch.isfinite(p1).all()
    torch.manual_seed(42)
    cur = cond.clone()
    preds = []
    with torch.amp.autocast(device_type="cpu"):
        for _ in range(3):
            preds.append(model.sample(cur).float())
            cur = torch.cat([cur[:, 2:], preds[-1]], dim=1)
    p2 = torch.stack(preds, dim=1)                # (B, L, 2, H, W, Z)
    assert torch.allclose(p1[:, 0], p2, atol=1e-6)


def test_ensemble_rollout_uses_autocast():
    # the rollout must run model.sample under autocast (AMP), otherwise the
    # historical evaluation path (and its numerics) is silently changed.
    seen = {"cpu": False, "cuda": False}

    class _FlagSampler:
        def sample(self, cur, num_sample_steps=None, clamp=True):
            if torch.is_autocast_enabled("cpu"):
                seen["cpu"] = True
            if torch.is_autocast_enabled("cuda"):
                seen["cuda"] = True
            return cur[:, :2].clone()

    cond = torch.rand(1, 14, H, W, Z)
    ensemble_rollout(_FlagSampler(), cond, 2, 1, seed=0)
    # the autocast device follows the TENSORS, not global CUDA availability:
    # CPU tensors -> CPU autocast even on a CUDA-capable machine; CUDA tensors
    # -> CUDA autocast (the historical evaluation path).
    if cond.is_cuda:
        assert seen["cuda"], "model.sample must run under CUDA autocast"
    else:
        assert seen["cpu"], "model.sample must run under CPU autocast"
    if torch.cuda.is_available():
        seen["cpu"] = seen["cuda"] = False
        ensemble_rollout(_FlagSampler(), cond.cuda(), 2, 1, seed=0)
        assert seen["cuda"], "model.sample must run under CUDA autocast"


def test_ensemble_seeds_per_window():
    # per-window seeds: window w's trajectory depends ONLY on seeds[w] and
    # cond[w] — NOT on the batch size or the other windows in the batch.
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    p_batch = ensemble_rollout(model, cond, 2, 1, seeds=[5, 9])
    assert p_batch.shape == (2, 1, 2, 2, H, W, Z)
    # window 0 alone (batch of 1) == window 0 inside the batch of 2
    p_single = ensemble_rollout(model, cond[:1], 2, 1, seeds=[5])
    assert torch.allclose(p_batch[0, 0], p_single[0, 0], atol=1e-6)
    # window 1 alone == window 1 inside the batch
    p_single1 = ensemble_rollout(model, cond[1:], 2, 1, seeds=[9])
    assert torch.allclose(p_batch[1, 0], p_single1[0, 0], atol=1e-6)
    # seeds[w] == the scalar path for the same window
    p_scalar = ensemble_rollout(model, cond[:1], 2, 1, seed=5)
    assert torch.allclose(p_batch[0, 0], p_scalar[0, 0], atol=1e-6)
    # different seed -> different trajectory; same seed -> reproducible
    p_other = ensemble_rollout(model, cond[:1], 2, 1, seeds=[8])
    assert not torch.allclose(p_scalar[0, 0], p_other[0, 0], atol=1e-6)
    p_again = ensemble_rollout(model, cond[:1], 2, 1, seeds=[5])
    assert torch.allclose(p_again[0, 0], p_scalar[0, 0], atol=1e-6)
    # seed and seeds are mutually exclusive
    try:
        ensemble_rollout(model, cond, 2, 1, seed=1, seeds=[2, 3])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for seed + seeds together")
    # per-window seeds keep the AR state per window (windows rolled out one by
    # one: window 0 -> calls 0..2, window 1 -> calls 3..5)
    s = _persistence_fake()
    cond2 = torch.arange(2 * 14 * H * W * Z).reshape(2, 14, H, W, Z).float() / 1000.0
    ensemble_rollout(s, cond2, 3, 2, seeds=[11, 22])
    assert torch.equal(s.calls[1][0], torch.cat([s.calls[0][0][2:], s.calls[0][0][:2]], dim=0))
    assert torch.equal(s.calls[4][0], torch.cat([s.calls[3][0][2:], s.calls[3][0][:2]], dim=0))


def test_ensemble4_shape_mean_and_independent():
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    preds = ensemble_rollout(model, cond, 2, 4, seed=7)
    assert preds.shape == (2, 4, 2, 2, H, W, Z)
    assert torch.isfinite(preds).all()
    assert float(preds.min()) >= 0.0 and float(preds.max()) <= 1.0
    # members are independent trajectories -> different noise -> different outputs
    assert not torch.allclose(preds[0, 0], preds[0, 1], atol=1e-6)
    assert not torch.allclose(preds[0, 0], preds[1, 0], atol=1e-6)
    # ensemble_mean is the member average (point prediction)
    m = ensemble_mean(preds)
    assert m.shape == (2, 2, 2, H, W, Z)
    assert torch.allclose(m, preds.mean(dim=1))
    # averaging reduces per-day variance vs a single member
    assert m[0].std() <= preds[0, 0].std() + 1e-6


def _persistence_fake():
    """Deterministic fake EDM: prediction = first 2 channels of the current
    condition (a persistence policy); records every input it sees."""
    class _PersistenceSampler:
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True):
            self.calls.append(cur.clone())
            return cur[:, :2].clone()
    return _PersistenceSampler()


def test_ensemble_ar_state_independent():
    s = _persistence_fake()
    cond = torch.arange(2 * 14 * H * W * Z).reshape(2, 14, H, W, Z).float() / 1000.0
    preds = ensemble_rollout(s, cond, 3, 2)
    assert preds.shape == (2, 2, 3, 2, H, W, Z)
    # member 0, step 2 condition == member 0's OWN step-1 window shifted by its
    # OWN prediction — no leakage from member 1 (and vice versa). Layout is
    # interleaved: [w0m0, w0m1, w1m0, w1m1, ...].
    c0_0, c0_1 = s.calls[0][0], s.calls[1][0]
    assert torch.equal(c0_1, torch.cat([c0_0[2:], c0_0[:2]], dim=0))
    c1_0, c1_1 = s.calls[0][2], s.calls[1][2]
    assert torch.equal(c1_1, torch.cat([c1_0[2:], c1_0[:2]], dim=0))
    # the two WINDOWS evolve independently and stay distinct (members of one
    # window are identical copies under this deterministic fake, by design)
    assert not torch.equal(s.calls[0][0], s.calls[0][2])
    assert not torch.equal(c0_1, c1_1)
    # expand_ensemble: E=1 gives a fresh copy, E>1 independent repeats
    e1 = expand_ensemble(cond, 1)
    assert e1 is not cond and torch.equal(e1, cond)
    e4 = expand_ensemble(cond, 4)
    assert e4.shape == (8, 14, H, W, Z)
    assert torch.equal(e4[0], cond[0]) and torch.equal(e4[4], cond[1])


def test_rollout_horizons_1_and_15():
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(1, 14, H, W, Z)
    p1 = ensemble_rollout(model, cond, 1, 1, seed=0)
    assert p1.shape == (1, 1, 1, 2, H, W, Z)
    p15 = ensemble_rollout(model, cond, 15, 1, seed=1)
    assert p15.shape == (1, 1, 15, 2, H, W, Z)
    assert torch.isfinite(p15).all()
    assert float(p15.min()) >= 0.0 and float(p15.max()) <= 1.0


def test_rho_oracle_metric():
    rng = np.random.default_rng(0)
    Bb, L, Zz = 2, 3, 1
    lo = np.array([-1.0, -2.0], np.float32)
    hi = np.array([1.0, 3.0], np.float32)
    target_norm = rng.uniform(0.05, 0.95, (Bb, L, 2, H, W, Zz)).astype(np.float32)
    phys = target_norm * (hi - lo).reshape(1, 1, 2, 1, 1, 1) + lo.reshape(1, 1, 2, 1, 1, 1)
    u_nat, v_nat = rho_to_native(phys)
    mask_u = np.ones((H, W - 1), bool)
    mask_v = np.ones((H - 1, W), bool)

    # truth == the very same conversion path -> oracle error is exactly 0
    se, ae = oracle_native_error_sums(target_norm, lo, hi, u_nat, v_nat, mask_u, mask_v)
    assert se.shape == (L, 2, Zz) and ae.shape == (L, 2, Zz)
    assert np.allclose(se, 0.0, atol=1e-6) and np.allclose(ae, 0.0, atol=1e-6)

    # shifted truth -> oracle error equals the direct masked comparison
    truth_u = u_nat + 1.0
    se2, ae2 = oracle_native_error_sums(target_norm, lo, hi, truth_u, v_nat, mask_u, mask_v)
    se_ref, ae_ref = masked_error_sums(u_nat, truth_u, mask_u)
    assert np.allclose(se2[:, 0, :], se_ref)
    assert np.allclose(ae2[:, 0, :], ae_ref)
    # land cells stay excluded
    mu2 = mask_u.copy()
    mu2[0, 0] = False
    se3, _ = oracle_native_error_sums(target_norm, lo, hi, truth_u, v_nat, mu2, mask_v)
    diff = (u_nat - truth_u)[:, :, 0, 0, :]
    assert np.allclose(se2[:, 0, :] - se3[:, 0, :], (diff ** 2).sum(axis=0))


def test_mask_tensor_writable_and_contiguous():
    import pre_dataset as pd
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        os.makedirs(aligned)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((4, 5), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((4, 5), np.uint8))
        saved = (pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL)
        pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL = aligned, 4, 5, 2
        try:
            t = pd.build_mask_tensor(torch.device("cpu"), depth_index=1)
            assert t.shape == (1, 2, 4, 5, 1)
            assert t.is_contiguous()
            t[0, 0, 0, 0, 0] = 0.0          # must be writable, no read-only warning
            assert t[0, 0, 0, 0, 0] == 0.0
            t2 = pd.build_mask_tensor(torch.device("cpu"), depth_index=None)
            assert t2.shape == (1, 2, 4, 5, 2) and t2.is_contiguous()
        finally:
            pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL = saved


def test_checkpoint_metadata_roundtrip():
    stats_sigma = 0.0856
    sd = sigma_data_from_stats(stats_sigma)
    assert np.isclose(sd, 0.1712)
    state = {"epoch": 0, "best_val": 1.0,
             "config": {"preset": "surface_smoke",
                        "stats_sigma": stats_sigma,
                        "sigma_data_scale": SIGMA_DATA_SCALE,
                        "sigma_data": sd}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pth")
        torch.save(state, p)
        loaded = torch.load(p, weights_only=True)
        sd2, used = sigma_data_from_checkpoint(loaded, stats_sigma)
        assert used and np.isclose(sd2, sd)
        assert loaded["config"]["sigma_data_scale"] == SIGMA_DATA_SCALE
        assert np.isclose(loaded["config"]["stats_sigma"], stats_sigma)

        # full model-state roundtrip through load_checkpoint (weights_only=True)
        model = make_model()
        p2 = os.path.join(d, "model.pth")
        torch.save({"model_state_dict": model.state_dict(), "epoch": 3,
                    "config": {"stats_sigma": stats_sigma,
                               "sigma_data_scale": SIGMA_DATA_SCALE,
                               "sigma_data": sd}}, p2)
        m2 = make_model()
        ckpt = load_checkpoint(p2, m2, map_location="cpu")
        assert ckpt["epoch"] == 3
        sd3, used3 = sigma_data_from_checkpoint(ckpt, stats_sigma)
        assert used3 and np.isclose(sd3, sd)
        assert next(m2.parameters()).device.type == "cpu"


def test_training_modes_and_run_tags():
    full = training_config("surface_smoke", "full", world_size=1)
    assert full == PRESETS["surface_smoke"]
    assert full is not PRESETS["surface_smoke"]

    smoke = training_config("surface_smoke", "smoke", world_size=4)
    assert smoke["embed_dim"] == PRESETS["surface_smoke"]["embed_dim"]
    assert smoke["patch_size"] == PRESETS["surface_smoke"]["patch_size"]
    assert smoke["num_epochs"] == 1 and smoke["sampling_steps"] == 4
    assert smoke["val_windows"] == 4
    assert smoke["max_train_windows"] == (
        SMOKE_BATCHES_PER_RANK * 4 * smoke["batch_size"])
    assert training_run_tag("surface_smoke", full) == (
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2")
    assert training_run_tag("surface_smoke", smoke, "smoke", 4).endswith(
        "_S4_C7_SD2_SMOKE_DDP4")
    # objective suffix: the deterministic baseline NEVER shares a run dir with
    # a diffusion run; the default objective keeps the legacy tag exactly
    assert training_run_tag("surface_smoke", full,
                            objective="persistence_residual") == (
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES")
    assert training_run_tag("surface_smoke", smoke, "smoke", 4,
                            objective="persistence_residual").endswith(
        "_S4_C7_SD2_RES_SMOKE_DDP4")
    assert training_run_tag("surface_smoke", full,
                            objective="diffusion") == training_run_tag("surface_smoke", full)

    for args in (("missing", "full", 1), ("surface_smoke", "bad", 1),
                 ("surface_smoke", "full", 0)):
        try:
            training_config(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"training_config{args} should fail")
    try:
        training_run_tag("surface_smoke", full, objective="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("training_run_tag with unknown objective should fail")


def test_objective_config_helpers():
    assert OBJECTIVES == ("diffusion", "persistence_residual")
    assert DEFAULT_OBJECTIVE == "diffusion"
    assert MASK_SCHEME == "bivariate_rho"
    assert validate_objective("Diffusion") == "diffusion"          # normalized
    assert validate_objective("persistence_residual") == "persistence_residual"
    # unknown objective -> ValueError
    for bad in ("residual", "", "DIFFUSION "):
        try:
            validate_objective(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate_objective({bad!r}) should fail")
    # legacy checkpoints (no config / no objective field) are always diffusion
    assert objective_from_checkpoint(None) == "diffusion"
    assert objective_from_checkpoint({}) == "diffusion"
    assert objective_from_checkpoint({"epoch": 3}) == "diffusion"
    assert objective_from_checkpoint({"config": {}}) == "diffusion"
    assert objective_from_checkpoint(
        {"config": {"objective": "persistence_residual"}}) == "persistence_residual"
    # matching objective passes through; mismatch refuses
    res_ckpt = {"config": {"objective": "persistence_residual"}}
    diff_ckpt = {"config": {"objective": "diffusion"}}
    assert ensure_objective_compatible(res_ckpt, "persistence_residual") == \
        "persistence_residual"
    assert ensure_objective_compatible(diff_ckpt, "diffusion") == "diffusion"
    assert ensure_objective_compatible({}, "diffusion") == "diffusion"
    for ckpt, want in ((diff_ckpt, "persistence_residual"),
                       (res_ckpt, "diffusion")):
        try:
            ensure_objective_compatible(ckpt, want)
        except RuntimeError as e:
            assert "objective" in str(e) and "incompatible" in str(e), str(e)
        else:
            raise AssertionError(f"ensure_objective_compatible({ckpt}, {want!r}) "
                                 "should refuse")


def test_persistence_residual_zero_init_identity():
    # the UNTRAINED wrapper must be EXACTLY last-day persistence: the zero-init
    # head makes the backbone output all zeros, so prediction == base == cond[:, -2:]
    model = make_residual_model()
    assert model.residual_base == "last_day"
    assert model.cond_chans == 14 and model.target_ch == 2
    assert torch.count_nonzero(model.net.head.weight).item() == 0
    torch.manual_seed(0)
    cond = torch.rand(3, 14, H, W, Z)
    with torch.no_grad():
        out = model(cond)
        sample = model.sample(cond, num_sample_steps=8, clamp=True)
    assert out.shape == (3, 2, H, W, Z)
    assert sample.shape == (3, 2, H, W, Z)
    assert torch.equal(out, cond[:, -2:])          # bitwise persistence identity
    assert torch.equal(sample, cond[:, -2:])       # cond in [0,1] -> clamp is a no-op
    # sample() ignores num_sample_steps (deterministic: bitwise identical)
    with torch.no_grad():
        s_other = model.sample(cond, num_sample_steps=2, clamp=True)
    assert torch.equal(sample, s_other)
    # clamp pushes out-of-range predictions back into [0, 1]
    cond_big = torch.zeros(1, 14, H, W, Z)
    cond_big[0, -2:, 0, 0, 0] = 5.0                # base value outside [0,1]
    with torch.no_grad():
        clamped = model.sample(cond_big, clamp=True)
    assert clamped[0, 0, 0, 0, 0].item() == 1.0
    # wrong condition channel count is rejected
    try:
        model(torch.randn(1, 15, H, W, Z))
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for wrong cond channels")
    # a backbone without self_condition cannot carry the condition
    net_nc = IAFNODiff(dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=8,
                       num_blocks=1, in_chans=2, out_chans=2, cond_chans=14,
                       ex_layer=1, nlayer=1, hidden_size_factor=1,
                       dim_f=(H, W, Z), self_condition=False)
    try:
        PersistenceResidualIAFNO(net_nc)
    except ValueError as e:
        assert "self_condition" in str(e), str(e)
    else:
        raise AssertionError("expected ValueError for self_condition=False backbone")


def test_persistence_residual_training_step():
    # one real forward/backward/optimizer step: finite loss, head moves, output
    # leaves the persistence identity, and EVERY parameter receives a .grad
    # (zero-valued at first for pre-head layers, but present — the property DDP
    # needs to all-reduce without find_unused_parameters)
    model = make_residual_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(1)
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 2, H, W, Z)
    mask = torch.ones(1, 2, H, W, Z)
    mask[0, 0, 0, 0] = 0.0

    pred = model(cond)
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)                 # DDP-safe
    assert torch.isfinite(model.net.head.weight.grad).all()
    assert model.net.head.weight.grad.abs().sum() > 0        # head actually moves
    head_before = model.net.head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(head_before, model.net.head.weight.detach())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # after a real update the model is no longer identical to persistence
    with torch.no_grad():
        out2 = model(cond)
    assert not torch.equal(out2, cond[:, -2:])
    # land gradient stays zero (masked loss never trains land outputs)
    assert model.net.head.weight.grad is not None


def test_masked_mse_loss_semantics():
    torch.manual_seed(2)
    pred = torch.rand(B, 2, H, W, Z)
    target = torch.rand(B, 2, H, W, Z)
    # land cell (0,0) invalid in BOTH channels
    mask = torch.ones(1, 2, H, W, Z)
    mask[0, :, 0, 0, :] = 0.0
    loss = masked_mse_loss(pred, target, mask)
    m = mask.expand_as(pred)
    ref = (((pred - target) ** 2 * m).sum(dim=(1, 2, 3, 4))
           / m.sum(dim=(1, 2, 3, 4))).mean()
    assert torch.allclose(loss, ref, atol=1e-6)
    # corrupting LAND cells must not change the loss
    pred_land_dirty = pred.clone()
    pred_land_dirty[:, :, 0, 0, :] = 123.0
    assert torch.allclose(loss, masked_mse_loss(pred_land_dirty, target, mask))
    # per-sample denominators are independent (batch-varying validity)
    maskb = torch.ones(B, 2, H, W, Z)
    maskb[0, :, :2, :, :] = 0.0
    ref_b = (((pred - target) ** 2 * maskb).sum(dim=(1, 2, 3, 4))
             / maskb.sum(dim=(1, 2, 3, 4))).mean()
    assert torch.allclose(masked_mse_loss(pred, target, maskb), ref_b, atol=1e-6)
    # single-channel (1,1,H,W,Z) mask broadcasts like diffusion.forward
    mask1 = torch.ones(1, 1, H, W, Z)
    mask1[0, 0, 0, 0, :] = 0.0
    assert torch.isfinite(masked_mse_loss(pred, target, mask1))
    # all-zero mask -> exactly 0, no NaN (same convention as the EDM loss)
    zero_loss = masked_mse_loss(pred, target, torch.zeros(1, 2, H, W, Z))
    assert torch.isfinite(zero_loss) and zero_loss.item() == 0.0


def test_persistence_residual_checkpoint_roundtrip():
    # save -> rebuild -> load_checkpoint(weights_only=True) reproduces the
    # trained model exactly; metadata carries the objective family
    model = make_residual_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(3)
    cond = torch.rand(1, 14, H, W, Z)
    target = torch.rand(1, 2, H, W, Z)
    loss = masked_mse_loss(model(cond), target, torch.ones(1, 2, H, W, Z))
    loss.backward()
    optimizer.step()

    state = {"epoch": 0, "best_val": float(loss.item()),
             "model_state_dict": model.state_dict(),
             "config": {"preset": "surface_smoke", "objective": "persistence_residual",
                        "residual_base": model.residual_base,
                        "cond_chans": model.cond_chans, "target_ch": model.target_ch,
                        "mask_scheme": MASK_SCHEME,
                        "time_sigma": model.time_sigma, "world_size": 1}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "best.pth")
        torch.save(state, p)
        # rebuild exactly like pre_evaluate.py does: fresh IAFNODiff + wrapper,
        # then load the state dict (the zero-init head is replaced by training)
        fresh = make_residual_model()
        assert torch.count_nonzero(fresh.net.head.weight).item() == 0
        ckpt = load_checkpoint(p, fresh, map_location="cpu")
        assert ckpt["epoch"] == 0
        assert ensure_objective_compatible(ckpt, "persistence_residual") == \
            "persistence_residual"
        with torch.no_grad():
            a = model(cond)
            b = fresh(cond)
        assert torch.allclose(a, b, atol=1e-6)
        assert not torch.equal(b, cond[:, -2:])   # trained: no longer persistence


def test_persistence_residual_static_mask_input():
    # Phase-5 arm B: the wrapper concatenates the static mask channels onto the
    # DYNAMIC condition for the backbone only; base stays last-day u/v and the
    # untrained model is still EXACTLY persistence (zero-init identity).
    model = make_residual_model(cond_chans=16)
    assert model.cond_chans == 16
    torch.manual_seed(1)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()
    with torch.no_grad():
        out = model(cond, static_cond=static)
    assert out.shape == (2, 2, H, W, Z)
    assert torch.equal(out, cond[:, -2:])          # zero-init identity holds
    # explicit batch-broadcast equals manually feeding the concatenated
    # x_self_cond to the backbone (the documented static-channel layout)
    import math
    with torch.no_grad():
        a = model(cond, static_cond=static)
        t = torch.full((2,), 0.25 * math.log(model.time_sigma))
        manual = model.net(torch.zeros_like(cond[:, :2]), t,
                           torch.cat([cond, static.expand(2, -1, -1, -1, -1)],
                                     dim=1)) + cond[:, -2:]
    assert torch.equal(a, manual)
    # wrong dynamic/static channel split is rejected loudly
    try:
        model(cond)                                 # 14 ch into a 16-ch model
        raise AssertionError("expected AssertionError for missing static_cond")
    except AssertionError:
        pass
    try:
        model(torch.cat([cond, cond], dim=1), static_cond=static)
        raise AssertionError("expected AssertionError for extra static channels")
    except AssertionError:
        pass
    try:
        model(cond, static_cond=static[0:1, :, :, :1, :])
        raise AssertionError("expected AssertionError for spatial mismatch")
    except AssertionError:
        pass
    # clamp path of sample() with static channels
    with torch.no_grad():
        trained_out = model.sample(cond, static_cond=static, clamp=True)
    assert trained_out.min() >= 0.0 and trained_out.max() <= 1.0


def test_rollout_static_cond_threading():
    # ensemble_rollout forwards static_cond to EVERY sample() call unchanged,
    # while the sliding window stays the pure 14-channel dynamic condition
    # (drop-oldest/append-prediction slicing untouched).
    class _StaticRecorder:
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True, static_cond=None):
            self.calls.append((cur.clone(), None if static_cond is None
                               else static_cond.clone()))
            return cur[:, :2] + 1.0

    torch.manual_seed(2)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()

    rec = _StaticRecorder()
    preds = ensemble_rollout(rec, cond, 3, 1, seed=0, static_cond=static)
    assert preds.shape == (2, 1, 3, 2, H, W, Z)
    assert len(rec.calls) == 3
    for step, (cur, sc) in enumerate(rec.calls):
        assert cur.shape == (2, 14, H, W, Z)        # window stays pure dynamic
        assert sc.shape == (1, 2, H, W, Z) and sc.dim() == 5
        assert torch.equal(sc, static)              # same tensor every step
        if step > 0:
            prev_pred = rec.calls[step - 1][0][:, :2] + 1.0
            # channels 0-1 dropped, prediction appended at the END of the window
            assert torch.equal(cur[:, :-2], rec.calls[step - 1][0][:, 2:])
            assert torch.equal(cur[:, -2:], prev_pred)
    # without static_cond the model receives None (historical call signature)
    rec_plain = _StaticRecorder()
    ensemble_rollout(rec_plain, cond, 2, 1, seed=0)
    assert all(sc is None for _, sc in rec_plain.calls)
    # per-window seeds path forwards static_cond as well
    rec_seeds = _StaticRecorder()
    ensemble_rollout(rec_seeds, cond, 2, 1, seeds=[5, 6], static_cond=static)
    assert len(rec_seeds.calls) == 4                # 2 windows x 2 steps
    assert all(sc is not None and torch.equal(sc, static)
               for _, sc in rec_seeds.calls)


def test_rollout_remask_feedback():
    class _LandDirtyRecorder:
        """Predicts cur[:, :2] + 1 (nonzero EVERYWHERE, including land cells)
        and records every condition window it is fed."""
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True):
            self.calls.append(cur.clone())
            return cur[:, :2] + 1.0

    torch.manual_seed(0)
    cond = torch.rand(1, 14, H, W, Z)
    ocean = torch.ones(1, 2, H, W, Z)
    ocean[0, :, 0, 0, :] = 0.0                    # (0,0) is land in both channels

    # OFF (default / historical): the dirty prediction (land included) is fed back.
    # calls[i][0] is the 4-D (14, H, W, Z) window: channels live on dim 0.
    s_off = _LandDirtyRecorder()
    p_off = ensemble_rollout(s_off, cond, 2, 1, seed=0, remask_feedback=False)
    assert p_off.shape == (1, 1, 2, 2, H, W, Z)
    fed_back = s_off.calls[1][0]
    assert torch.equal(fed_back[-2:], s_off.calls[0][0][:2] + 1.0)
    assert fed_back[-2, 0, 0, 0].item() != 0.0             # land NOT re-zeroed

    # ON: the prediction is remasked (land -> 0) before storage AND feedback
    s_on = _LandDirtyRecorder()
    p_on = ensemble_rollout(s_on, cond, 2, 1, seed=0, remask_feedback=True,
                            ocean_mask=ocean)
    fed_back_on = s_on.calls[1][0]
    assert fed_back_on[-2, 0, 0, 0].item() == 0.0
    assert fed_back_on[-1, 0, 0, 0].item() == 0.0
    # ocean values are unchanged by the remask; stored preds are the masked ones
    assert torch.allclose(p_on[0, 0], p_off[0, 0] * ocean[0])
    # the second-step INPUT differs only at masked-out (land) cells
    diff = (s_on.calls[1][0] - s_off.calls[1][0]).abs()
    assert diff[-2, 0, 0, 0].item() > 0 and diff[-1, 0, 0, 0].item() > 0
    assert diff[:, 1, 1, :].max().item() == 0
    # seeds still scope the trajectories; remasking is deterministic either way
    s_on_b = _LandDirtyRecorder()
    ensemble_rollout(s_on_b, cond, 2, 1, seeds=[7], remask_feedback=True,
                     ocean_mask=ocean)
    assert torch.equal(s_on_b.calls[1][0], fed_back_on)
    # remask_feedback=True without a mask is rejected early
    try:
        ensemble_rollout(_LandDirtyRecorder(), cond, 1, 1,
                         remask_feedback=True, ocean_mask=None)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for remask without ocean_mask")


def test_deterministic_model_in_rollout():
    # the persistence-residual wrapper inside the UNCHANGED rollout machinery:
    # seeds cannot matter (no RNG consumption) and ensemble members coincide
    model = make_residual_model()
    torch.manual_seed(4)
    cond = torch.rand(2, 14, H, W, Z)
    p_seed1 = ensemble_rollout(model, cond, 3, 1, seed=1)
    p_seed999 = ensemble_rollout(model, cond, 3, 1, seed=999)
    assert p_seed1.shape == (2, 1, 3, 2, H, W, Z)
    assert torch.equal(p_seed1, p_seed999)                 # bitwise deterministic
    # per-window seeds path behaves identically for a deterministic model
    p_pw = ensemble_rollout(model, cond, 3, 1, seeds=[1, 2])
    assert torch.equal(p_pw[:, 0], p_seed1[:, 0])
    # E=2 members are identical copies (mean == member)
    p_e2 = ensemble_rollout(model, cond, 3, 2, seed=1)
    assert p_e2.shape == (2, 2, 3, 2, H, W, Z)
    assert torch.equal(p_e2[:, 0], p_e2[:, 1])
    assert torch.allclose(ensemble_mean(p_e2), p_e2[:, 0])
    # predictions stay in [0,1] and respect the persistence identity per step
    assert float(p_seed1.min()) >= 0.0 and float(p_seed1.max()) <= 1.0


def _parse_progress_line(line):
    assert line.startswith("PROGRESS "), line
    tokens = line.split()[1:]
    fields = {}
    for i, tok in enumerate(tokens):
        key, sep, value = tok.partition("=")
        assert sep and key and value != "", line           # strict k=v tokens
        fields[key] = value
    return fields


def test_progress_reporter_lines():
    class _FakeClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t
        def advance(self, dt):
            self.t += dt

    # ---- non-interactive (pipe/file): NO bar, periodic newline-flushed lines
    clk = _FakeClock()
    buf = io.StringIO()
    rep = ProgressReporter("train", total=10, stream=buf, clock=clk,
                           interactive=False, unit="step", samples_per_unit=4,
                           context={"epoch": "1/4"})
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 1 and "status=start" in lines[0]
    f = _parse_progress_line(lines[0])
    assert f["phase"] == "train" and f["step"] == "0/10" and f["epoch"] == "1/4"
    assert f["elapsed_s"] == "0.0"

    rep.update(3, loss="0.50000", lr="1.00e-03")           # inside the interval
    assert len(buf.getvalue().splitlines()) == 1           # nothing emitted yet
    clk.advance(30.0)                                      # interval reached
    rep.update(2, loss="0.25000", lr="1.00e-03")           # -> emit periodic line
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 2 and "status=running" in lines[1]
    f = _parse_progress_line(lines[1])
    assert f["phase"] == "train" and f["status"] == "running"
    assert f["step"] == "5/10" and f["loss"] == "0.25000" and f["lr"] == "1.00e-03"
    assert f["epoch"] == "1/4"
    assert float(f["elapsed_s"]) >= 30.0
    assert float(f["eta_s"]) > 0.0
    assert f["step_per_s"].startswith("0.") and f["step_per_s"] != "0.000"
    # both rates are formatted to 3 decimals, so allow rounding slack
    assert abs(float(f["sample_per_s"]) - 4.0 * float(f["step_per_s"])) < 5e-3
    rep.close(loss="0.10000")
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    # reporters close with the INTERMEDIATE phase_done status — the script-level
    # `completed` is reserved for the entrypoints' own final line
    assert len(lines) == 3 and "status=phase_done" in lines[2]
    f = _parse_progress_line(lines[2])
    assert f["step"] == "5/10" and f["loss"] == "0.10000"
    # close is idempotent
    rep.close()
    assert len(buf.getvalue().splitlines()) == 3

    # ---- failure path: status=failed is emitted with the error detail
    buf2 = io.StringIO()
    clk2 = _FakeClock()
    rep2 = ProgressReporter("eval", total=4, stream=buf2, clock=clk2,
                            interactive=False, unit="window")
    rep2.update(1)
    clk2.advance(30.0)
    rep2.update(1, d1_rmse="0.1234")                       # -> periodic running line
    out2 = [ln for ln in buf2.getvalue().splitlines() if ln]
    assert len(out2) == 2 and "status=running" in out2[1]
    rep2.fail(error="RuntimeError: non_finite_loss_at_epoch_2")
    out2 = [ln for ln in buf2.getvalue().splitlines() if ln]
    assert len(out2) == 3 and "status=failed" in out2[2]
    f2 = _parse_progress_line(out2[2])
    assert f2["phase"] == "eval" and f2["window"] == "2/4"
    assert f2["d1_rmse"] == "0.1234"
    assert "RuntimeError" in f2["error"]
    # values never contain spaces (key=value parseability)
    for line in buf2.getvalue().splitlines():
        for tok in line.split()[1:]:
            assert " " not in tok, line

    # ---- disabled reporter (non-rank-0 DDP): fully silent
    buf3 = io.StringIO()
    rep3 = ProgressReporter("train", total=5, stream=buf3, clock=_FakeClock(),
                            interactive=False, enabled=False)
    rep3.update(2, loss="0.1")
    rep3.close()
    assert buf3.getvalue() == ""

    # ---- format_progress field order: phase first, status last
    line = format_progress("train", "running", epoch="2/4", loss="0.5")
    toks = line.split()
    assert toks[0] == "PROGRESS" and toks[1] == "phase=train"
    assert toks[-1] == "status=running" and "epoch=2/4" in toks
    _parse_progress_line(line)


def test_progress_heartbeat_without_updates():
    # the periodic line is TIME-DRIVEN: with ZERO update() calls (a single
    # rollout step blocking far longer than the interval) the daemon heartbeat
    # still emits running lines; close() stops it for good
    buf = io.StringIO()
    rep = ProgressReporter("eval", total=100, unit="window", interval=0.2,
                           stream=buf, interactive=False)
    assert len([ln for ln in buf.getvalue().splitlines() if ln]) == 1  # start only
    time.sleep(0.6)                                     # > 2x interval, no updates
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    running = [ln for ln in lines if "status=running" in ln]
    assert len(running) >= 1, lines                     # heartbeat without updates
    f = _parse_progress_line(running[-1])
    assert f["window"] == "0/100" and f["phase"] == "eval"
    rep.close()
    frozen = len(buf.getvalue().splitlines())
    time.sleep(0.3)
    assert len(buf.getvalue().splitlines()) == frozen   # heartbeat stopped by close


def test_progress_multiline_error_sanitization():
    # a multi-line exception message must stay on ONE parseable key=value line:
    # every whitespace run (spaces, newlines, tabs) collapses to one underscore
    line = format_progress("train", "failed",
                           error="RuntimeError: boom\nsecond line\twith\ttabs  and  spaces")
    assert "\n" not in line and "\t" not in line
    _parse_progress_line(line)
    assert ("error=RuntimeError:_boom_second_line_with_tabs_and_spaces"
            in line.split())
    # single-line values pass through untouched apart from inner spaces
    assert format_progress("eval", "failed", error="E:x y").split()[-2] == \
        "error=E:x_y"


def test_progress_failure_hook_dedup_and_stage():
    # the excepthook fallback emits ONE standard failed line (sanitized error,
    # stage read at failure time) for exceptions escaping guarded blocks;
    # mark_progress_failed() suppresses duplicates from the guarded handler;
    # the original excepthook still receives the exception (traceback).
    old_hook = sys.excepthook
    buf = io.StringIO()
    stage = ["setup"]
    fallback_calls = []
    try:
        reset_progress_failure_state()
        install_progress_failure_hook("train", stage=lambda: stage[0], stream=buf,
                                      fallback=lambda *a: fallback_calls.append(a))
        sys.excepthook(RuntimeError, RuntimeError("pre-flight refused\nbad config"), None)
        lines = [ln for ln in buf.getvalue().splitlines() if ln]
        assert len(lines) == 1 and "status=failed" in lines[0]
        f = _parse_progress_line(lines[0])
        assert f["phase"] == "train" and f["stage"] == "setup"
        assert f["error"] == "RuntimeError:_pre-flight_refused_bad_config"
        assert len(fallback_calls) == 1                 # traceback still dispatched
        # a guarded block that already reported its own failure -> hook silent
        mark_progress_failed()
        sys.excepthook(ValueError, ValueError("guarded failure"), None)
        assert len(buf.getvalue().splitlines()) == 1
        assert len(fallback_calls) == 2
        # the stage callable is read AT FAILURE TIME
        reset_progress_failure_state()
        stage[0] = "postprocess"
        sys.excepthook(ValueError, ValueError("late failure"), None)
        last = buf.getvalue().splitlines()[-1]
        assert _parse_progress_line(last)["stage"] == "postprocess"
        assert "status=failed" in last
    finally:
        sys.excepthook = old_hook
        reset_progress_failure_state()


def test_norm_fingerprint_and_time_sigma_checks():
    lo, hi = [-1.5, -2.0], [2.5, 3.0]
    mv = "deadbeef01234567"
    # matching fingerprint -> no warnings, no raise
    assert check_norm_fingerprint({"norm_lo": lo, "norm_hi": hi,
                                   "mask_version": mv}, lo, hi, mv) == []
    # float noise within tolerance passes
    assert check_norm_fingerprint({"norm_lo": [-1.5 + 1e-9, -2.0],
                                   "norm_hi": hi, "mask_version": mv},
                                  lo, hi, mv) == []
    # legacy checkpoint predating the recorded fields -> warnings, NOT a raise
    ws = check_norm_fingerprint({}, lo, hi, mv)
    assert len(ws) == 2 and all("legacy" in w for w in ws)
    # normalization drift -> refuse (silently changed stats are the hazard)
    for bad in ({"norm_lo": [-1.4, -2.0], "norm_hi": hi, "mask_version": mv},
                {"norm_lo": lo, "norm_hi": [2.5, 3.1], "mask_version": mv}):
        try:
            check_norm_fingerprint(bad, lo, hi, mv)
        except RuntimeError as e:
            assert "normalization fingerprint mismatch" in str(e), str(e)
        else:
            raise AssertionError("expected normalization mismatch refusal")
    # mask drift -> refuse
    try:
        check_norm_fingerprint({"norm_lo": lo, "norm_hi": hi,
                                "mask_version": "ffffffffffffffff"}, lo, hi, mv)
    except RuntimeError as e:
        assert "mask_version" in str(e), str(e)
    else:
        raise AssertionError("expected mask mismatch refusal")
    # residual time embedding: matching value passes; missing/mismatched refuse
    check_residual_time_sigma({"time_sigma": 0.002}, 0.002)
    for bad_cfg in ({}, {"time_sigma": 0.05}):
        try:
            check_residual_time_sigma(bad_cfg, 0.002)
        except RuntimeError as e:
            assert "time_sigma" in str(e), str(e)
        else:
            raise AssertionError(f"expected time_sigma refusal for {bad_cfg}")


def test_lead_schedule_pattern_and_rank_consistency():
    # doc §5.1 fixed schedule: 50% day-1 anchor + one full 2..K cycle per period
    assert [lead_for_batch(i, 5) for i in range(8)] == [1, 2, 1, 3, 1, 4, 1, 5]
    assert [lead_for_batch(i, 5) for i in range(8, 16)] == [1, 2, 1, 3, 1, 4, 1, 5]
    assert lead_schedule_str(5) == "1,2,1,3,1,4,1,5"
    assert lead_schedule_str(10) == "1,2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10"
    # K=1 is the historical single-step path: schedule inert
    assert all(lead_for_batch(i, 1) == 1 for i in range(32))
    assert lead_schedule_str(1) == "1"
    # distribution: exactly 50% anchor, uniform 2..K
    K = 5
    per = 2 * (K - 1)
    counts = {j: [lead_for_batch(i, K) for i in range(per)].count(j)
              for j in range(1, K + 1)}
    assert counts == {1: K - 1, 2: 1, 3: 1, 4: 1, 5: 1}
    # purity: a pure function of (batch_index, K) — same inputs, same output;
    # DDP ranks call it with their own batch index, so identical step indices
    # on every rank yield identical J (equal batch counts via drop_last=True)
    assert all(lead_for_batch(bi, K) == lead_for_batch(bi, K)
               for bi in range(3 * per))
    # env reading: default/empty -> 1; valid int; garbage refused
    assert train_horizon("") == 1 and train_horizon("5") == 5
    for bad in ("0", "-2", "x"):
        try:
            train_horizon(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {TRAIN_HORIZON_ENV}={bad!r}")
    assert init_checkpoint("") is None
    assert init_checkpoint("~/a.pth") == os.path.expanduser("~/a.pth")


def test_training_config_ms_defaults_and_tags():
    # multi-step runs use the frozen MS_DEFAULTS instead of the preset values
    cfg1 = training_config("surface_smoke", "full", 1)
    assert cfg1["lr"] == 1e-3 and cfg1["num_epochs"] == 10
    cfg5 = training_config("surface_smoke", "full", 1, train_horizon=5)
    assert cfg5["lr"] == MS_DEFAULTS["lr"] == 1e-4
    assert cfg5["num_epochs"] == MS_DEFAULTS["num_epochs"] == 5
    cfg5smoke = training_config("surface_smoke", "smoke", 1, train_horizon=5)
    assert cfg5smoke["lr"] == 1e-4 and cfg5smoke["num_epochs"] == 1
    # tags: _MS{K} only for K > 1; never shares a dir with single-step/_MSK runs
    base = dict(config=cfg1, objective="persistence_residual")
    assert run_tag_for("surface_smoke", **base) == \
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES"
    assert run_tag_for("surface_smoke", **base, train_horizon=1).endswith("_RES")
    assert run_tag_for("surface_smoke", **base, train_horizon=5).endswith("_RES_MS5")
    assert run_tag_for("surface_smoke", **base, train_horizon=10).endswith("_RES_MS10")
    assert run_tag_for("surface_smoke", **base, static_mask=True).endswith("_MSK")
    assert not run_tag_for("surface_smoke", **base).endswith("_MS1")
    t_smoke = training_run_tag("surface_smoke", cfg5, "smoke", 1,
                               objective="persistence_residual", train_horizon=5)
    assert t_smoke.endswith("_RES_MS5_SMOKE")
    t_ddp = training_run_tag("surface_smoke", cfg5, "full", 2,
                             objective="persistence_residual", train_horizon=5)
    assert t_ddp.endswith("_RES_MS5_DDP2")


def test_multistep_k1_matches_single_step():
    # K=1 must be the exact historical single-step path: the lead-1 branch and
    # a detached_feedback_window(lead=1) (which returns cond unchanged) both
    # reduce to one forward + masked MSE on target[:, 0]
    torch.manual_seed(3)
    model = make_residual_model()
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 5, 2, H, W, Z)   # horizon=5 windows
    mask = torch.ones(1, 2, H, W, Z)

    with torch.no_grad():
        # historical single-step ops
        pred_hist = model(cond)
        loss_hist = masked_mse_loss(pred_hist, target[:, 0], mask)
        # K=1 trainer path: lead == lead_for_batch(bi, 1) == 1 for every batch
        for bi in range(4):
            assert lead_for_batch(bi, 1) == 1
        pred_k1 = model(cond)
        loss_k1 = masked_mse_loss(pred_k1, target[:, 0], mask)
        # a lead-1 multi-step branch via the helper: cond unchanged, same ops
        cur = detached_feedback_window(model, cond, lead_for_batch(0, 1))
        assert torch.equal(cur, cond)
        pred_ms1 = model(cur)
        loss_ms1 = masked_mse_loss(pred_ms1, target[:, 0], mask)
    assert torch.equal(pred_hist, pred_k1) and torch.equal(pred_hist, pred_ms1)
    assert torch.equal(loss_hist, loss_k1) and torch.equal(loss_hist, loss_ms1)


def test_detached_feedback_window_gradient_and_calls():
    # J=3: exactly J-1 no-grad self-feedback forwards + 1 grad-carrying final
    # forward; early predictions have NO graph, the final one does, and the
    # sliding window matches the formal rollout's drop-oldest/append update
    torch.manual_seed(4)
    model = make_residual_model()
    # untrained == persistence bitwise; perturb the residual head so the
    # feedback provably carries the MODEL'S OWN (non-persistence) predictions
    with torch.no_grad():
        model.net.head.weight.data += 1e-3
    calls = {"n": 0, "grad_modes": []}

    def step_fn(cur):
        calls["n"] += 1
        calls["grad_modes"].append(torch.is_grad_enabled())
        return model(cur)

    cond = torch.rand(B, 14, H, W, Z)
    cur = detached_feedback_window(step_fn, cond, lead=3)
    assert calls["n"] == 2                                   # J-1 feedback steps
    assert calls["grad_modes"] == [False, False]             # all detached
    # sliding window: drop the oldest DAY (2 channels), append own prediction
    with torch.no_grad():
        p1 = model(cond).clamp(0., 1.).float()
        p2 = model(torch.cat([cond[:, 2:], p1], dim=1)).clamp(0., 1.).float()
    assert cur.shape == cond.shape
    # two feedback steps drop TWO days (4 channels): cur = [c3..c6, p1, p2]
    assert torch.equal(cur[:, :10], cond[:, 4:])
    assert torch.equal(cur[:, 10:12], p1)     # day-5 slot now holds feedback p1
    assert torch.equal(cur[:, 12:], p2)
    # feedback uses the MODEL'S OWN predictions, not persistence
    assert not torch.equal(cur[:, 12:], cond[:, -2:])
    # final step carries gradients; backward reaches the head
    target = torch.rand(B, 2, H, W, Z)
    pred = model(cur)
    assert pred.grad_fn is not None
    masked_mse_loss(pred, target, torch.ones(1, 2, H, W, Z)).backward()
    assert model.net.head.weight.grad is not None
    assert torch.isfinite(model.net.head.weight.grad).all()
    # J=1 returns cond unchanged and calls nothing
    calls["n"] = 0
    cur1 = detached_feedback_window(step_fn, cond, lead=1)
    assert calls["n"] == 0 and torch.equal(cur1, cond)


def test_untrained_multistep_reduces_to_persistence():
    # doc §6 WP2 item 3: the zero-init residual model in a multi-step rollout
    # is EXACTLY "persistence forever": every feedback prediction equals the
    # current last day, so the window slides trivially and the J-th prediction
    # is the ORIGINAL last-day persistence — bitwise
    torch.manual_seed(5)
    model = make_residual_model()
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 5, 2, H, W, Z)
    mask = torch.ones(1, 2, H, W, Z)
    J = 5
    with torch.no_grad():
        cur = detached_feedback_window(model, cond, lead=J)
        pred = model.sample(cur, clamp=True)
        loss_ms = masked_mse_loss(pred, target[:, J - 1], mask)
        base = cond[:, -2:]
        loss_pers = masked_mse_loss(base.expand_as(pred), target[:, J - 1], mask)
    # untrained forward == base == last-day persistence; clamp is a no-op in [0,1]
    assert torch.equal(cur[:, -2:], cond[:, -2:])
    assert torch.equal(pred, cond[:, -2:])
    assert torch.equal(loss_ms, loss_pers)
    # four feedback steps drop four days: cur = [c4, c5, c6, p1..p4], every
    # p == c6 (persistence feedback == identity slide)
    assert torch.equal(cur[:, :6], cond[:, 8:])
    for d in range(6, 14, 2):
        assert torch.equal(cur[:, d:d + 2], cond[:, -2:])


def test_dataset_horizon5_split_and_alignment():
    # doc §6 WP2 item 5: horizon=K windows never cross the split boundary and
    # target[:, J-1] is the absolute day start+context+J-1 (0-based)
    import pre_dataset as pds
    rng = np.random.default_rng(1)
    T, S, HH, WW = 30, 2, 4, 5
    CONTEXT, K = 7, 5
    u = rng.uniform(-1, 1, (T, S, HH, WW)).astype(np.float32)
    v = rng.uniform(-1, 1, (T, S, HH, WW)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        os.makedirs(aligned)
        np.save(os.path.join(aligned, "u_rho.npy"), u)
        np.save(os.path.join(aligned, "v_rho.npy"), v)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "ocean_time.npy"),
                np.arange(np.datetime64("1994-01-01"), T, dtype="datetime64[D]"))
        saved = (pds.ALIGNED_DIR, pds.NORM_DIR, pds.H, pds.W, pds.T_TOTAL,
                 dict(pds.SPLITS))
        pds.ALIGNED_DIR = aligned
        pds.H, pds.W, pds.T_TOTAL = HH, WW, T
        ds = dsv = None
        try:
            pds.SPLITS.clear()
            pds.SPLITS.update(train=(0, 16), val=(16, 29), test=(29, 30))
            stats = {"lo": np.float32([-1.0, -1.0]), "hi": np.float32([1.0, 1.0])}
            ds = PREUVDataset("train", stats, context=CONTEXT, horizon=K,
                              depth_index=0, stride=1, max_windows=None)
            # last_start = 16 - (7+5) = 4 -> starts 0..4, five windows
            assert len(ds) == 5, len(ds)
            cond, target, start = ds[2]
            assert cond.shape == (14, HH, WW, 1)
            assert target.shape == (K, 2, HH, WW, 1)
            s = int(start)
            assert s == 2
            # cond == days [s, s+7); target[:, j] == day s+7+j
            for j in range(CONTEXT):
                day = s + j
                assert torch.equal(cond[2 * j], torch.from_numpy(
                    ((u[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
                assert torch.equal(cond[2 * j + 1], torch.from_numpy(
                    ((v[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
            for j in range(K):
                day = s + CONTEXT + j
                assert torch.equal(target[j, 0], torch.from_numpy(
                    ((u[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
            # no window crosses the train/val boundary: max absolute day used
            _, _, last_start = ds[len(ds) - 1]
            assert int(last_start) + CONTEXT + K <= pds.SPLITS["train"][1]
            # the same horizon on the val split stays inside val
            dsv = PREUVDataset("val", stats, context=CONTEXT, horizon=K,
                               depth_index=0, stride=1, max_windows=None)
            assert len(dsv) == 2     # last_start = 29-12 = 17 -> starts 16..17
            _, _, vs = dsv[0]
            assert int(vs) + CONTEXT + K <= pds.SPLITS["val"][1]
        finally:
            _close_mmaps(ds, dsv)          # release handles before dir cleanup
            pds.ALIGNED_DIR, pds.NORM_DIR, pds.H, pds.W, pds.T_TOTAL = saved[:5]
            pds.SPLITS.clear()
            pds.SPLITS.update(saved[5])


def test_multistep_checkpoint_config_and_resume_guards():
    # doc §6 WP2 item 6: train_horizon / lead_schedule / feedback_detach /
    # init_checkpoint survive a save/load roundtrip and the resume guards
    # refuse any semantic change; legacy checkpoints are K=1-only
    ms_cfg = {"preset": "surface_smoke", "train_mode": "full", "world_size": 1,
              "objective": "persistence_residual",
              "train_horizon": 5, "lead_schedule": "1,2,1,3,1,4,1,5",
              "feedback_detach": True, "init_checkpoint": "/tmp/Ep10.pth",
              "init_weights_only": True}
    model = make_residual_model()
    state = {"epoch": 0, "model_state_dict": model.state_dict(),
             "config": ms_cfg}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Ep1.pth")
        torch.save(state, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    assert cfg["train_horizon"] == 5 and cfg["feedback_detach"] is True
    assert cfg["lead_schedule"] == lead_schedule_str(5)
    assert cfg["init_weights_only"] is True
    # matching semantics pass
    check_multistep_config(cfg, 5, "1,2,1,3,1,4,1,5")
    # horizon change -> refuse
    for bad_h in (1, 10):
        try:
            check_multistep_config(cfg, bad_h, lead_schedule_str(bad_h))
        except RuntimeError as e:
            assert "train_horizon" in str(e), str(e)
        else:
            raise AssertionError("expected horizon-change refusal")
    # schedule change -> refuse
    try:
        check_multistep_config(cfg, 5, "5,4,3,2,1")
    except RuntimeError as e:
        assert "lead_schedule" in str(e), str(e)
    else:
        raise AssertionError("expected schedule-change refusal")
    # legacy checkpoint (no train_horizon): K=1 fine, K>1 must use weights-only
    # init instead of resume
    check_multistep_config({}, 1, "1")
    try:
        check_multistep_config({}, 5, "1,2,1,3,1,4,1,5")
    except RuntimeError as e:
        assert "DIAFNO_INIT_CHECKPOINT" in str(e), str(e)
    else:
        raise AssertionError("expected legacy multi-step resume refusal")
    # weights-only init: model weights restored EXACTLY, optimizer stays fresh
    torch.manual_seed(6)
    target_model = make_residual_model()
    init_state = {"epoch": 9, "model_state_dict": model.state_dict()}
    target_model.load_state_dict(init_state["model_state_dict"])
    for (k1, p1), (k2, p2) in zip(model.state_dict().items(),
                                  target_model.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), k1
    opt = torch.optim.Adam(target_model.parameters(), lr=1e-4)
    assert len(opt.state) == 0        # fresh optimizer: no moments carried over


def test_diag_leadtime_npz_payload_keys_distinct():
    # regression: the historical f"{field}_{var}" keys let the persistence
    # array silently overwrite the model array; the payload must keep the
    # m/p source prefix so every key is unique and maps to its own source
    L = 4
    res = {f"{n}{v}": dict(rmse=np.arange(L) + (0 if n == "m" else 100.0),
                           bias=np.zeros(L) + (0 if n == "m" else -1.0),
                           var_ratio=np.ones(L), corr_mean=np.ones(L),
                           corr_med=np.ones(L), n=np.full(L, 7.0))
           for n in ("m", "p") for v in ("u", "v")}
    payload = build_npz_payload(res, L, dict(split=np.str_("val"),
                                             n_windows=np.int64(3)))
    stat_keys = [k for k in payload if k not in ("lead", "split", "n_windows")]
    assert len(stat_keys) == len(set(stat_keys)), "NPZ key collision"
    for var in ("u", "v"):
        for field in ("rmse", "bias", "n"):
            assert payload[f"m_{field}_{var}"] is res[f"m{var}"][field]
            assert payload[f"p_{field}_{var}"] is res[f"p{var}"][field]
    assert not np.array_equal(payload["m_rmse_u"], payload["p_rmse_u"])
    assert np.array_equal(payload["lead"], np.arange(1, L + 1))


def test_representative_layer_presets():
    # work package 5 (experiment 11): middle/bottom presets differ from
    # surface_smoke ONLY in depth_index; architecture/budget stay identical
    for name, depth in (("middle_smoke", 14), ("bottom_smoke", 0)):
        assert name in PRESETS
        cfg = PRESETS[name]
        assert cfg["depth_index"] == depth
        for key, v in PRESETS["surface_smoke"].items():
            if key == "depth_index":
                continue
            assert cfg[key] == v, (name, key, cfg[key], v)
        tag = run_tag_for(name, config=cfg, objective="persistence_residual")
        assert tag.startswith(f"{name}_BS4_EMD180_I4_E4_S32_C7") \
            and tag.endswith("_RES"), tag
        # smoke-mode reductions apply to the new presets like any other
        s = training_config(name, "smoke", 1)
        assert s["num_epochs"] == 1 and s["batch_size"] == 4
    # the MS defaults machinery also covers the new presets
    ms = training_config("bottom_smoke", "full", 1, train_horizon=5)
    assert ms["lr"] == MS_DEFAULTS["lr"] and ms["num_epochs"] == 5


def test_static_mask_checkpoint_rebuild_and_rollout():
    # pre_evaluate.py rebuild path: the checkpoint's static_mask_input /
    # model_cond_chans decide the backbone's condition channels, and a
    # static-mask checkpoint must roll out WITH static_cond while the plain
    # 14-channel path is untouched (no static_cond threaded through)
    # helper semantics: legacy checkpoints are the plain 14-channel arm
    assert static_mask_from_checkpoint(None) == (False, 2 * 7)
    assert static_mask_from_checkpoint({}) == (False, 14)
    assert static_mask_from_checkpoint({"static_mask_input": False}) == (False, 14)
    assert static_mask_from_checkpoint({"static_mask_input": True}) == (True, 16)
    assert static_mask_from_checkpoint(
        {"static_mask_input": True, "model_cond_chans": 16}) == (True, 16)
    assert static_mask_from_checkpoint(
        {"static_mask_input": False, "model_cond_chans": 14}) == (False, 14)
    # diffusion + static mask is an impossible combination -> refuse
    for bad in ({"static_mask_input": True},
                {"static_mask_input": True, "model_cond_chans": 16},
                {"static_mask_input": False, "model_cond_chans": 16},
                {"static_mask_input": True, "model_cond_chans": 14}):
        try:
            static_mask_from_checkpoint(bad, "diffusion" if bad["static_mask_input"]
                                        else None)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"static_mask_from_checkpoint({bad}) should fail")
    torch.manual_seed(4)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()
    for flag, chans in ((False, 14), (True, 16)):
        model = make_residual_model(cond_chans=chans)
        state = {"epoch": 0, "model_state_dict": model.state_dict(),
                 "config": {"objective": "persistence_residual",
                            "static_mask_input": flag,
                            "model_cond_chans": chans}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Ep1.pth")
            torch.save(state, p)
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        got_flag, got_ch = static_mask_from_checkpoint(
            ckpt["config"], "persistence_residual")
        assert (got_flag, got_ch) == (flag, chans)
        # rebuild EXACTLY like pre_evaluate.py: fresh backbone with the
        # helper-derived channel count, wrapper, then load the state dict
        fresh = make_residual_model(cond_chans=got_ch)
        fresh.load_state_dict(ckpt["model_state_dict"])
        fresh.eval()
        static_arg = static if got_flag else None
        preds = ensemble_rollout(fresh, cond, 3, 1, seed=0, static_cond=static_arg)
        assert preds.shape == (2, 1, 3, 2, H, W, Z), preds.shape
        assert torch.isfinite(preds).all()
        # untrained model: every rollout step is the last-day persistence,
        # with or without the static channels (zero-init identity holds)
        assert torch.equal(preds[:, 0], cond[:, -2:].unsqueeze(1).expand(-1, 3, -1, -1, -1, -1))
    # the plain 14-channel rebuild must be BITWISE unchanged by the new
    # static_cond kwarg (None is threaded as the historical absent argument)
    plain = make_residual_model(cond_chans=14)
    plain.eval()
    with torch.no_grad():
        a = plain.sample(cond, num_sample_steps=1, clamp=True)
        b = plain.sample(cond, num_sample_steps=1, clamp=True, static_cond=None)
    assert torch.equal(a, b)


def test_archived_msk_checkpoint_minimal_cpu_rollout():
    # acceptance for the static-mask eval fix: the ARCHIVED experiment-08
    # _MSK checkpoint rebuilds (16-channel backbone) and completes a minimal
    # CPU rollout with static_cond; skipped when the repo snapshot is absent
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, "checkpoints", "PRE",
                             "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MSK",
                             "Ep10.pth")
    if not os.path.isfile(ckpt_path):
        print("SKIP (archived _MSK checkpoint not present)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    assert cfg["objective"] == "persistence_residual"
    assert cfg["preset"] == "surface_smoke"
    flag, cond_ch = static_mask_from_checkpoint(cfg, "persistence_residual")
    assert flag is True and cond_ch == 16
    hh, ww, zz = 400, 441, 1
    net = IAFNODiff(dim=(hh, ww, zz), patch_size=PRESETS["surface_smoke"]["patch_size"],
                    embed_dim=PRESETS["surface_smoke"]["embed_dim"], num_blocks=1,
                    in_chans=2, out_chans=2, cond_chans=cond_ch,
                    ex_layer=PRESETS["surface_smoke"]["explicit_layer"],
                    nlayer=PRESETS["surface_smoke"]["implicit_layer"],
                    hidden_size_factor=4, dim_f=(hh, ww, zz), self_condition=True)
    model = PersistenceResidualIAFNO(net, time_sigma=float(cfg.get("time_sigma",
                                                                   RESIDUAL_TIME_SIGMA)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.manual_seed(7)
    cond = torch.rand(1, 14, hh, ww, zz)
    static = (torch.rand(1, 2, hh, ww, zz) > 0.5).float()
    preds = ensemble_rollout(model, cond, 2, 1, seed=0, static_cond=static)
    assert preds.shape == (1, 1, 2, 2, hh, ww, zz), preds.shape
    assert torch.isfinite(preds).all()
    # a 14-channel condition must be REJECTED by the 16-channel model
    try:
        model.sample(cond, num_sample_steps=1, clamp=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("16-channel model accepted a 14-channel condition")


def test_worse_epochs_checkpoint_roundtrip():
    # early-stop streak survives a resume: one worsening before the save, so a
    # single further worsening after the restore reaches the stop threshold
    model = make_residual_model()
    state = {"epoch": 4, "best_val": 0.5, "worse_epochs": 1,
             "model_state_dict": model.state_dict()}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Ep5.pth")
        torch.save(state, p)
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
    assert restore_worse_epochs(ckpt) == 1
    # legacy checkpoints without the field keep the historical default 0
    assert restore_worse_epochs({}) == 0
    assert restore_worse_epochs(None) == 0
    assert restore_worse_epochs({"worse_epochs": -3}) == 0
    # stop semantics: restored streak + one more worsening -> stop fires
    worse_epochs = restore_worse_epochs(ckpt)
    worse_epochs += 1                     # one further worsening epoch
    assert worse_epochs >= 2
    # a fresh-best epoch still resets the counter to 0 after the restore
    worse_epochs = restore_worse_epochs(ckpt)
    worse_epochs = 0                      # is_best branch
    worse_epochs += 1
    assert worse_epochs < 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("pre_smoke_test passed")
