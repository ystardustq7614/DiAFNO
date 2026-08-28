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
"""
import os
import tempfile
from math import exp, sqrt

import numpy as np
import torch

from scripts import preprocess_align_uv as pre_pp
from pre_dataset import NativeUVReader, _clip_range, compute_or_load_stats
from pre_metrics import (rho_to_native, masked_error_sums, pooled_rmse, masked_rel_l2,
                         oracle_native_error_sums)
from pre_rollout import expand_ensemble, ensemble_rollout, ensemble_mean
from pre_config import (SIGMA_DATA_SCALE, sigma_data_from_stats, sigma_data_from_checkpoint,
                        resume_sigma_decision)
from utilities3 import load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("pre_smoke_test passed")