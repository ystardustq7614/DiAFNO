"""Pure metric helpers shared by pre_trainer.py / pre_evaluate.py / pre_smoke_test.py.

No side effects at import time; no dependency on pre_dataset.py or the model.
Formal PRE metrics are computed on the NATIVE staggered u/v grids only.
"""
import numpy as np


def rho_to_native(rho_pred):
    """Map rho-grid predictions back to the native staggered u/v grids.

    Input rho_pred: (B, L, 2, H, W, Z) — channel 0 = u, channel 1 = v.
    Returns (u_nat, v_nat):
        u_nat: (B, L, H, W-1, Z) — mean of adjacent rho points along xi
        v_nat: (B, L, H-1, W, Z) — mean of adjacent rho points along eta
    (inverse of the Plan A colocation stencil, no rotation).
    """
    rho_pred = np.asarray(rho_pred)
    assert rho_pred.ndim == 6 and rho_pred.shape[2] == 2, rho_pred.shape
    up = rho_pred[:, :, 0]                       # (B, L, H, W, Z)
    vp = rho_pred[:, :, 1]
    u_nat = 0.5 * (up[:, :, :, :-1] + up[:, :, :, 1:])   # (B, L, H, W-1, Z)
    v_nat = 0.5 * (vp[:, :, :-1, :] + vp[:, :, 1:, :])   # (B, L, H-1, W, Z)
    return u_nat, v_nat


def masked_error_sums(pred, truth, mask):
    """Masked squared/absolute error sums over a batch of lead days.

    pred/truth must both be (B, L, H, W, Z) with identical shapes; mask is the
    corresponding 2-D native grid (H, W). Returns (se, ae), each (L, Z):
        se[l, z] = sum over batch, rows, cols of (pred - truth)**2 at valid cells
        ae[l, z] = sum over batch, rows, cols of |pred - truth| at valid cells
    Land cells (mask == 0) contribute exactly 0 to both sums. The strict shape
    assertions reject accidental re-use of (L, 1, Z)-style accumulators as
    pred/truth inputs.
    """
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    mask = np.asarray(mask)
    assert pred.ndim == 5 and truth.ndim == 5, (pred.shape, truth.shape)
    assert pred.shape == truth.shape, (pred.shape, truth.shape)
    assert mask.shape == (pred.shape[2], pred.shape[3]), mask.shape
    err = np.where(mask[None, None, :, :, None], pred - truth, np.float64(0.0))  # (B, L, H, W, Z)
    se = (err ** 2).sum(axis=(0, 2, 3))          # (L, Z)
    ae = np.abs(err).sum(axis=(0, 2, 3))         # (L, Z)
    return se, ae


def pooled_rmse(se, count):
    """sqrt(sum(se) / sum(count)) over everything passed in (0.0 if count == 0).

    se/count may be scalars or any array shapes; the aggregation is pooled, i.e.
    never an arithmetic mean of per-cell/per-layer RMSEs.
    """
    se = np.asarray(se, np.float64)
    count = np.asarray(count, np.float64)
    n = count.sum()
    if n <= 0:
        return 0.0
    return float(np.sqrt(se.sum() / n))


def oracle_native_error_sums(target_norm, y_lo, y_hi, truth_u, truth_v, mask_u, mask_v):
    """rho-oracle diagnostic: rho-grid truth -> native grids, then masked errors.

    target_norm: (B, L, 2, H, W, Z) NORMALIZED [0,1] rho-grid targets
        (channel 0 = u, 1 = v) — the dataset-provided real targets.
    y_lo/y_hi: per-variable clip range (len-2 sequences, [0]=u, [1]=v).
    truth_u: (B, L, H, W-1, Z), truth_v: (B, L, H-1, W, Z) — unclipped native
        physical truth (raw u.npy/v.npy).
    Returns (se, ae), each (L, 2, Z) [channel 0 = u, 1 = v], in the SAME
    accumulator layout as masked_error_sums results slot into.

    The prediction is the dataset's own rho target denormalized and mapped back
    with the identical rho_to_native stencil, so the result measures ONLY the
    irreversible error of the native -> rho -> native conversion (clipping,
    adjacent averaging, boundary copying).
    """
    t = np.asarray(target_norm, np.float32)
    assert t.ndim == 6 and t.shape[2] == 2, t.shape
    lo = np.asarray(y_lo, np.float32).reshape(1, 1, 2, 1, 1, 1)
    hi = np.asarray(y_hi, np.float32).reshape(1, 1, 2, 1, 1, 1)
    phys = t * (hi - lo) + lo                        # (B, L, 2, H, W, Z) physical
    u_nat, v_nat = rho_to_native(phys)
    se_u, ae_u = masked_error_sums(u_nat, truth_u, mask_u)
    se_v, ae_v = masked_error_sums(v_nat, truth_v, mask_v)
    L, Z = t.shape[1], t.shape[-1]
    se = np.zeros((L, 2, Z), np.float64)
    ae = np.zeros((L, 2, Z), np.float64)
    se[:, 0, :] = se_u
    se[:, 1, :] = se_v
    ae[:, 0, :] = ae_u
    ae[:, 1, :] = ae_v
    return se, ae


def masked_rel_l2(pred, tgt, mask):
    """Relative L2 over valid cells only, mean over batch (torch tensors).

    relL2 = sqrt(sum((pred - target)^2 * mask)) / sqrt(sum(target^2 * mask))
    per sample; mask is broadcastable to pred/tgt (1 = valid). Squared errors
    never cancel; the denominator is clamped to avoid 0/0 on empty masks.
    """
    diff2 = ((pred - tgt) ** 2 * mask).sum(dim=(1, 2, 3, 4))
    tgt2 = (tgt ** 2 * mask).sum(dim=(1, 2, 3, 4))
    return (diff2.sqrt() / tgt2.sqrt().clamp(min=1e-12)).mean().item()