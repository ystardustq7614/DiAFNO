#!/usr/bin/env python3
"""PRE_ocean_data Dataset + normalization for the DiAFNO 7-day -> 15-day current forecast task.

Data source (produced by scripts/preprocess_align_uv.py, Plan A colocation):
    /data2/user/zyq/data_processed/PRE/aligned/u_rho.npy  (10591, 30, 400, 441) float32, land=NaN
    /data2/user/zyq/data_processed/PRE/aligned/v_rho.npy  same shape
    /data2/user/zyq/data_processed/PRE/aligned/mask_uv.npy (400, 441) uint8 effective ocean mask

Splits are contiguous in time (NO random_split; overlapping windows must not leak):
    train: days [0, 8401)     1994-01-01 .. 2016-12-31
    val:   days [8401, 9496)  2017-01-01 .. 2019-12-31
    test:  days [9496, 10591) 2020-01-01 .. 2022-12-30

Sample layout (window of context+horizon consecutive days inside one split):
    cond:   (2*context, H, W, Z)  channel-first, day-major interleaved:
                                  ch 2k   = u of day (start+k)
                                  ch 2k+1 = v of day (start+k)
    target: (horizon, 2, H, W, Z) target[:, 0] = u, target[:, 1] = v
    Z = 1 if depth_index is an int (e.g. 29 = surface), else 30 (all sigma layers).

Normalization (train-ocean-points only, cached to disk):
    per variable: clip to [p0.1, p99.9] of train ocean values, then min-max to [0, 1];
    land/NaN filled with 0 AFTER normalization (loss/metrics must use the mask).
    sigma = std of normalized ocean values over both variables (EDM sigma_data).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

ALIGNED_DIR = "/data2/user/zyq/data_processed/PRE/aligned"
NORM_DIR = "/data2/user/zyq/data_processed/PRE/norm"

T_TOTAL, S_TOTAL, H, W = 10591, 30, 400, 441

# contiguous time splits, half-open day-index ranges
SPLITS = {
    "train": (0, 8401),
    "val": (8401, 9496),
    "test": (9496, 10591),
}

CONTEXT = 7


# ----------------------------------------------------------------------------- masks

def load_mask():
    """Effective ocean mask (H, W) bool: mask_rho==1 AND aligned u/v both have data."""
    return np.load(os.path.join(ALIGNED_DIR, "mask_uv.npy")).astype(bool)


# ----------------------------------------------------------------------------- stats

def _iter_ocean_values(arr, lo, hi, depth_index, mask, chunk):
    """Yield (n_ocean_in_chunk,) 1-D arrays of ocean values over days [lo, hi)."""
    for ts in range(lo, hi, chunk):
        te = min(ts + chunk, hi)
        if depth_index is None:
            a = np.asarray(arr[ts:te])          # (t, s, H, W)
            a = a[:, :, mask]                    # (t, s, n_ocean)
        else:
            a = np.asarray(arr[ts:te, depth_index])  # (t, H, W)
            a = a[:, mask]                            # (t, n_ocean)
        yield a.reshape(-1)


def compute_or_load_stats(depth_index=None, clip_pct=0.1, chunk=25, verbose=True):
    """Per-variable clip range + global sigma, from the TRAIN split, ocean points only.

    Returns dict with keys: lo/hi (np.float32 array [2], order u,v), sigma (float).
    Cached at NORM_DIR/stats_d{depth}_{clip}.npz ; delete cache to recompute.
    """
    os.makedirs(NORM_DIR, exist_ok=True)
    tag = f"d{'all' if depth_index is None else depth_index}_clip{clip_pct}"
    cache = os.path.join(NORM_DIR, f"stats_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if verbose:
            print(f"[stats] loaded {cache}: lo={z['lo']}, hi={z['hi']}, sigma={z['sigma'].item():.5f}")
        return {"lo": z["lo"], "hi": z["hi"], "sigma": float(z["sigma"])}

    lo_d, hi_d = SPLITS["train"]
    u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
    v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")
    mask = load_mask()

    lo_out, hi_out, stds = [], [], []
    for name, arr in (("u", u), ("v", v)):
        # pass 1: exact min/max (streaming)
        mn, mx = np.inf, -np.inf
        for vals in _iter_ocean_values(arr, lo_d, hi_d, depth_index, mask, chunk):
            mn = min(mn, float(vals.min()))
            mx = max(mx, float(vals.max()))
        # pass 2: histogram -> approximate percentiles
        bins = 4096
        hist = np.zeros(bins, dtype=np.int64)
        edges = np.linspace(mn, mx, bins + 1)
        for vals in _iter_ocean_values(arr, lo_d, hi_d, depth_index, mask, chunk):
            hist += np.histogram(vals, bins=edges)[0]
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1]
        clip_lo = float(np.interp(clip_pct / 100.0, cdf, edges[:-1]))
        clip_hi = float(np.interp(1.0 - clip_pct / 100.0, cdf, edges[:-1]))
        # pass 3: std of clipped, normalized values
        s1, s2, n = 0.0, 0.0, 0
        for vals in _iter_ocean_values(arr, lo_d, hi_d, depth_index, mask, chunk):
            x = np.clip(vals, clip_lo, clip_hi)
            x = (x - clip_lo) / (clip_hi - clip_lo)
            s1 += float(x.sum())
            s2 += float((x * x).sum())
            n += x.size
        mean = s1 / n
        std = float(np.sqrt(max(s2 / n - mean * mean, 0.0)))
        lo_out.append(clip_lo)
        hi_out.append(clip_hi)
        stds.append(std)
        if verbose:
            print(f"[stats] {name}: min={mn:.4f} max={mx:.4f} -> clip [{clip_lo:.4f}, {clip_hi:.4f}], "
                  f"normalized ocean std={std:.5f}", flush=True)

    # global sigma: pooled std assuming equal variable weight (same n per variable)
    sigma = float(np.sqrt(0.5 * (stds[0] ** 2 + stds[1] ** 2)))
    np.savez(cache, lo=np.float32(lo_out), hi=np.float32(hi_out), sigma=np.float32(sigma))
    print(f"[stats] saved {cache}; sigma_data={sigma:.5f}", flush=True)
    return {"lo": np.float32(lo_out), "hi": np.float32(hi_out), "sigma": sigma}


# ----------------------------------------------------------------------------- dataset

class PREUVDataset(Dataset):
    """Sliding-window dataset over one contiguous time split.

    __getitem__ -> (cond, target, start_day):
        cond:   (2*context, H, W, Z) float32, normalized to [0,1], land=0
        target: (horizon, 2, H, W, Z) float32, normalized to [0,1], land=0
        start_day: int, absolute day index of the first context frame
    """

    def __init__(self, split, stats, context=CONTEXT, horizon=1,
                 depth_index=None, stride=1, max_windows=None):
        assert split in SPLITS
        self.context = context
        self.horizon = horizon
        self.depth_index = depth_index
        self.lo_stats = stats["lo"]  # [2] float32 (u, v)
        self.hi_stats = stats["hi"]

        self.u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
        self.v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")

        lo, hi = SPLITS[split]
        last_start = hi - (context + horizon)  # inclusive
        starts = np.arange(lo, last_start + 1, stride)
        if max_windows is not None:
            starts = starts[:max_windows]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def _load_var(self, arr, i, k):
        """Days [i, i+k) -> (k, H, W, Z) float32, clipped+normalized, NaN->0."""
        if self.depth_index is None:
            a = np.asarray(arr[i:i + k])              # (k, s, H, W)
            a = np.transpose(a, (0, 2, 3, 1))         # (k, H, W, Z=s)
        else:
            a = np.asarray(arr[i:i + k, self.depth_index])  # (k, H, W)
            a = a[..., None]                               # (k, H, W, 1)
        return a

    def _norm(self, a, j):
        lo, hi = float(self.lo_stats[j]), float(self.hi_stats[j])
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo)
        return np.nan_to_num(a, nan=0.0).astype(np.float32)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        L = self.context + self.horizon
        u = self._norm(self._load_var(self.u, i, L), 0)  # (L,H,W,Z)
        v = self._norm(self._load_var(self.v, i, L), 1)

        # day-major interleave: [u0, v0, u1, v1, ...]
        uv = np.stack([u, v], axis=1)                    # (L, 2, H, W, Z)
        cond = uv[:self.context].reshape(2 * self.context, *uv.shape[2:])
        target = uv[self.context:]                       # (horizon, 2, H, W, Z)

        return (torch.from_numpy(np.ascontiguousarray(cond)),
                torch.from_numpy(np.ascontiguousarray(target)),
                i)


def build_mask_tensor(device, depth_index=None):
    """(1, 1, H, W, Z) float mask for broadcast over (B, C, H, W, Z) tensors."""
    m = load_mask().astype(np.float32)
    z = S_TOTAL if depth_index is None else 1
    m = np.broadcast_to(m[None, None, :, :, None], (1, 1, H, W, z))
    return torch.from_numpy(np.ascontiguousarray(m)).to(device)
