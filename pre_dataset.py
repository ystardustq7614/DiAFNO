#!/usr/bin/env python3
"""PRE_ocean_data Dataset + normalization for the DiAFNO 7-day -> 15-day current forecast task.

Data source (produced by scripts/preprocess_align_uv.py, Plan A colocation):
    <ALIGNED_DIR>/u_rho.npy   (10591, 30, 400, 441) float32, land=NaN
    <ALIGNED_DIR>/v_rho.npy   same shape
    <ALIGNED_DIR>/mask_u_rho.npy (400, 441) uint8 validity of u_rho
    <ALIGNED_DIR>/mask_v_rho.npy (400, 441) uint8 validity of v_rho
    <ALIGNED_DIR>/mask_uv.npy     (400, 441) uint8 intersection (compat)
    <ALIGNED_DIR>/ocean_time.npy         (10591,) datetime64[D] date view (compat)
    <ALIGNED_DIR>/ocean_time_seconds.npy (10591,) datetime64[s] precise verified times

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
    per variable: min-max to [0, 1] over train ocean points of THAT variable
    (u uses mask_u_rho, v uses mask_v_rho). Optional percentile clipping
    (clip_pct, e.g. 0.1) is DISABLED by default (clip_pct=None) and must be
    explicitly configured; the cache name and content record the clipping
    policy, the split boundaries and the mask hash (missing or changed
    'splits' also invalidate the cache), and hi <= lo raises. land/NaN is
    filled with 0 AFTER normalization (loss/metrics must use the mask).
    sigma = true pooled std of the CLIPPED, min-max NORMALIZED ocean values
    over both variables combined (values are clipped to [lo, hi] before
    normalization and pooling, exactly like the dataset normalization; the
    u/v concatenation means the u/v mean difference contributes via the
    between-group term) — used as EDM sigma_data.

Evaluation uses the UNCLIPPED raw native truth via NativeUVReader
(raw u.npy / v.npy on the staggered grids); normalized targets are never
denormalized to stand in for raw truth.
"""
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

ALIGNED_DIR = "/data2/user/zyq/data_processed/PRE/aligned"
NORM_DIR = "/data2/user/zyq/data_processed/PRE/norm"
NATIVE_DIR_CANDIDATES = [
    "/data2/user/zyq/datasets/PRE/processed/dyn_var",
    "/data/PRE_ocean_data/processed/dyn_var",
]

T_TOTAL, S_TOTAL, H, W = 10591, 30, 400, 441

# contiguous time splits, half-open day-index ranges
SPLITS = {
    "train": (0, 8401),
    "val": (8401, 9496),
    "test": (9496, 10591),
}

CONTEXT = 7


def native_dir():
    for p in NATIVE_DIR_CANDIDATES:
        if os.path.isdir(p):
            return p
    raise RuntimeError(f"raw native u/v dir not found (tried {NATIVE_DIR_CANDIDATES}); "
                       f"evaluation needs the unclipped original fields")


# ----------------------------------------------------------------------------- masks

def load_masks():
    """(mask_u_rho, mask_v_rho) as (H, W) bool arrays (bivariate validity)."""
    mu = np.load(os.path.join(ALIGNED_DIR, "mask_u_rho.npy")).astype(bool)
    mv = np.load(os.path.join(ALIGNED_DIR, "mask_v_rho.npy")).astype(bool)
    assert mu.shape == (H, W) and mv.shape == (H, W)
    return mu, mv


def mask_version():
    """Verifiable identifier of the bivariate mask files (SHA-256)."""
    h = hashlib.sha256()
    for name in ("mask_u_rho.npy", "mask_v_rho.npy"):
        with open(os.path.join(ALIGNED_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def native_masks():
    """(mask_u, mask_v) bool arrays on the NATIVE staggered grids.

    mask_u: (H, W-1), mask_v: (H-1, W) — the original land masks of u.npy/v.npy.
    """
    stat = os.path.join(os.path.dirname(native_dir()), "stat_var")
    mu = np.load(os.path.join(stat, "mask_u.npy")).astype(bool)
    mv = np.load(os.path.join(stat, "mask_v.npy")).astype(bool)
    assert mu.shape == (H, W - 1), mu.shape
    assert mv.shape == (H - 1, W), mv.shape
    return mu, mv


# ----------------------------------------------------------------------------- time

def load_ocean_time():
    """(T_TOTAL,) datetime64[D] dates verified by the preprocessing step."""
    ts = np.load(os.path.join(ALIGNED_DIR, "ocean_time.npy"))
    assert ts.shape == (T_TOTAL,), f"ocean_time shape {ts.shape}"
    return ts


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


def _clip_range(vals_iter_factory, clip_pct):
    """Exact (min, max) over the stream, or percentile-clipped if clip_pct set.

    vals_iter_factory must be a zero-arg callable returning a FRESH iterator
    (the stream is consumed once for min/max and again for the histogram).
    """
    mn, mx = np.inf, -np.inf
    for vals in vals_iter_factory():
        mn = min(mn, float(vals.min()))
        mx = max(mx, float(vals.max()))
    if clip_pct is None:
        return mn, mx
    # histogram -> approximate percentiles
    bins = 4096
    hist = np.zeros(bins, dtype=np.int64)
    edges = np.linspace(mn, mx, bins + 1)
    for vals in vals_iter_factory():
        hist += np.histogram(vals, bins=edges)[0]
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]
    lo = float(np.interp(clip_pct / 100.0, cdf, edges[:-1]))
    hi = float(np.interp(1.0 - clip_pct / 100.0, cdf, edges[:-1]))
    return lo, hi


def compute_or_load_stats(depth_index=None, clip_pct=None, chunk=25, verbose=True):
    """Per-variable clip range + pooled sigma, from the TRAIN split, ocean points only.

    Returns dict with keys: lo/hi (np.float32 array [2], order u,v), sigma (float).
    Cached at NORM_DIR/stats_d{all|idx}_clip{none|pct}.npz; the cache records
    clip_pct, depth preset, split boundaries and the mask version, and is
    recomputed automatically if any of them changes. Delete cache to force.
    """
    os.makedirs(NORM_DIR, exist_ok=True)
    dname = "all" if depth_index is None else f"d{depth_index}"
    cname = "none" if clip_pct is None else f"p{clip_pct}"
    cache = os.path.join(NORM_DIR, f"stats_{dname}_clip{cname}.npz")

    mv = mask_version()
    if os.path.exists(cache):
        z = np.load(cache)
        cur_splits = np.asarray([SPLITS["train"], SPLITS["val"], SPLITS["test"]])
        cached_splits = np.asarray(z["splits"]) if "splits" in z.files else None
        split_ok = (cached_splits is not None
                    and cached_splits.shape == cur_splits.shape
                    and np.array_equal(cached_splits, cur_splits))
        ok = (split_ok
              and z["mask_version"].item() == mv
              and z["clip_pct"].item() == (-1.0 if clip_pct is None else float(clip_pct))
              and int(z["depth_index"]) == (depth_index if depth_index is not None else -1))
        if ok:
            if verbose:
                print(f"[stats] loaded {cache}: lo={z['lo']}, hi={z['hi']}, "
                      f"sigma={z['sigma'].item():.5f}")
            return {"lo": z["lo"], "hi": z["hi"], "sigma": float(z["sigma"])}
        print(f"[stats] cache {cache} stale (splits/mask/clip/depth changed) -> recompute")

    lo_d, hi_d = SPLITS["train"]
    u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
    v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")
    mu, mv_ = load_masks()

    lo_out, hi_out = [], []
    for name, arr, m in (("u", u, mu), ("v", v, mv_)):
        lo, hi = _clip_range(lambda: _iter_ocean_values(arr, lo_d, hi_d, depth_index, m, chunk),
                             clip_pct)
        if hi <= lo:
            raise RuntimeError(f"[stats] {name}: clip range [{lo}, {hi}] is empty or "
                               f"degenerate (hi <= lo); check data or clipping policy")
        lo_out.append(lo)
        hi_out.append(hi)
        if verbose:
            print(f"[stats] {name}: clip [{lo:.5f}, {hi:.5f}] "
                  f"(clip_pct={clip_pct})", flush=True)

    # pooled sigma over BOTH variables combined (u/v concatenated): the pooled
    # mean enters the variance, so the u/v mean difference contributes.
    # Values are CLIPPED to the per-variable range and min-max normalized BEFORE
    # pooling, exactly like the dataset normalization.
    s1 = s2 = 0.0
    n = 0
    for arr, m, lo, hi in ((u, mu, lo_out[0], hi_out[0]), (v, mv_, lo_out[1], hi_out[1])):
        for vals in _iter_ocean_values(arr, lo_d, hi_d, depth_index, m, chunk):
            x = np.clip(vals, lo, hi)
            x = (x - lo) / (hi - lo)
            s1 += float(x.sum())
            s2 += float((x * x).sum())
            n += x.size
    if n == 0:
        raise RuntimeError("[stats] no valid ocean points in train split")
    mean = s1 / n
    sigma = float(np.sqrt(max(s2 / n - mean * mean, 0.0)))
    if verbose:
        print(f"[stats] pooled normalized sigma over u+v: {sigma:.5f} (n={n})", flush=True)

    np.savez(cache,
             lo=np.float32(lo_out), hi=np.float32(hi_out), sigma=np.float32(sigma),
             depth_index=np.int64(depth_index if depth_index is not None else -1),
             clip_pct=np.float64(-1.0 if clip_pct is None else float(clip_pct)),
             splits=np.int64([SPLITS["train"], SPLITS["val"], SPLITS["test"]]),
             mask_version=np.str_(mv))
    print(f"[stats] saved {cache}; sigma_data={sigma:.5f}", flush=True)
    return {"lo": np.float32(lo_out), "hi": np.float32(hi_out), "sigma": sigma}


# ----------------------------------------------------------------------------- dataset

class PREUVDataset(Dataset):
    """Sliding-window dataset over one contiguous time split (normalized rho grid).

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

        load_ocean_time()  # fail fast if the verified time file is missing/invalid
        self.u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
        self.v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")

        lo, hi = SPLITS[split]
        last_start = hi - (context + horizon)  # inclusive
        assert last_start >= lo, f"split {split} too short for context+horizon"
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


class NativeUVReader:
    """Unclipped raw native u/v truth on the staggered grids (evaluation only).

    u: (T_TOTAL, S_TOTAL, H, W-1), v: (T_TOTAL, S_TOTAL, H-1, W) — the original
    processed fields, land = NaN, never normalized or clipped. Reading the raw
    physical truth here is the ONLY sanctioned way to compute formal metrics.

    get(day, days=1) -> (u_sel, v_sel) with a UNIFIED (days, H, W, Z) layout
    (sigma axis moved to the last position, matching the model grid):
        depth_index=None: u_sel (days, H, W-1, Z=S_TOTAL), v_sel (days, H-1, W, Z=S_TOTAL)
        depth_index=int:  u_sel (days, H, W-1, 1),        v_sel (days, H-1, W, 1)
    """

    def __init__(self, depth_index=None, u_path=None, v_path=None, check_shape=True):
        d = native_dir() if (u_path is None or v_path is None) else None
        self.u = np.load(u_path or os.path.join(d, "u.npy"), mmap_mode="r")
        self.v = np.load(v_path or os.path.join(d, "v.npy"), mmap_mode="r")
        if check_shape:
            assert self.u.shape == (T_TOTAL, S_TOTAL, H, W - 1), self.u.shape
            assert self.v.shape == (T_TOTAL, S_TOTAL, H - 1, W), self.v.shape
        self.depth_index = depth_index

    def get(self, day, days=1):
        sl = (slice(day, day + days), self.depth_index if self.depth_index is not None else slice(None))
        u = np.asarray(self.u[sl])   # (days, S, H, W-1) or (days, H, W-1)
        v = np.asarray(self.v[sl])   # (days, S, H-1, W) or (days, H-1, W)
        if self.depth_index is not None:
            u = u[..., None]                       # (days, H, W-1, 1)
            v = v[..., None]                       # (days, H-1, W, 1)
        else:
            u = np.moveaxis(u, 1, -1)              # (days, H, W-1, S)
            v = np.moveaxis(v, 1, -1)              # (days, H-1, W, S)
        return u, v


def build_mask_tensor(device, depth_index=None):
    """(1, 2, H, W, Z) float bivariate mask for broadcast over (B, 2, H, W, Z).

    channel 0 = mask_u_rho, channel 1 = mask_v_rho.
    """
    mu, mv = load_masks()
    z = S_TOTAL if depth_index is None else 1
    m = np.stack([mu, mv])[:, None, :, :, None]           # (2, 1, H, W, 1)
    m = np.broadcast_to(m, (2, 1, H, W, z)).transpose(1, 0, 2, 3, 4)  # (1, 2, H, W, Z)
    # broadcast_to returns a READ-ONLY view; force a writable C-contiguous copy
    # so torch.from_numpy() does not warn and in-place ops never fail.
    m = np.array(m, copy=True, order="C")
    return torch.from_numpy(m).to(device)