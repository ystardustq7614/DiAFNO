#!/usr/bin/env python3
"""Plan A preprocessing: collocate raw staggered C-grid u/v onto the rho grid.

Raw fields (ROMS Arakawa-C):
    u: (T, s, 400, 440)  -- u[r,c] sits between rho(r,c) and rho(r,c+1)
    v: (T, s, 399, 441)  -- v[r,c] sits between rho(r,c) and rho(r+1,c)

Colocation (NaN-aware mean of the two adjacent faces; one-sided at boundaries):
    u_rho[r,c] = mean_valid(u[r,c-1], u[r,c])   (c=1..439)
    u_rho[r,0] = u[r,0];  u_rho[r,440] = u[r,439]
    v_rho[r,c] = mean_valid(v[r-1,c], v[r,c])   (r=1..398)
    v_rho[0,c] = v[0,c];  v_rho[399,c] = v[398,c]

No rotation is applied: u_rho/v_rho keep the raw grid-xi/eta component
semantics, only the sampling location moves to the rho points.

Bivariate validity masks are derived from mask_u/mask_v with the SAME stencil
(a rho point is valid iff at least one of its two adjacent face cells is
valid, one-sided at the boundary), so aligned NaN pattern == mask == 0 exactly:
    mask_u_rho.npy : (400, 441) validity of u_rho
    mask_v_rho.npy : (400, 441) validity of v_rho
    mask_uv.npy    : mask_u_rho & mask_v_rho & mask_rho (kept for compatibility)

Output (land kept as NaN, float32):
    <DST>/u_rho.npy, <DST>/v_rho.npy : (10591, 30, 400, 441)
    <DST>/mask_u_rho.npy, <DST>/mask_v_rho.npy, <DST>/mask_uv.npy
    <DST>/ocean_time.npy             : (10591,) datetime64[D] date view, verified from
                                      the authoritative ocean_time metadata of the
                                      raw NetCDF files (strictly increasing,
                                      exactly 24 h apart).
    <DST>/ocean_time_seconds.npy     : (10591,) datetime64[s] precise verified times
                                      (never downcast to days before verification).

Mask policy (the provided masks are authoritative):
    * NaN where mask == 1 (ocean cell without data) is DYNAMIC MISSING DATA:
      fail hard at the first (t, s, r, c), checked for every day and layer.
    * a value where mask == 0 (e.g. the 45 static land-boundary u-faces of
      this dataset) is DISCARDED (set to NaN before colocation) and counted;
      per-variable totals are reported at the end. After enforcement the
      aligned NaN pattern == mask == 0 exactly (asserted on the first chunk).
    * mask shapes/values, field dtypes and input shapes are asserted.
    * the ocean_time series must contain exactly T strictly increasing
      timestamps spaced by exactly 24 h (verify_daily_time).

The raw and aligned u/v extrema are recorded WITH their (day, layer, row, col)
location and value; extrema are not automatically treated as outliers.
"""
import os
import time
import numpy as np
import netCDF4

SRC_CANDIDATES = [
    "/data2/user/zyq/datasets/PRE/processed",
    "/data/PRE_ocean_data/processed",
]
RAW_DYN_CANDIDATES = [
    "/data2/user/zyq/datasets/PRE/raw/dyn",
    "/data/PRE_ocean_data/raw/dyn",
]
DST = "/data2/user/zyq/data_processed/PRE/aligned"
CHUNK = 50  # days per chunk

T, S, H, W = 10591, 30, 400, 441


def colocate(a, b):
    """NaN-aware mean of two same-shape float32 arrays; NaN where both are NaN."""
    na = ~np.isnan(a)
    nb = ~np.isnan(b)
    cnt = na.astype(np.float32) + nb.astype(np.float32)
    s = np.where(na, a, np.float32(0.0)) + np.where(nb, b, np.float32(0.0))
    out = np.full(a.shape, np.nan, np.float32)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


def u_rho_mask(mask_u):
    """(R, C-1) u-grid mask -> (R, C) rho mask under the u colocation stencil."""
    R, C = mask_u.shape
    out = np.empty((R, C + 1), np.bool_)
    out[:, 1:C] = mask_u[:, :-1] | mask_u[:, 1:]
    out[:, 0] = mask_u[:, 0]
    out[:, C] = mask_u[:, -1]
    return out


def v_rho_mask(mask_v):
    """(R-1, C) v-grid mask -> (R, C) rho mask under the v colocation stencil."""
    R, C = mask_v.shape
    out = np.empty((R + 1, C), np.bool_)
    out[1:R, :] = mask_v[:-1, :] | mask_v[1:, :]
    out[0, :] = mask_v[0, :]
    out[R, :] = mask_v[-1, :]
    return out


class ExtremumTracker:
    """Global min/max of a streaming array plus the location of each extremum.

    arr chunks have shape (t_len, S, R, C); the global (t, s, r, c) of the first
    occurrence is recorded. NaN cells (land) are ignored.
    """

    def __init__(self, name):
        self.name = name
        self.min_val = np.inf
        self.max_val = -np.inf
        self.min_loc = None  # (t, s, r, c)
        self.max_loc = None

    def update(self, arr, t0):
        flat = np.asarray(arr).ravel()
        if flat.size == 0:
            return
        mn = float(np.nanmin(flat))
        mx = float(np.nanmax(flat))
        if mn < self.min_val:
            i = int(np.nanargmin(flat))
            self.min_val = mn
            self.min_loc = self._loc(i, arr.shape, t0)
        if mx > self.max_val:
            i = int(np.nanargmax(flat))
            self.max_val = mx
            self.max_loc = self._loc(i, arr.shape, t0)

    @staticmethod
    def _loc(i, shape, t0):
        t_len, s, r, c = shape
        t, rem = divmod(i, s * r * c)
        s_, rem = divmod(rem, r * c)
        r_, c_ = divmod(rem, c)
        return (t0 + int(t), int(s_), int(r_), int(c_))

    def report(self):
        m = f"[extrema] {self.name}:"
        if self.min_loc is not None:
            m += (f" min={self.min_val:.5f} at (t={self.min_loc[0]}, s={self.min_loc[1]}, "
                  f"r={self.min_loc[2]}, c={self.min_loc[3]})")
            m += (f" max={self.max_val:.5f} at (t={self.max_loc[0]}, s={self.max_loc[1]}, "
                  f"r={self.max_loc[2]}, c={self.max_loc[3]})")
        else:
            m += " no finite values"
        return m


def enforce_land_mask(arr, mask, name, t0, discarded):
    """Enforce the (authoritative) land mask on a raw chunk, in place.

    Two mismatch directions are handled differently:
      * NaN where mask == 1 (ocean cell without data): dynamic missing data —
        fail hard with the first (t, s, r, c); never masked away silently.
      * value where mask == 0 (land cell carrying a value, e.g. static
        boundary/river u-faces): discarded (set to NaN) and counted in
        `discarded[name]`; the aligned output keeps NaN == (mask == 0).

    `arr` must be an in-memory chunk (t, s, R, C), NOT a mmap slice (it is
    modified in place). `discarded` is a dict accumulating per-variable counts.
    Returns arr.
    """
    nan = np.isnan(arr)
    ocean = mask != 0
    missing = nan & ocean[None, None]
    if missing.any():
        i = int(np.argmax(missing))
        t_len, s, r, c = arr.shape
        t, rem = divmod(i, s * r * c)
        s_, rem = divmod(rem, r * c)
        r_, c_ = divmod(rem, c)
        day = t0 + t
        raise RuntimeError(
            f"{name} dynamic missing data at (t={day}, s={s_}, r={r_}, c={c_}): "
            f"field is NaN but mask==1 (ocean); not masked away.")
    stray = ~nan & ~ocean[None, None]
    n_stray = int(stray.sum())
    if n_stray:
        arr[stray] = np.nan
        discarded[name] = discarded.get(name, 0) + n_stray
    return arr


def verify_daily_time(times):
    """Fail hard unless `times` is a 1-D datetime64 array of exactly daily steps.

    Adjacent timestamps must differ by EXACTLY 24 h (checked at datetime64[s]
    precision, so 23/25-hour gaps fail). On success returns `times` unchanged.
    The error reports the failing index, the two neighbouring timestamps and
    the actual interval.
    """
    times = np.asarray(times)
    if times.ndim != 1 or times.size == 0:
        raise RuntimeError(f"expected a non-empty 1-D datetime64 array, got shape {times.shape}")
    secs = times.astype("datetime64[s]").astype(np.int64)
    day = 24 * 3600
    gaps = np.diff(secs)
    bad = gaps != day
    if bad.any():
        j = int(np.argmax(bad))
        raise RuntimeError(
            f"ocean_time not daily at index {j}: {times[j]} -> {times[j + 1]} "
            f"(actual interval {int(gaps[j]) / 3600.0:g} h, expected 24 h)")
    return times


def extract_and_verify_time():
    """Read authoritative ocean_time from every raw NetCDF and cache the times.

    Verifies exactly T timestamps, strictly increasing, exactly 24 h apart.
    Raw times are kept at datetime64[s] precision (never downcast to days
    before verification); the date view is saved separately:
        ocean_time.npy         : (T,) datetime64[D]  date view (compat)
        ocean_time_seconds.npy : (T,) datetime64[s]  precise verified times
    Returns the precise (T,) datetime64[s] array.
    """
    files = sorted(f for f in os.listdir(RAW_DYN) if f.endswith(".nc"))
    if len(files) != T:
        raise RuntimeError(f"expected {T} raw NetCDF files in {RAW_DYN}, found {len(files)}")

    times = np.empty(T, dtype="datetime64[s]")
    units = None
    t0_wall = time.time()
    for i, fn in enumerate(files):
        fp = os.path.join(RAW_DYN, fn)
        with netCDF4.Dataset(fp) as ds:
            ot = ds.variables["ocean_time"]
            if units is None:
                units = getattr(ot, "units", None)
            vals = np.asarray(ot[:]).reshape(-1)
            if vals.size != 1:
                raise RuntimeError(f"{fp}: ocean_time has {vals.size} entries, expected 1")
            times[i] = np.datetime64(
                netCDF4.num2date(float(vals[0]), units, only_use_python_datetimes=True))
        if (i + 1) % 2000 == 0 or i + 1 == T:
            print(f"[time] {i + 1}/{T} files read ({time.time() - t0_wall:.0f}s)", flush=True)

    verify_daily_time(times)
    print(f"[time] verified {T} strictly increasing daily timestamps "
          f"{times[0]} .. {times[-1]} (units: {units})", flush=True)
    np.save(os.path.join(DST, "ocean_time.npy"), times.astype("datetime64[D]"))
    np.save(os.path.join(DST, "ocean_time_seconds.npy"), times)
    return times


def main():
    global SRC, RAW_DYN
    SRC = next((p for p in SRC_CANDIDATES if os.path.isdir(p)), None)
    if SRC is None:
        raise RuntimeError(f"processed data dir not found (tried {SRC_CANDIDATES})")
    RAW_DYN = next((p for p in RAW_DYN_CANDIDATES if os.path.isdir(p)), None)
    if RAW_DYN is None:
        raise RuntimeError(f"raw NetCDF dir not found (tried {RAW_DYN_CANDIDATES}); "
                           f"cannot verify ocean_time metadata")
    os.makedirs(DST, exist_ok=True)

    u = np.load(os.path.join(SRC, "dyn_var", "u.npy"), mmap_mode="r")
    v = np.load(os.path.join(SRC, "dyn_var", "v.npy"), mmap_mode="r")
    assert u.shape == (T, S, H, W - 1), f"u shape {u.shape}"
    assert v.shape == (T, S, H - 1, W), f"v shape {v.shape}"
    assert u.dtype == np.float32, f"u dtype {u.dtype}"
    assert v.dtype == np.float32, f"v dtype {v.dtype}"

    mask_rho = np.load(os.path.join(SRC, "stat_var", "mask_rho.npy"))
    mask_u = np.load(os.path.join(SRC, "stat_var", "mask_u.npy"))
    mask_v = np.load(os.path.join(SRC, "stat_var", "mask_v.npy"))
    assert mask_rho.shape == (H, W), f"mask_rho shape {mask_rho.shape}"
    assert mask_u.shape == (H, W - 1), f"mask_u shape {mask_u.shape}"
    assert mask_v.shape == (H - 1, W), f"mask_v shape {mask_v.shape}"
    for name, m in (("mask_rho", mask_rho), ("mask_u", mask_u), ("mask_v", mask_v)):
        assert set(np.unique(m)).issubset({0, 1}), f"{name} values {np.unique(m)}"
    # stored masks are float64 {0., 1.}; bitwise stencil ops below need booleans
    mask_rho = mask_rho.astype(bool)
    mask_u = mask_u.astype(bool)
    mask_v = mask_v.astype(bool)

    m_u_rho = u_rho_mask(mask_u)
    m_v_rho = v_rho_mask(mask_v)
    m_uv = m_u_rho & m_v_rho & (mask_rho == 1)
    np.save(os.path.join(DST, "mask_u_rho.npy"), m_u_rho.astype(np.uint8))
    np.save(os.path.join(DST, "mask_v_rho.npy"), m_v_rho.astype(np.uint8))
    np.save(os.path.join(DST, "mask_uv.npy"), m_uv.astype(np.uint8))
    print(f"[mask] mask_u_rho ocean pts: {m_u_rho.sum()}  "
          f"mask_v_rho: {m_v_rho.sum()}  mask_uv: {m_uv.sum()}", flush=True)

    # --- early probe on day 0 layer 0: fail fast on dynamic missing data and
    #     preview land-cell discards before the long pipeline starts (the probe
    #     slices are throwaway copies; their counts are NOT added to the final
    #     per-variable totals, which only accumulate over the main loop) ---
    probe = {}
    enforce_land_mask(np.array(u[0:1, 0:1]), mask_u, "u", 0, probe)
    enforce_land_mask(np.array(v[0:1, 0:1]), mask_v, "v", 0, probe)
    if probe:
        print(f"[mask] day0/layer0 probe discards (land cells carrying values): {probe}",
              flush=True)

    extract_and_verify_time()

    u_out = np.lib.format.open_memmap(
        os.path.join(DST, "u_rho.npy"), mode="w+", dtype=np.float32, shape=(T, S, H, W))
    v_out = np.lib.format.open_memmap(
        os.path.join(DST, "v_rho.npy"), mode="w+", dtype=np.float32, shape=(T, S, H, W))

    tr_u, tr_v = ExtremumTracker("u raw"), ExtremumTracker("v raw")
    tr_ur, tr_vr = ExtremumTracker("u_rho"), ExtremumTracker("v_rho")
    discarded = {}

    t0 = time.time()
    n_chunks = (T + CHUNK - 1) // CHUNK
    for ci, ts in enumerate(range(0, T, CHUNK)):
        te = min(ts + CHUNK, T)
        uc = np.array(u[ts:te])  # (t,s,400,440) writable copy (mmap slice is read-only)
        vc = np.array(v[ts:te])  # (t,s,399,441)
        tlen = te - ts

        # raw extrema describe the ORIGINAL chunk (before mask enforcement);
        # enforcement then fails hard on dynamic missing ocean data and sets
        # land-cell values to NaN so colocation sees the masked fields.
        tr_u.update(uc, ts)
        tr_v.update(vc, ts)
        enforce_land_mask(uc, mask_u, "u", ts, discarded)
        enforce_land_mask(vc, mask_v, "v", ts, discarded)

        ub = np.empty((tlen, S, H, W), np.float32)
        vb = np.empty((tlen, S, H, W), np.float32)

        ub[:, :, :, 1:W - 1] = colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
        ub[:, :, :, 0] = uc[:, :, :, 0]
        ub[:, :, :, W - 1] = uc[:, :, :, -1]

        vb[:, :, 1:H - 1, :] = colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
        vb[:, :, 0, :] = vc[:, :, 0, :]
        vb[:, :, H - 1, :] = vc[:, :, -1, :]

        if ci == 0:
            assert (np.isnan(ub) == (m_u_rho == 0)[None, None]).all(), \
                "u_rho NaN pattern does not match mask_u_rho"
            assert (np.isnan(vb) == (m_v_rho == 0)[None, None]).all(), \
                "v_rho NaN pattern does not match mask_v_rho"

        u_out[ts:te] = ub
        v_out[ts:te] = vb
        u_out.flush()
        v_out.flush()

        tr_ur.update(ub, ts)
        tr_vr.update(vb, ts)

        dt = time.time() - t0
        eta = dt / (ci + 1) * (n_chunks - ci - 1)
        print(f"[{ci + 1}/{n_chunks}] days {ts}:{te}  elapsed {dt / 60:.1f} min  "
              f"ETA {eta / 60:.1f} min", flush=True)
        del uc, vc, ub, vb

    print(tr_u.report(), flush=True)
    print(tr_v.report(), flush=True)
    print(tr_ur.report(), flush=True)
    print(tr_vr.report(), flush=True)

    if discarded:
        for name, n in sorted(discarded.items()):
            print(f"[mask] {name}: discarded {n} values at mask==0 (land) cells "
                  f"over all {T} days x {S} layers (mask authoritative)", flush=True)
    else:
        print("[mask] no values found at land cells", flush=True)

    print(f"DONE. total elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()