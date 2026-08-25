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

Output (land kept as NaN, float32):
    <DST>/u_rho.npy, <DST>/v_rho.npy : (10591, 30, 400, 441)
    <DST>/mask_uv.npy               : (400, 441) effective mask
                                      (mask_rho==1 AND both aligned u/v have data)

Also verifies raw-NaN vs mask_u/mask_v consistency and prints ocean-point stats
on a day subsample (to track the raw-u ~7 m/s outlier).
"""
import os
import time
import numpy as np

SRC = "/data2/user/zyq/datasets/PRE/processed"
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


def main():
    os.makedirs(DST, exist_ok=True)
    u = np.load(os.path.join(SRC, "dyn_var", "u.npy"), mmap_mode="r")
    v = np.load(os.path.join(SRC, "dyn_var", "v.npy"), mmap_mode="r")
    assert u.shape == (T, S, H, W - 1), u.shape
    assert v.shape == (T, S, H - 1, W), v.shape

    # --- one-time consistency check: raw NaN pattern vs mask_u/mask_v (day 0, layer 0)
    mask_u = np.load(os.path.join(SRC, "stat_var", "mask_u.npy"))
    mask_v = np.load(os.path.join(SRC, "stat_var", "mask_v.npy"))
    u0_nan = np.isnan(np.asarray(u[0, 0]))
    v0_nan = np.isnan(np.asarray(v[0, 0]))
    print(f"raw u NaN == (mask_u==0): {np.array_equal(u0_nan, mask_u == 0)}")
    print(f"raw v NaN == (mask_v==0): {np.array_equal(v0_nan, mask_v == 0)}", flush=True)

    u_out = np.lib.format.open_memmap(
        os.path.join(DST, "u_rho.npy"), mode="w+", dtype=np.float32, shape=(T, S, H, W))
    v_out = np.lib.format.open_memmap(
        os.path.join(DST, "v_rho.npy"), mode="w+", dtype=np.float32, shape=(T, S, H, W))

    t0 = time.time()
    n_chunks = (T + CHUNK - 1) // CHUNK
    for ci, ts in enumerate(range(0, T, CHUNK)):
        te = min(ts + CHUNK, T)
        uc = np.asarray(u[ts:te])  # (t,s,400,440)
        vc = np.asarray(v[ts:te])  # (t,s,399,441)
        tlen = te - ts

        ub = np.empty((tlen, S, H, W), np.float32)
        vb = np.empty((tlen, S, H, W), np.float32)

        ub[:, :, :, 1:W - 1] = colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
        ub[:, :, :, 0] = uc[:, :, :, 0]
        ub[:, :, :, W - 1] = uc[:, :, :, -1]

        vb[:, :, 1:H - 1, :] = colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
        vb[:, :, 0, :] = vc[:, :, 0, :]
        vb[:, :, H - 1, :] = vc[:, :, -1, :]

        u_out[ts:te] = ub
        v_out[ts:te] = vb
        u_out.flush()
        v_out.flush()

        dt = time.time() - t0
        eta = dt / (ci + 1) * (n_chunks - ci - 1)
        print(f"[{ci + 1}/{n_chunks}] days {ts}:{te}  elapsed {dt / 60:.1f} min  ETA {eta / 60:.1f} min",
              flush=True)
        del uc, vc, ub, vb

    # --- effective mask: ocean & both variables have data (checked on 3 spread days)
    mask_rho = np.load(os.path.join(SRC, "stat_var", "mask_rho.npy")) == 1
    eff = mask_rho.copy()
    for t in (0, T // 2, T - 1):
        eff &= ~np.isnan(np.asarray(u_out[t, 0]))
        eff &= ~np.isnan(np.asarray(v_out[t, 0]))
        eff &= ~np.isnan(np.asarray(u_out[t, S - 1]))
        eff &= ~np.isnan(np.asarray(v_out[t, S - 1]))
    np.save(os.path.join(DST, "mask_uv.npy"), eff.astype(np.uint8))
    n_lost = int((mask_rho & ~eff).sum())
    print(f"mask_rho ocean pts: {mask_rho.sum()}  effective pts: {eff.sum()}  "
          f"ocean-but-no-data pts: {n_lost}", flush=True)

    # --- sanity stats on a day subsample (ocean points only)
    rng_days = range(0, T, 500)
    for name, arr in (("u_rho", u_out), ("v_rho", v_out)):
        vals = []
        for t in rng_days:
            a = np.asarray(arr[t])  # (s,H,W)
            vals.append(a[:, eff])
        vals = np.concatenate(vals, axis=1).ravel()
        print(f"{name}: n={vals.size}  min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"p0.1={np.percentile(vals, 0.1):.4f}  p99.9={np.percentile(vals, 99.9):.4f}",
              flush=True)

    print(f"DONE. total elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
