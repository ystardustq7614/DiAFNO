#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect the PRE_ocean_data (GBA) dataset: file inventory, array shapes/dtypes,
missing-value rates, per-variable statistics and land-mask analysis.

All checks are read-only. Large .npy files are opened via mmap and only
sampled, so this script is safe to run repeatedly on the 4.1 TB dataset.

Usage:
    python scripts/inspect_pre_dataset.py [--full]
    --full: also compute min/max/mean/std over a larger time sample
            (10% of time steps per variable, ~22 GB read per big file).
"""

import os
import sys
import numpy as np
import argparse
import json
from datetime import datetime

DATA_ROOT = "/data/PRE_ocean_data"
PROCESSED = os.path.join(DATA_ROOT, "processed")
RAW_DYN = os.path.join(DATA_ROOT, "raw", "dyn")

DYNAMIC_VARS = [
    "ubar_eastward", "vbar_northward", "zeta", "ubar", "vbar",
    "u_eastward", "v_northward", "u", "v", "temp", "salt", "rho",
]
STATIC_VARS = [
    "angle", "f", "h", "lat_psi", "lat_rho", "lon_psi", "lon_rho",
    "mask_rho", "mask_u", "mask_v", "pm", "pn", "x_rho", "x_u", "x_v",
    "y_rho", "y_u", "y_v", "Cs_r", "Cs_w", "s_rho", "s_w", "hc",
    "Tcline", "theta_s", "theta_b", "meta",
]


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def fmt_shape(a):
    return f"{tuple(a.shape)} {a.dtype}"


def probe_stats(arr, t_sample=200, subsample=4):
    """NaN rate + min/max/mean/std over a time-sample and a spatial subsample."""
    t = min(t_sample, arr.shape[0])
    idx_t = np.linspace(0, arr.shape[0] - 1, t, dtype=int)
    sl = (idx_t,) + (slice(None, None, subsample),) * (arr.ndim - 1)
    a = np.asarray(arr[sl])
    nan_rate = float(np.isnan(a).mean())
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return nan_rate, (None, None, None, None)
    return nan_rate, (
        float(finite.min()), float(finite.max()),
        float(finite.mean()), float(finite.std()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="use 10%% of time steps for stats (default: 200)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"PRE_ocean_data inspection @ {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"root: {DATA_ROOT}  size: {human(sum(os.path.getsize(os.path.join(DATA_ROOT,f)) for f in os.listdir(DATA_ROOT) if os.path.isfile(os.path.join(DATA_ROOT,f))))}")
    print("=" * 78)

    # ---------- 1. raw dynamic files ----------
    print("\n[1] raw/dyn/ : daily COAWST/ROMS averages")
    if os.path.isdir(RAW_DYN):
        files = sorted(f for f in os.listdir(RAW_DYN) if f.endswith(".nc"))
        total = sum(os.path.getsize(os.path.join(RAW_DYN, f)) for f in files)
        print(f"    files: {len(files)}  total: {human(total)}  per-file: {human(total // max(len(files),1))}")
        first_last = [files[0], files[-1]]
        print(f"    first: {first_last[0]}   last: {first_last[1]}")
        ids = sorted(int(f.split("_")[-1].split(".")[0]) for f in files)
        gaps = [i for i in range(ids[0], ids[-1] + 1) if i not in set(ids)]
        print(f"    index range: {ids[0]}..{ids[-1]}  missing files: {len(gaps)} {gaps[:10] if gaps else ''}")
    else:
        print(f"    MISSING {RAW_DYN}")

    # ---------- 2. processed dynamic variables ----------
    print("\n[2] processed/dyn_var/ : merged .npy files")
    dyn_dir = os.path.join(PROCESSED, "dyn_var")
    n_sample_t = 200 if not args.full else max(100, int(10591 * 0.1))
    for var in DYNAMIC_VARS:
        fp = os.path.join(dyn_dir, f"{var}.npy")
        if not os.path.exists(fp):
            print(f"    {var:<16} MISSING")
            continue
        size = os.path.getsize(fp)
        arr = np.load(fp, mmap_mode="r")
        nan_rate, (mn, mx, mu, sd) = probe_stats(arr, t_sample=n_sample_t)
        print(f"    {var:<16} {fmt_shape(arr)}  {human(size):>8}  NaN={nan_rate*100:6.2f}%"
              + (f"  min={mn:.4g} max={mx:.4g} mean={mu:.4g} std={sd:.4g}" if mn is not None else "  (all NaN)"))

    # ---------- 3. processed static variables ----------
    print("\n[3] processed/stat_var/ : static fields")
    stat_dir = os.path.join(PROCESSED, "stat_var")
    for var in STATIC_VARS:
        fp = os.path.join(stat_dir, f"{var}.npy")
        if not os.path.exists(fp):
            continue
        try:
            arr = np.load(fp, mmap_mode="r")
            print(f"    {var:<16} {fmt_shape(arr)}")
        except ValueError as e:
            print(f"    {var:<16} ERR: {e}")

    # ---------- 4. land mask analysis ----------
    print("\n[4] land mask & bathymetry (rho grid)")
    mask_fp = os.path.join(stat_dir, "mask_rho.npy")
    h_fp = os.path.join(stat_dir, "h.npy")
    if os.path.exists(mask_fp) and os.path.exists(h_fp):
        mask = np.load(mask_fp)
        h = np.load(h_fp)
        wet = mask == 1
        print(f"    mask_rho {tuple(mask.shape)} wet={wet.sum()} ({100*wet.sum()/mask.size:.1f}%) land={mask.size-wet.sum()}")
        hw = h[wet]
        print(f"    h: min={np.nanmin(hw):.1f} m  max={np.nanmax(hw):.1f} m  mean={np.nanmean(hw):.1f} m")
        rows_wet = wet.any(axis=1)
        cols_wet = wet.any(axis=0)
        print(f"    wet rows: {rows_wet.sum()}/{mask.shape[0]}  first={rows_wet.argmax() if rows_wet.any() else None} last={mask.shape[0]-1-rows_wet[::-1].argmax() if rows_wet.any() else None}")
        print(f"    wet cols: {cols_wet.sum()}/{mask.shape[1]}  first={cols_wet.argmax() if cols_wet.any() else None} last={mask.shape[1]-1-cols_wet[::-1].argmax() if cols_wet.any() else None}")
        if os.path.exists(os.path.join(stat_dir, "lon_rho.npy")):
            lon = np.load(os.path.join(stat_dir, "lon_rho.npy"))
            lat = np.load(os.path.join(stat_dir, "lat_rho.npy"))
            print(f"    lon range: {np.nanmin(lon):.3f}..{np.nanmax(lon):.3f}  lat range: {np.nanmin(lat):.3f}..{np.nanmax(lat):.3f}")
            if wet.any():
                print(f"    wet-area lon: {lon[wet].min():.3f}..{lon[wet].max():.3f}  lat: {lat[wet].min():.3f}..{lat[wet].max():.3f}")

    # ---------- 5. sigma vertical coordinate ----------
    print("\n[5] vertical coordinate (sigma layers)")
    for v in ["s_rho", "s_w", "Cs_r", "Cs_w", "hc", "Tcline", "theta_s", "theta_b"]:
        fp = os.path.join(stat_dir, f"{v}.npy")
        if os.path.exists(fp):
            try:
                arr = np.load(fp, mmap_mode="r")
                print(f"    {v:<8} {fmt_shape(arr)}  {np.array(arr).ravel()[:6]} ...")
            except ValueError as e:
                print(f"    {v:<8} ERR: {e}")

    # ---------- 6. docs / metadata sanity ----------
    print("\n[6] docs & metadata")
    for fp in [os.path.join(DATA_ROOT, "metadata.json" if os.path.exists(os.path.join(DATA_ROOT,"metadata.json")) else "raw/metadata.json")]:
        if os.path.exists(fp):
            with open(fp) as f:
                md = json.load(f)
            print(f"    metadata.json: {md.get('dataset',{}).get('temporal_coverage',{})}")
    docs_dic = os.path.join(DATA_ROOT, "docs", "data_dic.md")
    print(f"    docs/data_dic.md exists: {os.path.exists(docs_dic)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
