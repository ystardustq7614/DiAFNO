#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep audit of PRE_ocean_data:
  1. Full variable inventory of raw NetCDF (dyn + static grid)
  2. Mask semantics verification (0=land / 1=ocean) via NaN correlation + plots
  3. Time evolution trends & value distributions
  4. Grid geometry: cell size (dx/dy in m and degrees), region extent, dimensions

Outputs plots to ./plots/ and prints a report. Read-only; big files use mmap.
"""

import os
import json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

DATA_ROOT = "/data/PRE_ocean_data"
RAW_DYN = os.path.join(DATA_ROOT, "raw", "dyn")
STAT = os.path.join(DATA_ROOT, "processed", "stat_var")
DYN = os.path.join(DATA_ROOT, "processed", "dyn_var")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(OUT, exist_ok=True)

DYN_NC_SAMPLE = os.path.join(RAW_DYN, "coawst_avg_00001.nc")
STAT_NC = os.path.join(DATA_ROOT, "raw", "PRE-90921-V2.nc")

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

CN_FONT = FontProperties(family=["Droid Sans Fallback", "AR PL KaitiM GB",
                                 "AR PL SungtiL GB", "AR PL Mingti2L Big5"])


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


# ------------------------------------------------------------------ 1. inventory
print("=" * 80)
print("[1] RAW NETCDF VARIABLE INVENTORY")
print("=" * 80)


def inventory(fp, label, field_ndim):
    ds = xr.open_dataset(fp)
    print(f"\n-- {label}: {os.path.basename(fp)}")
    print(f"   dims: {dict(ds.sizes)}  n_vars_total: {len(ds.data_vars)}")
    fields, scalars = [], []
    for name, da in ds.data_vars.items():
        if da.ndim >= field_ndim:
            fields.append((name, tuple(da.shape), str(da.encoding.get('dtype', da.dtype))))
        else:
            scalars.append(name)
    print(f"   FIELD variables ({len(fields)}):")
    for n, s, d in fields:
        print(f"      {n:<20} {str(s):<28} {d}")
    print(f"   scalar/config variables ({len(scalars)}):")
    print(f"      {', '.join(scalars)}")
    ds.close()
    return fields


fields_dyn = inventory(DYN_NC_SAMPLE, "DYNAMIC (daily average)", 3)
fields_stat = inventory(STAT_NC, "STATIC GRID", 2)

# ------------------------------------------------------------------ 2. mask semantics
print("\n" + "=" * 80)
print("[2] MASK SEMANTICS VERIFICATION")
print("=" * 80)
mask = np.load(os.path.join(STAT, "mask_rho.npy"))
h = np.load(os.path.join(STAT, "h.npy"))
print(f"mask_rho unique values: {np.unique(mask)}  -> 1=ocean, 0=land per docs")
print(f"h unique values at land (mask==0): {np.unique(h[mask == 0])[:8]}  (ocean: {np.unique(h[mask == 1])[:6]})")

t0 = np.load(os.path.join(DYN, "temp.npy"), mmap_mode="r")[0, -1]  # day0 surface (idx 29)
nan_fraction_land = np.isnan(t0[mask == 0]).mean()
nan_fraction_ocean = np.isnan(t0[mask == 1]).mean()
print(f"temp[0,29](surface): NaN at land cells = {nan_fraction_land*100:.2f}%  |  NaN at ocean cells = {nan_fraction_ocean*100:.2f}%")
agree = (np.isnan(t0) == (mask == 0)).mean()
print(f"agreement (field-NaN <-> mask==0): {agree*100:.2f}%")

# ------------------------------------------------------------------ plots: field sanity
lon = np.load(os.path.join(STAT, "lon_rho.npy"))
lat = np.load(os.path.join(STAT, "lat_rho.npy"))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
im = ax[0, 0].imshow(mask, origin="lower", cmap="gray_r", aspect="auto")
ax[0, 0].set_title("mask_rho (black=1 ocean, white=0 land)")
ax[0, 0].set_xlabel("经向网格索引 (xi)", fontproperties=CN_FONT); ax[0, 0].set_ylabel("纬向网格索引 (eta)", fontproperties=CN_FONT)
cb = plt.colorbar(im, ax=ax[0, 0], ticks=[0, 1])
cb.ax.set_yticklabels(["0 陆地 (land)", "1 海洋 (ocean)"], fontproperties=CN_FONT)
hm = np.ma.masked_where(mask == 0, h)
im = ax[0, 1].imshow(hm, origin="lower", cmap="terrain", aspect="auto")
ax[0, 1].set_title("bathymetry h [m] (land masked)")
ax[0, 1].set_xlabel("经向网格索引 (xi)", fontproperties=CN_FONT); ax[0, 1].set_ylabel("纬向网格索引 (eta)", fontproperties=CN_FONT)
cb = plt.colorbar(im, ax=ax[0, 1])
cb.set_label("水深 [m]", fontproperties=CN_FONT)
im = ax[1, 0].imshow(t0, origin="lower", cmap="RdBu_r", aspect="auto")
ax[1, 0].set_title("temp day0 surface [C] (NaN=white, land)")
ax[1, 0].set_xlabel("经向网格索引 (xi)", fontproperties=CN_FONT); ax[1, 0].set_ylabel("纬向网格索引 (eta)", fontproperties=CN_FONT)
cb = plt.colorbar(im, ax=ax[1, 0])
cb.set_label("温度 [°C]", fontproperties=CN_FONT)
im = ax[1, 1].imshow(np.ma.masked_where(mask == 0, t0), origin="lower", cmap="RdBu_r", aspect="auto")
ax[1, 1].set_title("temp day0 surface (land masked)")
ax[1, 1].set_xlabel("经向网格索引 (xi)", fontproperties=CN_FONT); ax[1, 1].set_ylabel("纬向网格索引 (eta)", fontproperties=CN_FONT)
cb = plt.colorbar(im, ax=ax[1, 1])
cb.set_label("温度 [°C]", fontproperties=CN_FONT)
fig.tight_layout()
fp = os.path.join(OUT, "01_field_mask_sanity.png")
fig.savefig(fp)
print(f"\nplot saved: {fp}")

# ------------------------------------------------------------------ 3. trends & distributions
print("\n" + "=" * 80)
print("[3] TIME TRENDS & DISTRIBUTIONS")
print("=" * 80)
wet = mask == 1
interior = wet.copy()
interior[:5] = False
interior[-5:] = False
interior[:, :5] = False
interior[:, -5:] = False
i_deep = np.unravel_index(np.nanargmax(np.where(interior, h, np.nan)), h.shape)
print(f"deepest interior wet cell: eta={i_deep[0]}, xi={i_deep[1]}, h={h[i_deep]:.1f} m, lon={lon[i_deep]:.2f}, lat={lat[i_deep]:.2f}")

zeta = np.load(os.path.join(DYN, "zeta.npy"), mmap_mode="r")
zs = zeta[:, i_deep[0], i_deep[1]]  # daily full-range
days = np.arange(zeta.shape[0])

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
ax[0].plot(days, zs, lw=0.5)
ax[0].set_title(f"zeta daily trend @ deep point ({lon[i_deep]:.2f}E,{lat[i_deep]:.2f}N)")
ax[0].set_xlabel("自1994-01-01起的天数", fontproperties=CN_FONT); ax[0].set_ylabel("海面高度 ζ [m]", fontproperties=CN_FONT)
# monthly means to see seasonal
ym = zeta.shape[0] // 365
mon = zs[:ym * 365].reshape(ym, 365).mean(axis=1)
ax[1].plot(np.arange(ym) + 1994, mon, marker=".", lw=1)
ax[1].set_title("annual-mean zeta")
ax[1].set_xlabel("年份", fontproperties=CN_FONT); ax[1].set_ylabel("海面高度 ζ [m]", fontproperties=CN_FONT)
fig.tight_layout()
fp = os.path.join(OUT, "02_zeta_trend.png")
fig.savefig(fp)
print(f"plot saved: {fp}")

# temp trend: surface (level -1) & bottom (level 0), every 7 days
temp = np.load(os.path.join(DYN, "temp.npy"), mmap_mode="r")
step = 7
tt = temp[::step, -1, i_deep[0], i_deep[1]]  # s_rho=-0.017 -> surface
tb = temp[::step, 0, i_deep[0], i_deep[1]]   # s_rho=-0.983 -> bottom
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(days[::step], tt, lw=0.8, label="surface (level 29)")
ax.plot(days[::step], tb, lw=0.8, label="bottom (level 0)")
ax.set_title(f"temp trend @ deep point; seasonal range {np.nanmin(tt):.1f}..{np.nanmax(tt):.1f} C (surf)")
ax.set_xlabel("自1994-01-01起的天数", fontproperties=CN_FONT); ax.set_ylabel("温度 [°C]", fontproperties=CN_FONT); ax.legend()
fig.tight_layout()
fp = os.path.join(OUT, "03_temp_trend.png")
fig.savefig(fp)
print(f"plot saved: {fp}")

# distributions over wet cells (sample ~400 days, surface + bottom)
rng = np.random.default_rng(0)
day_idx = np.sort(rng.choice(zeta.shape[0], 400, replace=False))
surf = np.concatenate([temp[d, -1][wet] for d in day_idx[::4]])  # surface = highest index
bot = np.concatenate([temp[d, 0][wet] for d in day_idx[::4]])    # bottom = index 0
salt = np.load(os.path.join(DYN, "salt.npy"), mmap_mode="r")
sal = np.concatenate([salt[d, 0][wet] for d in day_idx[::8]])
u = np.load(os.path.join(DYN, "u_eastward.npy"), mmap_mode="r")
ue = np.concatenate([u[d, 0][wet] for d in day_idx[::8]])
# zeta distribution from all sampled days
zz = np.concatenate([zeta[d][wet] for d in day_idx[::8]])

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
bins = 100
for ax_, data, ttl, xlb in zip(axes.ravel(),
                               [surf, bot, sal, ue, zz, ue[np.isfinite(ue)]],
                               ["temp surface [C]", "temp bottom [C]", "salt surface [PSU]",
                                "u_eastward surface [m/s]", "zeta [m]", "u_eastward (zoom)"],
                               ["表面温度 [°C]", "底层温度 [°C]", "表面盐度 [PSU]",
                                "东向流速 [m/s]", "海面高度 ζ [m]", "东向流速（放大）"]):
    ax_.hist(data[np.isfinite(data)], bins=bins, density=True)
    ax_.set_title(ttl)
    ax_.set_xlabel(xlb, fontproperties=CN_FONT); ax_.set_ylabel("概率密度（对数刻度）", fontproperties=CN_FONT)
    ax_.set_yscale("log")
fig.tight_layout()
fp = os.path.join(OUT, "04_distributions.png")
fig.savefig(fp)
print(f"plot saved: {fp}")

# ------------------------------------------------------------------ 4. geometry
print("\n" + "=" * 80)
print("[4] GRID GEOMETRY")
print("=" * 80)
pm = np.load(os.path.join(STAT, "pm.npy"))
pn = np.load(os.path.join(STAT, "pn.npy"))
dx = 1.0 / pm[wet]
dy = 1.0 / pn[wet]
print(f"horizontal dims: eta(rho)={mask.shape[0]} x xi(rho)={mask.shape[1]}   ({mask.shape[0]}x{mask.shape[1]})")
print(f"pm=1/dx: dx median={np.median(dx):.1f} m  (min {np.min(dx):.1f}, max {np.max(dx):.1f})")
print(f"pn=1/dy: dy median={np.median(dy):.1f} m  (min {np.min(dy):.1f}, max {np.max(dy):.1f})")

# lon/lat spacing in degrees (interior rho cells)
dlon = np.abs(np.diff(lon, axis=1))[wet[:, 1:]]
dlat = np.abs(np.diff(lat, axis=0))[wet[1:, :]]
print(f"lon spacing: median={np.nanmedian(dlon):.5f} deg  lat spacing: median={np.nanmedian(dlat):.5f} deg")
mlat = np.median(lat[wet])
print(f"-> approx cell size = {np.nanmedian(dlon)*111.32*np.cos(np.radians(mlat)):.2f} km (lon) x "
      f"{np.nanmedian(dlat)*111.32:.2f} km (lat)")

# region extent
lonmin, lonmax = np.nanmin(lon[wet]), np.nanmax(lon[wet])
latmin, latmax = np.nanmin(lat[wet]), np.nanmax(lat[wet])
span_lon_deg = lonmax - lonmin
span_lat_deg = latmax - latmin
km_lon = span_lon_deg * 111.32 * np.cos(np.radians((latmin + latmax) / 2))
km_lat = span_lat_deg * 111.32
print(f"\nregion (wet cells): lon {lonmin:.3f}..{lonmax:.3f} E ({span_lon_deg:.3f} deg, ~{km_lon:.0f} km)   "
      f"lat {latmin:.3f}..{latmax:.3f} N ({span_lat_deg:.3f} deg, ~{km_lat:.0f} km)")
print(f"cells-per-degree: ~{mask.shape[1]/span_lon_deg:.0f} (lon), {mask.shape[0]/span_lat_deg:.0f} (lat)")

# vertical
s_rho = np.load(os.path.join(STAT, "s_rho.npy"))
print(f"\nvertical: {len(s_rho)} sigma levels  s_rho {s_rho.min():.3f}(bottom, idx0)..{s_rho.max():.3f}(surface, idx{len(s_rho)-1})")
print(f"depth of sigma levels at deep point (h={h[i_deep]:.1f}m): "
      f"{[f'{hc*h[i_deep]:.1f}' for hc in s_rho[[0,5,14,24,29]]]} m for levels [0,5,14,24,29]  (level0=bottom)")

# temporal
print(f"\ntime: 10591 daily records, 1994-01-01..2022-12-30 (~{10591/365.25:.1f} years)")
print("\nDone. All plots in:", os.path.abspath(OUT))
