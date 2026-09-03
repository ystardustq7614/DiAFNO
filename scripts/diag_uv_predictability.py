#!/usr/bin/env python3
"""All-30-layer zero-training predictability profile for the PRE u/v fields
(work package 1 of docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md).

Read-only audit of the collocated rho-grid u/v (u_rho.npy / v_rho.npy):
scales, daily increments, validation persistence skill and the compression
the current unified min-max normalization applies to each layer. NO model,
NO GPU: everything streams from the memmapped arrays in day chunks.

Statistics (train split [0, 8401) ONLY for anything scale-related):
    per (variable, sigma layer): exact mean/std/min/max, exact valid counts,
    quantiles p0.1/p1/p50/p99/p99.9 from a deterministic stride subsample
    (every SAMPLE_STRIDE-th valid value in time-major arrival order), and the
    exact fraction of values outside the unified min-max range;
    train daily increments x[t+1]-x[t]: exact mean/std/min/max + subsampled
    quantiles per (variable, layer).

Predictability (validation split [8401, 9496)):
    persistence RMSE/MAE per (variable, layer, lead day 1..15) in physical
    m/s on the rho grid with the bivariate rho masks — the SAME window set
    and definition as the formal protocol (window start s in
    [val_lo, val_hi-22], persistence for lead l = day s+6 = target t-l),
    but computed on the rho grid (NOT the native staggered grid of
    pre_evaluate.py): this is a zero-training difficulty audit, and the
    convention is recorded in the outputs. Breakdowns: coastal/offshore
    (COASTAL_BUFFER-cell rule, identical to scripts/diag_region_breakdown.py)
    and the sigma index bands bottom=0..9 / middle=10..19 / upper=20..29
    (k=0 seabed, k=29 surface).

Gates (doc §6 WP1: ALL must pass or model training must not start):
    1. daily time continuity (scripts.preprocess_align_uv.verify_daily_time);
    2. no NaN/Inf inside the masks on ANY audited train/val day/layer
       (dynamic missing data would poison every downstream statistic);
    3. every (variable, layer) has a positive valid count in train and val.

Outputs (refused if they already exist), in
    $OUT_ROOT/diag_uv_predictability_<YYYYMMDD>/ :
        uv_predictability.npz   all statistics + provenance metadata
        summary.csv             one row per (variable, layer)
        SUMMARY.md              gate table + human-readable findings

Run from repo root:
    python -u scripts/diag_uv_predictability.py
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from pre_config import OUT_ROOT
from pre_dataset import (ALIGNED_DIR, NORM_DIR, S_TOTAL, SPLITS, T_TOTAL,
                         compute_or_load_stats, load_masks, load_ocean_time,
                         mask_version)
from scripts.preprocess_align_uv import verify_daily_time

# ----------------------------- configuration -----------------------------

CHUNK_DAYS = 10          # days per streamed train chunk (~211 MB per var)
SAMPLE_STRIDE = 1000     # deterministic quantile subsample stride (per layer)
COASTAL_BUFFER = 5       # cells to land; identical to diag_region_breakdown.py
VAL_MAX_LEAD = 15        # persistence audit horizon
BANDS = (("bottom", 0, 9), ("middle", 10, 19), ("upper", 20, 29))
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)   # p0.1 / p1 / p50 / p99 / p99.9 (%)
VARS = ("u", "v")
H, W = 400, 441

OUT_DIR = os.path.join(OUT_ROOT, "diag_uv_predictability_"
                       + time.strftime("%Y%m%d"))
NPZ_PATH = os.path.join(OUT_DIR, "uv_predictability.npz")
CSV_PATH = os.path.join(OUT_DIR, "summary.csv")
MD_PATH = os.path.join(OUT_DIR, "SUMMARY.md")

# unified min-max range per variable (resolved in main(), used by the train pass)
GLOBAL_LO = {}
GLOBAL_HI = {}


# ----------------------------- small helpers -----------------------------

class Moments:
    """Exact online moments/extrema + deterministic stride subsample.

    The subsample keeps every SAMPLE_STRIDE-th valid value (global order of
    arrival) so quantiles are stable, reproducible and need no second pass;
    mean/std/min/max/counts stay EXACT over all values.
    """

    def __init__(self):
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.mn = np.inf
        self.mx = -np.inf
        self.samples = []
        self._seen = 0         # values seen since the last sampled one

    def update(self, vals):
        vals = np.asarray(vals, np.float64).ravel()
        if vals.size == 0:
            return
        self.n += int(vals.size)
        self.s1 += float(vals.sum())
        self.s2 += float((vals * vals).sum())
        self.mn = min(self.mn, float(vals.min()))
        self.mx = max(self.mx, float(vals.max()))
        # deterministic global stride sampling: the phase continues across calls
        idx = np.arange(self._seen, self._seen + vals.size)
        take = vals[(idx % SAMPLE_STRIDE) == 0]
        if take.size:
            self.samples.append(take.astype(np.float32))
        self._seen += int(vals.size)

    def finalize(self):
        mean = self.s1 / max(self.n, 1)
        var = max(self.s2 / max(self.n, 1) - mean * mean, 0.0)
        samp = (np.concatenate(self.samples) if self.samples
                else np.zeros(0, np.float32))
        self.samples = None
        return dict(n=self.n, mean=mean, std=float(np.sqrt(var)),
                    min=self.mn, max=self.mx, samples=samp)


def quantiles_from_samples(samples):
    """Sample quantiles (fractions) -> physical values; NaN where empty."""
    if samples.size == 0:
        return np.full(len(QUANTILES), np.nan)
    return np.quantile(samples.astype(np.float64), QUANTILES)


def region_masks(mask2d):
    """(H, W) rho mask -> coastal/offshore cell masks.

    Identical rule to scripts/diag_region_breakdown.py: coastal = valid
    cells within COASTAL_BUFFER cells of land (land dilated with the default
    cross structuring element, iterations=COASTAL_BUFFER).
    """
    land = ~np.asarray(mask2d, bool)
    near_land = ndimage.binary_dilation(land, iterations=COASTAL_BUFFER)
    valid = np.asarray(mask2d, bool)
    return valid & near_land, valid & ~near_land


def gate(name, ok, detail=""):
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


# ----------------------------- train-scale pass -----------------------------

def train_scale_pass(masks, u, v):
    """Stream the train split once: per-(var, layer) exact scale stats,
    increment stats, exact valid/NaN counts and out-of-range counts.

    Chunks carry a one-day overlap so the boundary increment x[a0]-x[a0-1]
    is counted exactly once. Returns finalized per-var layer lists plus the
    exact count arrays (n_valid / n_nan_mask / n_below / n_above).
    """
    lo_t, hi_t = SPLITS["train"]
    scale = {v_: [Moments() for _ in range(S_TOTAL)] for v_ in VARS}
    inc = {v_: [Moments() for _ in range(S_TOTAL)] for v_ in VARS}
    n_valid = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}
    n_nan_mask = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}
    n_below = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}
    n_above = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}

    n_days = hi_t - lo_t
    n_chunks = (n_days + CHUNK_DAYS - 1) // CHUNK_DAYS
    t0_ = time.perf_counter()
    done = 0
    for ci in range(n_chunks):
        a0 = lo_t + ci * CHUNK_DAYS
        a1 = min(a0 + CHUNK_DAYS, hi_t)
        r0 = max(a0 - 1, lo_t)                     # one-day read overlap
        for name, arr in (("u", u), ("v", v)):
            m = masks[name]
            a = np.asarray(arr[r0:a1])             # (a1-r0, S, H, W)
            cur = a[(a0 - r0):]                    # days [a0, a1)
            # increments for every target day in [a0, a1) (target day a0 at the
            # very first chunk has no predecessor and is simply absent):
            # a[0] is day a0-1 (or a0 == lo_t), so a[1:] - a[:-1] pairs each
            # audited day with exactly its previous day, once.
            dif = a[1:] - a[:-1] if a.shape[0] > 1 else a[:0]
            for s in range(S_TOTAL):
                day_vals = cur[:, s][:, m]         # (days, n_ocean)
                fin = np.isfinite(day_vals)
                n_valid[name][s] += int(fin.sum())
                n_nan_mask[name][s] += int(day_vals.size - fin.sum())
                fv = day_vals[fin]
                scale[name][s].update(fv)
                n_below[name][s] += int((fv < GLOBAL_LO[name]).sum())
                n_above[name][s] += int((fv > GLOBAL_HI[name]).sum())
                if dif.shape[0] > 0:
                    d = dif[:, s][:, m]            # (inc_days, n_ocean)
                    ok = np.isfinite(d)
                    inc[name][s].update(d[ok])
        done += a1 - a0
        print(f"[train] chunk {ci + 1}/{n_chunks} days {a0}..{a1 - 1} "
              f"({done}/{n_days}) elapsed_s={time.perf_counter() - t0_:.0f}",
              flush=True)
    return scale, inc, n_valid, n_nan_mask, n_below, n_above, n_days


# ----------------------------- val persistence pass -----------------------------

def val_persistence_pass(masks, u, v, coastal_flat):
    """Stream the val split day-by-day; accumulate persistence RMSE/MAE
    per (var, layer, lead) plus the coastal split.

    Protocol-identical window set: window start s in [val_lo, val_hi-(7+L)],
    persistence for lead l = day s+6, target day t = s+6+l. Enumerated
    target-day-centric: for target day t the valid leads are
        l >= t - 6 - LAST_START   (window start t-6-l must be <= LAST_START)
        l <= min(L, t - 6 - val_lo)  (window start must be >= val_lo)
    A rolling buffer of the last L+1 days provides the persistence sources.
    """
    lo_v, hi_v = SPLITS["val"]
    L = VAL_MAX_LEAD
    LAST_START = hi_v - (7 + L)                # last valid window start
    first_t = lo_v + 7                         # earliest target day (s=lo_v, l=1)
    last_t = hi_v - 1                          # latest target day (s=LAST_START, l=L)
    n_windows = LAST_START - lo_v + 1

    se = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    ae = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    n = {v_: np.zeros((S_TOTAL, L), np.int64) for v_ in VARS}
    se_co = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    ae_co = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    n_co = {v_: np.zeros((S_TOTAL, L), np.int64) for v_ in VARS}
    n_valid = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}
    n_nan_mask = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}

    buffers = {v_: [] for v_ in VARS}          # rolling history, oldest first
    t0_ = time.perf_counter()
    for t in range(lo_v, last_t + 1):
        for name, arr in (("u", u), ("v", v)):
            day = np.asarray(arr[t])           # (S, H, W)
            m = masks[name]
            in_mask = m[None, :, :]
            fin = np.isfinite(day) & in_mask
            n_valid[name] += fin.reshape(S_TOTAL, -1).sum(axis=1)
            n_nan_mask[name] += (in_mask & ~np.isfinite(day)) \
                .reshape(S_TOTAL, -1).sum(axis=1)
            buffers[name].append(day)
            if len(buffers[name]) > L + 1:     # keep day t and L sources
                buffers[name].pop(0)
            if t < first_t:
                continue
            max_l = min(L, t - 6 - lo_v)
            min_l = max(1, t - 6 - LAST_START)
            truth = day
            hist = buffers[name]
            co = coastal_flat[name]            # (n_ocean,) within m
            for l in range(min_l, max_l + 1):
                src = hist[-1 - l]             # day t-l
                e = (src - truth)[:, m]        # (S, n_ocean)
                ok = np.isfinite(e)
                ec = np.where(ok, e, 0.0)
                se[name][:, l - 1] += (ec ** 2).sum(axis=1)
                ae[name][:, l - 1] += np.abs(ec).sum(axis=1)
                n[name][:, l - 1] += ok.sum(axis=1)
                eco = ec * co
                se_co[name][:, l - 1] += (eco ** 2).sum(axis=1)
                ae_co[name][:, l - 1] += np.abs(eco).sum(axis=1)
                n_co[name][:, l - 1] += (ok & co).sum(axis=1)
        if (t - lo_v) % 50 == 0 or t == last_t:
            print(f"[val] target day {t} ({t - lo_v}/{last_t - lo_v}) "
                  f"elapsed_s={time.perf_counter() - t0_:.0f}", flush=True)

    rmse = {v_: np.sqrt(se[v_] / np.maximum(n[v_], 1)) for v_ in VARS}
    mae = {v_: ae[v_] / np.maximum(n[v_], 1) for v_ in VARS}
    rmse_co = {v_: np.sqrt(se_co[v_] / np.maximum(n_co[v_], 1)) for v_ in VARS}
    mae_co = {v_: ae_co[v_] / np.maximum(n_co[v_], 1) for v_ in VARS}
    return dict(rmse=rmse, mae=mae, n=n, rmse_coastal=rmse_co, mae_coastal=mae_co,
                n_coastal=n_co, n_valid=n_valid, n_nan_mask=n_nan_mask,
                n_windows=n_windows)


# ----------------------------- main -----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in (NPZ_PATH, CSV_PATH, MD_PATH):
        if os.path.exists(path):
            raise RuntimeError(f"{path} already exists; delete it or change OUT_DIR")

    t_start = time.perf_counter()
    print(f"[setup] aligned data: {ALIGNED_DIR}", flush=True)
    print(f"[setup] out dir: {OUT_DIR}", flush=True)

    # ---- gate 1: time continuity ----
    times = verify_daily_time(load_ocean_time())
    assert times.shape[0] == T_TOTAL
    gate("daily time continuity", True,
         f"{times[0]} .. {times[-1]} ({T_TOTAL} days)")

    # ---- masks ----
    masks = dict(zip(VARS, load_masks()))
    mv = mask_version()
    shapes_ok = all(m.shape == (H, W) for m in masks.values())
    gate("mask shapes", shapes_ok, f"u{masks['u'].shape} v{masks['v'].shape} "
         f"version={mv}")
    coastal_flat = {}
    for name in VARS:
        co, _ = region_masks(masks[name])
        coastal_flat[name] = co[masks[name]]   # flattened within the ocean mask

    # unified (all-layer) min-max normalization actually in use: the
    # authoritative full3d stats cache (train-only, cached in NORM_DIR)
    print("[setup] loading the unified (depth_index=None) min-max stats...",
          flush=True)
    stats_all = compute_or_load_stats(depth_index=None, verbose=False)
    for j, v_ in enumerate(VARS):
        GLOBAL_LO[v_] = float(stats_all["lo"][j])
        GLOBAL_HI[v_] = float(stats_all["hi"][j])
    print(f"[setup] unified min-max: lo={GLOBAL_LO} hi={GLOBAL_HI}", flush=True)

    u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
    v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")

    # ---- train pass (scales + increments + exact counts) ----
    scale, inc, n_valid, n_nan, n_below, n_above, days_tr = \
        train_scale_pass(masks, u, v)

    # ---- val pass (persistence difficulty) ----
    val = val_persistence_pass(masks, u, v, coastal_flat)

    # ---- finalize + gates ----
    fin_scale = {v_: [scale[v_][s].finalize() for s in range(S_TOTAL)]
                 for v_ in VARS}
    fin_inc = {v_: [inc[v_][s].finalize() for s in range(S_TOTAL)]
               for v_ in VARS}

    ok_finite = all(int(n_nan[v_][s]) == 0 and int(val["n_nan_mask"][v_][s]) == 0
                    for v_ in VARS for s in range(S_TOTAL))
    gate("finite values inside masks (train+val, every layer)", ok_finite,
         "0 dynamic missing cells" if ok_finite else "NaN/Inf inside masks")
    ok_count = all(int(n_valid[v_][s]) > 0 and int(val["n_valid"][v_][s]) > 0
                   for v_ in VARS for s in range(S_TOTAL))
    gate("per-layer valid counts positive (train+val)", ok_count,
         ", ".join(f"{v_} min/layer="
                   f"{min(int(n_valid[v_].min()), int(val['n_valid'][v_].min()))}"
                   for v_ in VARS))
    gates_ok = shapes_ok and ok_finite and ok_count
    print(f"[gate] OVERALL: {'PASS' if gates_ok else 'FAIL'}", flush=True)

    # ---- NPZ payload ----
    def layer_stack(getter):
        return np.array([[getter(fin_scale[v_][s]) for s in range(S_TOTAL)]
                         for v_ in VARS], np.float64)

    payload = {}
    payload["train_mean"] = layer_stack(lambda d: d["mean"])
    payload["train_std"] = layer_stack(lambda d: d["std"])
    payload["train_min"] = layer_stack(lambda d: d["min"])
    payload["train_max"] = layer_stack(lambda d: d["max"])
    payload["train_sample_n"] = layer_stack(lambda d: d["samples"].size)
    qs_all = {v_: [quantiles_from_samples(fin_scale[v_][s]["samples"])
                   for s in range(S_TOTAL)] for v_ in VARS}
    qi_all = {v_: [quantiles_from_samples(fin_inc[v_][s]["samples"])
                   for s in range(S_TOTAL)] for v_ in VARS}
    for qi, q in enumerate(QUANTILES):
        payload[f"train_p{q * 100:g}"] = np.array(
            [[qs_all[v_][s][qi] for s in range(S_TOTAL)] for v_ in VARS])
        payload[f"inc_p{q * 100:g}"] = np.array(
            [[qi_all[v_][s][qi] for s in range(S_TOTAL)] for v_ in VARS])
    payload["inc_mean"] = np.array(
        [[fin_inc[v_][s]["mean"] for s in range(S_TOTAL)] for v_ in VARS])
    payload["inc_std"] = np.array(
        [[fin_inc[v_][s]["std"] for s in range(S_TOTAL)] for v_ in VARS])
    payload["train_valid_count"] = np.array([n_valid[v_] for v_ in VARS], np.int64)
    payload["train_days"] = np.int64(days_tr)
    payload["nan_in_mask_count"] = np.array([n_nan[v_] for v_ in VARS], np.int64)
    payload["clip_below_count"] = np.array([n_below[v_] for v_ in VARS], np.int64)
    payload["clip_above_count"] = np.array([n_above[v_] for v_ in VARS], np.int64)
    payload["norm_lo"] = np.array([GLOBAL_LO[v_] for v_ in VARS])
    payload["norm_hi"] = np.array([GLOBAL_HI[v_] for v_ in VARS])
    rng = payload["norm_hi"] - payload["norm_lo"]
    payload["norm_clip_frac"] = np.array(
        [[(int(n_below[v_][s]) + int(n_above[v_][s])) / max(int(n_valid[v_][s]), 1)
          for s in range(S_TOTAL)] for v_ in VARS])
    payload["norm_std"] = payload["train_std"] / rng[:, None]
    payload["norm_p1_p99_width"] = (payload["train_p1"] - payload["train_p99"]) \
        / rng[:, None]
    payload["val_rmse"] = np.array([val["rmse"][v_] for v_ in VARS])
    payload["val_mae"] = np.array([val["mae"][v_] for v_ in VARS])
    payload["val_n"] = np.array([val["n"][v_] for v_ in VARS], np.int64)
    payload["val_rmse_coastal"] = np.array([val["rmse_coastal"][v_] for v_ in VARS])
    payload["val_mae_coastal"] = np.array([val["mae_coastal"][v_] for v_ in VARS])
    payload["val_n_coastal"] = np.array([val["n_coastal"][v_] for v_ in VARS], np.int64)
    payload["val_valid_count"] = np.array([val["n_valid"][v_] for v_ in VARS], np.int64)
    payload["val_windows"] = np.int64(val["n_windows"])
    leads_csv = (1, 5, 10, 15)
    for bname, b0, b1 in BANDS:
        sel = slice(b0, b1 + 1)
        se_sum = np.array([np.maximum(val["n"][v_][sel], 1)
                           * val["rmse"][v_][sel] ** 2 for v_ in VARS])
        n_sum = np.array([np.maximum(val["n"][v_][sel], 1) for v_ in VARS])
        payload[f"val_rmse_band_{bname}"] = np.sqrt(
            se_sum.sum(axis=1) / n_sum.sum(axis=1))

    meta = dict(
        aligned_dir=np.str_(ALIGNED_DIR), norm_dir=np.str_(NORM_DIR),
        mask_version=np.str_(mv), splits=np.int64(
            [SPLITS["train"], SPLITS["val"], SPLITS["test"]]),
        chunk_days=np.int64(CHUNK_DAYS), sample_stride=np.int64(SAMPLE_STRIDE),
        coastal_buffer=np.int64(COASTAL_BUFFER), s_total=np.int64(S_TOTAL),
        quantiles=np.array(QUANTILES),
        bands=np.str_(";".join(b[0] for b in BANDS)),
        gate_pass=np.bool_(gates_ok), stats_date=np.str_(time.strftime("%Y-%m-%d")),
        elapsed_s=np.float64(time.perf_counter() - t_start))
    try:
        git_id = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        meta["git_commit"] = np.str_(git_id.stdout.strip() or "unknown")
    except Exception:
        meta["git_commit"] = np.str_("unknown")
    payload.update(meta)
    np.savez(NPZ_PATH, **payload)
    print(f"[out] saved {NPZ_PATH}", flush=True)

    # ---- CSV summary ----
    with open(CSV_PATH, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["var", "layer", "band", "train_valid_frac", "mean_ms",
                     "std_ms", "min_ms", "max_ms", "p0.1_ms", "p1_ms", "p50_ms",
                     "p99_ms", "p99.9_ms", "inc_mean_ms", "inc_std_ms",
                     "norm_clip_frac", "norm_std", "norm_p1_p99_width"]
                    + [f"val_pers_rmse_d{l}_ms" for l in leads_csv]
                    + [f"val_pers_rmse_coastal_d{l}_ms" for l in leads_csv])
        for j, v_ in enumerate(VARS):
            for s in range(S_TOTAL):
                band = next(b[0] for b in BANDS if b[1] <= s <= b[2])
                d = fin_scale[v_][s]
                wr.writerow(
                    [v_, s, band,
                     f"{d['n'] / (days_tr * int(masks[v_].sum())):.6g}",
                     f"{d['mean']:.6g}", f"{d['std']:.6g}",
                     f"{d['min']:.6g}", f"{d['max']:.6g}"]
                    + [f"{x:.6g}" for x in qs_all[v_][s]]
                    + [f"{fin_inc[v_][s]['mean']:.6g}",
                       f"{fin_inc[v_][s]['std']:.6g}",
                       f"{payload['norm_clip_frac'][j, s]:.6g}",
                       f"{payload['norm_std'][j, s]:.6g}",
                       f"{payload['norm_p1_p99_width'][j, s]:.6g}"]
                    + [f"{val['rmse'][v_][s, l - 1]:.6g}" for l in leads_csv]
                    + [f"{val['rmse_coastal'][v_][s, l - 1]:.6g}"
                       for l in leads_csv])
    print(f"[out] saved {CSV_PATH}", flush=True)

    # ---- Markdown summary ----
    lines = ["# 全层 u/v 可预测性画像（工作包 1）", "",
             f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；数据：`{ALIGNED_DIR}`",
             f"- mask version：`{mv}`；split：train {SPLITS['train']} / "
             f"val {SPLITS['val']} / test {SPLITS['test']}",
             f"- 统一 min-max（depth_index=None）：lo={GLOBAL_LO} hi={GLOBAL_HI}",
             f"- 分位数来自 stride={SAMPLE_STRIDE} 确定性子采样；"
             f"chunk={CHUNK_DAYS} 天；coastal=陆地 {COASTAL_BUFFER} 格内；"
             "persistence 指标为 rho 网格物理单位（零训练难度审计，非正式 native 协议）",
             f"- **门禁：{'PASS' if gates_ok else 'FAIL'}**"
             "（连续性 / mask 形状 / finite / 逐层 valid count）", "",
             "## validation persistence RMSE（m/s）按 band", "",
             "| var | band | d1 | d5 | d10 | d15 |", "|---|---|---|---|---|---|"]
    for j, v_ in enumerate(VARS):
        for bname, _, _ in BANDS:
            lines.append(
                f"| {v_} | {bname} | "
                + " | ".join(f"{payload[f'val_rmse_band_{bname}'][j, l - 1]:.4f}"
                             for l in leads_csv) + " |")
    lines += ["", "## 压缩程度（统一 min-max 下，norm_std 最小/最大各 3 层）", "",
              "| var | layer | norm_std | norm p1–p99 宽度 | clip_frac |",
              "|---|---|---|---|---|"]
    for j, v_ in enumerate(VARS):
        order = np.argsort(payload["norm_std"][j])
        for s in list(order[:3]) + list(order[-3:]):
            lines.append(
                f"| {v_} | {s} | {payload['norm_std'][j, s]:.4f} | "
                f"{payload['norm_p1_p99_width'][j, s]:.4f} | "
                f"{payload['norm_clip_frac'][j, s]:.2e} |")
    lines += ["", "## validation persistence RMSE（m/s）d15 最难/最易各 3 层", "",
              "| var | layer | d15 RMSE | d1 RMSE |", "|---|---|---|---|"]
    for j, v_ in enumerate(VARS):
        order = np.argsort(payload["val_rmse"][j][:, 14])
        for s in list(order[:3]) + list(order[-3:]):
            lines.append(f"| {v_} | {s} | "
                         f"{payload['val_rmse'][j, s, 14]:.4f} | "
                         f"{payload['val_rmse'][j, s, 0]:.4f} |")
    lines += ["", f"完整数值见 `summary.csv` / `uv_predictability.npz`；"
              f"耗时 {time.perf_counter() - t_start:.0f}s"]
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[out] saved {MD_PATH}", flush=True)
    print(f"PROGRESS phase=profile status={'completed' if gates_ok else 'failed'} "
          f"gate={'pass' if gates_ok else 'fail'} "
          f"elapsed_s={time.perf_counter() - t_start:.0f}")
    if not gates_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
