#!/usr/bin/env python3
"""模块职责：PRE u/v 全 30 个 sigma 层的零训练可预测性画像
（docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md 工作包 1）。

只读审计共定位 rho 网格 u/v（ALIGNED_DIR 下 u_rho.npy / v_rho.npy，
形状 (T=10591, S=30, H=400, W=441) float32，陆地保持 NaN）：量纲尺度、
逐日增量、验证段持续性基线技巧，以及当前统一 min-max 归一化对每层的
压缩程度。无模型、无 GPU：所有统计从 memmap 数组按天分块流式读出。

统计口径（凡尺度相关只用 train split [0, 8401)）：
    每 (变量, sigma 层)：exact mean/std/min/max 与精确有效计数；分位数
    p0.1/p1/p50/p99/p99.9 来自确定性定步长子采样（按时间主序到达顺序，
    每 SAMPLE_STRIDE 个有效值取 1 个，步进相位跨分块延续）；以及落在
    统一 min-max 区间之外的精确比例。train 段逐日增量 x[t+1]-x[t]：
    每 (变量, 层) 的 exact mean/std/min/max 与子采样分位数。

persistence 难度（val split [8401, 9496)）：
    每 (变量, 层, lead 1..15) 的持续性 RMSE/MAE，在 rho 网格物理单位
    （m/s）上用双变量 rho 掩膜计算 —— 窗口集合与正式协议一致（窗口起点
    s ∈ [val_lo, val_hi-22]，lead l 的持续性源 = 第 s+6 天 = 目标日
    t-l），但网格不是 pre_evaluate.py 的原生 staggered 网格：这是零训练
    难度审计，口径差异已随输出记录在案。分解口径：coastal/offshore
    （COASTAL_BUFFER 格规则，与 scripts/diag_region_breakdown.py 一致），
    以及 sigma 分带 bottom=0..9 / middle=10..19 / upper=20..29
    （k=0 海底，k=29 表层）。

门禁（§6 WP1：全部通过才允许开始模型训练）：
    1. 日时间连续性（scripts/preprocess_align_uv.py 的 verify_daily_time）；
    2. 任何被审计的 train/val 日/层在掩膜内无 NaN/Inf（动态缺测会污染
       所有下游统计）；
    3. 每个 (变量, 层) 在 train 与 val 均有正的有效计数。

不负责：不训练模型、不产生预测；不触碰任何生产输出（对齐数组、stats
缓存、checkpoints 均只读）。唯一写盘是本脚本自己的输出目录，且三个输出
文件已存在时拒绝运行（防覆盖历史归档）；门禁失败仍会写完全部输出，
再以退出码 1 结束。

依赖关系：pre_config.OUT_ROOT（输出根目录）；pre_dataset 的 SPLITS/
S_TOTAL/双变量掩膜/统一 stats 缓存（depth_index=None，缓存于 NORM_DIR，
仅按 train 段统计）；scripts/preprocess_align_uv.verify_daily_time。

输出位于 $OUT_ROOT/diag_uv_predictability_<YYYYMMDD>/：
    uv_predictability.npz   全部统计 + 溯源元数据
    summary.csv             每 (变量, 层) 一行
    SUMMARY.md              门禁表 + 人类可读结论

从仓库根目录运行：
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

# 配置常量

CHUNK_DAYS = 10          # 每个训练分块流式读取的天数（每变量约 211 MB 物化读入）
SAMPLE_STRIDE = 1000     # 确定性分位数子采样步长（全局步进相位跨 update 调用延续）
COASTAL_BUFFER = 5       # 距陆地的格数阈值；与 diag_region_breakdown.py 同一口径
VAL_MAX_LEAD = 15        # persistence 审计的最大 lead 天数
BANDS = (("bottom", 0, 9), ("middle", 10, 19), ("upper", 20, 29))
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)   # 分位数（分数形式）：p0.1/p1/p50/p99/p99.9
VARS = ("u", "v")
H, W = 400, 441

OUT_DIR = os.path.join(OUT_ROOT, "diag_uv_predictability_"
                       + time.strftime("%Y%m%d"))
NPZ_PATH = os.path.join(OUT_DIR, "uv_predictability.npz")
CSV_PATH = os.path.join(OUT_DIR, "summary.csv")
MD_PATH = os.path.join(OUT_DIR, "SUMMARY.md")

# 各变量的统一 min-max 归一化区间（在 main() 中由全层 stats 缓存解析后填充，
# 供训练段统计越界计数使用）
GLOBAL_LO = {}
GLOBAL_HI = {}


# 小工具

class Moments:
    """精确在线矩/极值 + 确定性定步长子采样。

    子采样保留"自开始以来第 SAMPLE_STRIDE*k 个"有效值（全局到达顺序），
    相位计数跨 update() 调用延续，分位数因此稳定、可复现且无需第二遍
    扫描；mean/std/min/max/计数始终对全部值精确。
    """

    def __init__(self):
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.mn = np.inf
        self.mx = -np.inf
        self.samples = []
        self._seen = 0         # 全局已读值计数，决定下一个子采样命中点

    def update(self, vals):
        vals = np.asarray(vals, np.float64).ravel()   # 展平为 1-D float64，物理单位 m/s
        if vals.size == 0:
            return
        self.n += int(vals.size)
        self.s1 += float(vals.sum())
        self.s2 += float((vals * vals).sum())
        self.mn = min(self.mn, float(vals.min()))
        self.mx = max(self.mx, float(vals.max()))
        # 全局定步长子采样：相位按展平后的到达顺序计算并跨调用延续，
        # 因此同一实例跨分块累积时分位数可复现
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
    """由子采样样本求分位数（分数形式）-> 物理值；子样本为空时返回全 NaN。"""
    if samples.size == 0:
        return np.full(len(QUANTILES), np.nan)
    return np.quantile(samples.astype(np.float64), QUANTILES)


def region_masks(mask2d):
    """(H, W) rho 掩膜 -> coastal / offshore 两个布尔格点掩膜。

    规则与 scripts/diag_region_breakdown.py 完全一致：coastal = 距陆地
    COASTAL_BUFFER 格以内的有效格点（陆地取掩膜补集，用默认十字结构元
    binary_dilation 迭代 COASTAL_BUFFER 次，即与陆地的 L1 距离 <=
    COASTAL_BUFFER），offshore = 其余有效格点。
    """
    land = ~np.asarray(mask2d, bool)
    near_land = ndimage.binary_dilation(land, iterations=COASTAL_BUFFER)
    valid = np.asarray(mask2d, bool)
    return valid & near_land, valid & ~near_land


def gate(name, ok, detail=""):
    """打印一行门禁结果并返回布尔值；只写终端，不写盘。"""
    print(f"[gate] {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


# 训练段统计

def train_scale_pass(masks, u, v):
    """单遍流式扫描 train split：每 (变量, 层) 的精确尺度统计、逐日增量
    统计、精确 valid/NaN 计数与统一 min-max 区间外的越界计数。

    分块自带一天读重叠，使边界增量 x[a0]-x[a0-1] 恰好计入一次。返回
    finalize 后的 per-var 层列表与精确计数数组（n_valid / n_nan_mask /
    n_below / n_above，形状均为 (S,) int64）。
    """
    lo_t, hi_t = SPLITS["train"]
    scale = {v_: [Moments() for _ in range(S_TOTAL)] for v_ in VARS}
    inc = {v_: [Moments() for _ in range(S_TOTAL)] for v_ in VARS}
    # 精确计数累加器（(S,) int64）：掩膜内有效 / 掩膜内 NaN / 越下界 / 越上界
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
        r0 = max(a0 - 1, lo_t)                     # 多读一天重叠，保证边界日的增量完整
        for name, arr in (("u", u), ("v", v)):
            m = masks[name]
            a = np.asarray(arr[r0:a1])             # 物化读入 (a1-r0, S, H, W)：天×层×行×列，物理 m/s
            cur = a[(a0 - r0):]                    # 本块需审计的天 [a0, a1)（不含重叠日）
            # 为 [a0, a1) 内每个目标日计算增量（首块的目标日 a0 无前一日，天然缺席）：
            # a[0] 是第 a0-1 天（首块时即 a0 本身），因此 a[1:] - a[:-1] 把每个
            # 被审计日与其前一背景日恰好配对一次。
            dif = a[1:] - a[:-1] if a.shape[0] > 1 else a[:0]   # 末块可能只有 1 天：dif 为空数组
            for s in range(S_TOTAL):
                day_vals = cur[:, s][:, m]         # 形状 (days, n_ocean)：仅审计日、掩膜内格点
                fin = np.isfinite(day_vals)
                n_valid[name][s] += int(fin.sum())
                n_nan_mask[name][s] += int(day_vals.size - fin.sum())
                fv = day_vals[fin]
                scale[name][s].update(fv)
                n_below[name][s] += int((fv < GLOBAL_LO[name]).sum())
                n_above[name][s] += int((fv > GLOBAL_HI[name]).sum())
                if dif.shape[0] > 0:
                    d = dif[:, s][:, m]            # 形状 (inc_days, n_ocean)：掩膜内逐日增量
                    ok = np.isfinite(d)
                    inc[name][s].update(d[ok])
        done += a1 - a0
        print(f"[train] chunk {ci + 1}/{n_chunks} days {a0}..{a1 - 1} "
              f"({done}/{n_days}) elapsed_s={time.perf_counter() - t0_:.0f}",
              flush=True)
    return scale, inc, n_valid, n_nan_mask, n_below, n_above, n_days


# 验证段持续性基线

def val_persistence_pass(masks, u, v, coastal_flat):
    """逐日流式扫描 val split：按 (变量, 层, lead) 累计持续性基线的
    RMSE/MAE，并给出 coastal 分解。

    窗口集合与正式协议一致：窗口起点 s ∈ [val_lo, val_hi-(7+L)]，lead l
    的持续性源 = 第 s+6 天，目标日 t = s+6+l。按目标日枚举，目标日 t 的
    合法 lead 满足
        l >= t - 6 - LAST_START      （窗口起点 t-6-l <= LAST_START）
        l <= min(L, t - 6 - val_lo)  （窗口起点 >= val_lo）
    滚动缓冲区保留最近 L+1 天，为各 lead 提供持续性源。
    """
    lo_v, hi_v = SPLITS["val"]
    L = VAL_MAX_LEAD
    LAST_START = hi_v - (7 + L)                # 最后一个合法窗口起点
    first_t = lo_v + 7                         # 最早目标日（s=lo_v, l=1）
    last_t = hi_v - 1                          # 最晚目标日（s=LAST_START, l=L）
    n_windows = LAST_START - lo_v + 1

    # 误差平方/绝对值/计数累计器，形状 (S, L)：第 l-1 列对应 lead l；
    # *_co 为 coastal 分解（同形状），n_valid/n_nan_mask 为 (S,) 当日计数
    se = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    ae = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    n = {v_: np.zeros((S_TOTAL, L), np.int64) for v_ in VARS}
    se_co = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    ae_co = {v_: np.zeros((S_TOTAL, L), np.float64) for v_ in VARS}
    n_co = {v_: np.zeros((S_TOTAL, L), np.int64) for v_ in VARS}
    n_valid = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}
    n_nan_mask = {v_: np.zeros(S_TOTAL, np.int64) for v_ in VARS}

    buffers = {v_: [] for v_ in VARS}          # 滚动历史（最旧在前），最多 L+1 天
    t0_ = time.perf_counter()
    for t in range(lo_v, last_t + 1):
        for name, arr in (("u", u), ("v", v)):
            day = np.asarray(arr[t])           # 形状 (S, H, W)：当日全场，物理 m/s
            m = masks[name]
            in_mask = m[None, :, :]
            fin = np.isfinite(day) & in_mask
            n_valid[name] += fin.reshape(S_TOTAL, -1).sum(axis=1)
            n_nan_mask[name] += (in_mask & ~np.isfinite(day)) \
                .reshape(S_TOTAL, -1).sum(axis=1)
            buffers[name].append(day)
            if len(buffers[name]) > L + 1:     # 缓冲区保留当日 t 与 L 个历史源日
                buffers[name].pop(0)
            if t < first_t:
                continue                       # 预热：first_t 之前只填缓冲区，不产生 lead
            max_l = min(L, t - 6 - lo_v)
            min_l = max(1, t - 6 - LAST_START)
            truth = day
            hist = buffers[name]
            co = coastal_flat[name]            # 形状 (n_ocean,)：与 m 选格对齐的 coastal 指示
            for l in range(min_l, max_l + 1):
                src = hist[-1 - l]             # 第 t-l 天：lead l 的持续性源
                e = (src - truth)[:, m]        # 形状 (S, n_ocean)：掩膜内误差
                ok = np.isfinite(e)
                # NaN 格点置 0 后再求和、计数只累加有限格点：
                # 分子分母严格来自同一有效集合，NaN 不会污染误差和
                ec = np.where(ok, e, 0.0)
                se[name][:, l - 1] += (ec ** 2).sum(axis=1)
                ae[name][:, l - 1] += np.abs(ec).sum(axis=1)
                n[name][:, l - 1] += ok.sum(axis=1)
                eco = ec * co                  # coastal 分解：误差逐格乘 coastal 指示
                se_co[name][:, l - 1] += (eco ** 2).sum(axis=1)
                ae_co[name][:, l - 1] += np.abs(eco).sum(axis=1)
                n_co[name][:, l - 1] += (ok & co).sum(axis=1)
        if (t - lo_v) % 50 == 0 or t == last_t:
            print(f"[val] target day {t} ({t - lo_v}/{last_t - lo_v}) "
                  f"elapsed_s={time.perf_counter() - t0_:.0f}", flush=True)

    # 先累计再 finalize：RMSE=sqrt(Σe²/Σn)、MAE=Σ|e|/Σn，绝不先取每段
    # RMSE 再平均；分母 max(n,1) 只防除零，门禁另行要求每层 n>0
    rmse = {v_: np.sqrt(se[v_] / np.maximum(n[v_], 1)) for v_ in VARS}
    mae = {v_: ae[v_] / np.maximum(n[v_], 1) for v_ in VARS}
    rmse_co = {v_: np.sqrt(se_co[v_] / np.maximum(n_co[v_], 1)) for v_ in VARS}
    mae_co = {v_: ae_co[v_] / np.maximum(n_co[v_], 1) for v_ in VARS}
    return dict(rmse=rmse, mae=mae, n=n, rmse_coastal=rmse_co, mae_coastal=mae_co,
                n_coastal=n_co, n_valid=n_valid, n_nan_mask=n_nan_mask,
                n_windows=n_windows)


# 主流程

def main():
    # 输出目录与三个输出文件：已存在则拒绝（防覆盖历史归档）
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in (NPZ_PATH, CSV_PATH, MD_PATH):
        if os.path.exists(path):
            raise RuntimeError(f"{path} already exists; delete it or change OUT_DIR")

    t_start = time.perf_counter()
    print(f"[setup] aligned data: {ALIGNED_DIR}", flush=True)
    print(f"[setup] out dir: {OUT_DIR}", flush=True)

    # 门禁 1：日时间连续性
    times = verify_daily_time(load_ocean_time())
    assert times.shape[0] == T_TOTAL
    gate("daily time continuity", True,
         f"{times[0]} .. {times[-1]} ({T_TOTAL} days)")

    # 双变量 rho 掩膜与 coastal 分解
    masks = dict(zip(VARS, load_masks()))
    mv = mask_version()
    shapes_ok = all(m.shape == (H, W) for m in masks.values())
    gate("mask shapes", shapes_ok, f"u{masks['u'].shape} v{masks['v'].shape} "
         f"version={mv}")
    coastal_flat = {}
    for name in VARS:
        co, _ = region_masks(masks[name])
        coastal_flat[name] = co[masks[name]]   # 在海洋掩膜内展平，与后续 [:, m] 取格一一对应

    # 当前实际使用的统一（全层）min-max 归一化：权威来源是全层 stats 缓存
    # （仅按 train 段统计，缓存于 NORM_DIR）
    print("[setup] loading the unified (depth_index=None) min-max stats...",
          flush=True)
    stats_all = compute_or_load_stats(depth_index=None, verbose=False)
    for j, v_ in enumerate(VARS):
        GLOBAL_LO[v_] = float(stats_all["lo"][j])
        GLOBAL_HI[v_] = float(stats_all["hi"][j])
    print(f"[setup] unified min-max: lo={GLOBAL_LO} hi={GLOBAL_HI}", flush=True)

    # 只读 memmap：(T, S, H, W) float32 物理场（m/s，陆地 NaN），整个脚本不物化全场
    u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
    v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")

    # 训练段（尺度 + 增量 + 精确计数）
    scale, inc, n_valid, n_nan, n_below, n_above, days_tr = \
        train_scale_pass(masks, u, v)

    # 验证段（persistence 难度）
    val = val_persistence_pass(masks, u, v, coastal_flat)

    # finalize 与门禁 2/3：掩膜内零 NaN/Inf；每层 train+val 有效计数为正
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

    # NPZ payload：全部统计 + 溯源元数据
    def layer_stack(getter):
        """把 per-var 层字典堆成 (2, S) 数组：第 0 轴 u/v，第 1 轴 sigma 层。"""
        return np.array([[getter(fin_scale[v_][s]) for s in range(S_TOTAL)]
                         for v_ in VARS], np.float64)

    # 尺度统计键均为 (2, S) float64；train_sample_n 是子采样样本数
    # （非全量计数，全量 n 见 train_valid_count）
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
    # 分位数键 train_p{...} / inc_p{...}：qi 按 QUANTILES 顺序对应
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
    rng = payload["norm_hi"] - payload["norm_lo"]   # (2,)：每变量的归一化区间宽度
    payload["norm_clip_frac"] = np.array(
        [[(int(n_below[v_][s]) + int(n_above[v_][s])) / max(int(n_valid[v_][s]), 1)
          for s in range(S_TOTAL)] for v_ in VARS])
    # 以下两个键把物理量换算到归一化空间：norm_std=train_std/(hi-lo)；
    # norm_p1_p99_width=(p99-p1)/(hi-lo) 为正的归一化 p1–p99 宽度。
    # 2026-09-05 勘误：旧实现误用 (p1-p99)，历史 NPZ/CSV/MD 中该键 60/60 全为
    # 负值（范围 -0.3273..-0.0695）；幅值仍是真宽度，但归档产物重生成前不得按
    # 正负号解读该键。
    payload["norm_std"] = payload["train_std"] / rng[:, None]
    payload["norm_p1_p99_width"] = (payload["train_p99"] - payload["train_p1"]) \
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
    # 分 band pooled RMSE：由每层 n 与 rmse 反解误差平方和 Σ(n·rmse²)，
    # 跨层求和后除以 Σn 再开方 —— 是池化，不是逐层 RMSE 的算术平均
    for bname, b0, b1 in BANDS:
        sel = slice(b0, b1 + 1)
        se_sum = np.array([np.maximum(val["n"][v_][sel], 1)
                           * val["rmse"][v_][sel] ** 2 for v_ in VARS])
        n_sum = np.array([np.maximum(val["n"][v_][sel], 1) for v_ in VARS])
        payload[f"val_rmse_band_{bname}"] = np.sqrt(
            se_sum.sum(axis=1) / n_sum.sum(axis=1))

    # 溯源元数据：数据路径、mask 版本、split 边界、采样超参、门禁结果与耗时；
    # git 提交号尽力获取（子进程失败记 unknown，不中断）
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

    # CSV 汇总：每 (变量, 层) 一行；数值均为物理单位（m/s）
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

    # Markdown 汇总：门禁结论 + band/d15 排名表（正文本身已是中文）
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
