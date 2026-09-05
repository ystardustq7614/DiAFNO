#!/usr/bin/env python3
"""模块职责：对 PRE u/v->rho 预处理流水线做 CPU/GPU/I/O 计时剖析；只写专用
scratch 目录，绝不触碰生产输出（u_rho.npy/v_rho.npy 等正式对齐产物）。

不负责：生成正式数据；不重复实现任何 kernel——CUDA 路径直接调用生产实现
（pp.torch_extrema_summary、pp.torch_enforce_land_mask、pp.torch_colocate_u/v），
保证剖析的就是生产 kernel 本身，计时结果不因重实现而失真。

关键约束：
- 在数据服务器上从仓库根目录运行，例如：

    python scripts/profile_preprocess_align_uv.py \\
        --src /data2/user/zyq/datasets/PRE/processed \\
        --scratch-root /data2/user/zyq/data_processed/PRE/profile_scratch \\
        --raw-dyn /data2/user/zyq/datasets/PRE/raw/dyn \\
        --profile-time-metadata \\
        --report-json /data2/user/zyq/data_processed/PRE/profile_report.json

- 从真实 mmap 输入采样完整 chunk，只写 chunk 大小的 scratch .npy；计时阶段与
  preprocess_align_uv.py 相同：read、极值、mask 强制、共定位、write、flush、
  输出极值。scratch 数据默认在成功或失败后删除；--keep-scratch 可保留供检查。
- --gpu-compare 时，把一个采样的 RAM 驻留 chunk 送入生产 CUDA 实现，分别计时
  H2D、GPU 各阶段与 D2H，同时做 NumPy 精确等价校验；同时报告 CUDA peak
  allocated 与 peak reserved 显存；CPU/GPU 各阶段计时外推为完整 10591 天
  运行的估计。
- torch 只在 --gpu-compare 分支内延迟 import，其余路径不需要 CUDA。

依赖关系：import 生产脚本 scripts.preprocess_align_uv（其 main 仅在
__main__ 下执行，import 无副作用）；--gpu-compare 分支延迟 import torch。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

try:
    import resource
except ImportError:  # Windows 没有 resource 模块；数据服务器是 Linux。
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import preprocess_align_uv as pp  # noqa: E402


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def percentile(values: list[float], q: float) -> float:
    """线性插值百分位数，不引入 pandas 依赖。"""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_stage_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """按阶段汇总多次运行的耗时统计（count/total/mean/median/p95），单位秒。"""
    values_by_stage: dict[str, list[float]] = {}
    for run in runs:
        for stage, value in run.items():
            values_by_stage.setdefault(stage, []).append(value)

    return {
        stage: {
            "count": len(values),
            "total_s": sum(values),
            "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "p95_s": percentile(values, 0.95),
        }
        for stage, values in sorted(values_by_stage.items())
    }


def rss_mib() -> float:
    """把 ru_maxrss 换算为 MiB；无 resource 模块时返回 0.0。"""
    # Linux 的 ru_maxrss 单位是 KiB，macOS 是字节。数据服务器是 Linux，
    # 但保持可移植可让本地 dry-run 的结果不意外。
    if resource is None:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def resource_snapshot() -> dict[str, float]:
    """当前进程的资源用量快照（peak RSS、缺页计数、块 I/O 计数）；无 resource 模块时全为 0。"""
    if resource is None:
        return {
            "peak_rss_mib": 0.0,
            "major_faults": 0.0,
            "minor_faults": 0.0,
            "input_blocks": 0.0,
            "output_blocks": 0.0,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_mib": rss_mib(),
        "major_faults": float(usage.ru_majflt),
        "minor_faults": float(usage.ru_minflt),
        "input_blocks": float(usage.ru_inblock),
        "output_blocks": float(usage.ru_oublock),
    }


def resource_delta(before: dict[str, float]) -> dict[str, float]:
    """快照差值：缺页与块 I/O 为区间增量；peak_rss_mib 是进程累计峰值而非差值。"""
    after = resource_snapshot()
    return {
        key: after[key] - before[key]
        for key in ("major_faults", "minor_faults", "input_blocks", "output_blocks")
    } | {"peak_rss_mib": after["peak_rss_mib"]}


def parse_chunk_indices(value: str, n_chunks: int) -> list[int]:
    """argparse 类型函数：解析逗号分隔的 chunk 索引；不允许重复，且必须落在 [0, n_chunks)。"""
    try:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--chunk-indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("--chunk-indices must not be empty")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("--chunk-indices must not contain duplicates")
    bad = [index for index in indices if index < 0 or index >= n_chunks]
    if bad:
        raise argparse.ArgumentTypeError(
            f"chunk indices {bad} outside [0, {n_chunks - 1}]")
    return indices


def timed(stages: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    """计时执行 fn()，把耗时（秒）记入 stages[name]，并返回 fn() 的结果。"""
    started = time.perf_counter()
    result = fn()
    stages[name] = seconds_since(started)
    return result


def colocate_u(uc: np.ndarray) -> np.ndarray:
    """CPU：u chunk (t, S, H, W-1) -> rho (t, S, H, W)；与 torch_colocate_u 同构，内部复用 pp.colocate。

    输出 shape 从输入推导（末轴 +1），不绑定全局网格常量；全局 pp.S/H/W 仅由
    main() 入口的完整网格断言保证与输入一致。
    """
    tlen = uc.shape[0]
    w_rho = uc.shape[-1] + 1
    ub = np.empty(uc.shape[:3] + (w_rho,), np.float32)
    ub[:, :, :, 1:w_rho - 1] = pp.colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
    ub[:, :, :, 0] = uc[:, :, :, 0]
    ub[:, :, :, w_rho - 1] = uc[:, :, :, -1]
    return ub


def colocate_v(vc: np.ndarray) -> np.ndarray:
    """CPU：v chunk (t, S, H-1, W) -> rho (t, S, H, W)；与 torch_colocate_v 同构，内部复用 pp.colocate。

    输出 shape 从输入推导（倒数第二轴 +1），不绑定全局网格常量；全局 pp.S/H/W
    仅由 main() 入口的完整网格断言保证与输入一致。
    """
    tlen = vc.shape[0]
    h_rho = vc.shape[-2] + 1
    vb = np.empty(vc.shape[:2] + (h_rho, vc.shape[-1]), np.float32)
    vb[:, :, 1:h_rho - 1, :] = pp.colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
    vb[:, :, 0, :] = vc[:, :, 0, :]
    vb[:, :, h_rho - 1, :] = vc[:, :, -1, :]
    return vb


def load_masks(src: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并校验 mask_rho/mask_u/mask_v 的 shape 与取值；返回原生 staggered 网格上的 (mask_u, mask_v) bool 数组。"""
    mask_rho = np.load(src / "stat_var" / "mask_rho.npy")
    mask_u = np.load(src / "stat_var" / "mask_u.npy")
    mask_v = np.load(src / "stat_var" / "mask_v.npy")
    assert mask_rho.shape == (pp.H, pp.W), f"mask_rho shape {mask_rho.shape}"
    assert mask_u.shape == (pp.H, pp.W - 1), f"mask_u shape {mask_u.shape}"
    assert mask_v.shape == (pp.H - 1, pp.W), f"mask_v shape {mask_v.shape}"
    for name, mask in (("mask_rho", mask_rho), ("mask_u", mask_u), ("mask_v", mask_v)):
        assert set(np.unique(mask)).issubset({0, 1}), f"{name} values {np.unique(mask)}"
    return mask_u.astype(bool), mask_v.astype(bool)


def profile_metadata(raw_dyn: Path) -> dict[str, float]:
    """对权威时间元数据扫描计时；只测耗时，不保存任何输出文件。

    与生产一致：逐 NetCDF 读取 ocean_time（保持 datetime64[s] 精度）后再做
    逐日间隔校验，两段分别计时。
    """
    files = sorted(path for path in raw_dyn.iterdir() if path.suffix == ".nc")
    if len(files) != pp.T:
        raise RuntimeError(f"expected {pp.T} raw NetCDF files in {raw_dyn}, found {len(files)}")

    started = time.perf_counter()
    times = np.empty(pp.T, dtype="datetime64[s]")
    units: str | None = None
    for i, path in enumerate(files):
        with pp.netCDF4.Dataset(path) as dataset:
            ocean_time = dataset.variables["ocean_time"]
            if units is None:
                units = getattr(ocean_time, "units", None)
            values = np.asarray(ocean_time[:]).reshape(-1)
            if values.size != 1:
                raise RuntimeError(f"{path}: ocean_time has {values.size} entries, expected 1")
            times[i] = np.datetime64(
                pp.netCDF4.num2date(float(values[0]), units, only_use_python_datetimes=True))
    read_s = seconds_since(started)

    started = time.perf_counter()
    pp.verify_daily_time(times)
    verify_s = seconds_since(started)
    return {"metadata_read_s": read_s, "metadata_verify_s": verify_s, "metadata_total_s": read_s + verify_s}


def profile_compute_only(
    u_base: np.ndarray,
    v_base: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    t0: int,
    repeats: int,
) -> list[dict[str, float]]:
    """对已 RAM 驻留的输入计时纯 CPU 计算，不含磁盘写入；重复 repeats 次取分布。"""
    runs: list[dict[str, float]] = []
    for _ in range(repeats):
        stages: dict[str, float] = {}
        uc = timed(stages, "ram_copy_u_s", u_base.copy)
        vc = timed(stages, "ram_copy_v_s", v_base.copy)
        raw_u, raw_v = pp.ExtremumTracker("u raw"), pp.ExtremumTracker("v raw")
        out_u, out_v = pp.ExtremumTracker("u_rho"), pp.ExtremumTracker("v_rho")
        discarded: dict[str, int] = {}

        timed(stages, "raw_extrema_s", lambda: (raw_u.update(uc, t0), raw_v.update(vc, t0)))
        timed(stages, "mask_check_s", lambda: (
            pp.enforce_land_mask(uc, mask_u, "u", t0, discarded),
            pp.enforce_land_mask(vc, mask_v, "v", t0, discarded),
        ))
        ub = timed(stages, "colocate_u_s", lambda: colocate_u(uc))
        vb = timed(stages, "colocate_v_s", lambda: colocate_v(vc))
        timed(stages, "aligned_extrema_s", lambda: (out_u.update(ub, t0), out_v.update(vb, t0)))
        stages["compute_only_total_s"] = sum(stages.values())
        runs.append(stages)
        del uc, vc, ub, vb
    return runs


def numpy_extrema_signature(arr: np.ndarray) -> tuple[float, int, float, int]:
    """NumPy 参照实现；返回元组的顺序与 pp.torch_extrema_summary 一致。"""
    flat = arr.ravel()
    return (
        float(np.nanmin(flat)), int(np.nanargmin(flat)),
        float(np.nanmax(flat)), int(np.nanargmax(flat)),
    )


def gpu_timed(stages: dict[str, float], name: str, fn: Callable[[], Any], torch: Any) -> Any:
    # CUDA 调用是异步的：fn 前后各同步一次，每个阶段计时才是真实墙钟耗时，
    # 而不是 kernel 启动时间。
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    stages[name] = seconds_since(started)
    return result


def cpu_gpu_reference(
    u_base: np.ndarray, v_base: np.ndarray, mask_u: np.ndarray, mask_v: np.ndarray,
) -> dict[str, Any]:
    """为 GPU 数值等价校验生成精确的 NumPy 参照结果。

    流程与生产一致：原始极值签名 -> mask 强制 -> 共定位；返回参照的极值签名、
    丢弃计数与对齐场 ub/vb（后续用 NaN 相等的数组比较校验）。
    """
    uc = u_base.copy()
    vc = v_base.copy()
    discarded: dict[str, int] = {}
    raw = (numpy_extrema_signature(uc), numpy_extrema_signature(vc))
    pp.enforce_land_mask(uc, mask_u, "u", 0, discarded)
    pp.enforce_land_mask(vc, mask_v, "v", 0, discarded)
    ub, vb = colocate_u(uc), colocate_v(vc)
    return {
        "raw": raw,
        "aligned": (numpy_extrema_signature(ub), numpy_extrema_signature(vb)),
        "discarded": discarded,
        "ub": ub,
        "vb": vb,
    }


def assert_gpu_matches_reference(reference: dict[str, Any], result: dict[str, Any]) -> None:
    """GPU 结果与 NumPy 参照逐项比对（丢弃计数、极值签名、对齐场 NaN 相等）；不一致即抛 AssertionError。"""
    if reference["discarded"] != result["discarded"]:
        raise AssertionError(f"GPU discarded counts differ: {result['discarded']} != {reference['discarded']}")
    for name in ("raw", "aligned"):
        if reference[name] != result[name]:
            raise AssertionError(f"GPU {name} extrema differ: {result[name]} != {reference[name]}")
    if not np.array_equal(reference["ub"], result["ub"], equal_nan=True):
        raise AssertionError("GPU u_rho differs from the NumPy reference")
    if not np.array_equal(reference["vb"], result["vb"], equal_nan=True):
        raise AssertionError("GPU v_rho differs from the NumPy reference")


def profile_gpu_once(
    u_base: np.ndarray,
    v_base: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    torch: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """单次完整 GPU 流水线计时：H2D -> 各 GPU 阶段 -> D2H，并收集显存峰值。"""
    device_index = 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    stages: dict[str, float] = {}
    # 目标环境的 PyTorch 2.4 构建中，这个显存分配器 API 会拒绝 torch.device
    # 对象（张量的 .to(device) 则接受它）；显式选定 cuda:0 之后，无参数形式
    # 是兼容的。
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    uc = gpu_timed(stages, "h2d_u_s", lambda: torch.from_numpy(u_base).to(device), torch)
    vc = gpu_timed(stages, "h2d_v_s", lambda: torch.from_numpy(v_base).to(device), torch)
    gpu_mask_u = torch.as_tensor(mask_u, dtype=torch.bool, device=device)
    gpu_mask_v = torch.as_tensor(mask_v, dtype=torch.bool, device=device)
    raw = gpu_timed(stages, "raw_extrema_s", lambda: (
        pp.torch_extrema_summary(uc), pp.torch_extrema_summary(vc)), torch)
    discarded: dict[str, int] = {}
    gpu_timed(stages, "mask_check_s", lambda: (
        pp.torch_enforce_land_mask(uc, gpu_mask_u, "u", 0, discarded),
        pp.torch_enforce_land_mask(vc, gpu_mask_v, "v", 0, discarded)), torch)
    ub = gpu_timed(stages, "colocate_u_s", lambda: pp.torch_colocate_u(uc), torch)
    vb = gpu_timed(stages, "colocate_v_s", lambda: pp.torch_colocate_v(vc), torch)
    aligned = gpu_timed(stages, "aligned_extrema_s", lambda: (
        pp.torch_extrema_summary(ub), pp.torch_extrema_summary(vb)), torch)
    ub_cpu = gpu_timed(stages, "d2h_u_s", lambda: ub.cpu().numpy(), torch)
    vb_cpu = gpu_timed(stages, "d2h_v_s", lambda: vb.cpu().numpy(), torch)
    stages["gpu_total_s"] = seconds_since(started)
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024 ** 2
    result = {
        "raw": raw,
        "aligned": aligned,
        "discarded": discarded,
        "ub": ub_cpu,
        "vb": vb_cpu,
        "gpu_peak_allocated_mib": peak_allocated_mib,
        "gpu_peak_reserved_mib": peak_reserved_mib,
    }
    del uc, vc, ub, vb, gpu_mask_u, gpu_mask_v
    torch.cuda.empty_cache()
    return stages, result


def profile_gpu_compare(
    u_base: np.ndarray,
    v_base: np.ndarray,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    repeats: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """warmup + repeats 次 GPU 计时，每次都与 NumPy 参照精确比对；返回各次阶段计时与汇总信息。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--gpu-compare requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-compare requested but torch.cuda.is_available() is False")

    reference = cpu_gpu_reference(u_base, v_base, mask_u, mask_v)
    # 预热运行不计入统计：先支付分配器/JIT 初始化成本，再开始正式计时。
    _, warmup = profile_gpu_once(u_base, v_base, mask_u, mask_v, torch)
    assert_gpu_matches_reference(reference, warmup)
    del warmup

    runs: list[dict[str, float]] = []
    peaks_allocated: list[float] = []
    peaks_reserved: list[float] = []
    for _ in range(repeats):
        stages, result = profile_gpu_once(u_base, v_base, mask_u, mask_v, torch)
        assert_gpu_matches_reference(reference, result)
        peaks_allocated.append(float(result.pop("gpu_peak_allocated_mib")))
        peaks_reserved.append(float(result.pop("gpu_peak_reserved_mib")))
        runs.append(stages)
        del result
    del reference
    return runs, {
        "device": torch.cuda.get_device_name(0),
        "numerically_verified": True,
        "peak_allocated_mib_median": statistics.median(peaks_allocated),
        "peak_allocated_mib_p95": percentile(peaks_allocated, 0.95),
        "peak_reserved_mib_median": statistics.median(peaks_reserved),
        "peak_reserved_mib_p95": percentile(peaks_reserved, 0.95),
    }


def profile_chunk(
    u: np.memmap,
    v: np.memmap,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    mask_u_rho: np.ndarray,
    mask_v_rho: np.ndarray,
    chunk_index: int,
    chunk_days: int,
    scratch: Path,
    trackers: tuple[pp.ExtremumTracker, pp.ExtremumTracker, pp.ExtremumTracker, pp.ExtremumTracker],
    validate_nan_pattern: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """对一个采样 chunk 完整跑一遍 CPU 流水线并逐阶段计时。

    阶段与生产一致：read -> 原始极值 -> mask 强制 -> 共定位 ->
    （validate_nan_pattern 为 True 时，即首个采样 chunk）NaN 图案断言 ->
    scratch memmap write -> flush -> 对齐极值。返回 (result, uc 副本, vc 副本)，
    副本供可选的 compute-only / GPU 对比基准复用。
    """
    ts = chunk_index * chunk_days
    te = min(ts + chunk_days, pp.T)
    tlen = te - ts
    stages: dict[str, float] = {}
    resource_before = resource_snapshot()
    chunk_started = time.perf_counter()

    uc = timed(stages, "read_u_s", lambda: np.array(u[ts:te]))
    vc = timed(stages, "read_v_s", lambda: np.array(v[ts:te]))
    raw_u, raw_v, aligned_u, aligned_v = trackers
    timed(stages, "raw_extrema_s", lambda: (raw_u.update(uc, ts), raw_v.update(vc, ts)))
    discarded: dict[str, int] = {}
    timed(stages, "mask_check_s", lambda: (
        pp.enforce_land_mask(uc, mask_u, "u", ts, discarded),
        pp.enforce_land_mask(vc, mask_v, "v", ts, discarded),
    ))
    ub = timed(stages, "colocate_u_s", lambda: colocate_u(uc))
    vb = timed(stages, "colocate_v_s", lambda: colocate_v(vc))

    validation_s = 0.0
    if validate_nan_pattern:
        validation_started = time.perf_counter()
        assert (np.isnan(ub) == (mask_u_rho == 0)[None, None]).all(), \
            "u_rho NaN pattern does not match mask_u_rho"
        assert (np.isnan(vb) == (mask_v_rho == 0)[None, None]).all(), \
            "v_rho NaN pattern does not match mask_v_rho"
        validation_s = seconds_since(validation_started)
        stages["first_chunk_validation_s"] = validation_s

    # scratch memmap 的 shape 直接取共定位结果（与输入 chunk 一致），不再绑定全局常量
    u_out = np.lib.format.open_memmap(
        scratch / f"u_rho_chunk{chunk_index:03d}.npy", mode="w+", dtype=np.float32,
        shape=ub.shape)
    v_out = np.lib.format.open_memmap(
        scratch / f"v_rho_chunk{chunk_index:03d}.npy", mode="w+", dtype=np.float32,
        shape=vb.shape)
    timed(stages, "write_u_s", lambda: u_out.__setitem__(slice(None), ub))
    timed(stages, "write_v_s", lambda: v_out.__setitem__(slice(None), vb))
    timed(stages, "flush_u_s", u_out.flush)
    timed(stages, "flush_v_s", v_out.flush)
    timed(stages, "aligned_extrema_s", lambda: (aligned_u.update(ub, ts), aligned_v.update(vb, ts)))
    stages["chunk_total_s"] = seconds_since(chunk_started)

    # 字节数按实际数组统计（与 shape 推导一致，部分网格 chunk 下仍成立）
    logical_read_bytes = uc.nbytes + vc.nbytes
    logical_write_bytes = ub.nbytes + vb.nbytes
    read_s = stages["read_u_s"] + stages["read_v_s"]
    write_flush_s = (stages["write_u_s"] + stages["write_v_s"] +
                     stages["flush_u_s"] + stages["flush_v_s"])

    # 仅为可选的 compute-only 基准返回 chunk 拷贝；在调用方可能清理 scratch
    # 之前，先用 del + gc.collect() 关闭 scratch memmap 句柄。
    result = {
        "chunk_index": chunk_index,
        "day_range": [ts, te],
        "discarded": discarded,
        "stages": stages,
        "logical_io": {
            "read_gib": logical_read_bytes / 1024 ** 3,
            "write_gib": logical_write_bytes / 1024 ** 3,
            "read_mib_s": logical_read_bytes / 1024 ** 2 / read_s if read_s else 0.0,
            "write_flush_mib_s": logical_write_bytes / 1024 ** 2 / write_flush_s if write_flush_s else 0.0,
        },
        "resources": resource_delta(resource_before),
    }
    del u_out, v_out
    gc.collect()
    return result, uc.copy(), vc.copy()


def environment_info() -> dict[str, Any]:
    """收集平台/Python/numpy/torch/CUDA 环境信息；torch 缺失时记录 None。"""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch"] = None
    return info


def print_summary(report: dict[str, Any]) -> None:
    """把剖析报告按 CPU/GPU/元数据/全量外推四段以可读表格打印到终端。"""
    print("\nCPU chunk-stage summary (seconds):")
    print(f"{'stage':<30} {'count':>5} {'median':>10} {'p95':>10} {'total':>10} {'% wall':>8}")
    wall = sum(run["stages"]["chunk_total_s"] for run in report["chunks"])
    for stage, summary in report["chunk_stage_summary"].items():
        if stage == "chunk_total_s":
            continue
        fraction = 100.0 * summary["total_s"] / wall if wall else 0.0
        print(f"{stage:<30} {summary['count']:>5} {summary['median_s']:>10.3f} "
              f"{summary['p95_s']:>10.3f} {summary['total_s']:>10.3f} {fraction:>7.1f}%")
    print(f"{'chunk_total_s':<30} {len(report['chunks']):>5}"
          f" {statistics.median(run['stages']['chunk_total_s'] for run in report['chunks']):>10.3f}"
          f" {'-':>10} {wall:>10.3f} {'100.0%':>8}")

    if report.get("compute_only_stage_summary"):
        print("\nRAM-resident CPU compute-only summary (seconds):")
        for stage, summary in report["compute_only_stage_summary"].items():
            print(f"{stage:<30} median={summary['median_s']:.3f}  p95={summary['p95_s']:.3f}")

    if report.get("gpu_stage_summary"):
        gpu = report["gpu_info"]
        total = report["gpu_stage_summary"]["gpu_total_s"]["median_s"]
        print(f"\nGPU comparison ({gpu['device']}; numerical check: {gpu['numerically_verified']}):")
        for stage, summary in report["gpu_stage_summary"].items():
            fraction = 100.0 * summary["median_s"] / total if total else 0.0
            print(f"{stage:<30} median={summary['median_s']:.3f}  p95={summary['p95_s']:.3f}"
                  f"  {fraction:5.1f}% of gpu_total")
        print(f"GPU peak allocated: median={gpu['peak_allocated_mib_median']:.1f} MiB "
              f"p95={gpu['peak_allocated_mib_p95']:.1f} MiB")
        print(f"GPU peak reserved : median={gpu['peak_reserved_mib_median']:.1f} MiB "
              f"p95={gpu['peak_reserved_mib_p95']:.1f} MiB")

    if report.get("metadata"):
        print("\nMetadata scan:")
        for name, value in report["metadata"].items():
            print(f"{name}: {value:.3f}s")

    if report["chunks"]:
        chunk_days = report["arguments"]["chunk_days"]
        n_chunks_total = (pp.T + chunk_days - 1) // chunk_days
        chunk_median = statistics.median(
            run["stages"]["chunk_total_s"] for run in report["chunks"])
        metadata_total = report.get("metadata", {}).get("metadata_total_s", 0.0) or 0.0
        lines = [
            f"\nFull-run estimates for T={pp.T} days, chunk={chunk_days} days "
            f"({n_chunks_total} chunks; metadata {metadata_total:.0f}s):",
            f"  CPU  pipeline @ median {chunk_median:.3f} s/chunk : "
            f"~{(chunk_median * n_chunks_total + metadata_total) / 60:.1f} min",
        ]
        if report.get("gpu_stage_summary"):
            gpu_total = report["gpu_stage_summary"]["gpu_total_s"]["median_s"]
            read_s = sum(
                report["chunk_stage_summary"].get(name, {}).get("median_s", 0.0)
                for name in ("read_u_s", "read_v_s"))
            write_flush = sum(
                report["chunk_stage_summary"].get(name, {}).get("median_s", 0.0)
                for name in ("write_u_s", "write_v_s", "flush_u_s", "flush_v_s"))
            lines.append(
                f"  GPU  pipeline @ median {gpu_total:.3f} s compute+transfer/chunk "
                f"+ {read_s:.3f} s read + {write_flush:.3f} s write+flush : "
                f"~{((gpu_total + read_s + write_flush) * n_chunks_total + metadata_total) / 60:.1f} min")
        print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="processed PRE root containing dyn_var/ and stat_var/")
    parser.add_argument("--scratch-root", type=Path, required=True,
                        help="existing or creatable directory on the intended output filesystem")
    parser.add_argument("--chunk-indices", default="0,1,2",
                        help="comma-separated chunk indices; default: 0,1,2")
    parser.add_argument("--chunk-days", type=int, default=pp.CHUNK,
                        help=f"days per chunk; default: {pp.CHUNK}")
    parser.add_argument("--compute-repeats", type=int, default=3,
                        help="RAM-resident CPU compute repeats after the sampled chunks; 0 disables")
    parser.add_argument("--gpu-compare", action="store_true",
                        help="benchmark one sampled RAM-resident chunk on cuda:0 and verify NumPy equivalence")
    parser.add_argument("--gpu-repeats", type=int, default=5,
                        help="measured GPU trials after one warmup; default: 5")
    parser.add_argument("--raw-dyn", type=Path,
                        help="raw NetCDF directory, required with --profile-time-metadata")
    parser.add_argument("--profile-time-metadata", action="store_true",
                        help="time the full authoritative ocean_time scan without saving files")
    parser.add_argument("--keep-scratch", action="store_true",
                        help="retain generated chunk outputs instead of deleting the private session directory")
    parser.add_argument("--report-json", type=Path,
                        help="write the machine-readable report to this path")
    args = parser.parse_args()

    if args.chunk_days <= 0:
        parser.error("--chunk-days must be positive")
    if args.compute_repeats < 0:
        parser.error("--compute-repeats must be non-negative")
    if args.gpu_repeats <= 0:
        parser.error("--gpu-repeats must be positive")
    if args.profile_time_metadata and args.raw_dyn is None:
        parser.error("--raw-dyn is required with --profile-time-metadata")
    if not args.src.is_dir():
        parser.error(f"--src does not exist or is not a directory: {args.src}")
    if args.raw_dyn is not None and not args.raw_dyn.is_dir():
        parser.error(f"--raw-dyn does not exist or is not a directory: {args.raw_dyn}")

    n_chunks = (pp.T + args.chunk_days - 1) // args.chunk_days
    chunk_indices = parse_chunk_indices(args.chunk_indices, n_chunks)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="preprocess-align-profile-", dir=args.scratch_root))
    print(f"[profile] scratch output: {scratch}", flush=True)

    try:
        u = np.load(args.src / "dyn_var" / "u.npy", mmap_mode="r")
        v = np.load(args.src / "dyn_var" / "v.npy", mmap_mode="r")
        assert u.shape == (pp.T, pp.S, pp.H, pp.W - 1), f"u shape {u.shape}"
        assert v.shape == (pp.T, pp.S, pp.H - 1, pp.W), f"v shape {v.shape}"
        assert u.dtype == np.float32, f"u dtype {u.dtype}"
        assert v.dtype == np.float32, f"v dtype {v.dtype}"
        mask_u, mask_v = load_masks(args.src)
        mask_u_rho = pp.u_rho_mask(mask_u)
        mask_v_rho = pp.v_rho_mask(mask_v)

        trackers = (
            pp.ExtremumTracker("u raw"), pp.ExtremumTracker("v raw"),
            pp.ExtremumTracker("u_rho"), pp.ExtremumTracker("v_rho"),
        )
        chunks: list[dict[str, Any]] = []
        compute_source: tuple[np.ndarray, np.ndarray, int] | None = None
        for index in chunk_indices:
            result, u_copy, v_copy = profile_chunk(
                u, v, mask_u, mask_v, mask_u_rho, mask_v_rho, index,
                args.chunk_days, scratch, trackers, validate_nan_pattern=(index == chunk_indices[0]))
            chunks.append(result)
            print(f"[profile] chunk {index} days {result['day_range'][0]}:{result['day_range'][1]} "
                  f"total {result['stages']['chunk_total_s']:.3f}s", flush=True)
            if compute_source is None:
                compute_source = (u_copy, v_copy, result["day_range"][0])
            else:
                del u_copy, v_copy

        compute_runs: list[dict[str, float]] = []
        gpu_runs: list[dict[str, float]] = []
        gpu_info: dict[str, Any] | None = None
        if compute_source is not None:
            u_base, v_base, t0 = compute_source
            if args.compute_repeats:
                compute_runs = profile_compute_only(
                    u_base, v_base, mask_u, mask_v, t0, args.compute_repeats)
            if args.gpu_compare:
                gpu_runs, gpu_info = profile_gpu_compare(
                    u_base, v_base, mask_u, mask_v, args.gpu_repeats)
            del u_base, v_base

        metadata = profile_metadata(args.raw_dyn) if args.profile_time_metadata else None
        stage_runs = [chunk["stages"] for chunk in chunks]
        report: dict[str, Any] = {
            "environment": environment_info(),
            "arguments": {
                "src": str(args.src),
                "scratch": str(scratch),
                "chunk_indices": chunk_indices,
                "chunk_days": args.chunk_days,
                "compute_repeats": args.compute_repeats,
                "gpu_compare": args.gpu_compare,
                "gpu_repeats": args.gpu_repeats,
                "profile_time_metadata": args.profile_time_metadata,
            },
            "chunks": chunks,
            "chunk_stage_summary": summarize_stage_runs(stage_runs),
            "compute_only_runs": compute_runs,
            "compute_only_stage_summary": summarize_stage_runs(compute_runs),
            "gpu_runs": gpu_runs,
            "gpu_stage_summary": summarize_stage_runs(gpu_runs),
            "gpu_info": gpu_info,
            "metadata": metadata,
            "trackers": [tracker.report() for tracker in trackers],
        }
        print_summary(report)
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            with args.report_json.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(f"\n[profile] report written to {args.report_json}", flush=True)
    finally:
        if args.keep_scratch:
            print(f"[profile] scratch retained: {scratch}", flush=True)
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            print(f"[profile] scratch removed: {scratch}", flush=True)


if __name__ == "__main__":
    main()
