#!/usr/bin/env python3
"""模块职责：Plan A 预处理——把原生 ROMS Arakawa-C staggered u/v 场共定位到
rho 网格，产出 rho 网格场、双变量有效性掩膜和经验证的时间轴；本脚本是
pre_dataset.py 读入的 u_rho/v_rho 训练数据的唯一生产者。

不负责：训练/评估/采样逻辑；不做 east/north 旋转——u_rho/v_rho 保留原始网格
xi/eta 方向分量的物理语义，只把采样位置移到 rho 点（Plan A 的核心决策，
下游所有网格换算都依赖该语义）。

关键约束：
- 原生输入（T 为天数，s 为 sigma 层轴）：
    u: (T, s, 400, 440)  u[r, c] 位于 rho(r, c) 与 rho(r, c+1) 之间
    v: (T, s, 399, 441)  v[r, c] 位于 rho(r, c) 与 rho(r+1, c) 之间
- rho 共定位 stencil：对相邻两个面元取 NaN 感知均值；边界单侧复制：
    u_rho[r, c] = mean_valid(u[r, c-1], u[r, c])   (c = 1..439)
    u_rho[r, 0] = u[r, 0];  u_rho[r, 440] = u[r, 439]
    v_rho[r, c] = mean_valid(v[r-1, c], v[r, c])   (r = 1..398)
    v_rho[0, c] = v[0, c];  v_rho[399, c] = v[398, c]
- mask 是权威有效性来源。双变量掩膜用与场相同的 stencil 从 mask_u/mask_v 推导
  （rho 点有效当且仅当相邻两个面元中至少一个有效，边界单侧复制），因此对齐后的
  NaN 图案与 mask==0 严格相等：
    mask_u_rho.npy : (400, 441)  u_rho 的有效性
    mask_v_rho.npy : (400, 441)  v_rho 的有效性
    mask_uv.npy    : mask_u_rho & mask_v_rho & mask_rho（仅为兼容保留）
- mask 策略（所给 mask 即权威）：
    * mask==1（海洋格）处出现 NaN 属于动态缺测数据：在首个 (t, s, r, c) 硬失败，
      逐日逐层检查，绝不静默掩除；
    * mask==0（陆地格，如本数据集 45 个静态陆地边界 u 面元）处出现数值属于丢弃值：
      在共定位前置为 NaN 并计数，结束时按变量报告总量。
  强制执行后，对齐场的 NaN 图案 == mask==0（首个 chunk 上断言）。mask 的
  shape/取值、场的 dtype 与输入 shape 均有断言。
- ocean_time 必须恰好为 T 个严格递增、间隔精确 24 h 的时间戳（verify_daily_time）。
- 原始与对齐 u/v 的极值连同其 (day, layer, row, col) 位置和数值一起记录；
  极值不自动当作离群值处理。
- 输出（陆地保持 NaN，float32）写入 <DST>：u_rho.npy/v_rho.npy 为
  (10591, 30, 400, 441)；三个掩膜为 uint8；ocean_time.npy 为 (10591,)
  datetime64[D] 日期视图，来自原始 NetCDF 的权威 ocean_time 元数据并经
  "严格递增 + 精确 24 h" 验证；ocean_time_seconds.npy 为 (10591,)
  datetime64[s] 精确经验证时间（验证前绝不降精度到日）。open_memmap 以 "w+"
  打开，已存在的对齐输出会被截断重写。
- 分块流水线在单块 CUDA GPU 上运行（逻辑索引 cuda:0，遵守
  CUDA_VISIBLE_DEVICES；无 CPU 回退路径）。每个 chunk 按序执行：
    mmap read (CPU) -> H2D -> 原始极值 -> 权威 mask 强制 -> NaN 感知共定位 ->
    首 chunk NaN 图案断言 -> 对齐极值 -> D2H -> memmap write -> flush
  只有标量（数值, 首个展平索引）摘要回传 CPU 供极值追踪器使用，整块数据绝不在
  CPU 上扫描。本文件中的 NumPy helper（colocate、u_rho_mask、v_rho_mask、
  enforce_land_mask、ExtremumTracker.update）只保留作单元测试参照与差分基线，
  主流水线一律走 torch_* CUDA 等价实现。
- 所有 GPU 计算均为 float32（不使用 AMP）；CUDA 显存在 chunk 间复用，mask
  只上传一次。

依赖关系：numpy / netCDF4 / torch；被 scripts/profile_preprocess_align_uv.py
以模块方式复用（main 仅在 __main__ 下执行，import 无副作用）；输出是
pre_dataset.py 的唯一数据源。
"""
import os
import time
import numpy as np
import netCDF4
import torch

SRC_CANDIDATES = [
    "/data2/user/zyq/datasets/PRE/processed",
    "/data/PRE_ocean_data/processed",
]
RAW_DYN_CANDIDATES = [
    "/data2/user/zyq/datasets/PRE/raw/dyn",
    "/data/PRE_ocean_data/raw/dyn",
]
DST = "/data2/user/zyq/data_processed/PRE/aligned"
CHUNK = 50  # 每 chunk 天数
DEVICE_INDEX = 0  # 逻辑 CUDA 索引；遵守 CUDA_VISIBLE_DEVICES

# T=总天数，S=sigma 层数，(H, W)=rho 网格尺寸
T, S, H, W = 10591, 30, 400, 441


def colocate(a, b):
    """两个同 shape float32 数组的 NaN 感知均值（单元测试参照；主流水线走 torch_colocate）。

    逐元素统计有效计数：只有一侧有效时直接取该值，两侧均 NaN 才输出 NaN。
    """
    na = ~np.isnan(a)
    nb = ~np.isnan(b)
    cnt = na.astype(np.float32) + nb.astype(np.float32)
    s = np.where(na, a, np.float32(0.0)) + np.where(nb, b, np.float32(0.0))
    out = np.full(a.shape, np.nan, np.float32)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


def u_rho_mask(mask_u):
    """(R, C-1) 的 u 面元掩膜 -> (R, C) 的 rho 掩膜；stencil 与 u 共定位一致（内部取 OR，边界单侧复制）。"""
    R, C = mask_u.shape
    out = np.empty((R, C + 1), np.bool_)
    out[:, 1:C] = mask_u[:, :-1] | mask_u[:, 1:]
    out[:, 0] = mask_u[:, 0]
    out[:, C] = mask_u[:, -1]
    return out


def v_rho_mask(mask_v):
    """(R-1, C) 的 v 面元掩膜 -> (R, C) 的 rho 掩膜；stencil 与 v 共定位一致（内部取 OR，边界单侧复制）。"""
    R, C = mask_v.shape
    out = np.empty((R + 1, C), np.bool_)
    out[1:R, :] = mask_v[:-1, :] | mask_v[1:, :]
    out[0, :] = mask_v[0, :]
    out[R, :] = mask_v[-1, :]
    return out


class ExtremumTracker:
    """流式数组的全局极值追踪器：记录全局 min/max 及其首次出现的全局位置。

    每个 chunk 的 shape 为 (t_len, S, R, C)，t0 为该 chunk 的起始绝对日；极值
    位置换算为全局 (t, s, r, c)。NaN（陆地）格被忽略。update() 接收 NumPy
    chunk（参照路径）；update_summary() 接收 GPU 上算好的标量（数值, 首个
    展平索引）摘要，整块数据因此无需回传 CPU。
    """

    def __init__(self, name):
        self.name = name
        self.min_val = np.inf
        self.max_val = -np.inf
        self.min_loc = None  # 首次出现的全局 (t, s, r, c)
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

    def update_summary(self, min_value, min_flat_index,
                       max_value, max_flat_index, shape, t0):
        """GPU 路径：把一个 chunk 的标量极值并入全局追踪器。

        min_flat_index/max_flat_index 是该 chunk 自身展平布局（C 序）中的索引，
        并列时取首次出现（与 NumPy 一致）；shape 为 chunk 形状 (t_len, S, R, C)，
        用于换算全局 (t, s, r, c)。
        """
        if min_value < self.min_val:
            self.min_val = float(min_value)
            self.min_loc = self._loc(int(min_flat_index), shape, t0)
        if max_value > self.max_val:
            self.max_val = float(max_value)
            self.max_loc = self._loc(int(max_flat_index), shape, t0)

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
    """在原始 chunk 上就地强制（权威）陆地掩膜；返回 arr。

    两种不匹配方向的处理不同：
      * mask==1（海洋格）处为 NaN：动态缺测数据——在首个 (t, s, r, c) 硬失败，
        绝不静默掩除；
      * mask==0（陆地格）处有数值（如静态边界/河道 u 面元）：丢弃（置 NaN）并
        计入 `discarded[name]`；对齐输出保持 NaN == (mask == 0)。

    `arr` 必须是已在内存中的 chunk (t, s, R, C)，不能是 mmap 切片（本函数就地
    修改）；`discarded` 是按变量累计丢弃数的 dict。
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


def torch_extrema_summary(arr):
    """chunk 上 nanmin/nanargmin/nanmax/nanargmax 的 CUDA 等价实现。

    返回 (min_value, min_flat_index, max_value, max_flat_index)；展平索引是该
    chunk 自身展平布局（C 序）中的位置，与 NumPy 一样并列时取首次出现。整块
    全 NaN 时抛 ValueError，与 np.nanmin/np.nanmax 一致。只有这四个标量回传
    CPU。
    """
    flat = arr.reshape(-1)
    nan = torch.isnan(flat)
    if bool(nan.all().item()):
        raise ValueError("All-NaN slice encountered")
    min_input = torch.where(nan, torch.full_like(flat, float("inf")), flat)
    max_input = torch.where(nan, torch.full_like(flat, float("-inf")), flat)
    min_value, min_index = torch.min(min_input, dim=0)
    max_value, max_index = torch.max(max_input, dim=0)
    return (float(min_value.item()), int(min_index.item()),
            float(max_value.item()), int(max_index.item()))


def torch_enforce_land_mask(arr, mask, name, t0, discarded):
    """enforce_land_mask 的 GPU 等价实现：策略与报错信息完全一致。

    `arr` 是 (t, s, R, C) 的 CUDA chunk，就地修改；`mask` 是 (R, C) 的布尔 GPU
    掩膜。返回 arr。
    """
    nan = torch.isnan(arr)
    ocean = mask != 0
    missing = nan & ocean[None, None]
    if bool(missing.any().item()):
        # torch.argmax 不支持 Bool 张量；nonzero 按 C 序返回首个 True，
        # 与 NumPy 侧 np.argmax 的取首语义一致。
        index = int(torch.nonzero(missing.reshape(-1))[0].item())
        t_len, s, r, c = arr.shape
        t, rem = divmod(index, s * r * c)
        s_, rem = divmod(rem, r * c)
        r_, c_ = divmod(rem, c)
        day = t0 + t
        raise RuntimeError(
            f"{name} dynamic missing data at (t={day}, s={s_}, r={r_}, c={c_}): "
            f"field is NaN but mask==1 (ocean); not masked away.")
    stray = ~nan & ~ocean[None, None]
    n_stray = int(stray.sum().item())
    if n_stray:
        arr.masked_fill_(stray, float("nan"))
        discarded[name] = discarded.get(name, 0) + n_stray
    return arr


def torch_colocate(a, b):
    """colocate 的 CUDA 等价实现：NaN 感知均值；两侧均 NaN 才输出 NaN。"""
    na = ~torch.isnan(a)
    nb = ~torch.isnan(b)
    cnt = na.to(torch.float32) + nb.to(torch.float32)
    s = torch.where(na, a, torch.zeros_like(a)) + torch.where(nb, b, torch.zeros_like(b))
    out = torch.full_like(a, float("nan"))
    valid = cnt > 0
    out[valid] = s[valid] / cnt[valid]
    return out


def torch_colocate_u(uc):
    """GPU：u chunk (t, s, r, c) -> rho u (t, s, r, c+1)（输出 shape 由输入推导）。"""
    t, s, r, c = uc.shape
    ub = torch.empty((t, s, r, c + 1), dtype=uc.dtype, device=uc.device)
    ub[:, :, :, 1:c] = torch_colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
    ub[:, :, :, 0] = uc[:, :, :, 0]
    ub[:, :, :, c] = uc[:, :, :, -1]
    return ub


def torch_colocate_v(vc):
    """GPU：v chunk (t, s, r, c) -> rho v (t, s, r+1, c)（输出 shape 由输入推导）。"""
    t, s, r, c = vc.shape
    vb = torch.empty((t, s, r + 1, c), dtype=vc.dtype, device=vc.device)
    vb[:, :, 1:r, :] = torch_colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
    vb[:, :, 0, :] = vc[:, :, 0, :]
    vb[:, :, r, :] = vc[:, :, -1, :]
    return vb


def verify_daily_time(times):
    """校验 `times` 是间隔恰好为一日的一维 datetime64 数组，否则硬失败。

    相邻时间戳必须精确相差 24 h（在 datetime64[s] 精度上检查，23/25 小时的
    间隔都会失败）；成功时原样返回 `times`。报错信息包含失败索引、相邻两个
    时间戳与实际间隔。
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
    """读取每个原始 NetCDF 的权威 ocean_time 并缓存到 DST。

    校验恰好 T 个时间戳、严格递增、间隔精确 24 h。原始时间保持 datetime64[s]
    精度（验证之前绝不降为日）；日期视图单独保存：
        ocean_time.npy         : (T,) datetime64[D]  日期视图（兼容）
        ocean_time_seconds.npy : (T,) datetime64[s]  精确且经验证的时间
    返回精确的 (T,) datetime64[s] 数组。

    副作用：向 DST 写入上述两个 .npy 文件。
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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available (torch.cuda.is_available() is False); the PRE "
            "alignment pipeline requires a GPU and has NO CPU fallback. Check the "
            "conda env and CUDA_VISIBLE_DEVICES.")
    torch.cuda.set_device(DEVICE_INDEX)
    device = torch.device("cuda", DEVICE_INDEX)
    print(f"[gpu] {torch.cuda.get_device_name(device)} "
          f"(logical device {DEVICE_INDEX})", flush=True)

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
    # 落盘 mask 存的是 float64 {0., 1.}；下面的 stencil 用到 | 与 &，必须先转 bool
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

    # 在流水线开始前对第 0 天第 0 层做早期探针：动态缺测尽早硬失败，并预览
    # 陆地格丢弃数。探针作用在一次性拷贝上，其计数不会并入最终按变量统计的
    # 丢弃总量（总量只在主循环中累计）。
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

    # 权威 mask 只上传 GPU 一次（bool 张量），供之后每个 chunk 复用
    gpu_mask_u = torch.as_tensor(mask_u, dtype=torch.bool, device=device)
    gpu_mask_v = torch.as_tensor(mask_v, dtype=torch.bool, device=device)
    gpu_mask_u_rho = torch.as_tensor(m_u_rho, dtype=torch.bool, device=device)
    gpu_mask_v_rho = torch.as_tensor(m_v_rho, dtype=torch.bool, device=device)

    t0 = time.time()
    n_chunks = (T + CHUNK - 1) // CHUNK
    with torch.inference_mode():
        for ci, ts in enumerate(range(0, T, CHUNK)):
            te = min(ts + CHUNK, T)
            # mmap 读（CPU）-> H2D：np.array 把只读 mmap 切片物化为 CPU 副本，
            # from_numpy 共享该副本，.to(device) 完成 H2D。uc: (t,s,400,440)，vc: (t,s,399,441)
            uc = torch.from_numpy(np.array(u[ts:te])).to(device)
            vc = torch.from_numpy(np.array(v[ts:te])).to(device)

            # 原始极值在 mask 强制之前记录，描述的是原始 chunk；随后 mask 强制
            # 对海洋格动态缺测硬失败、把陆地格数值置 NaN，使共定位看到掩膜后的场。
            mn, mi, mx, xi = torch_extrema_summary(uc)
            tr_u.update_summary(mn, mi, mx, xi, uc.shape, ts)
            mn, mi, mx, xi = torch_extrema_summary(vc)
            tr_v.update_summary(mn, mi, mx, xi, vc.shape, ts)

            torch_enforce_land_mask(uc, gpu_mask_u, "u", ts, discarded)
            torch_enforce_land_mask(vc, gpu_mask_v, "v", ts, discarded)

            ub = torch_colocate_u(uc)
            vb = torch_colocate_v(vc)

            # 首 chunk 自检：对齐 NaN 图案与 rho 掩膜严格相等（掩膜权威性的断言）
            if ci == 0:
                assert (torch.isnan(ub) == (gpu_mask_u_rho == 0)[None, None]).all().item(), \
                    "u_rho NaN pattern does not match mask_u_rho"
                assert (torch.isnan(vb) == (gpu_mask_v_rho == 0)[None, None]).all().item(), \
                    "v_rho NaN pattern does not match mask_v_rho"

            mn, mi, mx, xi = torch_extrema_summary(ub)
            tr_ur.update_summary(mn, mi, mx, xi, ub.shape, ts)
            mn, mi, mx, xi = torch_extrema_summary(vb)
            tr_vr.update_summary(mn, mi, mx, xi, vb.shape, ts)

            # D2H -> memmap write -> flush：.cpu() 产生 CPU float32 副本并完成
            # D2H 拷贝，写入磁盘 memmap 后 flush 落盘（输出保持 float32）
            ub_cpu = ub.cpu().numpy()
            vb_cpu = vb.cpu().numpy()
            u_out[ts:te] = ub_cpu
            v_out[ts:te] = vb_cpu
            u_out.flush()
            v_out.flush()

            dt = time.time() - t0
            eta = dt / (ci + 1) * (n_chunks - ci - 1)
            print(f"[{ci + 1}/{n_chunks}] days {ts}:{te}  elapsed {dt / 60:.1f} min  "
                  f"ETA {eta / 60:.1f} min", flush=True)
            del uc, vc, ub, vb, ub_cpu, vb_cpu

    print(f"[gpu] peak CUDA memory: allocated "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB, reserved "
          f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB", flush=True)
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