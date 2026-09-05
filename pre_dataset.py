#!/usr/bin/env python3
"""模块职责：PRE 海流数据集与归一化：提供滑窗数据集 PREUVDataset（7 日条件窗 ->
目标，绝对日索引，[0,1] min-max 归一化）、train-only 归一化统计缓存
（compute_or_load_stats）、未裁剪原生 staggered 真值读取（NativeUVReader）与
rho 网格双变量掩膜张量（build_mask_tensor）。

不负责：模型/损失/采样（pre_models.py、diffusion.py）、训练与评估流程
（pre_trainer.py、pre_evaluate.py）、指标计算（pre_metrics.py）；不做
east/north 旋转（数据自预处理起就保持 Plan A 的 xi/eta 分量语义）。

关键约束：
- 数据源（由 scripts/preprocess_align_uv.py 的 Plan A 共定位产出）：
    <ALIGNED_DIR>/u_rho.npy   (10591, 30, 400, 441) float32，land=NaN
    <ALIGNED_DIR>/v_rho.npy   同 shape
    <ALIGNED_DIR>/mask_u_rho.npy (400, 441) uint8  u_rho 有效性
    <ALIGNED_DIR>/mask_v_rho.npy (400, 441) uint8  v_rho 有效性
    <ALIGNED_DIR>/mask_uv.npy     (400, 441) uint8  交集（仅为兼容保留）
    <ALIGNED_DIR>/ocean_time.npy         (10591,) datetime64[D] 日期视图（兼容）
    <ALIGNED_DIR>/ocean_time_seconds.npy (10591,) datetime64[s] 精确经验证时间
- 时间切分按绝对日索引连续（绝不用 random_split；重叠窗口不得跨 split 泄漏）：
    train: 日 [0, 8401)      1994-01-01 .. 2016-12-31
    val:   日 [8401, 9496)   2017-01-01 .. 2019-12-31
    test:  日 [9496, 10591)  2020-01-01 .. 2022-12-30
- 样本布局（一个 split 内连续 context+horizon 天的窗口）：
    cond:   (2*context, H, W, Z)  channel-first、day-major 交错：
                                  ch 2k   = 第 (start+k) 天的 u
                                  ch 2k+1 = 第 (start+k) 天的 v
    target: (horizon, 2, H, W, Z) target[:, 0] = u，target[:, 1] = v
    Z = 1（depth_index 为 int，如 29 = 表层）或 30（全部 sigma 层）。
- 归一化（仅 train split 的海洋点，落盘缓存）：逐变量 min-max 到 [0, 1]
  （u 用 mask_u_rho，v 用 mask_v_rho）。可选百分位裁剪（clip_pct，如 0.1）默认
  禁用（clip_pct=None），必须显式配置；缓存名与内容记录裁剪策略、split 边界与
  mask 哈希（'splits' 字段缺失或变化同样判缓存 stale）；hi <= lo 直接报错。
  land/NaN 在归一化之后填 0（loss/指标必须改用 mask 区分）。
  sigma 为裁剪后 [0,1] 归一化值在 u+v 两变量合并上的真实 pooled std（取值先按
  逐变量范围 clip 再归一化，与 dataset 归一化完全一致；u/v 合并意味着两组均值
  差通过组间项进入方差）——用作 EDM 的 sigma_data。
- 评估使用 NativeUVReader 读取未裁剪的原始原生 staggered 真值；归一化 target
  绝不通过反归一化来冒充原始真值。

依赖关系：numpy / torch（torch.utils.data.Dataset）；数据由
scripts/preprocess_align_uv.py 产出；被 pre_trainer.py / pre_evaluate.py 与
诊断脚本 import（本模块无 import 副作用）。
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

# 连续时间切分；半开日索引区间 [lo, hi)
SPLITS = {
    "train": (0, 8401),
    "val": (8401, 9496),
    "test": (9496, 10591),
}

CONTEXT = 7


def native_dir():
    """返回第一个存在的原生 u/v 目录；都不存在则报错（评估需要未裁剪的原始场）。"""
    for p in NATIVE_DIR_CANDIDATES:
        if os.path.isdir(p):
            return p
    raise RuntimeError(f"raw native u/v dir not found (tried {NATIVE_DIR_CANDIDATES}); "
                       f"evaluation needs the unclipped original fields")


# 掩膜

def load_masks():
    """rho 网格双变量有效性掩膜：返回 (mask_u_rho, mask_v_rho)，各为 (H, W) bool 数组。"""
    mu = np.load(os.path.join(ALIGNED_DIR, "mask_u_rho.npy")).astype(bool)
    mv = np.load(os.path.join(ALIGNED_DIR, "mask_v_rho.npy")).astype(bool)
    assert mu.shape == (H, W) and mv.shape == (H, W)
    return mu, mv


def mask_version():
    """双变量掩膜文件的可验证标识符：两个 .npy 文件字节流 SHA-256 的前 16 位十六进制。"""
    h = hashlib.sha256()
    for name in ("mask_u_rho.npy", "mask_v_rho.npy"):
        with open(os.path.join(ALIGNED_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def native_masks():
    """原生 staggered 网格上的 (mask_u, mask_v) bool 数组。

    mask_u: (H, W-1)，mask_v: (H-1, W)——u.npy/v.npy 的原始陆地掩膜。
    """
    stat = os.path.join(os.path.dirname(native_dir()), "stat_var")
    mu = np.load(os.path.join(stat, "mask_u.npy")).astype(bool)
    mv = np.load(os.path.join(stat, "mask_v.npy")).astype(bool)
    assert mu.shape == (H, W - 1), mu.shape
    assert mv.shape == (H - 1, W), mv.shape
    return mu, mv


# 时间

def load_ocean_time():
    """(T_TOTAL,) datetime64[D] 日期数组；内容已经预处理步骤逐日校验。"""
    ts = np.load(os.path.join(ALIGNED_DIR, "ocean_time.npy"))
    assert ts.shape == (T_TOTAL,), f"ocean_time shape {ts.shape}"
    return ts


# 统计（train split 专用归一化统计）

def _iter_ocean_values(arr, lo, hi, depth_index, mask, chunk):
    """按 chunk 流式产出日区间 [lo, hi) 内海洋点的一维值数组。

    np.asarray 把 memmap 切片转为基础 ndarray 视图（与 mmap 页共享内存，不做
    立即拷贝），随后布尔掩膜索引产生 RAM 副本，最后 reshape(-1) 为一维流：
    全层 (t, s, H, W) -> (t, s, n_ocean)；单层 (t, H, W) -> (t, n_ocean)。
    """
    for ts in range(lo, hi, chunk):
        te = min(ts + chunk, hi)
        if depth_index is None:
            a = np.asarray(arr[ts:te])
            a = a[:, :, mask]
        else:
            a = np.asarray(arr[ts:te, depth_index])
            a = a[:, mask]
        yield a.reshape(-1)


def _clip_range(vals_iter_factory, clip_pct):
    """对值流求精确 (min, max)；clip_pct 设置时改为百分位裁剪范围。

    vals_iter_factory 必须是零参 callable 且每次调用返回一个全新迭代器：
    值流被消耗两遍（第一遍求 min/max，第二遍构建直方图）。
    """
    mn, mx = np.inf, -np.inf
    for vals in vals_iter_factory():
        mn = min(mn, float(vals.min()))
        mx = max(mx, float(vals.max()))
    if clip_pct is None:
        return mn, mx
    # 第二遍流式扫描：4096 个等宽 bin 的直方图 -> 累积 CDF -> np.interp 线性
    # 插值，取分位数所在的 bin 左边界作为近似 lo/hi
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
    """从 TRAIN split 的海洋点计算逐变量裁剪范围与 pooled sigma。

    返回 dict：lo/hi（shape [2] 的 float32 数组，顺序 u, v）、sigma（float）。
    缓存于 NORM_DIR/stats_d{all|idx}_clip{none|pct}.npz；缓存记录 clip_pct、
    深度 preset、split 边界与 mask 版本，任一变化即自动重算。删除缓存文件可
    强制重算。

    副作用：首次计算时向 NORM_DIR 写入缓存 .npz。
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

    # pooled sigma 在 u+v 两变量合并的值集上计算（等价于拼接后求 std）：pooled
    # 均值进入方差，因此 u/v 两组的均值差通过组间项贡献。取值先按逐变量范围
    # clip 再 min-max 归一化，然后才 pooled——与 dataset 的归一化完全一致。
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


# 数据集

class PREUVDataset(Dataset):
    """单一连续时间 split 上的滑窗数据集（归一化 rho 网格数据）。

    __getitem__ -> (cond, target, start_day)：
        cond:   (2*context, H, W, Z) float32，归一化到 [0,1]，land=0
        target: (horizon, 2, H, W, Z) float32，归一化到 [0,1]，land=0
        start_day: int，首个条件帧的绝对日索引
    """

    def __init__(self, split, stats, context=CONTEXT, horizon=1,
                 depth_index=None, stride=1, max_windows=None):
        """horizon > 1 服务于分离式多步训练（pre_config 的 train_horizon /
        lead_for_batch）：窗口绝不跨 split 边界（下方的 last_start 扣除了
        context+horizon），且 target[:, J-1] 对应绝对日 start+context+J-1
        （0 基），训练器由此直接索引被选中的训练 lead。评估仍用
        horizon=ROLLOUT_DAYS 的窗口做自回归 rollout。
        """
        assert split in SPLITS
        self.context = context
        self.horizon = horizon
        self.depth_index = depth_index
        self.lo_stats = stats["lo"]  # shape [2] float32，顺序 (u, v) 的归一化下界
        self.hi_stats = stats["hi"]

        load_ocean_time()  # 校验过的时间文件缺失/非法时立即失败，早于任何数据读取
        self.u = np.load(os.path.join(ALIGNED_DIR, "u_rho.npy"), mmap_mode="r")
        self.v = np.load(os.path.join(ALIGNED_DIR, "v_rho.npy"), mmap_mode="r")

        lo, hi = SPLITS[split]
        last_start = hi - (context + horizon)  # 含端点；保证窗口完整落在 split 内
        assert last_start >= lo, f"split {split} too short for context+horizon"
        starts = np.arange(lo, last_start + 1, stride)
        if max_windows is not None:
            starts = starts[:max_windows]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def _load_var(self, arr, i, k):
        """取日区间 [i, i+k) 的原始物理场并统一为 (k, H, W, Z) float32。

        NaN 保留、未裁剪未归一化；裁剪/归一化/NaN->0 由 _norm 负责。
        """
        if self.depth_index is None:
            # np.asarray 把 memmap 切片转为基础 ndarray 视图（共享 mmap 页），
            # transpose 仍为 view；真正的物化拷贝发生在 _norm 的 clip/astype
            # 全层 (k, s, H, W) -> transpose(0, 2, 3, 1) -> (k, H, W, Z=s)
            a = np.asarray(arr[i:i + k])
            a = np.transpose(a, (0, 2, 3, 1))
        else:
            # 单层 (k, H, W) -> 末尾补出 Z=1 轴 -> (k, H, W, 1)（均为 view）
            a = np.asarray(arr[i:i + k, self.depth_index])
            a = a[..., None]
        return a

    def _norm(self, a, j):
        """clip 到 [lo, hi] 后归一化到 [0,1]，NaN（land）填 0；j=0/1 选 u/v 的统计。

        clip 与 astype 各产生一个新数组，是 mmap 数据真正物化进 RAM 的位置。
        """
        lo, hi = float(self.lo_stats[j]), float(self.hi_stats[j])
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo)
        return np.nan_to_num(a, nan=0.0).astype(np.float32)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        L = self.context + self.horizon
        # u/v 各自 clip+归一化到 [0,1]（land 填 0）：(L,H,W,Z)，L = context + horizon 天
        u = self._norm(self._load_var(self.u, i, L), 0)
        v = self._norm(self._load_var(self.v, i, L), 1)

        # day-major 交错展平：stack 得 (L, 2, H, W, Z)，按 (日, 变量) 行主序展平成
        # [u0, v0, u1, v1, ...]，即 ch 2k = 第 k 天的 u、ch 2k+1 = 第 k 天的 v
        uv = np.stack([u, v], axis=1)
        cond = uv[:self.context].reshape(2 * self.context, *uv.shape[2:])   # 条件窗 (2*context, H, W, Z)
        target = uv[self.context:]                                          # 目标 (horizon, 2, H, W, Z)

        # ascontiguousarray 兜底保证可写且 C 连续，from_numpy 才能共享内存零拷贝
        return (torch.from_numpy(np.ascontiguousarray(cond)),
                torch.from_numpy(np.ascontiguousarray(target)),
                i)


class NativeUVReader:
    """未裁剪的原始原生 staggered u/v 真值（仅评估使用）。

    u: (T_TOTAL, S_TOTAL, H, W-1)，v: (T_TOTAL, S_TOTAL, H-1, W)——原始
    processed 场，land=NaN，从不归一化或裁剪。在这里读取原始物理真值是计算
    正式指标的唯一许可途径。

    get(day, days=1) -> (u_sel, v_sel)，统一 (days, H, W, Z) 布局（sigma 轴
    移到末尾，与模型网格一致）：
        depth_index=None: u_sel (days, H, W-1, Z=S_TOTAL)，v_sel (days, H-1, W, Z=S_TOTAL)
        depth_index=int:  u_sel (days, H, W-1, 1)，v_sel (days, H-1, W, 1)
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
        # np.asarray 把 memmap 切片转为基础 ndarray 视图（与 mmap 页共享内存，
        # 无立即拷贝）：
        # 全层 (days, S, H, W-1)/(days, S, H-1, W)；单层 (days, H, W-1)/(days, H-1, W)
        sl = (slice(day, day + days), self.depth_index if self.depth_index is not None else slice(None))
        u = np.asarray(self.u[sl])
        v = np.asarray(self.v[sl])
        if self.depth_index is not None:
            u = u[..., None]                       # 单层补 Z=1 轴：(days, H, W-1, 1)
            v = v[..., None]                       # 同上：(days, H-1, W, 1)
        else:
            # moveaxis 把 sigma 轴移到末尾以统一布局（返回 view，非连续，仅读取）
            u = np.moveaxis(u, 1, -1)              # sigma 移到末尾：(days, H, W-1, S)
            v = np.moveaxis(v, 1, -1)              # 同上：(days, H-1, W, S)
        return u, v


def build_mask_tensor(device, depth_index=None):
    """(1, 2, H, W, Z) 双变量掩膜张量，可直接对 (B, 2, H, W, Z) 广播。

    channel 0 = mask_u_rho，channel 1 = mask_v_rho；实际 dtype 为 torch.bool
    （0/1，海洋=1），与 float 条件拼接或相乘时依赖 PyTorch 类型提升转成
    0.0/1.0。
    """
    mu, mv = load_masks()
    z = S_TOTAL if depth_index is None else 1
    # 布局链：stack 产生新数组，其后插入空 batch/Z 轴、broadcast_to、transpose
    # 均为 view：(2, H, W) -> (2, 1, H, W, 1) -> broadcast_to (2, 1, H, W, z)
    # -> transpose 交换 batch/变量轴得 (1, 2, H, W, Z)
    m = np.stack([mu, mv])[:, None, :, :, None]           # bool (2, 1, H, W, 1)，变量轴在前
    m = np.broadcast_to(m, (2, 1, H, W, z)).transpose(1, 0, 2, 3, 4)  # 轴交换后 (1, 2, H, W, Z)
    # broadcast_to 返回只读 view；强制复制为可写 C 连续数组，torch.from_numpy()
    # 才不会告警、原地操作也不会失败。
    m = np.array(m, copy=True, order="C")
    return torch.from_numpy(m).to(device)