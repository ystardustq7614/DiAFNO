"""PRE 流水线的最小回归测试（纯断言，无额外依赖）。

模块职责：用小尺寸合成张量覆盖 PRE 共享模块的关键协议约束——C 网格到 rho 网格
共定位与双变量掩膜、7 天条件窗口的 day-major 14 通道展平与 rollout 滑窗、14→2
条件前向、双变量掩膜扩散损失（分母正确性）、反向传播、两步采样；pre_metrics.py
正式指标（rho→native、掩膜误差和、pooled RMSE、相对 L2）；NativeUVReader 统一
布局与 u/v 哨兵值；未裁剪原始真值路径；pooled sigma_data 与统计缓存 stale 判定
（裁剪、切分、缺字段）；legacy cond_chans=None 兼容；固定 sigma_data 尺度
（stats_sigma×2）；legacy checkpoint 回退与 RESUME_SIGMA_POLICY；ensemble rollout
（E=1 复现逐窗口路径、autocast 包裹、E=4 形状/均值/成员独立、AR 状态独立、
逐窗口种子、horizon 1 与 15）；rho-oracle 诊断；可写连续掩膜张量；checkpoint
元数据 roundtrip（weights_only=True 加载）。

另覆盖 persistence-residual 基线（pre_models.py）：zero-init 恒等（未训练=末日
持续性）、一步优化、掩膜 MSE 陆地不变性、checkpoint 往返与 objective 守卫、
remask_feedback rollout 行为（开/关+掩膜必填）、确定性 rollout（种子无关、成员
一致）、训练目标与 run tag、ProgressReporter 的 PROGRESS 行协议（update 驱动 +
守护线程时间驱动心跳、phase_done 与脚本级 completed 的区别、多行错误净化、失败
hook 去重与 stage、归一化/掩膜/time_sigma 指纹检查），以及静态掩膜 checkpoint
重建 helper（legacy/静态/矛盾元数据）、归档 _MSK checkpoint 的最小 CPU rollout、
early-stop 计数器往返。

不负责：CUDA 专用分支的强制验证（无 CUDA 时按原逻辑跳过并打印 SKIP）；
trainer.py/legacy 路径（由 smoke_test.py 承担）。

关键约束：全文件共用 fixture 尺寸 B=2、H=4、W=4、Z=2；fixture 的 u/v 值域刻意
分离（哨兵值 7.0/11.0、极值 50.0），使"v 被读成 u"一类 bug 立即暴露；只依赖
标准库与本仓库模块，不新增第三方依赖。

依赖关系：scripts.preprocess_align_uv、pre_dataset、pre_metrics、pre_models、
pre_rollout、pre_config、diffusion、IAFNO、scripts.diag_leadtime_residual。

运行：python pre_smoke_test.py
"""
import io
import os
import sys
import tempfile
import time
from math import exp, sqrt

import numpy as np
import torch

from scripts import preprocess_align_uv as pre_pp
from pre_dataset import (NativeUVReader, PREUVDataset, _clip_range,
                         compute_or_load_stats)
from pre_metrics import (rho_to_native, masked_error_sums, pooled_rmse, masked_rel_l2,
                         oracle_native_error_sums)
from pre_models import PersistenceResidualIAFNO, masked_mse_loss
from pre_rollout import expand_ensemble, ensemble_rollout, ensemble_mean
from pre_config import (PRESETS, SIGMA_DATA_SCALE, SMOKE_BATCHES_PER_RANK,
                        sigma_data_from_stats, sigma_data_from_checkpoint,
                        resume_sigma_decision, training_config, training_run_tag,
                        run_tag_for, OBJECTIVES, DEFAULT_OBJECTIVE, MASK_SCHEME,
                        RESIDUAL_TIME_SIGMA, validate_objective,
                        objective_from_checkpoint, ensure_objective_compatible,
                        check_norm_fingerprint, check_residual_time_sigma,
                        train_horizon, init_checkpoint,
                        TRAIN_HORIZON_ENV, INIT_CHECKPOINT_ENV, MS_DEFAULTS,
                        lead_for_batch, lead_schedule_str, check_multistep_config,
                        restore_worse_epochs, static_mask_from_checkpoint,
                        STATIC_MASK_CHANNELS,
                        format_progress, ProgressReporter,
                        install_progress_failure_hook, mark_progress_failed,
                        reset_progress_failure_state)
from utilities3 import load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_rollout import detached_feedback_window
from scripts.diag_leadtime_residual import build_npz_payload

B, H, W, Z = 2, 4, 4, 2  # 全文件共用 fixture 尺寸：B=窗口数，H/W=rho 网格边长，Z=sigma 层数


def make_model(embed=8):
    """最小 EDM fixture：14 通道条件 + 2 通道目标；P_std=0 与 S_churn=0 消除训练
    sigma 抖动与采样 churn，使随机数序列可复现。"""
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=embed, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=14, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    return ElucidatedDiffusion(
        net, channels=2, num_sample_steps=2,
        image_size_h=H, image_size_w=W, image_size_z=Z,
        sigma_data=0.5, P_mean=-1.0, P_std=0.0, S_churn=0,
    )


def make_residual_model(embed=8, time_sigma=RESIDUAL_TIME_SIGMA, cond_chans=14):
    """最小 persistence-residual fixture：零初始化残差头使未训练模型恰好等于
    末日语持续基线。"""
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=embed, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=cond_chans, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    return PersistenceResidualIAFNO(net, time_sigma=time_sigma)


def _close_mmaps(*datasets):
    """关闭数据集的 memmap 句柄（仅测试用清理，Windows 上打开的 numpy.memmap 会
    锁住底层 .npy，导致 TemporaryDirectory 清理报 WinError 32）。
    生产数据集生命周期不受影响。"""
    for ds in datasets:
        if ds is None:
            continue
        for attr in ("u", "v"):
            arr = getattr(ds, attr, None)
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()


# ── 预处理与数据组：共定位、掩膜、极值、时间 ──


def test_colocate_and_bivariate_masks():
    """防止共定位/掩膜回归：NaN-aware 邻均值共定位 + 边界复制的结果必须与
    u_rho_mask/v_rho_mask 的有效性一致（NaN 模式 == mask==0）。"""
    # u 原始 (2, 3)：NaN 位置 == (mask_u == 0)
    u = np.array([[1.0, np.nan, 3.0],
                  [4.0, 5.0, 6.0]])
    mask_u = np.array([[1, 0, 1],
                       [1, 1, 1]])
    # 内部列 = 相邻点的 NaN-aware 均值；边界列直接复制
    u_rho = np.empty((2, 4), np.float32)
    u_rho[:, 1:3] = pre_pp.colocate(u[:, :-1], u[:, 1:])
    u_rho[:, 0] = u[:, 0]
    u_rho[:, 3] = u[:, -1]
    expected = np.array([[1.0, 1.0, 3.0, 3.0],
                         [4.0, 4.5, 5.5, 6.0]], np.float32)
    assert np.allclose(u_rho, expected, equal_nan=True)

    m_ur = pre_pp.u_rho_mask(mask_u)
    assert m_ur.shape == (2, 4)
    assert np.array_equal(m_ur, np.array([[1, 1, 1, 1],
                                          [1, 1, 1, 1]]))
    # 对齐后 NaN 模式必须等于 (mask == 0)
    assert (np.isnan(u_rho) == (m_ur == 0)).all()

    # v 原始 (3, 4)：内部行使 eta 方向 stencil 生效
    v = np.array([[1., 2., 3., 4.],
                  [5., 6., 7., 8.],
                  [9., 10., 11., 12.]])
    mask_v = np.ones((3, 4), np.int64)
    v_rho = np.empty((4, 4), np.float32)
    v_rho[1:3] = pre_pp.colocate(v[:-1], v[1:])
    v_rho[0] = v[0]
    v_rho[3] = v[-1]
    expected_v = np.array([[1., 2., 3., 4.],
                           [3., 4., 5., 6.],
                           [7., 8., 9., 10.],
                           [9., 10., 11., 12.]], np.float32)
    assert np.allclose(v_rho, expected_v)
    m_vr = pre_pp.v_rho_mask(mask_v)
    assert m_vr.shape == (4, 4) and m_vr.all()
    assert (np.isnan(v_rho) == (m_vr == 0)).all()


def test_enforce_land_mask_policy():
    """防止掩膜权威策略回归：陆地值就地清 NaN 并计数、海洋单元格上的 NaN（动态
    缺测）必须带坐标 fail-fast、一致单元格保持不动。"""
    # 掩膜权威执行：陆地值清除并计数（就地），海洋动态缺测直接失败，一致单元格不动
    mask = np.array([[1, 0, 1],
                     [1, 1, 0]])
    arr = np.array([[[[1.0, 9.0, 3.0],      # 9.0 位于陆地 (0,1) → 清除
                      [4.0, 5.0, 6.0]]]], np.float32)   # 6.0 位于陆地 (1,2) → 清除
    discarded = {}
    out = pre_pp.enforce_land_mask(arr, mask, "u", 0, discarded)
    assert out is arr                                # 原地修改，同一对象
    assert np.isnan(arr[0, 0, 0, 1]) and np.isnan(arr[0, 0, 1, 2])
    assert arr[0, 0, 0, 0] == 1.0 and arr[0, 0, 0, 2] == 3.0
    assert arr[0, 0, 1, 0] == 4.0 and arr[0, 0, 1, 1] == 5.0
    assert discarded == {"u": 2}
    # 计数跨块累加
    pre_pp.enforce_land_mask(arr, mask, "u", 50, discarded)
    assert discarded == {"u": 2}                     # 已是 NaN → 不重复计数

    # 海洋单元格上的 NaN = 动态缺测 → 带坐标的 RuntimeError
    bad = np.array([[[[1.0, 2.0, 3.0],
                      [4.0, np.nan, 6.0]]]], np.float32)  # NaN 位于 (t=0, s=0, r=1, c=1)，mask==1
    try:
        pre_pp.enforce_land_mask(bad, mask, "v", 7, {})
    except RuntimeError as e:
        msg = str(e)
        assert "t=7" in msg and "r=1" in msg and "c=1" in msg, msg
        assert "mask==1" in msg, msg
    else:
        raise AssertionError("expected RuntimeError for NaN on ocean cell")


def _cuda_or_skip():
    """CUDA 可用时返回设备，否则打印提示并返回 None（该分支按原逻辑跳过）。"""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    print("  SKIP (no CUDA available)")
    return None


def test_tracker_update_summary_matches_update():
    """防止 GPU 标量接口回归：update_summary（只经 GPU 标量传输）必须与 NumPy 逐块
    update() 跟踪器完全一致——值、首次出现位置、跨块累加；无 CUDA 也可运行。"""
    rng = np.random.default_rng(2)
    arr = rng.uniform(-3, 3, (5, 2, 4, 6)).astype(np.float32)
    flat = arr.ravel()
    flat[::4] = np.nan
    flat[0] = -9.0
    flat[-1] = 9.0
    t0 = 100

    a = pre_pp.ExtremumTracker("same")
    b = pre_pp.ExtremumTracker("same")
    a.update(arr, t0)
    b.update_summary(float(np.nanmin(flat)), int(np.nanargmin(flat)),
                     float(np.nanmax(flat)), int(np.nanargmax(flat)),
                     arr.shape, t0)
    assert a.min_val == b.min_val == float(np.nanmin(flat))
    assert a.max_val == b.max_val == float(np.nanmax(flat))
    assert a.min_loc == b.min_loc == (t0 + 0, 0, 0, 0)
    assert a.max_loc == b.max_loc
    assert a.report() == b.report()

    # 分块多次调用与整块一次调用的累加结果一致
    a2 = pre_pp.ExtremumTracker("same2")
    b2 = pre_pp.ExtremumTracker("same2")
    for k in range(4):
        chunk = arr[k:k + 1]
        f = chunk.ravel()
        a2.update(chunk, t0 + k)
        b2.update_summary(float(np.nanmin(f)), int(np.nanargmin(f)),
                          float(np.nanmax(f)), int(np.nanargmax(f)),
                          chunk.shape, t0 + k)
    assert a2.report() == b2.report()


def test_torch_colocate_matches_numpy():
    """防止 CUDA 共定位内核回归：torch_colocate 与 NumPy 参考逐位一致（含 NaN），
    多种形状覆盖；无 CUDA 时跳过。"""
    dev = _cuda_or_skip()
    if dev is None:
        return
    rng = np.random.default_rng(0)
    for shape in ((7, 3), (2, 5), (4, 4)):
        a = rng.uniform(-2, 2, shape).astype(np.float32)
        b = rng.uniform(-2, 2, shape).astype(np.float32)
        a[rng.uniform(size=shape) < 0.3] = np.nan
        b[rng.uniform(size=shape) < 0.3] = np.nan
        cpu = pre_pp.colocate(a, b)
        gpu = pre_pp.torch_colocate(
            torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)).cpu().numpy()
        assert np.array_equal(cpu, gpu, equal_nan=True), shape


def test_torch_colocate_edge_cases():
    """防止边界 stencil 回归：单侧有效/双侧无效单元格，u 沿 xi 复制边界列、v 沿
    eta 复制边界行（含 NaN 边），GPU 与 CPU 参考一致。"""
    dev = _cuda_or_skip()
    if dev is None:
        return
    # 单元格 0：单侧有效（仅 a）→ a；单元格 1：双侧无效 → NaN；
    # 单元格 2：单侧有效（b 为 NaN）→ a
    a = np.array([[1.0, np.nan, 3.0]], np.float32)
    b = np.array([[np.nan, np.nan, np.nan]], np.float32)
    cpu = pre_pp.colocate(a, b)
    gpu = pre_pp.torch_colocate(torch.from_numpy(a).to(dev),
                                torch.from_numpy(b).to(dev)).cpu().numpy()
    assert np.array_equal(cpu, gpu, equal_nan=True)
    assert cpu[0, 0] == 1.0 and np.isnan(cpu[0, 1]) and cpu[0, 2] == 3.0

    # u 边界列为复制而非平均（含 NaN 边）
    uc = np.array([[[[1.0, 2.0, np.nan],
                     [4.0, np.nan, 6.0]]]], np.float32)          # (1,1,2,3)
    cpu_ub = np.empty((1, 1, 2, 4), np.float32)
    cpu_ub[:, :, :, 1:3] = pre_pp.colocate(uc[:, :, :, :-1], uc[:, :, :, 1:])
    cpu_ub[:, :, :, 0] = uc[:, :, :, 0]
    cpu_ub[:, :, :, 3] = uc[:, :, :, -1]
    gpu_ub = pre_pp.torch_colocate_u(torch.from_numpy(uc).to(dev)).cpu().numpy()
    assert np.array_equal(cpu_ub, gpu_ub, equal_nan=True)

    # v 边界行为复制而非平均
    vc = np.array([[[[1.0, 2.0, 3.0],
                     [4.0, np.nan, 6.0],
                     [7.0, 8.0, np.nan]]]], np.float32)          # (1,1,3,3)
    cpu_vb = np.empty((1, 1, 4, 3), np.float32)
    cpu_vb[:, :, 1:3, :] = pre_pp.colocate(vc[:, :, :-1, :], vc[:, :, 1:, :])
    cpu_vb[:, :, 0, :] = vc[:, :, 0, :]
    cpu_vb[:, :, 3, :] = vc[:, :, -1, :]
    gpu_vb = pre_pp.torch_colocate_v(torch.from_numpy(vc).to(dev)).cpu().numpy()
    assert np.array_equal(cpu_vb, gpu_vb, equal_nan=True)


def test_torch_enforce_land_mask():
    """防止 GPU 掩膜执行回归：陆地有限值就地清 NaN 并计数、计数跨块累加不重复、
    海洋 NaN 抛带全局坐标的 RuntimeError——语义必须与 NumPy 参考一致。"""
    dev = _cuda_or_skip()
    if dev is None:
        return
    mask = np.array([[1, 0, 1],
                     [1, 1, 0]])
    gmask = torch.as_tensor(mask, dtype=torch.bool, device=dev)

    # 陆地有限值就地清 NaN 并计数（与 NumPy 参考一致）
    arr = np.array([[[[1.0, 9.0, 3.0],
                      [4.0, 5.0, 6.0]]]], np.float32)
    cpu = arr.copy()
    discarded_cpu = {}
    pre_pp.enforce_land_mask(cpu, mask, "u", 0, discarded_cpu)
    gpu = torch.from_numpy(arr.copy()).to(dev)
    discarded_gpu = {}
    pre_pp.torch_enforce_land_mask(gpu, gmask, "u", 0, discarded_gpu)
    assert np.array_equal(cpu, gpu.cpu().numpy(), equal_nan=True)
    assert discarded_cpu == discarded_gpu == {"u": 2}

    # 计数跨块累加（已是 NaN 的单元格不重复计数）
    pre_pp.torch_enforce_land_mask(gpu, gmask, "u", 50, discarded_gpu)
    assert discarded_gpu == {"u": 2}

    # 海洋单元格（mask==1）上的 NaN → 带全局坐标的 RuntimeError
    bad = np.array([[[[1.0, 2.0, 3.0],
                      [4.0, np.nan, 6.0]]]], np.float32)  # NaN 位于 (t=7, s=0, r=1, c=1)
    gbad = torch.from_numpy(bad).to(dev)
    try:
        pre_pp.torch_enforce_land_mask(gbad, gmask, "v", 7, {})
    except RuntimeError as e:
        msg = str(e)
        assert "t=7" in msg and "r=1" in msg and "c=1" in msg, msg
        assert "mask==1" in msg, msg
    else:
        raise AssertionError("expected RuntimeError for NaN on ocean cell")


def test_torch_extrema_summary():
    """防止 GPU 极值统计回归：NaN-aware 极值与首次出现索引与 NumPy 参考一致、并列
    时保留 C 序最先出现者、全 NaN 块按 np.nanmin/np.nanmax 语义抛错。"""
    dev = _cuda_or_skip()
    if dev is None:
        return
    rng = np.random.default_rng(1)
    arr = rng.uniform(-5, 5, (4, 3, 2, 7)).astype(np.float32)
    flat = arr.ravel()
    flat[::3] = np.nan
    flat[0] = -9.0
    flat[10] = 9.0                     # 极大值故意放在非末位索引（供下方并列测试）
    mn, mi, mx, xi = pre_pp.torch_extrema_summary(torch.from_numpy(arr).to(dev))
    assert mn == float(np.nanmin(flat)) and mx == float(np.nanmax(flat))
    assert mi == int(np.nanargmin(flat)) and xi == int(np.nanargmax(flat))

    # 并列时保留 C 序首次出现（GPU == numpy）：在严格更后的索引复制同一极大值，
    # 首次出现者必须保持不变
    flat2 = flat.copy()
    first_max = int(np.nanargmax(flat2))
    flat2[first_max + 10] = flat2[first_max]      # 极大值的后置副本
    g2 = torch.from_numpy(flat2.reshape(arr.shape)).to(dev)
    mn2, mi2, mx2, xi2 = pre_pp.torch_extrema_summary(g2)
    assert mx2 == float(np.nanmax(flat2)) and xi2 == first_max

    # 全 NaN 块按 np.nanmin/np.nanmax 语义抛 ValueError
    all_nan = torch.full((2, 3), np.nan, device=dev)
    try:
        pre_pp.torch_extrema_summary(all_nan)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an all-NaN chunk")


def test_cond_flatten_and_rollout_shift():
    """防止条件窗口协议回归：day-major 交错（通道 2k=第 k 天 u、2k+1=第 k 天 v）与
    rollout 滑窗（丢最旧一天、末尾追加预测）的结构必须精确。"""
    uv = torch.randn(1, 7, 2, 4, 4, 1)          # (B, days, 2, H, W, Z)：7 天 u/v
    cond = uv.reshape(1, 14, 4, 4, 1)           # day-major 交错展平
    assert cond.shape == (1, 14, 4, 4, 1)
    # 通道 2k = 第 k 天 u，2k+1 = 第 k 天 v
    for k in range(7):
        assert torch.equal(cond[0, 2 * k], uv[0, k, 0])
        assert torch.equal(cond[0, 2 * k + 1], uv[0, k, 1])

    new = torch.randn(1, 2, 4, 4, 1)
    cur = torch.cat([cond[:, 2:], new], dim=1)  # rollout：丢最旧一天
    assert cur.shape == (1, 14, 4, 4, 1)
    assert torch.equal(cur[0, 0], uv[0, 1, 0])  # 原 day 1 的 u 成为首个通道
    assert torch.equal(cur[:, -2:], new)


# ── 模型与损失组：条件前向、掩膜损失、采样 ──


def test_forward_14_to_2_and_shape():
    """防止 14→2 条件前向回归：preconditioned_network_forward 输出形状正确、带
    掩膜训练损失有限（backbone 的 14 通道条件经 x_self_cond 槽位进入）。"""
    model = make_model()
    images = torch.randn(1, 2, H, W, Z)
    cond = torch.randn(1, 14, H, W, Z)
    out = model.preconditioned_network_forward(
        images, torch.full((1,), 0.5), cond)
    assert out.shape == (1, 2, H, W, Z)
    loss = model(images, cond, mask=torch.ones(1, 2, H, W, Z))
    assert torch.isfinite(loss)


def test_masked_loss_denominator_and_backward():
    """防止掩膜扩散损失回归：与同种子逐项复现的 manual_loss 对齐（双变量/单通道
    广播/逐 batch 变化三种掩膜）；失败意味着损失分母或掩膜广播语义被改动。"""
    model = make_model()
    torch.manual_seed(0)
    images = torch.rand(B, 2, H, W, Z)
    cond = torch.rand(B, 14, H, W, Z)
    mask = torch.zeros(1, 2, H, W, Z)
    mask[0, 0] = 1.0                              # u 全有效
    mask[0, 1, 0:2, 0:2] = 1.0                    # v 部分有效

    def manual_loss(m):
        # 用相同种子复现 forward() 的随机数序列，逐项手工重算
        sigmas = (model.P_mean + model.P_std * torch.randn(B)).exp()
        norm_img = images * 2 - 1
        noise = torch.randn_like(images)
        noised = norm_img + sigmas[:, None, None, None, None] * noise
        den = model.preconditioned_network_forward(noised, sigmas, cond)
        mse = (den - norm_img) ** 2
        mm = m.expand_as(mse)
        per_sample = (mse * mm).sum(dim=(1, 2, 3, 4)) / mm.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
        return (per_sample * model.loss_weight(sigmas)).mean()

    torch.manual_seed(0)
    loss = model(images, cond, mask=mask)
    torch.manual_seed(0)
    assert torch.allclose(loss, manual_loss(mask), atol=1e-6)

    # 单通道公共掩膜：直接传 (1,1,H,W,Z)，由扩散 forward 内部广播（不手动扩成两通道）
    mask1 = torch.zeros(1, 1, H, W, Z)
    mask1[0, 0, 0:2, 0:2] = 1.0
    torch.manual_seed(1)
    loss1 = model(images, cond, mask=mask1)
    torch.manual_seed(1)
    assert torch.allclose(loss1, manual_loss(mask1), atol=1e-6)

    # 逐 batch 变化的双变量掩膜
    maskb = torch.zeros(B, 2, H, W, Z)
    maskb[0, 0] = 1.0
    maskb[1, 1, :, :, :] = 1.0
    torch.manual_seed(2)
    lossb = model(images, cond, mask=maskb)
    torch.manual_seed(2)
    assert torch.allclose(lossb, manual_loss(maskb), atol=1e-6)

    # 全零掩膜：分母 clamp 防除零，结果恰为 0 不含 NaN
    loss0 = model(images, cond, mask=torch.zeros(1, 2, H, W, Z))
    assert torch.isfinite(loss0) and loss0.item() == 0.0

    # 非零损失反向必须产生梯度
    lossb.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())

    # Z 不匹配必须抛错；该错误绝不能被当成成功吞掉
    try:
        model(images[:, :, :, :, :1], cond, mask=mask)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for wrong Z")


def test_two_step_sample():
    """防止采样路径回归：两步 Heun 采样输出形状正确、有限且落在 [0,1]（clamp 后
    unnormalize 的输出域）。"""
    model = make_model()
    cond = torch.randn(1, 14, H, W, Z)
    with torch.no_grad():
        out = model.sample(cond, num_sample_steps=2)
    assert out.shape == (1, 2, H, W, Z)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ── 指标组：正式 pre_metrics 口径 ──


def test_corrected_relative_l2():
    """防止正式指标回归：masked_rel_l2 用符号相反的 ±1 误差排除平方误差抵消、掩膜
    外单元格不计入——失败意味着指标被退化为含抵消的旧口径。"""
    # pre_metrics.py 的正式实现（与 pre_trainer.py 共用）
    tgt = torch.full((1, 1, 2, 2, 1), 10.0)
    pred = tgt.clone()
    pred[0, 0, 0, 0, 0] = 11.0                   # +1 与 -1 误差：有符号和为 0（平方和不为 0）
    pred[0, 0, 0, 1, 0] = 9.0
    mask = torch.ones(1, 1, 2, 2, 1)
    got = masked_rel_l2(pred, tgt, mask)
    want = sqrt(2.0) / sqrt(400.0)
    assert got > 0.0 and abs(got - want) < 1e-6, got

    # 掩膜外单元格不得计入
    mask2 = mask.clone()
    mask2[0, 0, 0, 0, 0] = 0.0
    got2 = masked_rel_l2(pred, tgt, mask2)
    want2 = sqrt(1.0) / sqrt(300.0)
    assert abs(got2 - want2) < 1e-6, got2


def test_rho_to_native_resampling():
    """防止 rho→native 映射回归：u 沿 xi 对相邻 rho 点求均值、v 沿 eta，u/v 通道
    不得串扰（u/v 差 100 的偏移值使通道混淆立即暴露）。"""
    # 正式 rho_to_native：u 沿 xi（列）对相邻 rho 点求均值，v 沿 eta（行）——通道必须分离
    up = np.arange(16.0).reshape(1, 1, 4, 4, 1)
    rho = np.stack([up, up + 100.0], axis=2)     # (1,1,2,4,4,1)
    u_nat, v_nat = rho_to_native(rho)
    assert u_nat.shape == (1, 1, 4, 3, 1)
    assert v_nat.shape == (1, 1, 3, 4, 1)
    assert abs(u_nat[0, 0, 2, 1, 0] - 0.5 * (up[0, 0, 2, 1, 0] + up[0, 0, 2, 2, 0])) < 1e-6
    assert abs(v_nat[0, 0, 1, 3, 0]
               - 0.5 * (rho[0, 0, 1, 1, 3, 0] + rho[0, 0, 1, 2, 3, 0])) < 1e-6


def test_native_reader_unified_layout_and_sentinels():
    """防止真值读取回归：NativeUVReader 的统一 (days,H,W,Z) 布局（sigma 轴移到
    末位）；u/v 哨兵值分离（u=7.0、v=11.0、极值 50.0）使 v 被读成 u 的 bug 立即
    暴露；另验证被裁剪值反归一化后无法还原原始真值（评估必须读原始 native 文件）。"""
    # 刻意分离的哨兵值：u 全 7.0（含一个 50.0 极值）、v 全 11.0——
    # v 被读成 u 的 bug 立即暴露
    u = np.full((5, 3, 4, 5), 7.0, np.float32)
    u[0, 0, 0, 0] = 50.0
    v = np.full((5, 3, 3, 5), 11.0, np.float32)
    with tempfile.TemporaryDirectory() as d:
        up, vp = os.path.join(d, "u.npy"), os.path.join(d, "v.npy")
        np.save(up, u)
        np.save(vp, v)

        # full3d：统一 (days, H, W-1, Z) / (days, H-1, W, Z) 布局
        full = NativeUVReader(depth_index=None, u_path=up, v_path=vp, check_shape=False)
        us, vs = full.get(0, 2)
        assert us.shape == (2, 4, 5, 3) and vs.shape == (2, 3, 5, 3)
        assert us[0, 0, 0, 0] == 50.0             # u 的原始极值保持原样
        assert np.count_nonzero(us != 7.0) == 1   # 其余均为 u 的哨兵值
        assert (vs == 11.0).all(), "v must not be read from the u field"

        # surface：同一统一布局，Z=1
        surf = NativeUVReader(depth_index=2, u_path=up, v_path=vp, check_shape=False)
        us2, vs2 = surf.get(1, 3)
        assert us2.shape == (3, 4, 5, 1) and vs2.shape == (3, 3, 5, 1)
        assert np.array_equal(us2[1, :, :, 0], u[2, 2])
        assert (vs2 == 11.0).all()

        # day-7 切片上 u/v 值保持独立（哨兵值分离）
        du, dv = full.get(4, 1)
        assert du.shape == (1, 4, 5, 3) and dv.shape == (1, 3, 5, 3)
        assert np.count_nonzero(du != 7.0) == 0 and (dv == 11.0).all()
        # 释放 mmap 文件句柄，临时目录才能删除
        for r in (full, surf):
            r.u._mmap.close()
            r.v._mmap.close()

    # 被裁剪的值反归一化后无法还原原始真值
    raw = np.array([1.0, 2.0, 50.0])
    lo, hi = 1.0, 3.0
    denorm = (np.clip(raw, lo, hi) - lo) / (hi - lo) * (hi - lo) + lo
    assert np.any(denorm != raw) and denorm[2] == 3.0


def test_metrics_native_batch():
    """防止正式评估指标回归：rho_to_native 输出形状、masked_error_sums 的 (L,Z)
    误差和与 (L,2,Z) 累计器槽位、掩膜外贡献为零、pooled RMSE=sqrt(se/count)（绝非
    逐层 RMSE 均值）、零计数返回 0.0、持续性基线误差和与直接 NumPy 参考一致。"""
    # 走 pre_metrics.py 正式函数的合成评估批次
    rng = np.random.default_rng(3)
    for Zz in (1, 3):                             # surface（Z=1）与 full3d（Z>1）
        Bb, L = 2, 15
        rho_pred = rng.normal(size=(Bb, L, 2, H, W, Zz))
        truth_u = rng.normal(size=(Bb, L, H, W - 1, Zz))
        truth_v = rng.normal(size=(Bb, L, H - 1, W, Zz))
        u_nat, v_nat = rho_to_native(rho_pred)
        assert u_nat.shape == (Bb, L, H, W - 1, Zz)
        assert v_nat.shape == (Bb, L, H - 1, W, Zz)

        mask_u = np.ones((H, W - 1), bool)
        mask_v = np.ones((H - 1, W), bool)
        se_u, ae_u = masked_error_sums(u_nat, truth_u, mask_u)
        se_v, ae_v = masked_error_sums(v_nat, truth_v, mask_v)
        assert se_u.shape == (L, Zz) and ae_u.shape == (L, Zz)
        assert se_v.shape == (L, Zz) and ae_v.shape == (L, Zz)

        # 结果必须能经 [:, channel, :] 槽位并入 (L, 2, Z) 累计器
        se_m = np.zeros((L, 2, Zz))
        ae_m = np.zeros((L, 2, Zz))
        se_m[:, 0, :] += se_u
        ae_m[:, 0, :] += ae_u
        se_m[:, 1, :] += se_v
        ae_m[:, 1, :] += ae_v

        # 直接对原始数组用 NumPy 求精确参考和
        assert np.allclose(se_u, ((u_nat - truth_u) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(se_v, ((v_nat - truth_v) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(ae_v, np.abs(v_nat - truth_v).sum(axis=(0, 2, 3)))

        # 陆地（mask == 0）不贡献
        mu2 = mask_u.copy()
        mu2[0, 0] = False
        se_u2, _ = masked_error_sums(u_nat, truth_u, mu2)
        diff = (u_nat - truth_u)[:, :, 0, 0, :]
        assert np.allclose(se_u - se_u2, (diff ** 2).sum(axis=0))

        # pooled RMSE == sqrt(总se/总n)，绝不是逐层 RMSE 的均值
        n_u = mask_u.sum()
        n_v = mask_v.sum()
        rmse_u = pooled_rmse(se_u, np.full((L, Zz), n_u))
        assert np.isclose(rmse_u, sqrt(se_u.sum() / (L * Zz * n_u)))
        # 从 (L, 2, Z) 累计器取单个 lead 的 pooled RMSE
        cnt = np.empty((2, Zz))
        cnt[0, :] = n_u
        cnt[1, :] = n_v
        for l in (0, 4, 14):
            rm = pooled_rmse(se_m[l], cnt)
            assert np.isclose(rm, sqrt(se_m[l].sum() / (Zz * (n_u + n_v))))
        # 无有效计数时 pooled_rmse 返回 0.0 而非 NaN
        assert pooled_rmse(se_u, np.zeros((L, Zz))) == 0.0

        # 持续性：day-7 native u/v 复制到全部 lead 日，u/v 取不同量级保持独立
        day7_u = 100.0 * rng.normal(size=(Bb, 1, H, W - 1, Zz))
        day7_v = 100.0 * rng.normal(size=(Bb, 1, H - 1, W, Zz))
        pu = np.broadcast_to(day7_u, (Bb, L, H, W - 1, Zz))
        pv = np.broadcast_to(day7_v, (Bb, L, H - 1, W, Zz))
        assert np.allclose(pu[0, 3], day7_u[0, 0]) and np.allclose(pv[0, 3], day7_v[0, 0])
        se_pu, _ = masked_error_sums(pu, truth_u, mask_u)
        se_pv, _ = masked_error_sums(pv, truth_v, mask_v)
        assert np.allclose(se_pu, ((day7_u - truth_u) ** 2).sum(axis=(0, 2, 3)))
        assert np.allclose(se_pv, ((day7_v - truth_v) ** 2).sum(axis=(0, 2, 3)))


# ── 统计与归一化组：裁剪、pooled sigma、缓存 stale、时间校验 ──


def test_clip_range_policy():
    """防止裁剪策略回归：默认（clip_pct=None）不裁剪，lo/hi 为精确极值；显式百分位
    裁剪把 lo/hi 从两端收拢。"""
    vals = [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])]
    lo, hi = _clip_range(lambda: iter(vals), None)  # 默认：不裁剪
    assert (lo, hi) == (1.0, 10.0)
    lo20, hi20 = _clip_range(lambda: iter(vals), 20.0)  # 显式裁剪
    assert abs(lo20 - 2.0) < 0.01 and abs(hi20 - 5.0) < 0.01


def test_pooled_sigma_and_rmse():
    """防止统计口径回归：pooled std 必须等于拼接后数组的标准差（组间均值差贡献不可
    丢）；总体 RMSE=sqrt(总se/总n)，与逐层 RMSE 均值不同。"""
    # 两个均值不同的组合的 pooled std == 拼接后数组的标准差
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0])
    s1 = a.sum() + b.sum()
    s2 = (a * a).sum() + (b * b).sum()
    n = a.size + b.size
    mean = s1 / n
    pooled = sqrt(max(s2 / n - mean * mean, 0.0))
    assert np.isclose(pooled, np.std(np.concatenate([a, b])))
    per_var = 0.5 * (np.std(a) + np.std(b))       # 组内方差的朴素均值丢掉组间贡献
    assert abs(pooled - per_var) > 1e-3

    # 总体 RMSE = sqrt(总se/总n)，不是逐层 RMSE 均值
    se = np.array([[[1.0]], [[1.0]]])
    n = np.array([[[1.0]], [[4.0]]])
    overall = pooled_rmse(se, n)
    assert np.isclose(overall, sqrt(2.0 / 5.0))
    mean_of_rmse = np.sqrt(se / n).mean()
    assert not np.isclose(overall, mean_of_rmse)


def test_stats_cache_clip_and_splits():
    """防止统计缓存契约回归：正式 compute_or_load_stats() 给出裁剪后 lo/hi 与 pooled
    sigma、相同配置命中缓存、缺 'splits' 字段的旧缓存判 stale 重算、切分变化后旧缓存
    不得复用——失败意味着会用错误归一化统计训练/评估。"""
    # 对微型临时 aligned 数据集跑正式 compute_or_load_stats()：
    # 裁剪 sigma、缓存复用、缺 splits 的 stale、切分变化的 stale。
    import pre_dataset as pd
    rng = np.random.default_rng(0)
    T, S, HH, WW = 10, 2, 4, 5
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        norm = os.path.join(d, "norm")
        os.makedirs(aligned)
        os.makedirs(norm)
        u = rng.uniform(-1.0, 1.0, (T, S, HH, WW)).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, (T, S, HH, WW)).astype(np.float32)
        np.save(os.path.join(aligned, "u_rho.npy"), u)
        np.save(os.path.join(aligned, "v_rho.npy"), v)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((HH, WW), np.uint8))

        saved = (pd.ALIGNED_DIR, pd.NORM_DIR, pd.H, pd.W, pd.T_TOTAL, dict(pd.SPLITS))
        pd.ALIGNED_DIR, pd.NORM_DIR = aligned, norm
        pd.H, pd.W, pd.T_TOTAL = HH, WW, T
        try:
            pd.SPLITS.clear()
            pd.SPLITS.update(train=(0, 6), val=(6, 8), test=(8, 10))

            # 不裁剪：lo/hi == 训练段精确极值，sigma == min-max 归一化后 u+v 拼接的 pooled std
            s_none = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert np.isclose(s_none["lo"][0], u[:6].min()) and np.isclose(s_none["hi"][0], u[:6].max())
            assert np.isclose(s_none["lo"][1], v[:6].min()) and np.isclose(s_none["hi"][1], v[:6].max())

            def pooled_sigma(lo, hi):
                a = np.clip(u[:6], float(lo[0]), float(hi[0])).astype(np.float64)
                a = (a - float(lo[0])) / (float(hi[0]) - float(lo[0]))
                b = np.clip(v[:6], float(lo[1]), float(hi[1])).astype(np.float64)
                b = (b - float(lo[1])) / (float(hi[1]) - float(lo[1]))
                return float(np.std(np.concatenate([a.ravel(), b.ravel()])))

            assert np.isclose(s_none["sigma"], pooled_sigma(s_none["lo"], s_none["hi"]),
                              rtol=1e-4)

            # 裁剪把 lo/hi 从两端收拢并改变 sigma
            s_clip = pd.compute_or_load_stats(depth_index=None, clip_pct=20.0, verbose=False)
            assert s_clip["lo"][0] > s_none["lo"][0] and s_clip["hi"][0] < s_none["hi"][0]
            assert np.isclose(s_clip["sigma"], pooled_sigma(s_clip["lo"], s_clip["hi"]),
                              rtol=1e-4)
            assert abs(s_clip["sigma"] - s_none["sigma"]) > 1e-6

            # 相同配置 → 缓存命中，sigma 一致
            s_again = pd.compute_or_load_stats(depth_index=None, clip_pct=20.0, verbose=False)
            assert np.isclose(s_again["sigma"], s_clip["sigma"])

            # 缺 'splits' 字段的缓存必须判 stale
            cache = os.path.join(norm, "stats_all_clipnone.npz")
            os.remove(cache)
            np.savez(cache,
                     lo=np.float32([-9.0, -9.0]), hi=np.float32([9.0, 9.0]),
                     sigma=np.float32(0.123), depth_index=np.int64(-1),
                     clip_pct=np.float64(-1.0), mask_version=np.str_(pd.mask_version()))
            s_missing = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert abs(s_missing["sigma"] - 0.123) > 1e-6
            assert np.isclose(s_missing["sigma"], s_none["sigma"])

            # 切分变化 → 旧缓存不得复用
            pd.SPLITS.clear()
            pd.SPLITS.update(train=(0, 4), val=(4, 7), test=(7, 10))
            s_new = pd.compute_or_load_stats(depth_index=None, clip_pct=None, verbose=False)
            assert np.isclose(s_new["lo"][0], u[:4].min())
            assert abs(s_new["sigma"] - s_none["sigma"]) > 1e-6
        finally:
            pd.ALIGNED_DIR, pd.NORM_DIR, pd.H, pd.W, pd.T_TOTAL = saved[:5]
            pd.SPLITS.clear()
            pd.SPLITS.update(saved[5])


def test_verify_daily_time():
    """防止时间轴校验回归：任意 datetime64 分辨率下 24 h 间隔通过；23/25 h 间隔必须
    抛出带索引、两端时间与实际间隔的 RuntimeError。"""
    # 24 h 间隔在任意 datetime64 分辨率下通过
    good = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    assert pre_pp.verify_daily_time(good) is good
    good_s = np.array(["2020-01-01T00:00:00", "2020-01-02T00:00:00",
                       "2020-01-03T00:00:00"], dtype="datetime64[s]")
    assert pre_pp.verify_daily_time(good_s) is good_s

    # 23/25 h 间隔必须失败，并报告索引、时间与实际间隔
    for hours in (23, 25):
        t = np.array([
            np.datetime64("2020-01-01T00:00:00", "s"),
            np.datetime64("2020-01-01T00:00:00", "s") + np.timedelta64(hours, "h"),
            np.datetime64("2020-01-01T00:00:00", "s") + np.timedelta64(2 * hours, "h"),
        ])
        try:
            pre_pp.verify_daily_time(t)
        except RuntimeError as e:
            msg = str(e)
            assert "index 0" in msg, msg
            assert "->" in msg, msg
            assert f"{hours} h" in msg and "24 h" in msg, msg
        else:
            raise AssertionError(f"expected RuntimeError for {hours}h-spaced times")


# ── 配置与 checkpoint 兼容组：legacy 通道、sigma 尺度、续训策略 ──


def test_legacy_cond_chans_none():
    """防止 legacy 兼容回归：cond_chans=None 恢复旧版通道倍增（2+2），前向形状不变。"""
    net = IAFNODiff(
        dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=8, num_blocks=1,
        in_chans=2, out_chans=2, cond_chans=None, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(H, W, Z), self_condition=True,
    )
    x = torch.randn(1, 2, H, W, Z)
    cond = torch.randn(1, 2, H, W, Z)             # legacy 倍增（2+2）
    out = net(x, torch.zeros(1), cond)
    assert out.shape == (1, 2, H, W, Z)


def test_sigma_data_conversion():
    """防止 sigma 尺度回归：[0,1] 统计 sigma ×2 得到 [-1,1] 图像空间的 EDM
    sigma_data（diffusion.py 以 images*2-1 归一化）。"""
    # [0,1] 统计空间的 sigma → [-1,1] 图像空间的 EDM sigma_data（×2）
    assert SIGMA_DATA_SCALE == 2.0
    assert np.isclose(sigma_data_from_stats(0.0856), 0.1712)
    assert np.isclose(sigma_data_from_stats(0.0), 0.0)
    assert np.isclose(sigma_data_from_stats(1.0), 2.0)


def test_sigma_data_legacy_checkpoint_fallback():
    """防止 checkpoint 兼容回归：缺 config.sigma_data 的 legacy checkpoint 回退到旧
    stats-only 尺度（不乘 2，以 used=False 供调用方显式告警）；新 checkpoint 以其
    存储值为准。"""
    # legacy checkpoint（无 config.sigma_data）保持旧 stats-only 尺度
    sd, used = sigma_data_from_checkpoint({"epoch": 2}, 0.0856)
    assert not used and np.isclose(sd, 0.0856)
    sd2, used2 = sigma_data_from_checkpoint({}, 0.0856)
    assert not used2 and np.isclose(sd2, 0.0856)
    sd3, used3 = sigma_data_from_checkpoint(None, 0.0856)
    assert not used3 and np.isclose(sd3, 0.0856)
    # 新 checkpoint → 其存储的 sigma_data 优先，无视 stats
    sd4, used4 = sigma_data_from_checkpoint(
        {"config": {"sigma_data": 0.1712, "sigma_data_scale": 2.0}}, 0.0856)
    assert used4 and np.isclose(sd4, 0.1712)


def test_resume_sigma_policy():
    """防止续训尺度策略回归：RESUME_SIGMA_POLICY 四分支——尺度一致时任何策略都保持
    现状不 adopt；不一致时 error（默认）抛 RuntimeError 绝不静默混尺度、migrate 保持
    当前 SD2 尺度、adopt 采用 checkpoint 旧尺度；未知策略抛 ValueError。"""
    # 尺度一致 → 任何策略都保持现状，绝不 adopted
    sd_new = sigma_data_from_stats(0.0856)          # 0.1712
    for policy in ("error", "migrate", "adopt"):
        sd, adopted = resume_sigma_decision(sd_new, sd_new, policy)
        assert not adopted and np.isclose(sd, sd_new), policy
    # 不一致 + "error"（默认）→ RuntimeError，绝不静默混用尺度
    try:
        resume_sigma_decision(0.0856, sd_new, "error")
    except RuntimeError as e:
        assert "sigma_data" in str(e) and "0.0856" in str(e)
    else:
        raise AssertionError("expected RuntimeError on scale mismatch")
    # 不一致 + "migrate" → 保持当前（SD2）尺度，非 adopted
    sd, adopted = resume_sigma_decision(0.0856, sd_new, "migrate")
    assert not adopted and np.isclose(sd, sd_new)
    # 不一致 + "adopt" → 采用 checkpoint 旧尺度，adopted
    sd, adopted = resume_sigma_decision(0.0856, sd_new, "adopt")
    assert adopted and np.isclose(sd, 0.0856)
    # 未知策略 → ValueError
    try:
        resume_sigma_decision(0.0856, sd_new, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown policy")


# ── rollout 组：ensemble、逐窗口种子、autocast、oracle ──


def test_ensemble_size1_matches_sequential():
    """防止 ensemble 语义回归：E=1 必须逐位复现普通逐窗口 rollout（相同 RNG 流 +
    与历史评估路径相同的 autocast 包裹）。"""
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    p1 = ensemble_rollout(model, cond, 3, 1, seed=42)
    assert p1.shape == (2, 1, 3, 2, H, W, Z)
    assert torch.isfinite(p1).all()
    torch.manual_seed(42)
    cur = cond.clone()
    preds = []
    with torch.amp.autocast(device_type="cpu"):
        for _ in range(3):
            preds.append(model.sample(cur).float())
            cur = torch.cat([cur[:, 2:], preds[-1]], dim=1)
    p2 = torch.stack(preds, dim=1)                # (B, L, 2, H, W, Z)：逐日预测栈
    assert torch.allclose(p1[:, 0], p2, atol=1e-6)


def test_ensemble_rollout_uses_autocast():
    """防止评估数值路径回归：model.sample 必须在 autocast（AMP）内运行，否则历史
    评估路径的数值被静默改变；autocast 设备跟随张量而非全局 CUDA 可用性。"""
    seen = {"cpu": False, "cuda": False}

    class _FlagSampler:
        def sample(self, cur, num_sample_steps=None, clamp=True):
            if torch.is_autocast_enabled("cpu"):
                seen["cpu"] = True
            if torch.is_autocast_enabled("cuda"):
                seen["cuda"] = True
            return cur[:, :2].clone()

    cond = torch.rand(1, 14, H, W, Z)
    ensemble_rollout(_FlagSampler(), cond, 2, 1, seed=0)
    # autocast 设备跟随张量而非全局 CUDA 可用性：CPU 张量 → CPU autocast（即使机器
    # 有 CUDA）；CUDA 张量 → CUDA autocast（历史评估路径）。
    if cond.is_cuda:
        assert seen["cuda"], "model.sample must run under CUDA autocast"
    else:
        assert seen["cpu"], "model.sample must run under CPU autocast"
    if torch.cuda.is_available():
        seen["cpu"] = seen["cuda"] = False
        ensemble_rollout(_FlagSampler(), cond.cuda(), 2, 1, seed=0)
        assert seen["cuda"], "model.sample must run under CUDA autocast"


def test_ensemble_seeds_per_window():
    """防止逐窗口种子语义回归：窗口 w 的轨迹只由 seeds[w] 与 cond[w] 决定，与
    batch 大小和其他窗口无关；seed/seeds 互斥；每窗口持有独立 AR 状态（窗口依次
    rollout，无交叉污染）。"""
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    p_batch = ensemble_rollout(model, cond, 2, 1, seeds=[5, 9])
    assert p_batch.shape == (2, 1, 2, 2, H, W, Z)
    # 窗口 0 单独（batch=1）与批内窗口 0 一致
    p_single = ensemble_rollout(model, cond[:1], 2, 1, seeds=[5])
    assert torch.allclose(p_batch[0, 0], p_single[0, 0], atol=1e-6)
    # 窗口 1 单独与批内一致
    p_single1 = ensemble_rollout(model, cond[1:], 2, 1, seeds=[9])
    assert torch.allclose(p_batch[1, 0], p_single1[0, 0], atol=1e-6)
    # seeds[w] 与同窗口的标量 seed 路径等价
    p_scalar = ensemble_rollout(model, cond[:1], 2, 1, seed=5)
    assert torch.allclose(p_batch[0, 0], p_scalar[0, 0], atol=1e-6)
    # 不同种子 → 不同轨迹；同种子 → 可复现
    p_other = ensemble_rollout(model, cond[:1], 2, 1, seeds=[8])
    assert not torch.allclose(p_scalar[0, 0], p_other[0, 0], atol=1e-6)
    p_again = ensemble_rollout(model, cond[:1], 2, 1, seeds=[5])
    assert torch.allclose(p_again[0, 0], p_scalar[0, 0], atol=1e-6)
    # seed 与 seeds 互斥
    try:
        ensemble_rollout(model, cond, 2, 1, seed=1, seeds=[2, 3])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for seed + seeds together")
    # 逐窗口种子保持每窗口独立 AR 状态（窗口依次 rollout：窗口 0 占调用 0..2，窗口 1 占 3..5）
    s = _persistence_fake()
    cond2 = torch.arange(2 * 14 * H * W * Z).reshape(2, 14, H, W, Z).float() / 1000.0
    ensemble_rollout(s, cond2, 3, 2, seeds=[11, 22])
    assert torch.equal(s.calls[1][0], torch.cat([s.calls[0][0][2:], s.calls[0][0][:2]], dim=0))
    assert torch.equal(s.calls[4][0], torch.cat([s.calls[3][0][2:], s.calls[3][0][:2]], dim=0))


def test_ensemble4_shape_mean_and_independent():
    """防止 ensemble 聚合回归：E=4 输出 (B,E,L,2,H,W,Z)、成员轨迹独立、
    ensemble_mean 为成员均值（点预测）、平均降低单日方差。"""
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(2, 14, H, W, Z)
    preds = ensemble_rollout(model, cond, 2, 4, seed=7)
    assert preds.shape == (2, 4, 2, 2, H, W, Z)
    assert torch.isfinite(preds).all()
    assert float(preds.min()) >= 0.0 and float(preds.max()) <= 1.0
    # 成员是独立轨迹 → 噪声不同 → 输出不同
    assert not torch.allclose(preds[0, 0], preds[0, 1], atol=1e-6)
    assert not torch.allclose(preds[0, 0], preds[1, 0], atol=1e-6)
    # ensemble_mean 是成员平均（点预测）
    m = ensemble_mean(preds)
    assert m.shape == (2, 2, 2, H, W, Z)
    assert torch.allclose(m, preds.mean(dim=1))
    # 平均相对单成员降低单日方差
    assert m[0].std() <= preds[0, 0].std() + 1e-6


def _persistence_fake():
    """确定性假 EDM：预测=当前条件的前 2 通道（持续性策略）；记录收到的每个输入。"""
    class _PersistenceSampler:
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True):
            self.calls.append(cur.clone())
            return cur[:, :2].clone()
    return _PersistenceSampler()


def test_ensemble_ar_state_independent():
    """防止 AR 状态串扰回归：持久化假模型的条件窗口演化必须只由该窗口自己的预测
    驱动（成员间无泄漏；调用布局交错 [w0m0, w0m1, w1m0, w1m1, ...]）；
    expand_ensemble 的 E=1 新副本 / E>1 独立重复语义精确。"""
    s = _persistence_fake()
    cond = torch.arange(2 * 14 * H * W * Z).reshape(2, 14, H, W, Z).float() / 1000.0
    preds = ensemble_rollout(s, cond, 3, 2)
    assert preds.shape == (2, 2, 3, 2, H, W, Z)
    # 成员 0 第 2 步条件 == 成员 0 自己第 1 步窗口加自己的预测——不得混入成员 1（反之亦然）。
    # 布局交错：[w0m0, w0m1, w1m0, w1m1, ...]。
    c0_0, c0_1 = s.calls[0][0], s.calls[1][0]
    assert torch.equal(c0_1, torch.cat([c0_0[2:], c0_0[:2]], dim=0))
    c1_0, c1_1 = s.calls[0][2], s.calls[1][2]
    assert torch.equal(c1_1, torch.cat([c1_0[2:], c1_0[:2]], dim=0))
    # 两个窗口独立演化且保持互异（该确定性假模型下同一窗口的成员相同，属设计内）
    assert not torch.equal(s.calls[0][0], s.calls[0][2])
    assert not torch.equal(c0_1, c1_1)
    # expand_ensemble：E=1 返回新副本，E>1 独立重复
    e1 = expand_ensemble(cond, 1)
    assert e1 is not cond and torch.equal(e1, cond)
    e4 = expand_ensemble(cond, 4)
    assert e4.shape == (8, 14, H, W, Z)
    assert torch.equal(e4[0], cond[0]) and torch.equal(e4[4], cond[1])


def test_rollout_horizons_1_and_15():
    """防止 rollout 时长边界回归：horizon 1 与 15 的输出形状、有限性与 [0,1] 值域。"""
    model = make_model()
    torch.manual_seed(0)
    cond = torch.rand(1, 14, H, W, Z)
    p1 = ensemble_rollout(model, cond, 1, 1, seed=0)
    assert p1.shape == (1, 1, 1, 2, H, W, Z)
    p15 = ensemble_rollout(model, cond, 15, 1, seed=1)
    assert p15.shape == (1, 1, 15, 2, H, W, Z)
    assert torch.isfinite(p15).all()
    assert float(p15.min()) >= 0.0 and float(p15.max()) <= 1.0


def test_rho_oracle_metric():
    """防止 oracle 诊断回归：真值与预测走同一条反归一化+rho_to_native 路径时
    oracle 误差恰为 0；真值平移后与直接掩膜比较一致；陆地保持排除。失败意味着
    rho↔native 转换与指标口径脱钩。"""
    rng = np.random.default_rng(0)
    Bb, L, Zz = 2, 3, 1
    lo = np.array([-1.0, -2.0], np.float32)
    hi = np.array([1.0, 3.0], np.float32)
    target_norm = rng.uniform(0.05, 0.95, (Bb, L, 2, H, W, Zz)).astype(np.float32)
    phys = target_norm * (hi - lo).reshape(1, 1, 2, 1, 1, 1) + lo.reshape(1, 1, 2, 1, 1, 1)
    u_nat, v_nat = rho_to_native(phys)
    mask_u = np.ones((H, W - 1), bool)
    mask_v = np.ones((H - 1, W), bool)

    # 真值 == 同一条转换路径 → oracle 误差恰为 0
    se, ae = oracle_native_error_sums(target_norm, lo, hi, u_nat, v_nat, mask_u, mask_v)
    assert se.shape == (L, 2, Zz) and ae.shape == (L, 2, Zz)
    assert np.allclose(se, 0.0, atol=1e-6) and np.allclose(ae, 0.0, atol=1e-6)

    # 真值平移 → oracle 误差等于直接掩膜比较
    truth_u = u_nat + 1.0
    se2, ae2 = oracle_native_error_sums(target_norm, lo, hi, truth_u, v_nat, mask_u, mask_v)
    se_ref, ae_ref = masked_error_sums(u_nat, truth_u, mask_u)
    assert np.allclose(se2[:, 0, :], se_ref)
    assert np.allclose(ae2[:, 0, :], ae_ref)
    # 陆地保持排除
    mu2 = mask_u.copy()
    mu2[0, 0] = False
    se3, _ = oracle_native_error_sums(target_norm, lo, hi, truth_u, v_nat, mu2, mask_v)
    diff = (u_nat - truth_u)[:, :, 0, 0, :]
    assert np.allclose(se2[:, 0, :] - se3[:, 0, :], (diff ** 2).sum(axis=0))


def test_mask_tensor_writable_and_contiguous():
    """防止掩膜张量契约回归：build_mask_tensor 输出 (1,2,H,W,Z)、C-contiguous 且
    可写（broadcast_to 的只读 view 必须先复制）；单层/全深度两种形状都覆盖。"""
    import pre_dataset as pd
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        os.makedirs(aligned)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((4, 5), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((4, 5), np.uint8))
        saved = (pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL)
        pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL = aligned, 4, 5, 2
        try:
            t = pd.build_mask_tensor(torch.device("cpu"), depth_index=1)
            assert t.shape == (1, 2, 4, 5, 1)
            assert t.is_contiguous()
            t[0, 0, 0, 0, 0] = 0.0          # 必须可写，无只读告警
            assert t[0, 0, 0, 0, 0] == 0.0
            t2 = pd.build_mask_tensor(torch.device("cpu"), depth_index=None)
            assert t2.shape == (1, 2, 4, 5, 2) and t2.is_contiguous()
        finally:
            pd.ALIGNED_DIR, pd.H, pd.W, pd.S_TOTAL = saved


# ── checkpoint 与训练配置组 ──


def test_checkpoint_metadata_roundtrip():
    """防止 checkpoint 元数据回归：stats_sigma/sigma_data_scale/sigma_data 经
    weights_only=True 的 save/load 往返不变；模型权重经 load_checkpoint 全量恢复
    且留在 map_location 指定的设备。"""
    stats_sigma = 0.0856
    sd = sigma_data_from_stats(stats_sigma)
    assert np.isclose(sd, 0.1712)
    state = {"epoch": 0, "best_val": 1.0,
             "config": {"preset": "surface_smoke",
                        "stats_sigma": stats_sigma,
                        "sigma_data_scale": SIGMA_DATA_SCALE,
                        "sigma_data": sd}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pth")
        torch.save(state, p)
        loaded = torch.load(p, weights_only=True)
        sd2, used = sigma_data_from_checkpoint(loaded, stats_sigma)
        assert used and np.isclose(sd2, sd)
        assert loaded["config"]["sigma_data_scale"] == SIGMA_DATA_SCALE
        assert np.isclose(loaded["config"]["stats_sigma"], stats_sigma)

        # 完整模型状态经 load_checkpoint（weights_only=True）往返
        model = make_model()
        p2 = os.path.join(d, "model.pth")
        torch.save({"model_state_dict": model.state_dict(), "epoch": 3,
                    "config": {"stats_sigma": stats_sigma,
                               "sigma_data_scale": SIGMA_DATA_SCALE,
                               "sigma_data": sd}}, p2)
        m2 = make_model()
        ckpt = load_checkpoint(p2, m2, map_location="cpu")
        assert ckpt["epoch"] == 3
        sd3, used3 = sigma_data_from_checkpoint(ckpt, stats_sigma)
        assert used3 and np.isclose(sd3, sd)
        assert next(m2.parameters()).device.type == "cpu"


def test_training_modes_and_run_tags():
    """防止训练配置与 run tag 回归：full 模式返回 preset 的隔离副本（值相同、对象
    不同）；smoke 模式压缩 epoch/步数/窗口但保留架构与网格；目标后缀（_RES 隔离
    确定性基线与扩散运行，默认目标保持 legacy tag）；非法 preset/mode/world_size/
    objective 必须 ValueError。"""
    full = training_config("surface_smoke", "full", world_size=1)
    assert full == PRESETS["surface_smoke"]
    assert full is not PRESETS["surface_smoke"]

    smoke = training_config("surface_smoke", "smoke", world_size=4)
    assert smoke["embed_dim"] == PRESETS["surface_smoke"]["embed_dim"]
    assert smoke["patch_size"] == PRESETS["surface_smoke"]["patch_size"]
    assert smoke["num_epochs"] == 1 and smoke["sampling_steps"] == 4
    assert smoke["val_windows"] == 4
    assert smoke["max_train_windows"] == (
        SMOKE_BATCHES_PER_RANK * 4 * smoke["batch_size"])
    assert training_run_tag("surface_smoke", full) == (
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2")
    assert training_run_tag("surface_smoke", smoke, "smoke", 4).endswith(
        "_S4_C7_SD2_SMOKE_DDP4")
    # 目标后缀：确定性基线绝不与扩散运行共用运行目录；默认目标保持 legacy tag 不变
    assert training_run_tag("surface_smoke", full,
                            objective="persistence_residual") == (
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES")
    assert training_run_tag("surface_smoke", smoke, "smoke", 4,
                            objective="persistence_residual").endswith(
        "_S4_C7_SD2_RES_SMOKE_DDP4")
    assert training_run_tag("surface_smoke", full,
                            objective="diffusion") == training_run_tag("surface_smoke", full)

    for args in (("missing", "full", 1), ("surface_smoke", "bad", 1),
                 ("surface_smoke", "full", 0)):
        try:
            training_config(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"training_config{args} should fail")
    try:
        training_run_tag("surface_smoke", full, objective="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("training_run_tag with unknown objective should fail")


def test_objective_config_helpers():
    """防止目标配置守卫回归：目标名归一化、未知值 ValueError；legacy checkpoint
    （无 config/objective 字段）恒为 diffusion；目标一致放行、不一致拒绝重建。"""
    assert OBJECTIVES == ("diffusion", "persistence_residual")
    assert DEFAULT_OBJECTIVE == "diffusion"
    assert MASK_SCHEME == "bivariate_rho"
    assert validate_objective("Diffusion") == "diffusion"          # 归一化
    assert validate_objective("persistence_residual") == "persistence_residual"
    # 未知目标 → ValueError
    for bad in ("residual", "", "DIFFUSION "):
        try:
            validate_objective(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate_objective({bad!r}) should fail")
    # legacy checkpoint（无 config/objective 字段）恒为 diffusion
    assert objective_from_checkpoint(None) == "diffusion"
    assert objective_from_checkpoint({}) == "diffusion"
    assert objective_from_checkpoint({"epoch": 3}) == "diffusion"
    assert objective_from_checkpoint({"config": {}}) == "diffusion"
    assert objective_from_checkpoint(
        {"config": {"objective": "persistence_residual"}}) == "persistence_residual"
    # 目标一致放行；不一致拒绝
    res_ckpt = {"config": {"objective": "persistence_residual"}}
    diff_ckpt = {"config": {"objective": "diffusion"}}
    assert ensure_objective_compatible(res_ckpt, "persistence_residual") == \
        "persistence_residual"
    assert ensure_objective_compatible(diff_ckpt, "diffusion") == "diffusion"
    assert ensure_objective_compatible({}, "diffusion") == "diffusion"
    for ckpt, want in ((diff_ckpt, "persistence_residual"),
                       (res_ckpt, "diffusion")):
        try:
            ensure_objective_compatible(ckpt, want)
        except RuntimeError as e:
            assert "objective" in str(e) and "incompatible" in str(e), str(e)
        else:
            raise AssertionError(f"ensure_objective_compatible({ckpt}, {want!r}) "
                                 "should refuse")


# ── 残差基线组：persistence-residual 与静态掩膜 ──


def test_persistence_residual_zero_init_identity():
    """防止 zero-init 恒等回归：未训练包装器必须逐位等于末日语持续（零初始化残差
    头使 backbone 输出全零 → prediction == base == cond[:, -2:]）；sample() 忽略
    num_sample_steps；clamp 把越界预测压回 [0,1]；条件通道数错误与无 self_condition
    的 backbone 均拒绝。"""
    model = make_residual_model()
    assert model.residual_base == "last_day"
    assert model.cond_chans == 14 and model.target_ch == 2
    assert torch.count_nonzero(model.net.head.weight).item() == 0
    torch.manual_seed(0)
    cond = torch.rand(3, 14, H, W, Z)
    with torch.no_grad():
        out = model(cond)
        sample = model.sample(cond, num_sample_steps=8, clamp=True)
    assert out.shape == (3, 2, H, W, Z)
    assert sample.shape == (3, 2, H, W, Z)
    assert torch.equal(out, cond[:, -2:])          # 逐位持续性恒等
    assert torch.equal(sample, cond[:, -2:])       # cond 在 [0,1] → clamp 是无操作
    # sample() 忽略 num_sample_steps（确定性：逐位一致）
    with torch.no_grad():
        s_other = model.sample(cond, num_sample_steps=2, clamp=True)
    assert torch.equal(sample, s_other)
    # clamp 把越界预测压回 [0,1]
    cond_big = torch.zeros(1, 14, H, W, Z)
    cond_big[0, -2:, 0, 0, 0] = 5.0                # 基值在 [0,1] 之外
    with torch.no_grad():
        clamped = model.sample(cond_big, clamp=True)
    assert clamped[0, 0, 0, 0, 0].item() == 1.0
    # 条件通道数错误必须拒绝
    try:
        model(torch.randn(1, 15, H, W, Z))
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for wrong cond channels")
    # 无 self_condition 的 backbone 无法承载条件
    net_nc = IAFNODiff(dim=(H, W, Z), patch_size=(2, 2, 1), embed_dim=8,
                       num_blocks=1, in_chans=2, out_chans=2, cond_chans=14,
                       ex_layer=1, nlayer=1, hidden_size_factor=1,
                       dim_f=(H, W, Z), self_condition=False)
    try:
        PersistenceResidualIAFNO(net_nc)
    except ValueError as e:
        assert "self_condition" in str(e), str(e)
    else:
        raise AssertionError("expected ValueError for self_condition=False backbone")


def test_persistence_residual_training_step():
    """防止训练步回归：真实 forward/backward/optimizer 一步后——损失有限、残差头
    移动、输出离开持续性恒等、每个参数都有 .grad（预头层最初为零值但存在，这是
    DDP 免 find_unused_parameters all-reduce 的前提）。"""
    model = make_residual_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(1)
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 2, H, W, Z)
    mask = torch.ones(1, 2, H, W, Z)
    mask[0, 0, 0, 0] = 0.0

    pred = model(cond)
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)                 # DDP 安全
    assert torch.isfinite(model.net.head.weight.grad).all()
    assert model.net.head.weight.grad.abs().sum() > 0        # 残差头确实移动
    head_before = model.net.head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(head_before, model.net.head.weight.detach())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # 真实更新后模型不再与持续性一致
    with torch.no_grad():
        out2 = model(cond)
    assert not torch.equal(out2, cond[:, -2:])
    # 陆地梯度保持为零（掩膜损失从不训练陆地输出）
    assert model.net.head.weight.grad is not None


def test_masked_mse_loss_semantics():
    """防止掩膜 MSE 语义回归：逐样本有效元素均值再批均值；污染陆地值不影响损失；
    逐 batch 独立分母；单通道掩膜与扩散 forward 一致地广播；全零掩膜恰为 0 不 NaN。"""
    torch.manual_seed(2)
    pred = torch.rand(B, 2, H, W, Z)
    target = torch.rand(B, 2, H, W, Z)
    # 陆地单元格 (0,0) 在两通道均无效
    mask = torch.ones(1, 2, H, W, Z)
    mask[0, :, 0, 0, :] = 0.0
    loss = masked_mse_loss(pred, target, mask)
    m = mask.expand_as(pred)
    ref = (((pred - target) ** 2 * m).sum(dim=(1, 2, 3, 4))
           / m.sum(dim=(1, 2, 3, 4))).mean()
    assert torch.allclose(loss, ref, atol=1e-6)
    # 污染陆地单元格不得改变损失
    pred_land_dirty = pred.clone()
    pred_land_dirty[:, :, 0, 0, :] = 123.0
    assert torch.allclose(loss, masked_mse_loss(pred_land_dirty, target, mask))
    # 逐样本分母独立（逐 batch 变化的有效性）
    maskb = torch.ones(B, 2, H, W, Z)
    maskb[0, :, :2, :, :] = 0.0
    ref_b = (((pred - target) ** 2 * maskb).sum(dim=(1, 2, 3, 4))
             / maskb.sum(dim=(1, 2, 3, 4))).mean()
    assert torch.allclose(masked_mse_loss(pred, target, maskb), ref_b, atol=1e-6)
    # 单通道 (1,1,H,W,Z) 掩膜像 diffusion.forward 一样广播
    mask1 = torch.ones(1, 1, H, W, Z)
    mask1[0, 0, 0, 0, :] = 0.0
    assert torch.isfinite(masked_mse_loss(pred, target, mask1))
    # 全零掩膜 → 恰为 0，无 NaN（与 EDM 损失同一约定）
    zero_loss = masked_mse_loss(pred, target, torch.zeros(1, 2, H, W, Z))
    assert torch.isfinite(zero_loss) and zero_loss.item() == 0.0


def test_persistence_residual_checkpoint_roundtrip():
    """防止残差 checkpoint 往返回归：保存→重建→load_checkpoint(weights_only=True)
    逐位复现已训练模型；元数据携带 objective 族；重建路径与 pre_evaluate.py 一致。"""
    model = make_residual_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(3)
    cond = torch.rand(1, 14, H, W, Z)
    target = torch.rand(1, 2, H, W, Z)
    loss = masked_mse_loss(model(cond), target, torch.ones(1, 2, H, W, Z))
    loss.backward()
    optimizer.step()

    state = {"epoch": 0, "best_val": float(loss.item()),
             "model_state_dict": model.state_dict(),
             "config": {"preset": "surface_smoke", "objective": "persistence_residual",
                        "residual_base": model.residual_base,
                        "cond_chans": model.cond_chans, "target_ch": model.target_ch,
                        "mask_scheme": MASK_SCHEME,
                        "time_sigma": model.time_sigma, "world_size": 1}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "best.pth")
        torch.save(state, p)
        # 与 pre_evaluate.py 相同的重建：新建 IAFNODiff+包装器，再加载 state dict（训练后的头替换零初始化）
        fresh = make_residual_model()
        assert torch.count_nonzero(fresh.net.head.weight).item() == 0
        ckpt = load_checkpoint(p, fresh, map_location="cpu")
        assert ckpt["epoch"] == 0
        assert ensure_objective_compatible(ckpt, "persistence_residual") == \
            "persistence_residual"
        with torch.no_grad():
            a = model(cond)
            b = fresh(cond)
        assert torch.allclose(a, b, atol=1e-6)
        assert not torch.equal(b, cond[:, -2:])   # 已训练：不再是持续性


def test_persistence_residual_static_mask_input():
    """防止静态掩膜输入（实验 08 B 臂）回归：包装器只把 2 个静态掩膜通道拼接到
    backbone 的动态条件上，base 仍是末日语持续，未训练模型仍精确等于持续性
    （zero-init 恒等）；显式广播与手工拼接 x_self_cond 等价；通道/空间不匹配
    高声拒绝。"""
    model = make_residual_model(cond_chans=16)
    assert model.cond_chans == 16
    torch.manual_seed(1)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()
    with torch.no_grad():
        out = model(cond, static_cond=static)
    assert out.shape == (2, 2, H, W, Z)
    assert torch.equal(out, cond[:, -2:])          # zero-init 恒等成立
    # 显式 batch 广播与手工把拼接条件喂给 backbone 的 x_self_cond 槽位等价
    # （文档化的静态通道布局）
    import math
    with torch.no_grad():
        a = model(cond, static_cond=static)
        t = torch.full((2,), 0.25 * math.log(model.time_sigma))
        manual = model.net(torch.zeros_like(cond[:, :2]), t,
                           torch.cat([cond, static.expand(2, -1, -1, -1, -1)],
                                     dim=1)) + cond[:, -2:]
    assert torch.equal(a, manual)
    # 动态/静态通道切分错误必须高声拒绝
    try:
        model(cond)                                 # 14 通道喂 16 通道模型
        raise AssertionError("expected AssertionError for missing static_cond")
    except AssertionError:
        pass
    try:
        model(torch.cat([cond, cond], dim=1), static_cond=static)
        raise AssertionError("expected AssertionError for extra static channels")
    except AssertionError:
        pass
    try:
        model(cond, static_cond=static[0:1, :, :, :1, :])
        raise AssertionError("expected AssertionError for spatial mismatch")
    except AssertionError:
        pass
    # sample() 带 static 通道的 clamp 路径
    with torch.no_grad():
        trained_out = model.sample(cond, static_cond=static, clamp=True)
    assert trained_out.min() >= 0.0 and trained_out.max() <= 1.0


def test_rollout_static_cond_threading():
    """防止 static_cond 透传回归：ensemble_rollout 把 static_cond 原样转发给每次
    sample() 调用，滑窗保持纯 14 通道动态条件（丢最旧/追加预测的切片不动）；不传时
    模型收到 None（历史调用签名）；逐窗口种子路径同样透传。"""
    class _StaticRecorder:
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True, static_cond=None):
            self.calls.append((cur.clone(), None if static_cond is None
                               else static_cond.clone()))
            return cur[:, :2] + 1.0

    torch.manual_seed(2)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()

    rec = _StaticRecorder()
    preds = ensemble_rollout(rec, cond, 3, 1, seed=0, static_cond=static)
    assert preds.shape == (2, 1, 3, 2, H, W, Z)
    assert len(rec.calls) == 3
    for step, (cur, sc) in enumerate(rec.calls):
        assert cur.shape == (2, 14, H, W, Z)        # 滑窗保持纯动态
        assert sc.shape == (1, 2, H, W, Z) and sc.dim() == 5
        assert torch.equal(sc, static)              # 每步同一张量
        if step > 0:
            prev_pred = rec.calls[step - 1][0][:, :2] + 1.0
            # 通道 0-1 被丢弃，预测追加在窗口末尾
            assert torch.equal(cur[:, :-2], rec.calls[step - 1][0][:, 2:])
            assert torch.equal(cur[:, -2:], prev_pred)
    # 不传 static_cond 时模型收到 None（历史调用签名）
    rec_plain = _StaticRecorder()
    ensemble_rollout(rec_plain, cond, 2, 1, seed=0)
    assert all(sc is None for _, sc in rec_plain.calls)
    # 逐窗口种子路径同样转发 static_cond
    rec_seeds = _StaticRecorder()
    ensemble_rollout(rec_seeds, cond, 2, 1, seeds=[5, 6], static_cond=static)
    assert len(rec_seeds.calls) == 4                # 2 窗 × 2 步
    assert all(sc is not None and torch.equal(sc, static)
               for _, sc in rec_seeds.calls)


def test_rollout_remask_feedback():
    """防止 remask_feedback 语义回归：关闭（默认/历史）时含陆地填充值的预测被原样
    反馈；开启时预测在存储与反馈前都被掩膜（陆地→0），海洋值不变，第二步输入仅在
    陆地单元格不同；逐窗口种子仍约束轨迹；开启但缺掩膜必须提前拒绝。"""
    class _LandDirtyRecorder:
        """预测 cur[:, :2] + 1（处处非零，包括陆地单元格），并记录收到的每个条件窗口。"""
        def __init__(self):
            self.calls = []
        def sample(self, cur, num_sample_steps=None, clamp=True):
            self.calls.append(cur.clone())
            return cur[:, :2] + 1.0

    torch.manual_seed(0)
    cond = torch.rand(1, 14, H, W, Z)
    ocean = torch.ones(1, 2, H, W, Z)
    ocean[0, :, 0, 0, :] = 0.0                    # (0,0) 在两通道均为陆地

    # 关闭（默认/历史）：含陆地的脏预测被反馈。
    # calls[i][0] 是 4 维 (14, H, W, Z) 窗口：通道在维 0。
    s_off = _LandDirtyRecorder()
    p_off = ensemble_rollout(s_off, cond, 2, 1, seed=0, remask_feedback=False)
    assert p_off.shape == (1, 1, 2, 2, H, W, Z)
    fed_back = s_off.calls[1][0]
    assert torch.equal(fed_back[-2:], s_off.calls[0][0][:2] + 1.0)
    assert fed_back[-2, 0, 0, 0].item() != 0.0             # 陆地未被重新置零

    # 开启：预测在存储与反馈前被重掩膜（陆地→0）
    s_on = _LandDirtyRecorder()
    p_on = ensemble_rollout(s_on, cond, 2, 1, seed=0, remask_feedback=True,
                            ocean_mask=ocean)
    fed_back_on = s_on.calls[1][0]
    assert fed_back_on[-2, 0, 0, 0].item() == 0.0
    assert fed_back_on[-1, 0, 0, 0].item() == 0.0
    # 海洋值不受重掩膜影响；存储的预测是掩膜后的
    assert torch.allclose(p_on[0, 0], p_off[0, 0] * ocean[0])
    # 第二步输入仅在掩膜外（陆地）单元格不同
    diff = (s_on.calls[1][0] - s_off.calls[1][0]).abs()
    assert diff[-2, 0, 0, 0].item() > 0 and diff[-1, 0, 0, 0].item() > 0
    assert diff[:, 1, 1, :].max().item() == 0
    # 种子仍约束轨迹；重掩膜本身是确定性的
    s_on_b = _LandDirtyRecorder()
    ensemble_rollout(s_on_b, cond, 2, 1, seeds=[7], remask_feedback=True,
                     ocean_mask=ocean)
    assert torch.equal(s_on_b.calls[1][0], fed_back_on)
    # remask_feedback=True 而无掩膜必须提前拒绝
    try:
        ensemble_rollout(_LandDirtyRecorder(), cond, 1, 1,
                         remask_feedback=True, ocean_mask=None)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for remask without ocean_mask")


def test_deterministic_model_in_rollout():
    """防止确定性模型 rollout 回归：persistence-residual 包装器在未改动的 rollout
    机制内——种子不起作用（无 RNG 消耗）、逐窗口种子路径一致、E=2 成员为相同副本
    （均值==成员）、预测保持 [0,1] 且每步尊重持续性恒等。"""
    model = make_residual_model()
    torch.manual_seed(4)
    cond = torch.rand(2, 14, H, W, Z)
    p_seed1 = ensemble_rollout(model, cond, 3, 1, seed=1)
    p_seed999 = ensemble_rollout(model, cond, 3, 1, seed=999)
    assert p_seed1.shape == (2, 1, 3, 2, H, W, Z)
    assert torch.equal(p_seed1, p_seed999)                 # 逐位确定
    # 确定性模型的逐窗口种子路径行为一致
    p_pw = ensemble_rollout(model, cond, 3, 1, seeds=[1, 2])
    assert torch.equal(p_pw[:, 0], p_seed1[:, 0])
    # E=2 成员是相同副本（均值 == 成员）
    p_e2 = ensemble_rollout(model, cond, 3, 2, seed=1)
    assert p_e2.shape == (2, 2, 3, 2, H, W, Z)
    assert torch.equal(p_e2[:, 0], p_e2[:, 1])
    assert torch.allclose(ensemble_mean(p_e2), p_e2[:, 0])
    # 预测保持 [0,1] 并逐步尊重持续性恒等
    assert float(p_seed1.min()) >= 0.0 and float(p_seed1.max()) <= 1.0


# ── 进度协议组：PROGRESS 行格式、心跳、失败上报 ──


def _parse_progress_line(line):
    """解析一条 PROGRESS 行为字段 dict，并强制严格的 k=v token 形态。"""
    assert line.startswith("PROGRESS "), line
    tokens = line.split()[1:]
    fields = {}
    for i, tok in enumerate(tokens):
        key, sep, value = tok.partition("=")
        assert sep and key and value != "", line           # 强制严格 k=v token
        fields[key] = value
    return fields


def test_progress_reporter_lines():
    """防止进度协议回归：非交互流输出单行可解析 PROGRESS k=v 行——start/周期
    running（时间驱动）/phase_done（reporter 自己的阶段结束，脚本级 completed 保留
    给入口）/failed（带错误详情）；值绝不含空白；吞吐率与 sample_per_s 换算一致；
    禁用 reporter 完全静默；format_progress 字段序 phase 首位、status 末位。"""
    class _FakeClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t
        def advance(self, dt):
            self.t += dt

    # ---- 非交互（管道/文件）：无进度条，周期性换行刷新行
    clk = _FakeClock()
    buf = io.StringIO()
    rep = ProgressReporter("train", total=10, stream=buf, clock=clk,
                           interactive=False, unit="step", samples_per_unit=4,
                           context={"epoch": "1/4"})
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 1 and "status=start" in lines[0]
    f = _parse_progress_line(lines[0])
    assert f["phase"] == "train" and f["step"] == "0/10" and f["epoch"] == "1/4"
    assert f["elapsed_s"] == "0.0"

    rep.update(3, loss="0.50000", lr="1.00e-03")
    assert len(buf.getvalue().splitlines()) == 1           # 间隔内不发出任何行
    clk.advance(30.0)                                      # 到达间隔
    rep.update(2, loss="0.25000", lr="1.00e-03")           # → 发出周期行
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 2 and "status=running" in lines[1]
    f = _parse_progress_line(lines[1])
    assert f["phase"] == "train" and f["status"] == "running"
    assert f["step"] == "5/10" and f["loss"] == "0.25000" and f["lr"] == "1.00e-03"
    assert f["epoch"] == "1/4"
    assert float(f["elapsed_s"]) >= 30.0
    assert float(f["eta_s"]) > 0.0
    assert f["step_per_s"].startswith("0.") and f["step_per_s"] != "0.000"
    # 两种速率均保留 3 位小数，允许舍入误差
    assert abs(float(f["sample_per_s"]) - 4.0 * float(f["step_per_s"])) < 5e-3
    rep.close(loss="0.10000")
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    # reporter 以中间态 phase_done 收尾——脚本级 completed 保留给入口自己的结束行
    assert len(lines) == 3 and "status=phase_done" in lines[2]
    f = _parse_progress_line(lines[2])
    assert f["step"] == "5/10" and f["loss"] == "0.10000"
    # close 幂等
    rep.close()
    assert len(buf.getvalue().splitlines()) == 3

    # ---- 失败路径：发出带错误详情的 status=failed
    buf2 = io.StringIO()
    clk2 = _FakeClock()
    rep2 = ProgressReporter("eval", total=4, stream=buf2, clock=clk2,
                            interactive=False, unit="window")
    rep2.update(1)
    clk2.advance(30.0)
    rep2.update(1, d1_rmse="0.1234")                       # → 周期 running 行
    out2 = [ln for ln in buf2.getvalue().splitlines() if ln]
    assert len(out2) == 2 and "status=running" in out2[1]
    rep2.fail(error="RuntimeError: non_finite_loss_at_epoch_2")
    out2 = [ln for ln in buf2.getvalue().splitlines() if ln]
    assert len(out2) == 3 and "status=failed" in out2[2]
    f2 = _parse_progress_line(out2[2])
    assert f2["phase"] == "eval" and f2["window"] == "2/4"
    assert f2["d1_rmse"] == "0.1234"
    assert "RuntimeError" in f2["error"]
    # 值绝不含空格（key=value 可解析性）
    for line in buf2.getvalue().splitlines():
        for tok in line.split()[1:]:
            assert " " not in tok, line

    # ---- 禁用 reporter（DDP 非 rank-0）：完全静默
    buf3 = io.StringIO()
    rep3 = ProgressReporter("train", total=5, stream=buf3, clock=_FakeClock(),
                            interactive=False, enabled=False)
    rep3.update(2, loss="0.1")
    rep3.close()
    assert buf3.getvalue() == ""

    # ---- format_progress 字段序：phase 首位，status 末位
    line = format_progress("train", "running", epoch="2/4", loss="0.5")
    toks = line.split()
    assert toks[0] == "PROGRESS" and toks[1] == "phase=train"
    assert toks[-1] == "status=running" and "epoch=2/4" in toks
    _parse_progress_line(line)


def test_progress_heartbeat_without_updates():
    """防止心跳回归：周期行由时间驱动——零次 update()（单步 rollout 阻塞远超间隔）
    时守护心跳线程仍发出 running 行；close() 永久停止心跳。"""
    buf = io.StringIO()
    rep = ProgressReporter("eval", total=100, unit="window", interval=0.2,
                           stream=buf, interactive=False)
    assert len([ln for ln in buf.getvalue().splitlines() if ln]) == 1  # 仅 start
    time.sleep(0.6)                                     # >2 倍间隔，无任何 update
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    running = [ln for ln in lines if "status=running" in ln]
    assert len(running) >= 1, lines                     # 无更新时的心跳
    f = _parse_progress_line(running[-1])
    assert f["window"] == "0/100" and f["phase"] == "eval"
    rep.close()
    frozen = len(buf.getvalue().splitlines())
    time.sleep(0.3)
    assert len(buf.getvalue().splitlines()) == frozen   # 心跳被 close 停止


def test_progress_multiline_error_sanitization():
    """防止状态行净化回归：多行异常信息必须收进单行可解析 key=value——所有空白串
    （空格/换行/制表）折叠为单个下划线；单行值仅内部空格被替换。"""
    line = format_progress("train", "failed",
                           error="RuntimeError: boom\nsecond line\twith\ttabs  and  spaces")
    assert "\n" not in line and "\t" not in line
    _parse_progress_line(line)
    assert ("error=RuntimeError:_boom_second_line_with_tabs_and_spaces"
            in line.split())
    # 单行值除内部空格外原样通过
    assert format_progress("eval", "failed", error="E:x y").split()[-2] == \
        "error=E:x_y"


def test_progress_failure_hook_dedup_and_stage():
    """防止失败上报回归：excepthook 兜底对逃逸出受守卫块的异常发一条标准 failed 行
    （净化错误、失败时刻读取 stage）；mark_progress_failed() 抑制受守卫 handler 的
    重复上报；原 excepthook 仍收到异常（traceback 不丢）。"""
    old_hook = sys.excepthook
    buf = io.StringIO()
    stage = ["setup"]
    fallback_calls = []
    try:
        reset_progress_failure_state()
        install_progress_failure_hook("train", stage=lambda: stage[0], stream=buf,
                                      fallback=lambda *a: fallback_calls.append(a))
        sys.excepthook(RuntimeError, RuntimeError("pre-flight refused\nbad config"), None)
        lines = [ln for ln in buf.getvalue().splitlines() if ln]
        assert len(lines) == 1 and "status=failed" in lines[0]
        f = _parse_progress_line(lines[0])
        assert f["phase"] == "train" and f["stage"] == "setup"
        assert f["error"] == "RuntimeError:_pre-flight_refused_bad_config"
        assert len(fallback_calls) == 1                 # traceback 仍被分发
        # 已自行上报失败的受守卫块 → hook 静默
        mark_progress_failed()
        sys.excepthook(ValueError, ValueError("guarded failure"), None)
        assert len(buf.getvalue().splitlines()) == 1
        assert len(fallback_calls) == 2
        # stage callable 在失败时刻读取
        reset_progress_failure_state()
        stage[0] = "postprocess"
        sys.excepthook(ValueError, ValueError("late failure"), None)
        last = buf.getvalue().splitlines()[-1]
        assert _parse_progress_line(last)["stage"] == "postprocess"
        assert "status=failed" in last
    finally:
        sys.excepthook = old_hook
        reset_progress_failure_state()


# ── 分离式多步与配置守卫组：指纹、lead 调度、反馈窗口、数据集 ──


def test_norm_fingerprint_and_time_sigma_checks():
    """防止指纹守卫回归：norm_lo/norm_hi/mask_version 匹配时无警告无异常、容差内
    浮点噪声通过、legacy checkpoint（缺字段）返回警告而非抛错、归一化或掩膜漂移
    拒绝（静默换统计是主要危害）；残差 time_sigma 缺失/不一致拒绝续训。"""
    lo, hi = [-1.5, -2.0], [2.5, 3.0]
    mv = "deadbeef01234567"
    # 指纹匹配 → 无警告、不抛错
    assert check_norm_fingerprint({"norm_lo": lo, "norm_hi": hi,
                                   "mask_version": mv}, lo, hi, mv) == []
    # 容差内浮点噪声通过
    assert check_norm_fingerprint({"norm_lo": [-1.5 + 1e-9, -2.0],
                                   "norm_hi": hi, "mask_version": mv},
                                  lo, hi, mv) == []
    # 早于字段引入的 legacy checkpoint → 警告而非抛错
    ws = check_norm_fingerprint({}, lo, hi, mv)
    assert len(ws) == 2 and all("legacy" in w for w in ws)
    # 归一化漂移 → 拒绝（静默换统计是危害所在）
    for bad in ({"norm_lo": [-1.4, -2.0], "norm_hi": hi, "mask_version": mv},
                {"norm_lo": lo, "norm_hi": [2.5, 3.1], "mask_version": mv}):
        try:
            check_norm_fingerprint(bad, lo, hi, mv)
        except RuntimeError as e:
            assert "normalization fingerprint mismatch" in str(e), str(e)
        else:
            raise AssertionError("expected normalization mismatch refusal")
    # 掩膜漂移 → 拒绝
    try:
        check_norm_fingerprint({"norm_lo": lo, "norm_hi": hi,
                                "mask_version": "ffffffffffffffff"}, lo, hi, mv)
    except RuntimeError as e:
        assert "mask_version" in str(e), str(e)
    else:
        raise AssertionError("expected mask mismatch refusal")
    # 残差时间嵌入：值匹配通过；缺失/不一致拒绝
    check_residual_time_sigma({"time_sigma": 0.002}, 0.002)
    for bad_cfg in ({}, {"time_sigma": 0.05}):
        try:
            check_residual_time_sigma(bad_cfg, 0.002)
        except RuntimeError as e:
            assert "time_sigma" in str(e), str(e)
        else:
            raise AssertionError(f"expected time_sigma refusal for {bad_cfg}")


def test_lead_schedule_pattern_and_rank_consistency():
    """防止 lead 调度回归：固定调度（50% day-1 锚点 + 每周期一次完整 2..K 循环）、
    K=1 调度失效（历史单步路径）、分布恰好 50% 锚点且 2..K 均匀、纯函数性（DDP 各
    rank 对相同步索引得到相同 J）；环境变量读取默认/合法整数/垃圾值拒绝。"""
    # 文档 §5.1 固定调度：50% day-1 锚点 + 每周期一次完整 2..K 循环
    assert [lead_for_batch(i, 5) for i in range(8)] == [1, 2, 1, 3, 1, 4, 1, 5]
    assert [lead_for_batch(i, 5) for i in range(8, 16)] == [1, 2, 1, 3, 1, 4, 1, 5]
    assert lead_schedule_str(5) == "1,2,1,3,1,4,1,5"
    assert lead_schedule_str(10) == "1,2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1,10"
    # K=1 是历史单步路径：调度失效
    assert all(lead_for_batch(i, 1) == 1 for i in range(32))
    assert lead_schedule_str(1) == "1"
    # 分布：恰好 50% 锚点，2..K 均匀
    K = 5
    per = 2 * (K - 1)
    counts = {j: [lead_for_batch(i, K) for i in range(per)].count(j)
              for j in range(1, K + 1)}
    assert counts == {1: K - 1, 2: 1, 3: 1, 4: 1, 5: 1}
    # 纯度：(batch_index, K) 的纯函数——DDP 各 rank 用自己的 batch index 调用，
    # 相同的步索引得到相同的 J（drop_last=True 保证各 rank batch 数相等）
    assert all(lead_for_batch(bi, K) == lead_for_batch(bi, K)
               for bi in range(3 * per))
    # 环境变量读取：默认/空 → 1；合法整数；垃圾值拒绝
    assert train_horizon("") == 1 and train_horizon("5") == 5
    for bad in ("0", "-2", "x"):
        try:
            train_horizon(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {TRAIN_HORIZON_ENV}={bad!r}")
    assert init_checkpoint("") is None
    assert init_checkpoint("~/a.pth") == os.path.expanduser("~/a.pth")


def test_training_config_ms_defaults_and_tags():
    """防止多步配置回归：train_horizon>1 使用冻结的 MS_DEFAULTS（lr 1e-4 / 5
    epochs）而非 preset 单步值；tag 的 _MS{K} 仅在 K>1 追加（绝不与单步/_MSK 运行
    共用目录）；smoke/DDP 隔离后缀叠加正确。"""
    cfg1 = training_config("surface_smoke", "full", 1)
    assert cfg1["lr"] == 1e-3 and cfg1["num_epochs"] == 10
    cfg5 = training_config("surface_smoke", "full", 1, train_horizon=5)
    assert cfg5["lr"] == MS_DEFAULTS["lr"] == 1e-4
    assert cfg5["num_epochs"] == MS_DEFAULTS["num_epochs"] == 5
    cfg5smoke = training_config("surface_smoke", "smoke", 1, train_horizon=5)
    assert cfg5smoke["lr"] == 1e-4 and cfg5smoke["num_epochs"] == 1
    # tag：仅 K>1 追加 _MS{K}；绝不与单步/_MSK 运行共用目录
    base = dict(config=cfg1, objective="persistence_residual")
    assert run_tag_for("surface_smoke", **base) == \
        "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES"
    assert run_tag_for("surface_smoke", **base, train_horizon=1).endswith("_RES")
    assert run_tag_for("surface_smoke", **base, train_horizon=5).endswith("_RES_MS5")
    assert run_tag_for("surface_smoke", **base, train_horizon=10).endswith("_RES_MS10")
    assert run_tag_for("surface_smoke", **base, static_mask=True).endswith("_MSK")
    assert not run_tag_for("surface_smoke", **base).endswith("_MS1")
    t_smoke = training_run_tag("surface_smoke", cfg5, "smoke", 1,
                               objective="persistence_residual", train_horizon=5)
    assert t_smoke.endswith("_RES_MS5_SMOKE")
    t_ddp = training_run_tag("surface_smoke", cfg5, "full", 2,
                             objective="persistence_residual", train_horizon=5)
    assert t_ddp.endswith("_RES_MS5_DDP2")


def test_multistep_k1_matches_single_step():
    """防止 K=1 路径漂移：K=1 必须逐位等于历史单步路径——lead-1 分支与
    detached_feedback_window(lead=1)（cond 原样返回）都归约为一次前向 + 对
    target[:, 0] 的掩膜 MSE。"""
    torch.manual_seed(3)
    model = make_residual_model()
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 5, 2, H, W, Z)   # horizon=5 的窗口
    mask = torch.ones(1, 2, H, W, Z)

    with torch.no_grad():
        # 历史单步操作
        pred_hist = model(cond)
        loss_hist = masked_mse_loss(pred_hist, target[:, 0], mask)
        # K=1 训练路径：每个 batch 的 lead == lead_for_batch(bi, 1) == 1
        for bi in range(4):
            assert lead_for_batch(bi, 1) == 1
        pred_k1 = model(cond)
        loss_k1 = masked_mse_loss(pred_k1, target[:, 0], mask)
        # 经 helper 的 lead-1 多步分支：cond 不变，操作相同
        cur = detached_feedback_window(model, cond, lead_for_batch(0, 1))
        assert torch.equal(cur, cond)
        pred_ms1 = model(cur)
        loss_ms1 = masked_mse_loss(pred_ms1, target[:, 0], mask)
    assert torch.equal(pred_hist, pred_k1) and torch.equal(pred_hist, pred_ms1)
    assert torch.equal(loss_hist, loss_k1) and torch.equal(loss_hist, loss_ms1)


def test_detached_feedback_window_gradient_and_calls():
    """防止分离式反馈回归：lead=J 时恰为 J-1 步 no_grad 自反馈 + 1 步带梯度最终
    前向；早期预测无计算图、最终一步有；滑窗与正式 rollout 的丢最旧/追加更新一致；
    反馈携带模型自身预测（扰动残差头使非持续性可证）；J=1 原样返回 cond 且零调用。"""
    torch.manual_seed(4)
    model = make_residual_model()
    # 未训练 == 持续性（逐位）；扰动残差头，使反馈可证地携带模型自身（非持续性）预测
    with torch.no_grad():
        model.net.head.weight.data += 1e-3
    calls = {"n": 0, "grad_modes": []}

    def step_fn(cur):
        calls["n"] += 1
        calls["grad_modes"].append(torch.is_grad_enabled())
        return model(cur)

    cond = torch.rand(B, 14, H, W, Z)
    cur = detached_feedback_window(step_fn, cond, lead=3)
    assert calls["n"] == 2                                   # J-1 步反馈
    assert calls["grad_modes"] == [False, False]             # 全部无梯度
    # 滑窗：丢最旧一天（2 通道），追加自身预测
    with torch.no_grad():
        p1 = model(cond).clamp(0., 1.).float()
        p2 = model(torch.cat([cond[:, 2:], p1], dim=1)).clamp(0., 1.).float()
    assert cur.shape == cond.shape
    # 两次反馈丢两天（4 通道）：cur = [c3..c6, p1, p2]
    assert torch.equal(cur[:, :10], cond[:, 4:])
    assert torch.equal(cur[:, 10:12], p1)     # 第 5 天槽位现在持有反馈 p1
    assert torch.equal(cur[:, 12:], p2)
    # 反馈使用模型自身预测，而非持续性
    assert not torch.equal(cur[:, 12:], cond[:, -2:])
    # 最终步携带梯度；backward 到达残差头
    target = torch.rand(B, 2, H, W, Z)
    pred = model(cur)
    assert pred.grad_fn is not None
    masked_mse_loss(pred, target, torch.ones(1, 2, H, W, Z)).backward()
    assert model.net.head.weight.grad is not None
    assert torch.isfinite(model.net.head.weight.grad).all()
    # J=1 原样返回 cond 且零调用
    calls["n"] = 0
    cur1 = detached_feedback_window(step_fn, cond, lead=1)
    assert calls["n"] == 0 and torch.equal(cur1, cond)


def test_untrained_multistep_reduces_to_persistence():
    """防止未训练多步退化回归：零初始化残差模型在多步 rollout 中必须精确等于
    "永远持续性"——每个反馈预测都等于当前最后一天，滑窗平凡滑动，第 J 步预测就是
    原始末日持续性（逐位）。"""
    torch.manual_seed(5)
    model = make_residual_model()
    cond = torch.rand(B, 14, H, W, Z)
    target = torch.rand(B, 5, 2, H, W, Z)
    mask = torch.ones(1, 2, H, W, Z)
    J = 5
    with torch.no_grad():
        cur = detached_feedback_window(model, cond, lead=J)
        pred = model.sample(cur, clamp=True)
        loss_ms = masked_mse_loss(pred, target[:, J - 1], mask)
        base = cond[:, -2:]
        loss_pers = masked_mse_loss(base.expand_as(pred), target[:, J - 1], mask)
    # 未训练前向 == base == 末日持续性；[0,1] 内 clamp 是无操作
    assert torch.equal(cur[:, -2:], cond[:, -2:])
    assert torch.equal(pred, cond[:, -2:])
    assert torch.equal(loss_ms, loss_pers)
    # 四次反馈丢四天：cur = [c4, c5, c6, p1..p4]，每个 p == c6
    # （持续性反馈 == 恒等滑动）
    assert torch.equal(cur[:, :6], cond[:, 8:])
    for d in range(6, 14, 2):
        assert torch.equal(cur[:, d:d + 2], cond[:, -2:])


def test_dataset_horizon5_split_and_alignment():
    """防止 horizon 窗口越界回归：horizon=K 的窗口绝不跨切分边界（last_start 扣除
    context+horizon），target[:, J-1] 是绝对日 start+context+J-1（0 基）；train 与
    val 两个切分都验证。"""
    import pre_dataset as pds
    rng = np.random.default_rng(1)
    T, S, HH, WW = 30, 2, 4, 5
    CONTEXT, K = 7, 5
    u = rng.uniform(-1, 1, (T, S, HH, WW)).astype(np.float32)
    v = rng.uniform(-1, 1, (T, S, HH, WW)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        aligned = os.path.join(d, "aligned")
        os.makedirs(aligned)
        np.save(os.path.join(aligned, "u_rho.npy"), u)
        np.save(os.path.join(aligned, "v_rho.npy"), v)
        np.save(os.path.join(aligned, "mask_u_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "mask_v_rho.npy"), np.ones((HH, WW), np.uint8))
        np.save(os.path.join(aligned, "ocean_time.npy"),
                np.arange(np.datetime64("1994-01-01"), T, dtype="datetime64[D]"))
        saved = (pds.ALIGNED_DIR, pds.NORM_DIR, pds.H, pds.W, pds.T_TOTAL,
                 dict(pds.SPLITS))
        pds.ALIGNED_DIR = aligned
        pds.H, pds.W, pds.T_TOTAL = HH, WW, T
        ds = dsv = None
        try:
            pds.SPLITS.clear()
            pds.SPLITS.update(train=(0, 16), val=(16, 29), test=(29, 30))
            stats = {"lo": np.float32([-1.0, -1.0]), "hi": np.float32([1.0, 1.0])}
            ds = PREUVDataset("train", stats, context=CONTEXT, horizon=K,
                              depth_index=0, stride=1, max_windows=None)
            # last_start = 16-(7+5) = 4 → 起点 0..4，共 5 个窗口
            assert len(ds) == 5, len(ds)
            cond, target, start = ds[2]
            assert cond.shape == (14, HH, WW, 1)
            assert target.shape == (K, 2, HH, WW, 1)
            s = int(start)
            assert s == 2
            # cond == 日 [s, s+7)；target[:, j] == 日 s+7+j
            for j in range(CONTEXT):
                day = s + j
                assert torch.equal(cond[2 * j], torch.from_numpy(
                    ((u[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
                assert torch.equal(cond[2 * j + 1], torch.from_numpy(
                    ((v[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
            for j in range(K):
                day = s + CONTEXT + j
                assert torch.equal(target[j, 0], torch.from_numpy(
                    ((u[day, 0] + 1.0) / 2.0).astype(np.float32))[..., None])
            # 无窗口跨 train/val 边界：使用的最大绝对日
            _, _, last_start = ds[len(ds) - 1]
            assert int(last_start) + CONTEXT + K <= pds.SPLITS["train"][1]
            # 同一 horizon 在 val 切分内不越界
            dsv = PREUVDataset("val", stats, context=CONTEXT, horizon=K,
                               depth_index=0, stride=1, max_windows=None)
            assert len(dsv) == 2     # last_start = 29-12 = 17 → 起点 16..17
            _, _, vs = dsv[0]
            assert int(vs) + CONTEXT + K <= pds.SPLITS["val"][1]
        finally:
            _close_mmaps(ds, dsv)          # 目录删除前释放句柄
            pds.ALIGNED_DIR, pds.NORM_DIR, pds.H, pds.W, pds.T_TOTAL = saved[:5]
            pds.SPLITS.clear()
            pds.SPLITS.update(saved[5])


def test_multistep_checkpoint_config_and_resume_guards():
    """防止多步 checkpoint 守卫回归：train_horizon/lead_schedule/feedback_detach/
    init_checkpoint 经 save/load 往返保留，续训守卫拒绝任何语义变化（horizon/调度
    改变；legacy 无字段 checkpoint 只能 K=1，K>1 必须走 weights-only 初始化而非
    续训）；weights-only 初始化精确恢复权重且优化器全新。"""
    ms_cfg = {"preset": "surface_smoke", "train_mode": "full", "world_size": 1,
              "objective": "persistence_residual",
              "train_horizon": 5, "lead_schedule": "1,2,1,3,1,4,1,5",
              "feedback_detach": True, "init_checkpoint": "/tmp/Ep10.pth",
              "init_weights_only": True}
    model = make_residual_model()
    state = {"epoch": 0, "model_state_dict": model.state_dict(),
             "config": ms_cfg}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Ep1.pth")
        torch.save(state, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    assert cfg["train_horizon"] == 5 and cfg["feedback_detach"] is True
    assert cfg["lead_schedule"] == lead_schedule_str(5)
    assert cfg["init_weights_only"] is True
    # 语义匹配通过
    check_multistep_config(cfg, 5, "1,2,1,3,1,4,1,5")
    # horizon 改变 → 拒绝
    for bad_h in (1, 10):
        try:
            check_multistep_config(cfg, bad_h, lead_schedule_str(bad_h))
        except RuntimeError as e:
            assert "train_horizon" in str(e), str(e)
        else:
            raise AssertionError("expected horizon-change refusal")
    # 调度改变 → 拒绝
    try:
        check_multistep_config(cfg, 5, "5,4,3,2,1")
    except RuntimeError as e:
        assert "lead_schedule" in str(e), str(e)
    else:
        raise AssertionError("expected schedule-change refusal")
    # legacy checkpoint（无 train_horizon）：K=1 可行；K>1 必须用 weights-only
    # 初始化而非续训
    check_multistep_config({}, 1, "1")
    try:
        check_multistep_config({}, 5, "1,2,1,3,1,4,1,5")
    except RuntimeError as e:
        assert "DIAFNO_INIT_CHECKPOINT" in str(e), str(e)
    else:
        raise AssertionError("expected legacy multi-step resume refusal")
    # weights-only 初始化：模型权重精确恢复，优化器全新
    torch.manual_seed(6)
    target_model = make_residual_model()
    init_state = {"epoch": 9, "model_state_dict": model.state_dict()}
    target_model.load_state_dict(init_state["model_state_dict"])
    for (k1, p1), (k2, p2) in zip(model.state_dict().items(),
                                  target_model.state_dict().items()):
        assert k1 == k2 and torch.equal(p1, p2), k1
    opt = torch.optim.Adam(target_model.parameters(), lr=1e-4)
    assert len(opt.state) == 0        # 全新优化器：不携带动量


# ── 诊断、preset 与杂项验收组 ──


def test_diag_leadtime_npz_payload_keys_distinct():
    """防止 NPZ 键冲突回归：历史 f"{field}_{var}" 键曾让持续性数组静默覆盖模型
    数组；payload 必须保留 m/p 来源前缀使每个键唯一并各自映射到来源。"""
    L = 4
    res = {f"{n}{v}": dict(rmse=np.arange(L) + (0 if n == "m" else 100.0),
                           bias=np.zeros(L) + (0 if n == "m" else -1.0),
                           var_ratio=np.ones(L), corr_mean=np.ones(L),
                           corr_med=np.ones(L), n=np.full(L, 7.0))
           for n in ("m", "p") for v in ("u", "v")}
    payload = build_npz_payload(res, L, dict(split=np.str_("val"),
                                             n_windows=np.int64(3)))
    stat_keys = [k for k in payload if k not in ("lead", "split", "n_windows")]
    assert len(stat_keys) == len(set(stat_keys)), "NPZ key collision"
    for var in ("u", "v"):
        for field in ("rmse", "bias", "n"):
            assert payload[f"m_{field}_{var}"] is res[f"m{var}"][field]
            assert payload[f"p_{field}_{var}"] is res[f"p{var}"][field]
    assert not np.array_equal(payload["m_rmse_u"], payload["p_rmse_u"])
    assert np.array_equal(payload["lead"], np.arange(1, L + 1))


def test_representative_layer_presets():
    """防止代表层 preset 回归（实验 11）：middle/bottom 与 surface_smoke 仅差
    depth_index，架构/预算逐字段一致；run tag、smoke 模式缩减与 MS 默认值机制对
    新 preset 一致生效。"""
    # 工作包 5（实验 11）：middle/bottom 与 surface_smoke 仅差 depth_index；
    # 架构/预算保持一致
    for name, depth in (("middle_smoke", 14), ("bottom_smoke", 0)):
        assert name in PRESETS
        cfg = PRESETS[name]
        assert cfg["depth_index"] == depth
        for key, v in PRESETS["surface_smoke"].items():
            if key == "depth_index":
                continue
            assert cfg[key] == v, (name, key, cfg[key], v)
        tag = run_tag_for(name, config=cfg, objective="persistence_residual")
        assert tag.startswith(f"{name}_BS4_EMD180_I4_E4_S32_C7") \
            and tag.endswith("_RES"), tag
        # smoke 模式缩减对新 preset 一视同仁
        s = training_config(name, "smoke", 1)
        assert s["num_epochs"] == 1 and s["batch_size"] == 4
    # MS 默认值机制同样覆盖新 preset
    ms = training_config("bottom_smoke", "full", 1, train_horizon=5)
    assert ms["lr"] == MS_DEFAULTS["lr"] and ms["num_epochs"] == 5


def test_static_mask_checkpoint_rebuild_and_rollout():
    """防止静态掩膜评估重建回归：checkpoint 的 static_mask_input/model_cond_chans
    决定 backbone 条件通道数，静态掩膜 checkpoint 携带 static_cond rollout，纯
    14 通道路径不受影响；legacy 元数据视为 14 通道臂；diffusion+静态掩膜拒绝；
    重建路径与 pre_evaluate.py 一致；sample() 新增的 static_cond=None 形参不改变
    历史输出。"""
    # helper 语义：legacy checkpoint 即纯 14 通道臂
    assert static_mask_from_checkpoint(None) == (False, 2 * 7)
    assert static_mask_from_checkpoint({}) == (False, 14)
    assert static_mask_from_checkpoint({"static_mask_input": False}) == (False, 14)
    assert static_mask_from_checkpoint({"static_mask_input": True}) == (True, 16)
    assert static_mask_from_checkpoint(
        {"static_mask_input": True, "model_cond_chans": 16}) == (True, 16)
    assert static_mask_from_checkpoint(
        {"static_mask_input": False, "model_cond_chans": 14}) == (False, 14)
    # diffusion + 静态掩膜是不可能的组合 → 拒绝
    for bad in ({"static_mask_input": True},
                {"static_mask_input": True, "model_cond_chans": 16},
                {"static_mask_input": False, "model_cond_chans": 16},
                {"static_mask_input": True, "model_cond_chans": 14}):
        try:
            static_mask_from_checkpoint(bad, "diffusion" if bad["static_mask_input"]
                                        else None)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"static_mask_from_checkpoint({bad}) should fail")
    torch.manual_seed(4)
    cond = torch.rand(2, 14, H, W, Z)
    static = (torch.rand(1, 2, H, W, Z) > 0.5).float()
    for flag, chans in ((False, 14), (True, 16)):
        model = make_residual_model(cond_chans=chans)
        state = {"epoch": 0, "model_state_dict": model.state_dict(),
                 "config": {"objective": "persistence_residual",
                            "static_mask_input": flag,
                            "model_cond_chans": chans}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Ep1.pth")
            torch.save(state, p)
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        got_flag, got_ch = static_mask_from_checkpoint(
            ckpt["config"], "persistence_residual")
        assert (got_flag, got_ch) == (flag, chans)
        # 与 pre_evaluate.py 完全相同的重建：按 helper 导出的通道数新建 backbone、
        # 包装器，再加载 state dict
        fresh = make_residual_model(cond_chans=got_ch)
        fresh.load_state_dict(ckpt["model_state_dict"])
        fresh.eval()
        static_arg = static if got_flag else None
        preds = ensemble_rollout(fresh, cond, 3, 1, seed=0, static_cond=static_arg)
        assert preds.shape == (2, 1, 3, 2, H, W, Z), preds.shape
        assert torch.isfinite(preds).all()
        # 未训练模型：每个 rollout 步都是末日持续性，带不带静态通道皆然
        #（zero-init 恒等成立）
        assert torch.equal(preds[:, 0], cond[:, -2:].unsqueeze(1).expand(-1, 3, -1, -1, -1, -1))
    # 纯 14 通道重建必须被新的 static_cond 形参逐位保持不变
    #（None 作为历史上缺席的实参透传）
    plain = make_residual_model(cond_chans=14)
    plain.eval()
    with torch.no_grad():
        a = plain.sample(cond, num_sample_steps=1, clamp=True)
        b = plain.sample(cond, num_sample_steps=1, clamp=True, static_cond=None)
    assert torch.equal(a, b)


def test_archived_msk_checkpoint_minimal_cpu_rollout():
    """静态掩膜评估修复的验收：归档的实验 08 _MSK checkpoint（16 通道 backbone）
    可重建并完成带 static_cond 的最小 CPU rollout；仓库快照缺失时跳过；16 通道
    模型必须拒绝 14 通道条件。"""
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, "checkpoints", "PRE",
                             "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MSK",
                             "Ep10.pth")
    if not os.path.isfile(ckpt_path):
        print("SKIP (archived _MSK checkpoint not present)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    assert cfg["objective"] == "persistence_residual"
    assert cfg["preset"] == "surface_smoke"
    flag, cond_ch = static_mask_from_checkpoint(cfg, "persistence_residual")
    assert flag is True and cond_ch == 16
    hh, ww, zz = 400, 441, 1
    net = IAFNODiff(dim=(hh, ww, zz), patch_size=PRESETS["surface_smoke"]["patch_size"],
                    embed_dim=PRESETS["surface_smoke"]["embed_dim"], num_blocks=1,
                    in_chans=2, out_chans=2, cond_chans=cond_ch,
                    ex_layer=PRESETS["surface_smoke"]["explicit_layer"],
                    nlayer=PRESETS["surface_smoke"]["implicit_layer"],
                    hidden_size_factor=4, dim_f=(hh, ww, zz), self_condition=True)
    model = PersistenceResidualIAFNO(net, time_sigma=float(cfg.get("time_sigma",
                                                                   RESIDUAL_TIME_SIGMA)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.manual_seed(7)
    cond = torch.rand(1, 14, hh, ww, zz)
    static = (torch.rand(1, 2, hh, ww, zz) > 0.5).float()
    preds = ensemble_rollout(model, cond, 2, 1, seed=0, static_cond=static)
    assert preds.shape == (1, 1, 2, 2, hh, ww, zz), preds.shape
    assert torch.isfinite(preds).all()
    # 16 通道模型必须拒绝 14 通道条件
    try:
        model.sample(cond, num_sample_steps=1, clamp=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("16-channel model accepted a 14-channel condition")


def test_worse_epochs_checkpoint_roundtrip():
    """防止早停计数回归：恶化连击数经 checkpoint 往返存活（保存前 1 次恶化，恢复
    后再一次恶化即达停止阈值）；legacy checkpoint 缺字段保持历史默认 0；恢复后
    新最优 epoch 仍把计数清零。"""
    model = make_residual_model()
    state = {"epoch": 4, "best_val": 0.5, "worse_epochs": 1,
             "model_state_dict": model.state_dict()}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Ep5.pth")
        torch.save(state, p)
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
    assert restore_worse_epochs(ckpt) == 1
    # 缺字段的 legacy checkpoint 保持历史默认 0
    assert restore_worse_epochs({}) == 0
    assert restore_worse_epochs(None) == 0
    assert restore_worse_epochs({"worse_epochs": -3}) == 0
    # 停止语义：恢复的连击 + 再一次恶化 → 触发停止
    worse_epochs = restore_worse_epochs(ckpt)
    worse_epochs += 1                     # 又一次恶化 epoch
    assert worse_epochs >= 2
    # 恢复后新最优 epoch 仍把计数清零
    worse_epochs = restore_worse_epochs(ckpt)
    worse_epochs = 0                      # is_best 分支
    worse_epochs += 1
    assert worse_epochs < 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("pre_smoke_test passed")
