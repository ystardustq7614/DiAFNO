#!/usr/bin/env python3
"""模块职责：PRE_ocean_data 训练入口：7 天条件窗口 -> 次日 u/v（条件 EDM 或
确定性 persistence-residual；可选分离式多步训练）。

任务（见 docs/operations/PRE_runbook.md）：
    cond   = rho 网格共定位的连续 7 天原始 u/v -> 14 通道（day-major u/v 交错）
    target = 第 8 天 u/v                                ->  2 通道
    15 天预报由 pre_evaluate.py 的自回归 rollout 产出。

四个预设（DIAFNO_PRESET）：
    'surface_smoke' : 仅表层（depth_index=29），网格 400x441x1，patch (4,3,1)
    'middle_smoke'  : 中层 sigma 层（depth_index=14），其余与 surface_smoke 相同
    'bottom_smoke'  : 底层 sigma 层（depth_index=0），  其余与 surface_smoke 相同
    'full3d'        : 全部 30 层 sigma，网格 400x441x30，patch (4,3,2)
所有 patch 选项都整除网格，因此 IAFNO 不触发 padding。

训练目标（DIAFNO_OBJECTIVE）：
    'diffusion'            : 条件 EDM（legacy 默认）
    'persistence_residual' : 确定性 PersistenceResidualIAFNO 基线
                             （末日持续性 + 零初始化残差头，masked-MSE 目标，
                             run 标签后缀 _RES）

分离式多步（DIAFNO_TRAIN_HORIZON=K，doc
docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md §5；run 标签后缀 _MS{K}）：
    仅 persistence_residual，且无静态掩膜。对 batch i 训练 lead
    J = lead_for_batch(i, K)（K=5 的固定调度 1,2,1,3,1,4,1,5,...；50% day-1
    锚点）；模型自己的预测在 torch.no_grad() 下前推 J-1 步（clamp [0,1]、rf0、
    与正式 rollout 相同的滑窗），只反传第 J 步。K=1 即历史单步路径。MS 运行
    默认 lr 1e-4 / 5 个 epoch（pre_config.MS_DEFAULTS），支持经
    DIAFNO_INIT_CHECKPOINT 从已结束的运行做仅权重初始化（全新 optimizer/
    scheduler/history；与 DIAFNO_CHECKPOINT 互斥）。

不负责：不可作为库导入——本文件是 module-top-level 脚本，import 即执行 DDP
初始化、数据加载与训练；共享逻辑一律放 pre_config.py / pre_rollout.py 等
模块，绝不从本文件 import。

从仓库根目录运行（安全默认是短 smoke 运行）：
    python pre_trainer.py
    DIAFNO_TRAIN_MODE=full python pre_trainer.py
    DIAFNO_TRAIN_MODE=full torchrun --standalone --nproc_per_node=4 pre_trainer.py
"""
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from utilities3 import count_params, load_checkpoint
from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff
from pre_models import PersistenceResidualIAFNO, masked_mse_loss
from pre_config import (OUT_ROOT, CONTEXT, TARGET_CH, training_config,
                        training_run_tag, static_mask_input,
                        SIGMA_DATA_SCALE, sigma_data_from_stats,
                        sigma_data_from_checkpoint, resume_sigma_decision,
                        DEFAULT_OBJECTIVE, MASK_SCHEME, RESIDUAL_TIME_SIGMA,
                        STATIC_MASK_CHANNELS,
                        train_horizon, init_checkpoint,
                        lead_for_batch, lead_schedule_str,
                        check_multistep_config, restore_worse_epochs,
                        validate_objective, ensure_objective_compatible,
                        check_norm_fingerprint, check_residual_time_sigma,
                        ProgressReporter, format_progress,
                        install_progress_failure_hook, mark_progress_failed)
from pre_dataset import (PREUVDataset, build_mask_tensor, compute_or_load_stats,
                         mask_version)
from pre_metrics import masked_rel_l2
from pre_rollout import detached_feedback_window

# torchrun 注入的 DDP 环境变量；多进程仅支持 CUDA/NCCL（CPU 上多进程拒绝启动）。
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DISTRIBUTED = WORLD_SIZE > 1
if DISTRIBUTED:
    if not torch.cuda.is_available():
        raise RuntimeError("multi-process training requires CUDA/NCCL; launch one process on CPU")
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl")
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", LOCAL_RANK) if DISTRIBUTED else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN = RANK == 0


def log(*args, **kwargs):
    """仅 rank 0 打印；其余 DDP rank 静默，避免进度输出重复。"""
    if IS_MAIN:
        print(*args, **kwargs)


# 所有 rank 必须初始化出相同的权重；rank 专属的 RNG 流在下方 DDP 同步模型之后
# 才选择。
torch.manual_seed(123)
log(f"Using device: {device}  world_size={WORLD_SIZE} rank={RANK}")

# 逃出受保护训练块的异常统一发标准 status=failed 行（初始化 / 数据 / 模型 /
# pre-flight 失败时没有活跃 reporter）。受保护块自身的 handler 经
# mark_progress_failed() 去重；非 rank-0 的 DDP rank 打印普通 traceback。
if IS_MAIN:
    install_progress_failure_hook("train")

# rank-0 进度行的诚实 scope 标注：DDP 下训练与验证 loader 都按 rank 分片，其
# step/batch 总量是每 rank 的、绝不是全局的（全局样本吞吐单独标注为 sample_per_s）
PROGRESS_SCOPE = f"rank{RANK}_shard_of_{WORLD_SIZE}" if DISTRIBUTED else "whole_split"

# 预设配置

PRESET = os.environ.get("DIAFNO_PRESET", "surface_smoke")
TRAIN_MODE = os.environ.get("DIAFNO_TRAIN_MODE", "smoke").lower()
OBJECTIVE = validate_objective(os.environ.get("DIAFNO_OBJECTIVE", DEFAULT_OBJECTIVE))
# Phase-5 掩膜输入 A/B（B 臂）：静态掩膜通道只对确定性目标实现；diffusion
# 路径保持其精确历史布局（拒绝而不是静默改变 EDM 输入形状）。
STATIC_MASK = static_mask_input()
if STATIC_MASK and OBJECTIVE != "persistence_residual":
    raise RuntimeError(
        "DIAFNO_STATIC_MASK=1 is only supported with "
        "DIAFNO_OBJECTIVE=persistence_residual (the diffusion path keeps its "
        "historical 14-channel layout)")

# 分离式多步（工作包 2）：K=1 即历史单步 teacher-forcing 路径（按位一致）；
# K>1 用模型自己的分离反馈复刻正式确定性 rollout（doc §5）。仅允许确定性
# 目标、且不带（实验 08/09 已否决的）掩膜臂。
TRAIN_HORIZON = train_horizon()
LEAD_SCHEDULE = lead_schedule_str(TRAIN_HORIZON)
if TRAIN_HORIZON > 1 and (OBJECTIVE != "persistence_residual" or STATIC_MASK):
    raise RuntimeError(
        f"DIAFNO_TRAIN_HORIZON={TRAIN_HORIZON} (detached multi-step) is only "
        "supported with DIAFNO_OBJECTIVE=persistence_residual and "
        "DIAFNO_STATIC_MASK unset: the feedback must mirror the formal "
        "deterministic rollout (rf0, no static mask channels)")
cfg = training_config(PRESET, TRAIN_MODE, WORLD_SIZE, train_horizon=TRAIN_HORIZON)

# 供一次性运行使用的按预设覆盖项。常规默认值只存在于 pre_config.py，避免
# 调度视界与文档化预设发生漂移。
EPOCH_OVERRIDES = {}
VAL_SEED = 1234            # 验证期 diffusion 采样的固定种子

# 检查点 sigma_data 与当前（SD2）尺度不一致时的 resume 策略——默认拒绝，因为
# 静默混用尺度既会破坏 EDM 预条件，又会产生矛盾的元数据
# （目录=*_SD2、scale=2.0、实际 sigma_data=旧值）：
#   "error"   : 抛 RuntimeError（安全默认）
#   "migrate" : 显式尺度迁移——保留当前（SD2）尺度，继续写入 SD2 运行目录
#   "adopt"   : 显式 legacy 延续——严格沿用检查点的（旧）尺度，输出写入检查点
#               目录的专属子目录，并在 config 中记录真实尺度（1.0）；resume 出的
#               运行绝不会被误认成 SD2，也绝不覆盖原实验的 Ep{n}.pth / loss.dat
RESUME_SIGMA_POLICY = "error"

# "adopt" 延续使用的子目录（位于被 resume 检查点旁）
LEGACY_RESUME_DIR = "legacy_resume"

# 任务固定常量

COND_CH = 2 * CONTEXT              # 14 通道，day-major u/v 交错（见 pre_dataset.py）
# backbone 条件通道数：动态窗口加上（B 臂）经 static_cond 单独转发的两个静态
# 掩膜通道
MODEL_COND_CH = COND_CH + (STATIC_MASK_CHANNELS if STATIC_MASK else 0)
H, W = 400, 441
Z = 30 if cfg["depth_index"] is None else 1

hidden_size_factor = 4
num_blocks = 1                     # AFNO 通道分组数
checkpoint_path = os.environ.get("DIAFNO_CHECKPOINT") or None
if checkpoint_path is not None:
    checkpoint_path = os.path.expanduser(checkpoint_path)

# 仅权重初始化（全新 optimizer/scheduler/history）与完整 resume 互斥，且限定
# 于它规划的确定性目标（doc §6 WP3：从已完成的实验 07 Ep10 权重初始化 MS5）
INIT_CHECKPOINT = init_checkpoint()
if INIT_CHECKPOINT is not None:
    if checkpoint_path is not None:
        raise RuntimeError(
            "DIAFNO_INIT_CHECKPOINT (weights-only init) and DIAFNO_CHECKPOINT "
            "(full resume) are mutually exclusive; remove one of them")
    if OBJECTIVE != "persistence_residual":
        raise RuntimeError(
            "DIAFNO_INIT_CHECKPOINT is only supported with "
            "DIAFNO_OBJECTIVE=persistence_residual")

run_tag = training_run_tag(PRESET, cfg, TRAIN_MODE, WORLD_SIZE, OBJECTIVE,
                           static_mask=STATIC_MASK, train_horizon=TRAIN_HORIZON)
# "adopt" 延续时该目录会改指向被 resume 检查点自己的目录（见下方 resume 段）
run_dir = os.path.join(OUT_ROOT, run_tag)

# 数据

# stats 缓存缺失或过期时构建代价高，且多 rank 并发写不安全。由 rank 0 创建，
# 其余 rank 再加载。
if DISTRIBUTED:
    stats = compute_or_load_stats(depth_index=cfg["depth_index"]) if IS_MAIN else None
    dist.barrier()
    if not IS_MAIN:
        stats = compute_or_load_stats(depth_index=cfg["depth_index"], verbose=False)
else:
    stats = compute_or_load_stats(depth_index=cfg["depth_index"])
# 逐变量 min-max 反归一化范围，reshape 成 (1,2,1,1,1) 以便与 (B,2,H,W,Z) 广播
y_lo = torch.tensor(stats["lo"], device=device).reshape(1, 2, 1, 1, 1)
y_hi = torch.tensor(stats["hi"], device=device).reshape(1, 2, 1, 1, 1)

# 多步：训练窗口覆盖 lead 1..K（target[:, J-1] 选取训练 lead）；验证窗口保持
# 单步，因为每 epoch 的 val_masked_relL2 只是训练健康信号（正式选型 =
# pre_evaluate.py 的验证 15 天确定性协议）。
train_dataset = PREUVDataset("train", stats, context=CONTEXT, horizon=TRAIN_HORIZON,
                             depth_index=cfg["depth_index"], stride=cfg["train_stride"],
                             max_windows=cfg["max_train_windows"])
val_dataset = PREUVDataset("val", stats, context=CONTEXT, horizon=1,
                           depth_index=cfg["depth_index"], stride=1)
log(f"train windows: {len(train_dataset)}   val windows: {len(val_dataset)}")

# DistributedSampler 以固定种子(123)打乱窗口并按 rank 均分；drop_last=True
#（sampler 与 loader 两处）使各 rank 的 batch 数一致——lead_for_batch 依赖
# 这一点来保证各 rank 集体同步安全（collective-safe）。
train_sampler = DistributedSampler(
    train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True,
    seed=123, drop_last=True) if DISTRIBUTED else None
# batch_size 是 per-device 语义；LR 不随 world_size 自动缩放。
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=cfg["batch_size"],
    shuffle=train_sampler is None, sampler=train_sampler,
    num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", drop_last=True)
if len(train_loader) == 0:
    raise RuntimeError(
        f"training loader is empty: windows={len(train_dataset)}, world_size={WORLD_SIZE}, "
        f"per_device_batch={cfg['batch_size']}")
# 验证：固定数量的窗口均匀铺满整个验证期（确定性 linspace，无 RNG），因此各
# epoch 的检查点可相互比较。
val_idx = np.linspace(0, len(val_dataset) - 1, cfg["val_windows"]).astype(int)
rank_val_idx = val_idx[RANK::WORLD_SIZE]   # 验证窗口按 rank 交错分片，epoch 末经 all_reduce 聚合
val_subset = torch.utils.data.Subset(val_dataset, rank_val_idx.tolist())
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=cfg["batch_size"],
                                         shuffle=False, num_workers=cfg["num_workers"],
                                         pin_memory=device.type == "cuda", drop_last=False)
log(f"val subset: {len(val_idx)} windows at indices {val_idx[0]}..{val_idx[-1]} "
    f"({WORLD_SIZE} rank shard(s))")

mask = build_mask_tensor(device, cfg["depth_index"])   # (1,2,H,W,Z) 双变量掩膜

# 模型

dm_backbone = IAFNODiff(
    dim=(H, W, Z),
    patch_size=cfg["patch_size"],
    embed_dim=cfg["embed_dim"],
    num_blocks=num_blocks,
    in_chans=TARGET_CH,
    out_chans=TARGET_CH,
    cond_chans=MODEL_COND_CH,
    ex_layer=cfg["explicit_layer"],
    nlayer=cfg["implicit_layer"],
    hidden_size_factor=hidden_size_factor,
    dim_f=(H, W, Z),
    self_condition=True,
).to(device)

if OBJECTIVE == "diffusion":
    model = ElucidatedDiffusion(
        dm_backbone,
        channels=TARGET_CH,
        num_sample_steps=cfg["sampling_steps"],
        image_size_h=H,
        image_size_w=W,
        image_size_z=Z,
        sigma_data=sigma_data_from_stats(stats["sigma"]),   # [-1,1] 图像空间尺度
    )
else:
    # 确定性 persistence-residual 基线：prediction = 末日持续性 + 残差；零初始化
    # 的残差头使未训练模型恰好等于持续性（下方在任何训练发生前验证）。
    model = PersistenceResidualIAFNO(dm_backbone, time_sigma=RESIDUAL_TIME_SIGMA)
    with torch.no_grad():
        probe = torch.rand(1, COND_CH, H, W, Z, device=device)
        probe_static = mask if STATIC_MASK else None
        ident = model(probe, static_cond=probe_static)
    if not torch.equal(ident, probe[:, -TARGET_CH:]):
        raise RuntimeError(
            "zero-initialized persistence-residual model does not reduce to "
            "last-day persistence; refusing to train")
    log("zero-init check passed: untrained residual model == last-day persistence"
        + (" (with static mask input)" if STATIC_MASK else ""))

optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
n_epochs = EPOCH_OVERRIDES.get(PRESET) or cfg["num_epochs"]
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * len(train_loader))
scaler = GradScaler(device.type)   # torch.amp.GradScaler（新 AMP API）

# resume（history 与 best_val 必须存续）

hist = {"train": [], "val_rel": [], "time": []}
best_val = float("inf")
start_epoch = 0
worse_epochs = 0   # val_masked_relL2 连续严格高于最佳的 epoch 数
sigma_scale = SIGMA_DATA_SCALE      # 实际的 stats_sigma -> sigma_data 乘数
adopted = False                     # legacy "adopt" 延续（仅 diffusion）
if checkpoint_path is not None:
    ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                           map_location=device)
    ckpt_cfg = ckpt.get("config") or {}
    if "preset" in ckpt_cfg and ckpt_cfg["preset"] != PRESET:
        raise RuntimeError(
            f"checkpoint preset={ckpt_cfg['preset']!r} vs current {PRESET!r}")
    if "train_mode" in ckpt_cfg and ckpt_cfg["train_mode"] != TRAIN_MODE:
        raise RuntimeError(
            f"checkpoint train_mode={ckpt_cfg['train_mode']!r} cannot resume in "
            f"{TRAIN_MODE!r}; smoke checkpoints are pipeline gates, not full-run starts")
    if "world_size" in ckpt_cfg and int(ckpt_cfg["world_size"]) != WORLD_SIZE:
        raise RuntimeError(
            f"checkpoint world_size={ckpt_cfg['world_size']} vs current {WORLD_SIZE}; "
            "resume with the original GPU count so optimizer/scheduler semantics stay fixed")
    if DISTRIBUTED and "world_size" not in ckpt_cfg:
        raise RuntimeError(
            "checkpoint predates DDP world-size metadata; resume it on one GPU or "
            "start a fresh multi-GPU run")
    # 绝不把一个模型类加载进另一类，也绝不跨结构变更 resume（缺失这些字段的
    # legacy 检查点早于目标拆分存在，只能是 diffusion 运行——下方有防线）
    ckpt_objective = ensure_objective_compatible(ckpt, OBJECTIVE)
    # 多步语义必须在 resume 后原样存续（doc §6 WP2 item 6）
    check_multistep_config(ckpt_cfg, TRAIN_HORIZON, LEAD_SCHEDULE)
    for key, current in (("cond_chans", COND_CH), ("target_ch", TARGET_CH),
                         ("mask_scheme", MASK_SCHEME),
                         ("static_mask_input", STATIC_MASK)):
        if key in ckpt_cfg and ckpt_cfg[key] != current:
            raise RuntimeError(
                f"checkpoint {key}={ckpt_cfg[key]!r} vs current {current!r}; "
                "refusing to resume across a structural change")
    if OBJECTIVE == "persistence_residual" and "residual_base" in ckpt_cfg \
            and ckpt_cfg["residual_base"] != "last_day":
        raise RuntimeError(
            f"checkpoint residual_base={ckpt_cfg['residual_base']!r} is not "
            "supported (only 'last_day')")
    # 数据语义指纹：记录的归一化范围与掩膜版本必须与当前 stats/masks 一致，否则
    # resume 出的运行会在与检查点不同的数据语义上静默训练（缺失记录字段的
    # legacy 检查点只能告警）
    for fp_warning in check_norm_fingerprint(ckpt_cfg, stats["lo"], stats["hi"],
                                             mask_version()):
        log(f"WARNING: {checkpoint_path}: {fp_warning}")
    if OBJECTIVE == "persistence_residual":
        check_residual_time_sigma(ckpt_cfg, model.time_sigma)
        if "stats_sigma" in ckpt_cfg and \
                abs(float(ckpt_cfg["stats_sigma"]) - float(stats["sigma"])) > 1e-6:
            raise RuntimeError(
                f"checkpoint stats_sigma={float(ckpt_cfg['stats_sigma']):.6f} vs "
                f"current {float(stats['sigma']):.6f}; the residual objective has "
                "no sigma migration policy — refusing to resume")
    if OBJECTIVE == "diffusion":
        # sigma_data 预条件只存在于 EDM 目标
        sd_ckpt, sd_in_ckpt = sigma_data_from_checkpoint(ckpt, stats["sigma"])
        if not sd_in_ckpt:
            log(f"WARNING: {checkpoint_path} has no config.sigma_data (legacy "
                f"checkpoint); its sigma_data is the old stats-only scale {sd_ckpt:.5f}")
        model.sigma_data, adopted = resume_sigma_decision(
            sd_ckpt, model.sigma_data, RESUME_SIGMA_POLICY)
        if adopted:
            # legacy 延续：写入检查点旁的专属子目录——原实验的 Ep{n}.pth /
            # loss.dat 绝不被触碰，且该运行绝不会被误认成 SD2 运行
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            if os.path.basename(ckpt_dir) == LEGACY_RESUME_DIR:
                run_dir = ckpt_dir        # 正在 resume 上一次的 legacy 延续
            else:
                run_dir = os.path.join(ckpt_dir, LEGACY_RESUME_DIR)
            sigma_scale = model.sigma_data / float(stats["sigma"])
            log(f"adopted checkpoint scale: sigma_data={model.sigma_data:.5f} "
                f"(stats_sigma x {sigma_scale:.3f}); outputs -> {run_dir}")
        elif abs(sd_ckpt - model.sigma_data) <= 1e-6:
            log(f"checkpoint sigma_data {sd_ckpt:.5f} matches the current "
                f"(SD2) scale")
    else:
        log(f"residual checkpoint objective={ckpt_objective!r}; sigma_data "
            "policy not applicable to the deterministic objective")
    start_epoch = ckpt.get("epoch", -1) + 1
    # 早停连续段必须在 resume 后存续：既有的变差计数仍导向同样的连续 2 个
    # epoch 停止（缺失字段的 legacy 检查点保持历史默认 0）
    worse_epochs = restore_worse_epochs(ckpt)
    if worse_epochs:
        log(f"restored early-stop counter: {worse_epochs} consecutive "
            f"worsening epoch(s) from {checkpoint_path}")

loss_file = os.path.join(run_dir, "loss.dat")
# adopt 时历史从原实验读取（延续目录是全新的），因此写出的 loss.dat 总是包含
# 完整历史；其余模式的历史来源就是输出目录本身。
hist_src = loss_file
if checkpoint_path is not None and adopted:
    hist_src = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                            "loss.dat")

if checkpoint_path is not None:
    best_val = ckpt.get("best_val")
    if best_val is None:
        # 更早的检查点没有 best_val：从 loss.dat 历史重算
        if os.path.exists(hist_src):
            arr = np.loadtxt(hist_src).reshape(-1, 3)
            best_val = float(arr[:start_epoch, 2].min())
            log(f"recomputed best_val={best_val:.5f} from {hist_src}")
        else:
            best_val = float("inf")
            log("WARNING: checkpoint has no best_val and loss.dat is missing; "
                "starting best_val from inf")
    log(f"resumed from {checkpoint_path} (epoch {start_epoch}, "
        f"best_val={best_val:.5f})")
    if os.path.exists(hist_src):
        arr = np.loadtxt(hist_src).reshape(-1, 3)
        n_old = min(start_epoch, len(arr))
        hist["time"] = list(arr[:n_old, 0])
        hist["train"] = list(arr[:n_old, 1])
        hist["val_rel"] = list(arr[:n_old, 2])
        log(f"restored {n_old} epochs of history from {hist_src}")

# 仅权重初始化（与上方 resume 互斥）：只从已结束的运行加载模型权重——来源的
# optimizer/scheduler/scaler/epoch/history 一概不恢复（来源 cosine 调度已结束；
# doc §6 WP3）。这里的一切保持全新：hist/best_val/start_epoch 保持初始值，
# 运行写入自己的 _MS{K} 目录。
if INIT_CHECKPOINT is not None:
    init = torch.load(INIT_CHECKPOINT, map_location=device, weights_only=True)
    init_cfg = init.get("config") or {}
    ensure_objective_compatible(init, OBJECTIVE)
    if "preset" in init_cfg and init_cfg["preset"] != PRESET:
        raise RuntimeError(
            f"init checkpoint preset={init_cfg['preset']!r} vs current {PRESET!r}; "
            "weights-only init must stay within the same architecture preset")
    if "static_mask_input" in init_cfg and init_cfg["static_mask_input"] != STATIC_MASK:
        raise RuntimeError(
            f"init checkpoint static_mask_input={init_cfg['static_mask_input']!r} "
            f"vs current {STATIC_MASK!r}; refusing to init across a structural change")
    for fp_warning in check_norm_fingerprint(init_cfg, stats["lo"], stats["hi"],
                                             mask_version()):
        log(f"WARNING: {INIT_CHECKPOINT}: {fp_warning}")
    if OBJECTIVE == "persistence_residual":
        check_residual_time_sigma(init_cfg, model.time_sigma)
        if "stats_sigma" in init_cfg and \
                abs(float(init_cfg["stats_sigma"]) - float(stats["sigma"])) > 1e-6:
            raise RuntimeError(
                f"init checkpoint stats_sigma={float(init_cfg['stats_sigma']):.6f} "
                f"vs current {float(stats['sigma']):.6f}; the residual objective "
                "has no sigma migration policy — refusing to init")
    model.load_state_dict(init["model_state_dict"])
    log(f"weights-only init from {INIT_CHECKPOINT} "
        f"(source epoch {init.get('epoch')}); optimizer/scheduler/scaler/history "
        "are FRESH", flush=True)

os.makedirs(run_dir, exist_ok=True)

log("Model Total Params:", count_params(model))
if OBJECTIVE == "diffusion":
    scale_info = (f"stats_sigma={stats['sigma']:.5f} "
                  f"sigma_data={model.sigma_data:.5f} (scale {sigma_scale:.3f}x)")
else:
    scale_info = (f"stats_sigma={stats['sigma']:.5f} objective=persistence_residual "
                  f"(residual_base={model.residual_base}, time_sigma={model.time_sigma:g}; "
                  "sigma_data not applicable)")
log(f"preset={PRESET} mode={TRAIN_MODE} objective={OBJECTIVE} grid=({H},{W},{Z}) "
    f"patch={cfg['patch_size']} cond_ch={COND_CH} model_cond_ch={MODEL_COND_CH} "
    f"static_mask_input={STATIC_MASK} target_ch={TARGET_CH} "
    f"mask_scheme={MASK_SCHEME} {scale_info} epochs={n_epochs} "
    f"train_horizon={TRAIN_HORIZON} lead_schedule={LEAD_SCHEDULE} "
    f"init_checkpoint={INIT_CHECKPOINT} "
    f"world_size={WORLD_SIZE} per_device_batch={cfg['batch_size']} "
    f"effective_batch={cfg['batch_size'] * WORLD_SIZE} run_dir={run_dir}")

# pre-flight 检查（在浪费一次训练之前拒绝）

# 本次运行要写的每个 epoch 文件位必须空闲：冲突会静默改写历史（例如 resume
# 一个早于同实验后续 epoch 的检查点）。只在上前方检查一次。
for ep in range(start_epoch, n_epochs):
    ep_out = os.path.join(run_dir, f"Ep{ep + 1}.pth")
    if os.path.exists(ep_out):
        raise RuntimeError(
            f"{ep_out} already exists; refusing to overwrite. Delete it or "
            f"resume from a checkpoint that leaves epoch {ep + 1} free")

# loss.dat 绝不允许被截断：若连完整运行都无法超越既有历史长度，就上前方拒绝
#（早停导致的截断仍由每 epoch 的防线在任何 checkpoint 保存之前捕获）。
if os.path.exists(loss_file):
    n_existing = len(np.loadtxt(loss_file).reshape(-1, 3))
    n_written = len(hist["train"]) + (n_epochs - start_epoch)
    if n_existing > n_written:
        raise RuntimeError(
            f"{loss_file} has {n_existing} rows but at most {n_written} epochs "
            f"of history will be written — refusing to truncate")

train_model = DistributedDataParallel(
    model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
) if DISTRIBUTED else model
torch.manual_seed(123 + RANK)   # DDP 同步完成之后才分化 rank 专属 RNG 流


# 辅助函数

def unnormalize(x):
    """功能：(B,2,H,W,Z) 的 [0,1] 归一化值 -> 物理 m/s（逐变量 clip 范围）。"""
    return x * (y_hi - y_lo) + y_lo


# 训练循环

if start_epoch >= n_epochs:
    raise RuntimeError(
        f"checkpoint already completed epoch {start_epoch}, but this run has only "
        f"{n_epochs} epoch(s)")

# rank-0 的运行生命周期（供监控代理）：现在发 PROGRESS status=start；过 smoke
# 门后发 status=completed；下方受保护块抛出任何异常（非有限值、OOM、config
# 拒绝）时发 status=failed。
run_t0 = time.perf_counter()
if IS_MAIN:
    log(format_progress("train", "start", objective=OBJECTIVE, preset=PRESET,
                        mode=TRAIN_MODE, world=WORLD_SIZE, epochs=n_epochs,
                        steps_per_epoch=len(train_loader),
                        train_horizon=TRAIN_HORIZON, lead_schedule=LEAD_SCHEDULE,
                        run_dir=run_dir), flush=True)

last_updates = last_skipped = 0
last_train_loss = last_val_rel = float("nan")
try:
    for ep in range(start_epoch, n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        train_model.train()
        t1 = time.time()
        t_batch = time.time()
        train_loss_sum = 0.0
        n_batch = 0
        succ_updates = 0
        skipped_updates = 0
        max_lead_seen = 0   # 实际执行过的最高训练 lead J（smoke 门）
        # rank-0 的交互进度条 + 周期性代理可读状态行；其他 rank 保持沉默（绝不
        # 重复 DDP 进度）。scope 如实标注每 rank 分片；sample_per_s 是全局吞吐。
        train_rep = ProgressReporter(
            "train", total=len(train_loader), unit="step",
            samples_per_unit=cfg["batch_size"] * WORLD_SIZE,   # 全局 sample/s
            desc=f"train ep{ep + 1}/{n_epochs}",
            context={"epoch": f"{ep + 1}/{n_epochs}", "scope": PROGRESS_SCOPE}
        ) if IS_MAIN else None
        for bi, (cond, target, _) in enumerate(train_loader):
            xx = cond.to(device, non_blocking=True)          # (B,14,H,W,Z)，[0,1] 归一化
            yy = target[:, 0].to(device, non_blocking=True)  # (B,2,H,W,Z)，[0,1] 归一化

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                if OBJECTIVE == "diffusion":
                    loss = train_model(yy, xx, mask=mask)
                elif TRAIN_HORIZON == 1:
                    # 历史单步路径，保持按位一致
                    pred = train_model(xx, static_cond=mask if STATIC_MASK else None)
                    loss = masked_mse_loss(pred, yy, mask)
                else:
                    # 分离式多步反馈（doc §5）：lead J 按固定 batch 调度取值；
                    # J-1 个分离自反馈步把训练输入分布对齐到 15 天 rollout，然后
                    # 只有第 J 个预测携带梯度。J 是 batch 索引的纯函数，因此每个
                    # DDP rank 每步执行相同次数的前向（collective-safe）。
                    lead = lead_for_batch(bi, TRAIN_HORIZON)
                    max_lead_seen = max(max_lead_seen, lead)
                    if lead > 1:
                        # 反馈推理必须在 autocast 权重缓存之外运行：autocast+
                        # no_grad 的前向会把 Linear 系权重的 fp16 副本缓存为
                        # detached 张量，同一 autocast 上下文内的最终梯度前向会
                        # 复用它们，使那些参数与 loss 图断开（其 DDP hook 永不
                        # 触发 -> 下一次迭代报 "Expected to have finished
                        # reduction"）。嵌套的 autocast(enabled=False) 帧让反馈
                        # 以 fp32 运行且不触碰缓存；最终前向在梯度开启下重新
                        # cast，保持连接。（2026-09-03 修复，见 CHANGELOG）
                        with autocast(device_type=device.type, enabled=False):
                            cur = detached_feedback_window(train_model, xx, lead)
                        pred = train_model(cur, static_cond=mask if STATIC_MASK else None)
                        loss = masked_mse_loss(
                            pred, target[:, lead - 1].to(device, non_blocking=True), mask)
                    else:
                        pred = train_model(xx, static_cond=mask if STATIC_MASK else None)
                        loss = masked_mse_loss(pred, yy, mask)
            finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
            if DISTRIBUTED:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError(
                    f"non-finite training loss {loss.detach().item()} at epoch {ep + 1} "
                    f"batch {bi} rank {RANK} "
                    f"(grad_scale={scaler.get_scale():.4e}, n_batch={n_batch}); aborting")
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                # 检测到 inf/nan 梯度：本步被跳过（无参数更新）
                skipped_updates += 1
            else:
                succ_updates += 1
                scheduler.step()   # 只在真实优化器更新之后步进
            train_loss_sum += loss.detach().item()
            n_batch += 1
            if train_rep is not None:
                train_rep.update(
                    1, loss=f"{train_loss_sum / n_batch:.5f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    updates=succ_updates, skipped=skipped_updates)
            if (bi + 1) % 100 == 0:
                dt_b = time.time() - t_batch
                if train_rep is not None:
                    train_rep.note(
                        f"  [ep {ep + 1}] batch {bi + 1}/{len(train_loader)}  "
                        f"avg_loss {train_loss_sum / n_batch:.5f}  "
                        f"{dt_b / 100:.2f}s/batch  "
                        f"scale {scaler.get_scale():.4e}")
                t_batch = time.time()
        if train_rep is not None:
            train_rep.close(updates=succ_updates, skipped=skipped_updates)

        # 跨 rank 汇总：损失与 batch 数求和；succ_updates 取各 rank 最小值（任一
        # rank 跳步都如实反映）；skipped_updates 求和。
        loss_stats = torch.tensor([train_loss_sum, n_batch], dtype=torch.float64,
                                  device=device)
        if DISTRIBUTED:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            min_updates = torch.tensor(succ_updates, device=device)
            total_skipped = torch.tensor(skipped_updates, device=device)
            dist.all_reduce(min_updates, op=dist.ReduceOp.MIN)
            dist.all_reduce(total_skipped, op=dist.ReduceOp.SUM)
            succ_updates = int(min_updates.item())
            skipped_updates = int(total_skipped.item())
        train_loss = float((loss_stats[0] / loss_stats[1].clamp(min=1)).item())

        # 验证窗口跨 rank 分片并归约，每张 GPU 都有贡献，没有 rank 会闲置到触发
        # collective 超时。fork_rng 隔离 CPU（以及 CUDA 上当前设备）的 RNG，使
        # 固定的 rank 种子不会扰动训练 RNG 流。
        train_model.eval()
        val_rel_sum, n_val = 0.0, 0
        # DDP 下验证窗口按 rank 分片：该 reporter 的 batch 总量只覆盖本 rank 的
        # 分片（scope 字段已明确说明）
        val_rep = ProgressReporter(
            "val", total=len(val_loader), unit="batch",
            desc=f"val ep{ep + 1}/{n_epochs}",
            context={"epoch": f"{ep + 1}/{n_epochs}", "scope": PROGRESS_SCOPE}
        ) if IS_MAIN else None
        # torch.device("cuda").index 为 None，且 fork_rng(devices=[None]) 会崩溃；
        # current_device() 给出模型所在设备的实际序号。
        rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(VAL_SEED + RANK)
            for cond, target, _ in val_loader:
                xx = cond.to(device, non_blocking=True)
                yy = target[:, 0].to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    pred = model.sample(xx, static_cond=mask if STATIC_MASK else None)
                batch_n = xx.shape[0]
                val_rel_sum += (masked_rel_l2(unnormalize(pred.float()), unnormalize(yy),
                                              mask) * batch_n)
                n_val += batch_n
                if val_rep is not None:
                    val_rep.update(1, rel_l2=f"{val_rel_sum / max(n_val, 1):.5f}")
        if val_rep is not None:
            val_rep.close()
        val_stats = torch.tensor([val_rel_sum, n_val], dtype=torch.float64, device=device)
        if DISTRIBUTED:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        if val_stats[1].item() == 0:
            raise RuntimeError("validation subset is empty across all ranks")
        val_rel = float((val_stats[0] / val_stats[1]).item())
        if not np.isfinite(val_rel):
            raise RuntimeError(f"non-finite validation metric {val_rel} at epoch {ep + 1}")

        dt = time.time() - t1
        hist["train"].append(train_loss)
        hist["val_rel"].append(val_rel)
        hist["time"].append(dt)
        log(f"epoch {ep + 1}/{n_epochs}  {dt:.1f}s  "
            f"train_loss {train_loss:.5f}  val_masked_relL2 {val_rel:.5f}  "
            f"updates/rank {succ_updates} (skipped across ranks {skipped_updates})  "
            f"max_lead {max_lead_seen if TRAIN_HORIZON > 1 else 1}  "
            f"grad_scale {scaler.get_scale():.4e}  "
            f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # 检查点顺序：先判 is_best，再更新 best_val，然后构建 Ep{n}.pth 与
        # best.pth 共享的同一个 state dict——新的最佳 epoch 因此绝不会写出带
        # 陈旧 best_val 的 best.pth。
        is_best = val_rel < best_val
        if is_best:
            best_val = val_rel
            worse_epochs = 0
        else:
            worse_epochs += 1
        if IS_MAIN:
            state = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val": best_val,
                "worse_epochs": worse_epochs,
                "config": {
                    "preset": PRESET, **cfg, "context": CONTEXT,
                    "train_mode": TRAIN_MODE,
                    "world_size": WORLD_SIZE,
                    "per_device_batch_size": cfg["batch_size"],
                    "effective_batch_size": cfg["batch_size"] * WORLD_SIZE,
                    "objective": OBJECTIVE,
                    "cond_chans": COND_CH,
                    "model_cond_chans": MODEL_COND_CH,
                    "static_mask_input": STATIC_MASK,
                    "target_ch": TARGET_CH,
                    "mask_scheme": MASK_SCHEME,
                    "train_horizon": TRAIN_HORIZON,
                    "lead_schedule": LEAD_SCHEDULE,
                    "feedback_detach": True,
                    "init_checkpoint": INIT_CHECKPOINT,
                    "init_weights_only": INIT_CHECKPOINT is not None,
                    "stats_sigma": float(stats["sigma"]),
                    "norm_lo": [float(x) for x in stats["lo"]],
                    "norm_hi": [float(x) for x in stats["hi"]],
                    "mask_version": mask_version(),
                },
            }
            if OBJECTIVE == "diffusion":
                state["config"]["sigma_data_scale"] = sigma_scale
                state["config"]["sigma_data"] = model.sigma_data
            else:
                state["config"]["residual_base"] = model.residual_base
                state["config"]["time_sigma"] = model.time_sigma
            # loss.dat 绝不允许被截断：在本 epoch 任何 checkpoint 保存之前检查
            #（早停仍可能把历史缩到既有文件之下；上前方检查覆盖的是完整运行）。
            if os.path.exists(loss_file):
                n_existing = len(np.loadtxt(loss_file).reshape(-1, 3))
                if n_existing > len(hist["train"]):
                    raise RuntimeError(
                        f"{loss_file} has {n_existing} rows but only {len(hist['train'])} "
                        f"epochs of history will be written — refusing to truncate")
            ckpt_out = os.path.join(run_dir, f"Ep{ep + 1}.pth")
            torch.save(state, ckpt_out)
            if is_best:
                torch.save(state, os.path.join(run_dir, "best.pth"))
            # loss.dat 总是包含完整历史（resume 时已恢复），因此 resume 出的运行
            # 绝不会静默覆盖先前的 epoch。
            np.savetxt(loss_file,
                       np.dstack((hist["time"], hist["train"], hist["val_rel"])).squeeze(),
                       fmt="%16.7f")
        if DISTRIBUTED:
            dist.barrier()

        last_updates, last_skipped = succ_updates, skipped_updates
        last_train_loss, last_val_rel = train_loss, val_rel

        if worse_epochs >= 2 and ep >= 1:
            log(f"early stop: val_masked_relL2 worsened for {worse_epochs} consecutive "
                f"epochs (best {best_val:.5f})")
            break

    if TRAIN_MODE == "smoke":
        if (not np.isfinite(last_train_loss) or not np.isfinite(last_val_rel)
                or last_updates < 1 or last_skipped != 0):
            raise RuntimeError(
                f"SMOKE FAIL: train_loss={last_train_loss}, val_rel={last_val_rel}, "
                f"updates/rank={last_updates}, skipped={last_skipped}")
        if TRAIN_HORIZON > 1 and max_lead_seen <= 1:
            # smoke batch 必须真实执行到分离反馈路径：若调度/开窗回归使 J 静默
            # 坍缩为 1，门会空转通过而什么都没测到（doc §6 WP3）
            raise RuntimeError(
                f"SMOKE FAIL: DIAFNO_TRAIN_HORIZON={TRAIN_HORIZON} but no J>1 "
                f"batch was executed in {n_batch} smoke batches "
                f"(max_lead_seen={max_lead_seen}); the lead schedule or the "
                "window alignment is broken")
        if IS_MAIN:
            required = (os.path.join(run_dir, "Ep1.pth"),
                        os.path.join(run_dir, "best.pth"), loss_file)
            missing = [path for path in required if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(f"SMOKE FAIL: missing outputs {missing}")
        log(f"SMOKE PASS: finite train/val, {last_updates} updates/rank, no AMP skips, "
            f"checkpoint outputs complete in {run_dir}")
except BaseException as exc:
    mark_progress_failed()          # 兜底 hook 不得重复这条
    if IS_MAIN:
        print(format_progress("train", "failed", stage="run",
                              error=f"{type(exc).__name__}: {exc}",
                              elapsed_s=f"{time.perf_counter() - run_t0:.1f}"),
              flush=True)
    raise

if IS_MAIN:
    log(format_progress("train", "completed", objective=OBJECTIVE,
                        epochs_done=len(hist["train"]),
                        best_val=f"{best_val:.5f}",
                        elapsed_s=f"{time.perf_counter() - run_t0:.1f}",
                        run_dir=run_dir), flush=True)

log(f"done. best val_masked_relL2 = {best_val:.5f}; checkpoints in {run_dir}")
if DISTRIBUTED:
    dist.destroy_process_group()
