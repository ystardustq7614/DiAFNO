#!/usr/bin/env python3
"""模块职责：PRE_ocean_data 预报任务的共享配置（被 pre_trainer.py 与
pre_evaluate.py 导入——本模块必须保持无副作用）。

同时承载 trainer 与评估共享的两个运行时小件：训练目标配置（diffusion 与确定性
persistence_residual）以及 rank-0 终端进度报告（`ProgressReporter`：交互 tqdm 条
+ 可解析的 PROGRESS key=value 状态行；不接监控服务，除 tqdm 外无新依赖）。

不负责：数据集实现（pre_dataset.py）、模型、训练循环与评估循环本身；
ProgressReporter 只做进度呈现，不承载任何训练语义。

关键约束：import 时零副作用（不读环境变量、不建线程、不写盘）——所有环境变量
读取都封装在函数内（static_mask_input / train_horizon / init_checkpoint），
测试可以注入 env 参数而不触碰进程环境。

依赖关系：标准库 + tqdm；被 pre_trainer.py、pre_evaluate.py、pre_smoke_test.py
与多个 scripts/diag_*.py 导入。
"""
import re
import sys
import threading
import time

from tqdm import tqdm

OUT_ROOT = "/data2/user/zyq/checkpoints/PRE"

PRESETS = {
    # 冒烟测试：在表层验证整条流水线。
    "surface_smoke": dict(
        depth_index=29,            # 29 = 海表层（0 = 海底）
        patch_size=(4, 3, 1),      # 400/4=100, 441/3=147, 1/1=1 -> 14,700 个 token
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,            # 训练集窗口二次抽样步长
        max_train_windows=None,    # 例如设 2000 可加快试运行
        sampling_steps=32,
        val_windows=24,            # 每个 epoch 均匀抽取的验证窗口数（覆盖整个验证期）
        lr=1e-3,
    ),
    # full3d：30 层 sigma，400/4 x 441/3 x 30/2 = 100x147x15 = 220,500 个 token。
    # 24GB 显存吃紧：从 batch_size=1 起步；若 OOM，先降 embed_dim 或
    # implicit_layer，再考虑其他参数。
    "full3d": dict(
        depth_index=None,
        patch_size=(4, 3, 2),
        embed_dim=128,
        implicit_layer=2,
        explicit_layer=4,
        batch_size=1,
        num_workers=2,
        num_epochs=50,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=16,
        lr=1e-3,
    ),
    # 工作包 5 代表层（实验 11）：MIDDLE（sigma 索引 14）与 BOTTOM（sigma 索引 0）。
    # 架构、patch、预算与协议与 surface_smoke 完全一致——唯一差别是探测的深度
    # 索引（sigma 索引绝不换算成固定米深）。run 标签：middle_smoke_*/bottom_smoke_*。
    "middle_smoke": dict(
        depth_index=14,            # 中层代表 sigma 层
        patch_size=(4, 3, 1),
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=24,
        lr=1e-3,
    ),
    "bottom_smoke": dict(
        depth_index=0,             # 底层代表 sigma 层（海床）
        patch_size=(4, 3, 1),
        embed_dim=180,
        implicit_layer=4,
        explicit_layer=4,
        batch_size=4,
        num_workers=4,
        num_epochs=10,
        train_stride=1,
        max_train_windows=None,
        sampling_steps=32,
        val_windows=24,
        lr=1e-3,
    ),
}

# 流水线 smoke 运行保留生产架构/网格，但每个 rank 只执行少量真实优化器步数。
# 用于在不造第二个玩具模型的前提下，暴露数据、显存、AMP、采样与 checkpoint 失败。
SMOKE_BATCHES_PER_RANK = 4

CONTEXT = 7        # 条件窗口天数
HORIZON = 15       # rollout 天数
TARGET_CH = 2      # u、v 两个通道

# 训练目标配置

# "diffusion"            ：条件 EDM（legacy 默认；不带 objective 字段的 legacy 检查点
#                          一律归入此路径）
# "persistence_residual" ：确定性 PersistenceResidualIAFNO 基线
#                          （prediction = 末日持续性 + 零初始化残差头）
OBJECTIVES = ("diffusion", "persistence_residual")
DEFAULT_OBJECTIVE = "diffusion"

# 写入检查点的静态掩膜输入方案标识；当前只有双变量 rho 掩膜
# （mask_u_rho 与 mask_v_rho 两个通道）
MASK_SCHEME = "bivariate_rho"

# 确定性模型 c_noise 时间嵌入背后的常数 sigma
# （time = 0.25 * log(time_sigma)；不存在噪声 schedule，取任意固定常数均有效
# ——该值写入检查点以便精确重建）
RESIDUAL_TIME_SIGMA = 0.002

# Phase-5 掩膜输入 A/B（B 臂）：把双变量 rho 掩膜的两个通道
# （mask_u_rho / mask_v_rho）拼接到 backbone 条件。动态窗口在各处（dataset、
# rollout 滑窗、持续性基）都保持 14 通道；掩膜经 pre_rollout 的 `static_cond`
# 单独转发给模型。按运行以 DIAFNO_STATIC_MASK=1 启用，并以 `static_mask_input`
# 记录在检查点 config 中（run 标签后缀 "_MSK"）。
STATIC_MASK_ENV = "DIAFNO_STATIC_MASK"
STATIC_MASK_CHANNELS = 2


def static_mask_input(env=None):
    """读取 DIAFNO_STATIC_MASK 开关（"1"/"true"/"yes" -> True）。"""
    import os
    value = (env if env is not None else os.environ.get(STATIC_MASK_ENV, ""))
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def static_mask_from_checkpoint(ckpt_cfg, objective=None):
    """功能：从检查点 config 解析静态掩膜输入臂（实验 08）。

    返回：(static_mask_input, model_cond_chans)——开关标志与该检查点训练时使用的
    backbone 条件通道数，评估据此重建完全一致的架构。缺失字段的 legacy 检查点
    走普通 14 通道路径（False, 2*CONTEXT）。不可能或互相矛盾的元数据直接拒绝，
    绝不静默重建出另一个模型：
      - diffusion 检查点记录 static_mask_input=True（diffusion 路径保持历史
        布局，trainer 从不允许该臂）；
      - 记录的 model_cond_chans 与解析结果不符。

    异常：上述两种矛盾各抛 RuntimeError。
    """
    ckpt_cfg = ckpt_cfg or {}
    flag = bool(ckpt_cfg.get("static_mask_input", False))
    cond_ch = 2 * CONTEXT + (STATIC_MASK_CHANNELS if flag else 0)
    if objective == "diffusion" and flag:
        raise RuntimeError(
            "checkpoint records static_mask_input=True with the diffusion "
            "objective; the diffusion path has no static-mask arm and this "
            "combination must never be rebuilt")
    recorded = ckpt_cfg.get("model_cond_chans")
    if recorded is not None and int(recorded) != cond_ch:
        raise RuntimeError(
            f"checkpoint model_cond_chans={int(recorded)} contradicts "
            f"static_mask_input={flag} (resolved {cond_ch}); refusing to "
            "rebuild a mismatched model")
    return flag, cond_ch


# 分离式多步训练（工作包 2；doc §5）

# 分离式自回归多步训练视界 K（"MS{K}"）：trainer 把模型自己的（no_grad、clamp
# 后）预测前推 J-1 步，只反传第 J 步（doc §5 伪代码）。K=1 与历史单步
# teacher-forcing 路径按位一致。仅允许确定性 persistence_residual 目标、
# 且 static_mask_input 为 False。
TRAIN_HORIZON_ENV = "DIAFNO_TRAIN_HORIZON"

# 仅权重初始化来源（如实验 07 的 Ep10）：只加载模型权重，optimizer/scheduler/
# scaler/epoch/history 均不加载（来源的 cosine 调度已走完；MS 运行以更低 LR
# 启动全新 optimizer）。与 DIAFNO_CHECKPOINT（完整 resume）互斥。
INIT_CHECKPOINT_ENV = "DIAFNO_INIT_CHECKPOINT"

# 仅在 train_horizon > 1 时应用的默认值（doc §6 WP3 冻结配置：全新 optimizer、
# LR 1e-4、最多 5 个 epoch；smoke 模式仍会覆盖 epoch 数）。与其他超参数一样
# 写入检查点 config。
MS_DEFAULTS = dict(lr=1e-4, num_epochs=5)


def train_horizon(env=None):
    """读取 DIAFNO_TRAIN_HORIZON（"MS{K}"，整数 >= 1；默认 1 = 单步）。

    异常：非整数或小于 1 抛 ValueError（fail-fast，避免把非法视界带进训练）。
    """
    import os
    value = (env if env is not None else os.environ.get(TRAIN_HORIZON_ENV, ""))
    value = str(value).strip()
    if not value:
        return 1
    try:
        horizon = int(value)
    except ValueError:
        raise ValueError(f"{TRAIN_HORIZON_ENV}={value!r} is not an integer")
    if horizon < 1:
        raise ValueError(f"{TRAIN_HORIZON_ENV}={horizon} must be >= 1")
    return horizon


def init_checkpoint(env=None):
    """读取 DIAFNO_INIT_CHECKPOINT（仅权重初始化来源；默认 None）。"""
    import os
    value = (env if env is not None else os.environ.get(INIT_CHECKPOINT_ENV, ""))
    value = str(value).strip()
    return os.path.expanduser(value) if value else None


def lead_for_batch(batch_index, train_horizon):
    """功能：按 batch 索引给出训练 lead J（doc §5.1 固定调度）。

    K=1  -> 恒为 1（历史单步路径，调度不起作用）。
    K>1  -> 偶数 batch 保持 day-1 锚点（占 50%）；奇数 batch 在 2..K 上轮换，
            即 MS5 产生 1,2,1,3,1,4,1,5,1,2,...

    关键约束：J 是 (batch_index, K) 的纯函数——无 RNG、无全局状态，因此每个
    DDP rank 对同一步推出相同的 J（sampler 与 loader 都用 drop_last=True，
    各 rank 的 batch 数相同），collective-safe。
    """
    k = int(train_horizon)
    if k <= 1:
        return 1
    bi = int(batch_index)
    if bi % 2 == 0:
        return 1
    return 2 + ((bi // 2) % (k - 1))


def lead_schedule_str(train_horizon):
    """功能：生成规范的一个周期调度字符串，供日志与检查点元数据使用
    （如 MS5 -> "1,2,1,3,1,4,1,5"；K=1 -> "1"）。"""
    k = int(train_horizon)
    if k <= 1:
        return "1"
    return ",".join(str(lead_for_batch(i, k)) for i in range(2 * (k - 1)))


def check_multistep_config(ckpt_cfg, train_horizon_now, schedule_now):
    """功能：resume 时校验检查点中记录的多步语义。

    带（或不带）分离式多步反馈训练出的检查点，绝不允许在不同语义下 resume：
      - train_horizon 必须精确一致；缺失该字段的 legacy 检查点只与 K=1 兼容
        （它们早于多步机制存在）；
      - 两侧都记录 lead_schedule 时，规范字符串必须一致（调度变更会静默改变
        resume 后的训练分布）。

    异常：任何不匹配抛 RuntimeError。
    """
    ckpt_cfg = ckpt_cfg or {}
    recorded = ckpt_cfg.get("train_horizon")
    if recorded is None:
        if int(train_horizon_now) != 1:
            raise RuntimeError(
                "checkpoint has no config.train_horizon (pre-multi-step) but "
                f"DIAFNO_TRAIN_HORIZON={train_horizon_now}; multi-step training "
                "cannot resume a single-step run — use DIAFNO_INIT_CHECKPOINT "
                "(weights-only init) instead")
        return
    if int(recorded) != int(train_horizon_now):
        raise RuntimeError(
            f"checkpoint train_horizon={int(recorded)} vs current "
            f"{int(train_horizon_now)}; refusing to resume across a "
            "multi-step horizon change")
    recorded_schedule = ckpt_cfg.get("lead_schedule")
    if recorded_schedule is not None and \
            str(recorded_schedule) != str(schedule_now):
        raise RuntimeError(
            f"checkpoint lead_schedule={recorded_schedule!r} vs current "
            f"{schedule_now!r}; refusing to resume across a schedule change")


def restore_worse_epochs(checkpoint):
    """功能：从检查点恢复早停计数器（resume 语义）。

    `worse_epochs` 统计 val_masked_relL2 连续严格高于历史最佳的 epoch 数；
    trainer 在达到 2 时提前停止。resume 后该计数必须存续：既有的变差连续段
    仍然计入。缺失该字段的 legacy 检查点保持历史默认 0。
    """
    ckpt = checkpoint or {}
    return max(0, int(ckpt.get("worse_epochs", 0)))


def validate_objective(objective):
    """功能：归一化并校验训练目标名称（未知名称抛 ValueError）。"""
    objective = str(objective).lower()
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    return objective


def objective_from_checkpoint(checkpoint, default=DEFAULT_OBJECTIVE):
    """功能：读取检查点记录的训练目标（入参为含可选 "config" 的 dict）。

    缺失 objective 字段的 legacy 检查点一律视为 diffusion。
    """
    cfg = (checkpoint or {}).get("config") or {}
    return validate_objective(cfg.get("objective", default))


def ensure_objective_compatible(checkpoint, objective):
    """功能：拒绝把检查点加载进另一类目标的模型（resume 或评估重建）。

    返回检查点的 objective；不一致抛 RuntimeError。
    """
    ckpt_obj = objective_from_checkpoint(checkpoint)
    if ckpt_obj != objective:
        raise RuntimeError(
            f"checkpoint objective={ckpt_obj!r} is incompatible with the "
            f"requested {objective!r}; refusing to load a different model class")
    return ckpt_obj


def check_norm_fingerprint(ckpt_cfg, lo, hi, mask_version_now, tol=1e-6):
    """功能：在 resume 或评估使用检查点之前，校验其数据语义指纹（归一化范围 +
    海洋掩膜版本）与当前 stats/masks 是否一致：不一致说明该检查点训练时的归一化
    语义与本次运行将使用的不同，绝不允许静默发生。

    异常 / 前置条件：
    - 任何不匹配抛 RuntimeError；
    - 对缺失记录字段的 legacy 检查点返回 WARNING 字符串列表（无法校验——调用方
      必须打印它们）。

    参数：`lo`/`hi` 为当前逐变量统计（任意浮点序列）；`mask_version_now` 为
    pre_dataset.mask_version()。
    """
    ckpt_cfg = ckpt_cfg or {}
    warnings = []
    lo_now = [float(x) for x in lo]
    hi_now = [float(x) for x in hi]
    if "norm_lo" in ckpt_cfg and "norm_hi" in ckpt_cfg:
        lo_ck = [float(x) for x in ckpt_cfg["norm_lo"]]
        hi_ck = [float(x) for x in ckpt_cfg["norm_hi"]]
        same = (len(lo_ck) == len(lo_now) and len(hi_ck) == len(hi_now)
                and all(abs(a - b) <= tol for a, b in zip(lo_ck, lo_now))
                and all(abs(a - b) <= tol for a, b in zip(hi_ck, hi_now)))
        if not same:
            raise RuntimeError(
                f"checkpoint normalization fingerprint mismatch: "
                f"config.norm_lo/norm_hi={lo_ck}/{hi_ck} vs current {lo_now}/{hi_now}; "
                "the stats cache changed since this checkpoint was trained — "
                "refusing to continue/evaluate with different normalization semantics")
    else:
        warnings.append("checkpoint has no config.norm_lo/norm_hi (legacy); "
                        "normalization range could NOT be verified")
    if "mask_version" in ckpt_cfg:
        if str(ckpt_cfg["mask_version"]) != str(mask_version_now):
            raise RuntimeError(
                f"checkpoint mask_version={ckpt_cfg['mask_version']!r} vs current "
                f"{str(mask_version_now)!r}; the ocean masks changed since this "
                "checkpoint was trained — refusing")
    else:
        warnings.append("checkpoint has no config.mask_version (legacy); "
                        "mask fingerprint could NOT be verified")
    return warnings


def check_residual_time_sigma(ckpt_cfg, time_sigma):
    """功能：确定性模型的常数时间嵌入是其语义的一部分：拒绝从记录了不同（或未
    记录）time_sigma 的 persistence-residual 检查点 resume。（评估侧采用检查点
    自身的值，因此该防线只用于训练 resume。）

    异常：缺失 time_sigma 或与当前值偏差超过 1e-9 抛 RuntimeError。
    """
    ckpt_cfg = ckpt_cfg or {}
    if "time_sigma" not in ckpt_cfg:
        raise RuntimeError(
            "persistence-residual checkpoint has no config.time_sigma; the "
            "constant time embedding cannot be verified — refusing to resume")
    if abs(float(ckpt_cfg["time_sigma"]) - float(time_sigma)) > 1e-9:
        raise RuntimeError(
            f"checkpoint time_sigma={float(ckpt_cfg['time_sigma'])!r} vs current "
            f"{float(time_sigma)!r}; the residual time embedding changed — "
            "refusing to resume")

# EDM 的 sigma_data 生活在 ElucidatedDiffusion 实际使用的图像空间：diffusion.py
# 用 `images * 2 - 1` 归一化训练图像，即 EDM 看到的数据分布是 [-1, 1]，其标准差
# 是 [0, 1] 归一化统计缓存标准差的两倍。stats["sigma"] 继续存 [0, 1] 空间的值；
# 训练与评估必须都经由下方的 sigma_data_from_stats() / sigma_data_from_checkpoint()。
SIGMA_DATA_SCALE = 2.0


def sigma_data_from_stats(stats_sigma):
    """功能：[0,1] 空间的池化 sigma -> [-1,1] 图像空间的 EDM sigma_data。"""
    return SIGMA_DATA_SCALE * float(stats_sigma)


def sigma_data_from_checkpoint(checkpoint, stats_sigma):
    """功能：解析检查点使用的 EDM sigma_data。

    优先级：检查点自身的 config["sigma_data"]（定尺度 trainer 写入）。缺失
    config 或字段的 legacy 检查点回退到旧尺度 stats["sigma"]（不是加倍值），并
    返回 used_checkpoint=False 供调用方显式告警。

    返回：(sigma_data: float, used_checkpoint_value: bool)。
    """
    cfg = (checkpoint or {}).get("config") or {}
    if "sigma_data" in cfg:
        return float(cfg["sigma_data"]), True
    return float(stats_sigma), False


def resume_sigma_decision(sd_ckpt, sd_current, policy):
    """功能：决定 resume 运行必须使用的 sigma_data。

    参数：
    - sd_ckpt：检查点的 sigma_data（经 sigma_data_from_checkpoint 解析）；
    - sd_current：当前运行的 sigma_data（sigma_data_from_stats）；
    - policy 三选一：
      "error"   （默认）：不一致 -> RuntimeError；绝不静默混用尺度。
      "migrate"           ：显式尺度迁移——沿用 sd_current。
      "adopt"             ：显式 legacy 延续——沿用 sd_ckpt。

    返回：(sigma_data: float, adopted: bool)。尺度一致时无论 policy 一律返回
    (sd_current, False)。未知 policy 抛 ValueError。
    """
    sd_ckpt = float(sd_ckpt)
    sd_current = float(sd_current)
    mismatch = abs(sd_ckpt - sd_current) > 1e-6
    if not mismatch:
        return sd_current, False
    if policy == "error":
        raise RuntimeError(
            f"resume scale mismatch: checkpoint sigma_data={sd_ckpt:.5f} vs "
            f"current sigma_data={sd_current:.5f}; refusing to continue. Set "
            f"RESUME_SIGMA_POLICY='migrate' to keep the current scale (explicit "
            f"scale migration) or 'adopt' to continue in the checkpoint's old "
            f"scale (outputs written back into the checkpoint's directory)")
    if policy == "migrate":
        return sd_current, False
    if policy == "adopt":
        return sd_ckpt, True
    raise ValueError(f"unknown RESUME_SIGMA_POLICY {policy!r} "
                     f"(expected 'error', 'migrate' or 'adopt')")


def training_config(preset, mode="full", world_size=1, train_horizon=1):
    """功能：为 smoke 或 full 训练返回隔离的可变 config 副本。

    ``batch_size`` 保持 per-device 语义。smoke 运行在每个 DDP rank 上恰好包含
    ``SMOKE_BATCHES_PER_RANK`` 个完整 batch，使用 1 个 epoch 与短采样，同时保留
    所选预设的模型架构与物理网格。

    分离式多步运行（train_horizon > 1）改用冻结的 MS_DEFAULTS（lr/num_epochs），
    不取预设的单步值；smoke 模式仍随后缩减 epoch 数。

    异常：未知 preset/mode、world_size < 1 或 train_horizon < 1 抛 ValueError。
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {tuple(PRESETS)}")
    if mode not in ("smoke", "full"):
        raise ValueError(f"unknown training mode {mode!r}; expected 'smoke' or 'full'")
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    train_horizon = int(train_horizon)
    if train_horizon < 1:
        raise ValueError(f"train_horizon must be >= 1, got {train_horizon}")

    cfg = dict(PRESETS[preset])
    if train_horizon > 1:
        cfg.update(MS_DEFAULTS)
    if mode == "smoke":
        cfg.update(
            num_epochs=1,
            max_train_windows=(SMOKE_BATCHES_PER_RANK * world_size
                               * cfg["batch_size"]),
            sampling_steps=4,
            val_windows=max(2, world_size),
        )
    return cfg


def run_tag_for(preset, sd2=True, config=None, objective=DEFAULT_OBJECTIVE,
                static_mask=False, train_horizon=1):
    """功能：检查点/输出目录标签。sd2=True 追加定尺度后缀，使重训运行绝不与
    legacy（sd1）运行共享目录；objective="persistence_residual" 追加 "_RES"，
    使确定性基线绝不与 diffusion 运行共享目录；static_mask=True 追加 "_MSK"，
    使 Phase-5 掩膜输入臂绝不与 14 通道基线共享目录；train_horizon > 1 追加
    "_MS{K}"，使分离式多步运行绝不与单步运行共享目录（K=1 保持历史标签不变）。"""
    cfg = PRESETS[preset] if config is None else config
    tag = (f"{preset}_BS{cfg['batch_size']}_EMD{cfg['embed_dim']}"
           f"_I{cfg['implicit_layer']}_E{cfg['explicit_layer']}"
           f"_S{cfg['sampling_steps']}_C{CONTEXT}")
    if sd2:
        tag += "_SD2"
    if validate_objective(objective) == "persistence_residual":
        tag += "_RES"
    if static_mask:
        tag += "_MSK"
    if int(train_horizon) > 1:
        tag += f"_MS{int(train_horizon)}"
    return tag


def training_run_tag(preset, config, mode="full", world_size=1,
                     objective=DEFAULT_OBJECTIVE, static_mask=False,
                     train_horizon=1):
    """功能：带 smoke/DDP 隔离的 run 标签；单卡 full 标签与 legacy 保持兼容。
    smoke 追加 "_SMOKE"；多进程追加 "_DDP{world_size}"。"""
    tag = run_tag_for(preset, config=config, objective=objective,
                      static_mask=static_mask, train_horizon=train_horizon)
    if mode == "smoke":
        tag += "_SMOKE"
    if int(world_size) > 1:
        tag += f"_DDP{int(world_size)}"
    return tag


# rank-0 终端进度报告（trainer + evaluation）

# 阶段运行期间周期性 PROGRESS 状态行之间的最小间隔
#（start/close/failed 行总是发出，不受该间隔限制）
PROGRESS_INTERVAL_S = 30.0


def _progress_value(value):
    """功能：key=value token 不得含任何空白：所有连续空格/换行/制表符折叠为单个
    下划线，使多行异常消息也保持状态行单行可解析。"""
    return re.sub(r"\s+", "_", str(value))


def format_progress(phase, status, **fields):
    """功能：生成一行可解析状态行，例如
    'PROGRESS phase=train epoch=1/4 step=120/2101 elapsed_s=91.2 eta_s=1506.4
    step_per_s=1.31 sample_per_s=5.24 loss=0.0187 lr=1e-4 status=running'。
    `phase` 恒为首字段、`status` 恒为末字段，因此简单 split() 与逐 token 的
    key=value 解析都保持稳定。"""
    parts = ["PROGRESS", f"phase={_progress_value(phase)}"]
    for key, value in fields.items():
        parts.append(f"{key}={_progress_value(value)}")
    parts.append(f"status={_progress_value(status)}")
    return " ".join(parts)


class ProgressReporter:
    """表示：单个阶段（一个训练 epoch、一次验证、一次评估运行）的 rank-0 进度
    报告。不是日志框架：tqdm 薄包装加上非交互运行所需的 PROGRESS 行。

    交互式 TTY（stream.isatty()）：单行 tqdm 进度条（desc、count/total、rate、
    ETA）加由每次 update() 字段构成的后缀。

    非交互（管道/文件重定向/监控代理）：无进度条、无回车动画；改为在 start、
    close、失败时以及运行期间至少每 `interval` 秒发出一条完整、换行结束且立即
    flush 的 `PROGRESS key=value` 行。周期行由时间驱动而非更新驱动：守护心跳
    线程在单次 batch/rollout 步阻塞调用方超过 `interval` 时也会发出（批中不会
    沉默）。

    状态词表（稳定的解析契约）：
      start       阶段启动（两种模式、必定发出）
      running     周期心跳/更新驱动的进度行
      phase_done  本 reporter 的阶段结束（中间态：一个训练 epoch、评估的
                  rollout 循环）——不是脚本结束
      failed      运行中止（任何异常）
    `status=completed` 保留给脚本级结束，由入口脚本（pre_trainer/pre_evaluate）
    自己发出，绝不由单阶段 reporter 发出，监控端因此不会把 epoch 边界误判为
    运行结束。

    两种模式总是发出 start/close/failed 行（即使不足 30 秒的 smoke 运行也有
    status=start 与一条终止状态）。`enabled=False`（非 rank-0 的 DDP rank）使
    一切变为静默空操作。`stream` 与 `clock` 可注入用于测试；`context` 字段合并
    进每条发出的行（如 epoch=k/n、split=test、scope=rank0_shard_of_4）。

    线程安全：全部状态变更与发射都在 `self._lock` 内；心跳是 daemon 线程，由
    close() 经 _stop_evt 停止。
    """

    def __init__(self, phase, total, stream=None, clock=None,
                 interval=PROGRESS_INTERVAL_S, interactive=None, enabled=True,
                 desc=None, unit="step", samples_per_unit=None, context=None):
        self.phase = phase
        self.total = int(total)
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.perf_counter
        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.samples_per_unit = None if samples_per_unit is None else int(samples_per_unit)
        if interactive is None:
            interactive = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.interactive = bool(interactive)
        self.desc = desc or phase
        self.unit = unit
        self.context = dict(context or {})
        self.done = 0
        self._t0 = self.clock()
        self._last_emit = None
        self._bar = None
        self._closed = False
        self._last_fields = {}          # 最近一次 update() 的指标字段
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._hb_thread = None
        if not self.enabled:
            return
        with self._lock:
            self._emit_nolock("start")
        if self.interactive:
            self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit,
                             dynamic_ncols=True, leave=False, file=self.stream)
        else:
            self._last_emit = self._t0
            # 时间驱动心跳：interval 内没有 update() 调用也发出 status=running
            #（daemon 线程；close() 停止）
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True,
                name=f"progress-heartbeat-{phase}")
            self._hb_thread.start()

    # 内部实现

    def _heartbeat_loop(self):
        """守护线程：距上次发射超过 `interval` 就发出一条 running 行，即使调用方
        阻塞在单个 batch 内也不沉默。轮询周期取 interval/4，夹在 [0.05, 5] 秒
        之间，避免高频自旋或漏检。"""
        poll = min(max(self.interval / 4.0, 0.05), 5.0)
        while not self._stop_evt.wait(poll):
            with self._lock:
                if self._closed or self.interactive:
                    return
                now = self.clock()
                if self._last_emit is None or now - self._last_emit >= self.interval:
                    self._last_emit = now
                    self._emit_nolock("running")

    def _throughput_fields(self, now, status_fields):
        """功能：构成每条发射行共享的进度 + elapsed/ETA/rate 字段。

        最近一次 update() 的指标字段（loss、lr、d1_rmse、...）合并进来，使
        close/failed 行也携带最后已知指标；本次发射调用显式传入的字段优先。
        """
        fields = dict(self.context)
        fields[self.unit] = f"{self.done}/{self.total}"
        elapsed = max(now - self._t0, 0.0)
        fields["elapsed_s"] = f"{elapsed:.1f}"
        rate = self.done / elapsed if elapsed > 0 and self.done > 0 else None
        if rate is not None:
            fields[f"{self.unit}_per_s"] = f"{rate:.3f}"
            if self.samples_per_unit:
                fields["sample_per_s"] = f"{rate * self.samples_per_unit:.3f}"
            if self.total > self.done:
                fields["eta_s"] = f"{(self.total - self.done) / rate:.1f}"
        fields.update(self._last_fields)
        fields.update(status_fields)
        return fields

    def _emit_nolock(self, status, **fields):
        """功能：格式化并写出一条状态行。调用方必须已持有 `self._lock`。"""
        line = format_progress(self.phase, status,
                               **self._throughput_fields(self.clock(), fields))
        if self._bar is not None:
            self._bar.write(line, file=self.stream)
        else:
            print(line, file=self.stream, flush=True)

    # 对外 API

    def update(self, n=1, **fields):
        """功能：推进计数并刷新后缀/周期状态行。

        `fields`（如 loss、lr、updates、d1_rmse）立即出现在交互后缀中，并按
        原文出现在周期 PROGRESS 行里。`n` 必须如实按 reporter 自身单位计数
        （如部分满的末 batch 里的实际窗口数），而不是调用次数。
        """
        if not self.enabled or self._closed:
            return
        with self._lock:
            self.done += int(n)
            now = self.clock()
            if fields:
                self._last_fields = dict(fields)
            if self._bar is not None:
                self._bar.update(int(n))
                if fields:
                    postfix = "  ".join(f"{k}={_progress_value(v)}"
                                        for k, v in fields.items())
                    self._bar.set_postfix_str(postfix)
            if not self.interactive and (self._last_emit is None
                                         or now - self._last_emit >= self.interval):
                self._last_emit = now
                self._emit_nolock("running", **fields)

    def note(self, message):
        """功能：打印普通行而不破坏活动进度条（经 tqdm.write）；非交互模式下为
        立即 flush 的普通 print。"""
        if not self.enabled or self._closed:
            return
        with self._lock:
            if self._bar is not None:
                self._bar.write(str(message), file=self.stream)
            else:
                print(str(message), file=self.stream, flush=True)

    def close(self, status="phase_done", **fields):
        """功能：发出本阶段的终止状态行（默认 `phase_done`——脚本级 `completed`
        由入口脚本自己发出）并停止心跳线程。幂等。"""
        if not self.enabled or self._closed:
            return
        with self._lock:
            self._closed = True
            self._stop_evt.set()
            self._emit_nolock(status, **fields)
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    def fail(self, **fields):
        """功能：发出 status=failed（字段通常为 error=...、stage=...）并关闭。"""
        self.close(status="failed", **fields)


# 未受保护脚本段的统一失败上报

# 每进程最多一条失败行：受保护块的 handler 在自己发行前先经
# mark_progress_failed() 置位，因此下方兜底 hook 不会重复。
_PROGRESS_FAILURE_REPORTED = False


def mark_progress_failed():
    """功能：记录本进程已发出过标准 `status=failed` PROGRESS 行（见
    install_progress_failure_hook），使兜底 hook 去重。"""
    global _PROGRESS_FAILURE_REPORTED
    _PROGRESS_FAILURE_REPORTED = True


def reset_progress_failure_state():
    """功能：清除去重标志（测试缝隙；无生产调用方）。"""
    global _PROGRESS_FAILURE_REPORTED
    _PROGRESS_FAILURE_REPORTED = False


def install_progress_failure_hook(phase, stage=None, stream=None, fallback=None):
    """功能：安装一个 `sys.excepthook` 兜底，为逃出脚本受保护块的异常发出唯一
    一条标准 `PROGRESS ... status=failed` 行——初始化、数据/模型搭建、
    pre-flight 拒绝与后处理失败没有自己的活跃 reporter。

    参数：
    - phase：PROGRESS 的 phase 字段（如 "train" / "eval"）；
    - stage：静态阶段名，或零参 callable——在失败时刻读取，使脚本能经可变
      变量跟踪当前段（setup -> rollout -> postprocess）；
    - stream/fallback：测试可注入（fallback 默认 sys.__excepthook__，仍打印
      完整 traceback）。

    副作用：替换 sys.excepthook。去重：mark_progress_failed() 已调用时保持
    沉默（受保护块已报告自身失败）。只在负责上报的 rank（rank 0 / 单进程
    入口）调用。

    返回：安装的 hook。
    """
    def _hook(exc_type, exc, tb):
        if not _PROGRESS_FAILURE_REPORTED:
            mark_progress_failed()
            stage_value = stage() if callable(stage) else (stage or "setup")
            print(format_progress(phase, "failed", stage=stage_value,
                                  error=f"{exc_type.__name__}: {exc}"),
                  file=stream if stream is not None else sys.stdout, flush=True)
        (fallback if fallback is not None else sys.__excepthook__)(
            exc_type, exc, tb)
    sys.excepthook = _hook
    return _hook
