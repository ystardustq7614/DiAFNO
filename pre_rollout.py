#!/usr/bin/env python3
"""模块职责：提供 PRE 评估用的无副作用自回归 ensemble rollout，以及训练侧的
分离式多步反馈窗口 detached_feedback_window（doc §5 detached MS，
docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md）。

不负责：指标计算（pre_metrics.py）、数据集加载（pre_dataset.py）、模型实现
（diffusion.py / pre_models.py）；import 时无任何副作用（不写盘、不设种子、
不建线程），可被 pre_smoke_test.py 安全导入。

关键约束：模型按鸭子类型调用——只调用 `model.sample(cur, num_sample_steps=...,
clamp=...)`，条件 EDM（ElucidatedDiffusion）与确定性基线
（PersistenceResidualIAFNO.sample，忽略 num_sample_steps 且不消耗 RNG）都无需
改动即可接入；因此对确定性模型，逐窗口种子与 ensemble 尺寸均不起作用：轨迹
按位可复现，所有 member 相同。条件窗口 `cur` 始终是纯动态通道（2*CONTEXT，
day-major u/v 交错）；可选的 static_cond（双变量 rho 掩膜通道）不进入滑窗，
每一步原样转发。采样统一在 torch.amp.autocast 下运行（device_type 跟随张量
所在设备），复现历史评估路径（CUDA 启用 AMP；对无 autocast 敏感算子的 FP32
模型等价于空操作，但两端语义一致）。

依赖关系：仅依赖 torch；被 pre_evaluate.py（ensemble rollout）与
pre_trainer.py（detached_feedback_window）调用。

shape 记号（全文件一符一义）：cond 为 (B, C, H, W, Z)，B=窗口数，
C=2*CONTEXT=14（day-major u/v 交错），H/W 为 rho 网格 400x441，Z 为 sigma
层数（surface=1，full3d=30），E=ensemble member 数，L=lead 天数。数值语义：
条件与预测均为 [0,1] min-max 归一化；掩膜 1=海洋、0=陆地。

种子严格逐窗口：传入 `seeds`（每个 batch 行一个 int）时，每个窗口的轨迹由它
自己的种子决定，因此与 batch 大小、装载器分组以及同批其他窗口无关。标量
`seed` 保留给单窗口/legacy 路径（整个 batch 一次性施加一个种子）。
"""
import torch


def _sample(model, cur, num_sample_steps, clamp, static_cond=None):
    """功能：在 autocast 下调用 model.sample，复现历史评估路径。

    参数：
    - cur：条件窗口 (B, C, H, W, Z)，[0,1] 归一化。
    - static_cond：仅在给定值时经 kwarg 转发，因此不接受该参数的模型（如 EDM
      采样器）保持与历史完全相同的调用签名。

    返回：model.sample 的次日预测 (B, 2, H, W, Z)。
    """
    device_type = "cuda" if cur.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type):
        if static_cond is None:
            return model.sample(cur, num_sample_steps=num_sample_steps, clamp=clamp)
        return model.sample(cur, num_sample_steps=num_sample_steps, clamp=clamp,
                            static_cond=static_cond)


def _rollout_one(model, cond, horizon, num_sample_steps, clamp,
                 remask_feedback=False, ocean_mask=None, static_cond=None):
    """功能：在一个种子下 rollout 一批（已按 member 展开的）条件窗口。

    参数：
    - cond：条件窗口 (B, C, H, W, Z)，[0,1] 归一化（调用方已 expand_ensemble）。

    返回：
    - (B, horizon, 2, H, W, Z) 的 float32 预测（通道 0/1 为 rho 网格 u/v）。

    关键转换：
    - 自回归滑窗：每步丢弃窗口最旧一天的 2 个通道（cur[:, 2:]），并在末尾追加
      本模型的预测 p——即"模型消费自己的输出"的结构变化点；
    - remask_feedback=True 时，每个预测在存入结果之前、也在进入下一个条件窗口
      之前，都乘以海洋掩膜（陆地 -> 0）；默认 False = 历史无掩膜反馈（陆地填充
      值原样反馈）；该开关的最终取舍属于 Phase-5 A/B，记录在评估元数据中；
    - static_cond 不参与滑窗切片、每步原样转发，因此 `cur` 始终是纯动态条件。
    """
    if remask_feedback:
        assert ocean_mask is not None, "remask_feedback=True requires ocean_mask"
    cur = cond
    preds = []
    for _ in range(int(horizon)):
        p = _sample(model, cur, num_sample_steps, clamp,
                    static_cond=static_cond).float()
        if remask_feedback:
            p = p * ocean_mask
        preds.append(p)
        cur = torch.cat([cur[:, 2:], p], dim=1)     # 丢最旧一天 2 通道，末尾追加自身预测
    return torch.stack(preds, dim=1)                # 沿新 lead 轴堆叠 -> (B, L, 2, H, W, Z)


def detached_feedback_window(step_fn, cond, lead, clamp=True):
    """功能：训练侧分离式自回归反馈窗口（分离式多步反馈，doc §5）。

    把模型自己的预测在 torch.no_grad() 下前推 lead-1 步并返回最终条件窗口；
    调用方随后在该窗口上做携带梯度的最终预测。语义与正式确定性 rollout
    （_rollout_one）逐项对齐：
      - 滑窗更新：丢弃最旧一天（2 通道）、追加预测（切片 cur[:, pred.shape[1]:]
        与正式 rollout 的 cur[:, target_ch:] 完全一致）；
      - clamp=[0,1] 对应 model.sample(clamp=True)；
      - rf0：反馈前不做掩膜再处理（实验 09 历史语义）。

    参数：
    - step_fn(cur)：输入一个条件窗口，返回次日制预测（如 DDP 包装的
      PersistenceResidualIAFNO forward）；autocast 上下文由调用方决定
      （trainer 用嵌套的 autocast(enabled=False) 帧包裹整个多步块，原因见下）。

    返回：前推 lead-1 步后的条件窗口，shape 与输入相同 (B, C, H, W, Z)；
    lead == 1 时原样返回 cond（无反馈步，调度不起作用）。

    梯度与显存：前 lead-1 次前向不保留梯度图（no_grad，detach 语义），峰值训练
    显存与单步路径接近，与 lead 无关；仅最终第 J 步携带梯度。

    AUTOCAST 陷阱（DDP）：当最终前向在 autocast 下运行时，反馈前向必须运行在
    autocast 权重缓存之外——例如把本调用包在
    `with torch.amp.autocast(..., enabled=False):` 中。原因：autocast+no_grad 的
    前向会把 Linear 系权重的 fp16 副本以 DETACHED 张量形式缓存；同一 autocast
    上下文内的梯度前向会复用这些副本，使这些参数与 loss 图断开（其 DDP 梯度
    hook 永不触发，下一次迭代报 "Expected to have finished reduction in the
    prior iteration"）。默认反馈推理是 fp32 精确计算（不做 cast），也比 fp16
    近似更贴近确定性 rollout 的底层算术。此为 2026-09-03 修复
    （docs/project/CHANGELOG.md）。
    """
    cur = cond
    for _ in range(int(lead) - 1):
        with torch.no_grad():
            pred = step_fn(cur).float()   # 与 _rollout_one 的样本相同的 .float() cast
            if clamp:
                pred = pred.clamp(0., 1.)
        cur = torch.cat([cur[:, pred.shape[1]:], pred], dim=1)
    return cur


def expand_ensemble(cond, ensemble_size):
    """功能：把每个条件窗口复制 ensemble_size 份 -> (B*E, C, H, W, Z)。

    每个 ensemble member 是同一条件窗口的独立副本：member 之间只共享初始条件，
    此后各 member 的预测只更新自己的条件窗口，rollout 结束前互不相见（meeting
    点仅在 ensemble_mean 的最终均值）。

    所有权：clone()/repeat_interleave 都物化新内存，不与输入共享存储；E=1 返回
    输入的（新）副本而非原张量，保证 rollout 消耗的 RNG 流与普通单轨迹循环
    逐位一致。

    异常 / 前置条件：cond 必须为 5 维；ensemble_size >= 1（assert 失败）。
    """
    assert cond.dim() == 5, cond.shape
    assert int(ensemble_size) >= 1
    if ensemble_size == 1:
        return cond.clone()
    return cond.repeat_interleave(int(ensemble_size), dim=0)


def ensemble_rollout(model, cond, horizon, ensemble_size=1, num_sample_steps=None,
                     seed=None, seeds=None, clamp=True,
                     remask_feedback=False, ocean_mask=None, static_cond=None):
    """功能：带 ensemble_size 个完全独立 member 的自回归 rollout。

    参数：
    - model：具备 sample(cur, num_sample_steps=None, clamp=True) ->
      (B*, 2, H, W, Z) 次日制 [0,1] 预测的对象（如 ElucidatedDiffusion 或
      PersistenceResidualIAFNO）。
    - cond：(B, 2*CONTEXT, H, W, Z) [0,1] 归一化条件窗口，day-major u/v 交错。
    - horizon：rollout 步数（lead 天数），>= 1。
    - ensemble_size：独立 member 数，>= 1（对确定性模型不起作用：member 全同）。
    - num_sample_steps：采样器步数（None -> 模型默认值；确定性模型忽略）。
    - seed：标量 RNG 种子，在整个 batch rollout 前施加一次（legacy 路径；member
      共享一条 RNG 流但各自消耗独立抽取）。与 `seeds` 互斥。
    - seeds：逐窗口种子（len == B）；每个窗口用 seeds[w] 播种自己的 RNG 流，因此
      窗口 w 的轨迹不依赖 batch 大小或其他窗口的装载分组（逐窗口种子协议）。
    - clamp：透传给 model.sample。
    - remask_feedback：为 True 时，每个预测在存储前、也在进入下一条件窗口前都
      乘以 ocean_mask（陆地 -> 0）；默认 False = 历史无掩膜反馈。
    - ocean_mask：可广播的海洋掩膜（1=有效海洋，0=陆地），如 (1, 2, H, W, Z)；
      当且仅当 remask_feedback 为 True 时必填。
    - static_cond：可选静态条件通道（如 (1, 2, H, W, Z) 的双变量 rho 掩膜），经
      model.sample 的 static_cond kwarg 转发给每一次调用；滑窗 `cur` 保持纯动态
      条件。只能与接受该 kwarg 的模型（确定性基线）一起使用。

    返回：
    - (B, E, horizon, 2, H, W, Z) float32 归一化预测，member 维在轴 1。E=1 时与
      普通顺序 rollout 完全一致（相同 RNG 消耗、相同种子下相同数值）。

    异常 / 前置条件：
    - cond 必须为 5 维且通道数为偶数（day-major 成对 u/v）；seed 与 seeds 不得
      同时给出；remask_feedback=True 必须提供 ocean_mask；seeds 长度必须等于 B，
      否则 assert 失败。
    """
    assert cond.dim() == 5, cond.shape
    assert cond.shape[1] % 2 == 0, "condition must be day-major interleaved"
    B = cond.shape[0]
    E = int(ensemble_size)
    assert E >= 1
    assert not (seed is not None and seeds is not None), "pass either seed or seeds"
    if remask_feedback:
        assert ocean_mask is not None, "remask_feedback=True requires ocean_mask"

    if seeds is not None:
        seeds = [int(s) for s in seeds]
        assert len(seeds) == B, (len(seeds), B)
        outs = []
        for w in range(B):
            torch.manual_seed(seeds[w])                       # 逐窗口播种：窗口 w 的整条轨迹只由 seeds[w] 决定
            outs.append(_rollout_one(model, expand_ensemble(cond[w:w + 1], E),
                                     horizon, num_sample_steps, clamp,
                                     remask_feedback, ocean_mask,
                                     static_cond=static_cond))
        return torch.stack(outs, dim=0)                       # 沿新 batch 轴堆叠各窗口 -> (B, E, L, 2, H, W, Z)

    if seed is not None:
        torch.manual_seed(seed)
    out = _rollout_one(model, expand_ensemble(cond, E), horizon,
                       num_sample_steps, clamp, remask_feedback, ocean_mask,
                       static_cond=static_cond)
    return out.view(B, E, out.shape[1], *out.shape[2:])       # view 只重切 batch 轴 (B*E,...)->(B, E, L, 2, H, W, Z)，不复制内存


def ensemble_mean(preds):
    """功能：member 均值 -> 点预测：(B, E, L, C, H, W, Z) -> (B, L, C, H, W, Z)。

    在 E 轴（轴 1）上求均值；确定性模型的 E 个 member 全同，此时均值为恒等操作。
    """
    assert preds.dim() == 7, preds.shape
    return preds.mean(dim=1)
