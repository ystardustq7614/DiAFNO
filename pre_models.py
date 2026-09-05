#!/usr/bin/env python3
"""模块职责：PRE 任务确定性"持续性-残差"模型（PersistenceResidualIAFNO）与掩膜
MSE 损失（masked_mse_loss）；被 pre_trainer.py 与 pre_evaluate.py 导入使用。

不负责：扩散目标与采样（diffusion.py）；数据加载、归一化与统计缓存
（pre_dataset.py）；rollout 循环（pre_rollout.py 按鸭子类型调用 sample()）。

关键约束：
- 本模块必须保持无副作用：导入即执行的部分只允许定义，不触碰数据、
  设备与 checkpoint。
- prediction = base + residual，base = condition[:, -target_ch:]（day-major
  u/v 条件的最后一天，即持续性基线）；残差头零初始化，未训练的 forward()
  输出恰为持续性。零权重使首个优化步只更新 head（更深层首步梯度为 0），
  更深层从第二步开始收到梯度。
- 条件经 IAFNODiff 的 x_self_cond 槽位进入（历史接口，禁止按名称"修正"）：
  residual = net(x=全零 target, time=常数 c_noise, x_self_cond=cond)，
  patch-embed 看到 cond(14 通道) + 全零(2 通道) = in_chans(16 通道)，与扩散
  路径的通道布局逐位一致。
- time 为 EDM c_noise 形式的常数 0.25·log(time_sigma)；确定性模型没有噪声
  调度，任何固定常数都成立，time_sigma 记录进 checkpoint 供重建校验。
- sample() 按鸭子类型兼容 rollout：num_sample_steps 接受并忽略，
  clamp=True 时把 [0,1] 归一化预测钳制到 [0,1]。
- masked_mse_loss 与 ElucidatedDiffusion.forward 的掩膜口径一致：掩膜可
  广播、1=有效海洋 0=陆地，逐样本只在有效元素取均值，再对 batch 取均值。

依赖关系：IAFNO.IAFNODiff（须 self_condition=True，in_chans = target 通道 +
条件通道）；包装后的状态字典为 net.* 前缀，键布局与 ElucidatedDiffusion 一致。
"""
import math

import torch
import torch.nn as nn


def masked_mse_loss(pred, target, mask):
    """掩膜 MSE：逐样本只在有效元素上取均值，再对 batch 取均值。

    pred/target 形状 (B, C, H, W, Z)（B=窗口数，C=通道，H/W 为 rho 网格
    空间轴，Z 为 sigma 层数）；mask 可广播到该形状，1=有效海洋、0=陆地。
    陆地格不产生损失，分母是每个样本自己的有效元素计数（clamp(min=1)
    防空掩膜除零），与 diffusion.ElucidatedDiffusion.forward 的掩膜分支
    同一口径。
    """
    mse = (pred - target) ** 2
    m = mask.expand_as(mse)
    per_sample = (mse * m).sum(dim=(1, 2, 3, 4)) / m.sum(dim=(1, 2, 3, 4)).clamp(min=1.)
    return per_sample.mean()


class PersistenceResidualIAFNO(nn.Module):
    """薄确定性包装：最后日持续性 + 零初始化 IAFNO 残差。

    net 必须是为条件 EDM 构建的 IAFNODiff：self_condition=True 且
    in_chans 等于 target 通道 + 外部条件通道，与扩散路径的构建完全一致。
    包装后的状态字典为 net.* 前缀，键布局与 ElucidatedDiffusion 一致。

    异常 / 前置条件：net.self_condition 为 False，或 in_chans 在 target
    通道之外没有剩余条件通道时，构造即抛 ValueError（fail-fast）。
    """

    def __init__(self, net, time_sigma=0.002):
        super().__init__()
        if not net.self_condition:
            raise ValueError(
                "PersistenceResidualIAFNO requires an IAFNODiff with "
                "self_condition=True (the condition enters through the "
                "x_self_cond slot)")
        self.net = net
        self.target_ch = int(net.out_chans)
        self.cond_chans = int(net.in_chans) - self.target_ch
        if self.cond_chans <= 0:
            raise ValueError(
                f"IAFNODiff in_chans={net.in_chans} leaves no condition "
                f"channels beyond out_chans={net.out_chans}")
        self.time_sigma = float(time_sigma)
        self.residual_base = "last_day"
        # 残差头零初始化：未训练的 forward() 输出恰为 base；零权重使首个
        # 优化步只更新 head，更深层从第二步开始收到梯度
        nn.init.zeros_(net.head.weight)

    def forward(self, cond, static_cond=None):
        """(B, cond_ch, H, W, Z) 归一化条件 -> (B, target_ch, H, W, Z) 预测。

        传入 static_cond 时（静态掩膜输入实验 08 的 B 臂，如 (1, 2, H, W, Z)
        的双变量 rho 掩膜，沿 batch 广播）：先做形状校验（5 维、batch 为 1
        或与 cond 一致、空间形状一致），batch=1 时 expand 为广播 view（不
        复制内存），再 cat([cond, static_cond], dim=1) 送入骨干的 x_self_cond
        槽位；动态滑窗条件保持纯净，base = cond[:, -target_ch:] 始终是最后
        日持续性。不传 static_cond 时行为与历史 14 通道路径逐位一致。

        异常 / 前置条件：static_cond 形状不可广播或条件通道数不符时抛
        AssertionError（fail-fast，防止错配条件被当作正常输入）。
        """
        if static_cond is not None:
            if static_cond.dim() != 5 or \
                    static_cond.shape[0] not in (1, cond.shape[0]):
                raise AssertionError(
                    f"static_cond shape {tuple(static_cond.shape)} is not "
                    f"broadcastable to batch {cond.shape[0]}")
            if static_cond.shape[2:] != cond.shape[2:]:
                raise AssertionError(
                    f"static_cond spatial shape {tuple(static_cond.shape[2:])} "
                    f"!= condition {tuple(cond.shape[2:])}")
            if static_cond.shape[0] == 1 and cond.shape[0] > 1:
                static_cond = static_cond.expand(cond.shape[0], -1, -1, -1, -1)
            x_self_cond = torch.cat([cond, static_cond], dim=1)
        else:
            x_self_cond = cond
        if x_self_cond.shape[1] != self.cond_chans:
            raise AssertionError(
                f"condition channels {x_self_cond.shape[1]} != expected "
                f"{self.cond_chans}")
        base = cond[:, -self.target_ch:]  # 持续性基线：day-major 条件的最后一天（最后 target_ch 通道）
        batch = cond.shape[0]
        # 常数 c_noise（=0.25·log σ 的 EDM 形式）：确定性路径无噪声调度，
        # 固定常数即可，time_sigma 已记录进 checkpoint
        time = torch.full((batch,), 0.25 * math.log(self.time_sigma),
                          device=cond.device)
        # target 槽位传全零、条件走 x_self_cond 槽位：与扩散路径的
        # [条件, 全零 target] 通道布局一致（见模块 docstring）
        residual = self.net(torch.zeros_like(base), time, x_self_cond)
        return base + residual

    def sample(self, cond, batch_size=None, num_sample_steps=None, clamp=True,
               static_cond=None):
        """确定性预测；num_sample_steps 仅接受并忽略（与 EDM 采样器的调用
        签名兼容，pre_rollout 按鸭子类型调用）。clamp=True 时把 [0,1] 归一化
        预测钳制到 [0,1]，与 EDM 采样器反归一化后的输出值域一致。"""
        pred = self.forward(cond, static_cond=static_cond)
        if clamp:
            pred = pred.clamp(0., 1.)
        return pred
