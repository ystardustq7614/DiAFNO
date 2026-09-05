"""模块职责：PRE 正式指标与网格换算的纯函数库：rho -> native 反共定位
（rho_to_native）、掩膜误差累计（masked_error_sums）、pooled RMSE
（pooled_rmse）、rho-oracle 诊断（oracle_native_error_sums）与训练侧
masked rel-L2（masked_rel_l2）。

不负责：数据读取、模型、采样与输出文件；无 import 副作用，也不依赖
pre_dataset.py 或模型定义。

关键约束：正式 PRE 指标只在原生 staggered u/v 网格上计算；误差按
(L, 2, Z)（lead x 变量 x 层）逐格累计为 se/ae 之和，总体 RMSE 必须
sqrt(总 se / 总有效计数)（pooled），绝不能对逐层 RMSE 求算术平均；
陆地格（mask==0）对误差和的贡献恒为 0。

依赖关系：numpy（masked_rel_l2 接收 torch 张量，其设备与 dtype 由调用方
负责）；被 pre_trainer.py / pre_evaluate.py / pre_smoke_test.py 与诊断脚本
import。
"""
import numpy as np


def rho_to_native(rho_pred):
    """把 rho 网格 u/v 预测映射回原生 staggered 网格；stencil 与 Plan A
    共定位同构（相邻均值），不做 east/north 旋转。

    参数：
    - rho_pred：shape (B, L, 2, H, W, Z)；channel 0 = u，channel 1 = v，
      物理单位 m/s（调用方负责先反归一化）。

    返回：
    - (u_nat, v_nat)：u_nat (B, L, H, W-1, Z)（沿列轴收缩一维）；
      v_nat (B, L, H-1, W, Z)（沿行轴收缩一维）。

    关键转换：
    - u 沿 xi（列）轴对相邻 rho 点取均值，v 沿 eta（行）轴同理；共定位一侧
      的相邻均值平滑与单侧边界复制使该映射不可逆（该误差由
      oracle_native_error_sums 单独度量）。

    异常 / 前置条件：
    - 输入必须为六维且变量通道数 = 2，否则断言失败。
    """
    rho_pred = np.asarray(rho_pred)
    assert rho_pred.ndim == 6 and rho_pred.shape[2] == 2, rho_pred.shape
    up = rho_pred[:, :, 0]                       # (B, L, H, W, Z) u 通道
    vp = rho_pred[:, :, 1]
    u_nat = 0.5 * (up[:, :, :, :-1] + up[:, :, :, 1:])   # (B, L, H, W-1, Z) 沿 xi 相邻均值
    v_nat = 0.5 * (vp[:, :, :-1, :] + vp[:, :, 1:, :])   # (B, L, H-1, W, Z) 沿 eta 相邻均值
    return u_nat, v_nat


def masked_error_sums(pred, truth, mask):
    """按 lead/层累计掩膜平方/绝对误差和。

    参数：
    - pred/truth：shape 均为 (B, L, H, W, Z) 且完全一致（原生 staggered 的
      单变量场，物理单位 m/s；truth 为未裁剪原始场，land=NaN）。H/W 属于
      原生网格。
    - mask：对应变量的 2-D 原生网格掩膜 (H, W)；1/True = 有效海洋格。

    返回：
    - (se, ae)：各为 (L, Z) 的 float64 累计器；se[l, z] = batch x 行 x 列上
      有效格的 (pred - truth)**2 之和，ae[l, z] 同理为 |pred - truth| 之和。

    关键约束：
    - 陆地格（mask==0）对两个和的贡献恒为 0（NaN 随掩膜剔除，不进入求和）。
    - 严格的 shape 断言用于拒绝把 (L, 1, Z) 之类的累计器误当成 pred/truth
      传入。
    """
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    mask = np.asarray(mask)
    assert pred.ndim == 5 and truth.ndim == 5, (pred.shape, truth.shape)
    assert pred.shape == truth.shape, (pred.shape, truth.shape)
    assert mask.shape == (pred.shape[2], pred.shape[3]), mask.shape
    err = np.where(mask[None, None, :, :, None], pred - truth, np.float64(0.0))  # (B, L, H, W, Z) 无效格恒 0
    se = (err ** 2).sum(axis=(0, 2, 3))          # 按 batch/H/W 求和 -> (L, Z)
    ae = np.abs(err).sum(axis=(0, 2, 3))         # 同上 -> (L, Z)
    return se, ae


def pooled_rmse(se, count):
    """pooled RMSE = sqrt(sum(se) / sum(count))；总有效计数为 0 时返回 0.0。

    se/count 可为标量或任意数组形状（如 (L, 2, Z) 或其子块）；聚合范围由调用
    方传入的内容决定（整体 / 单 lead / 单变量 / 单层均可），但聚合本身始终是
    pooled 的——即平方误差与点数分别求和再相除，绝不能改为对逐层/逐格 RMSE
    求算术平均（各层有效点数不同时，RMSE 平均在数学上不等于总体 RMSE）。
    """
    se = np.asarray(se, np.float64)
    count = np.asarray(count, np.float64)
    n = count.sum()
    if n <= 0:
        return 0.0
    return float(np.sqrt(se.sum() / n))


def oracle_native_error_sums(target_norm, y_lo, y_hi, truth_u, truth_v, mask_u, mask_v):
    """rho-oracle 诊断：rho 网格真值 -> 原生网格，再算掩膜误差。

    参数：
    - target_norm：(B, L, 2, H, W, Z)，[0,1] 归一化的 rho 网格真值
      （channel 0 = u，1 = v）——数据集给出的真实目标（land 已填 0，之后会被
      原生掩膜剔除）。
    - y_lo/y_hi：逐变量裁剪范围（长度 2 的序列，[0]=u，[1]=v）。
    - truth_u：(B, L, H, W-1, Z)，truth_v：(B, L, H-1, W, Z)——未裁剪的
      原生物理真值（原始 u.npy/v.npy）。

    返回：
    - (se, ae)：各为 (L, 2, Z)（channel 0 = u，1 = v）的 float64 累计器，
      布局与 masked_error_sums 的结果按通道槽位直接可加。

    关键转换：
    - 这里的"预测"就是数据集自身的 rho 目标：先按逐变量范围反归一化为物理值，
      再用与 rho_to_native 完全相同的 stencil 映射回原生网格，因此结果只度量
      native -> rho -> native 往返的不可逆误差（裁剪、相邻均值平滑、边界单侧
      复制），不含任何模型误差。
    """
    t = np.asarray(target_norm, np.float32)
    assert t.ndim == 6 and t.shape[2] == 2, t.shape
    lo = np.asarray(y_lo, np.float32).reshape(1, 1, 2, 1, 1, 1)   # 逐变量广播到 (B, L, 2, H, W, Z)
    hi = np.asarray(y_hi, np.float32).reshape(1, 1, 2, 1, 1, 1)
    phys = t * (hi - lo) + lo                        # 反归一化为物理值 (B, L, 2, H, W, Z)
    u_nat, v_nat = rho_to_native(phys)
    se_u, ae_u = masked_error_sums(u_nat, truth_u, mask_u)
    se_v, ae_v = masked_error_sums(v_nat, truth_v, mask_v)
    L, Z = t.shape[1], t.shape[-1]
    se = np.zeros((L, 2, Z), np.float64)
    ae = np.zeros((L, 2, Z), np.float64)
    se[:, 0, :] = se_u
    se[:, 1, :] = se_v
    ae[:, 0, :] = ae_u
    ae[:, 1, :] = ae_v
    return se, ae


def masked_rel_l2(pred, tgt, mask):
    """逐样本相对 L2（仅在有效格上），对 batch 取均值；输入为 torch 张量。

    relL2 = sqrt(sum((pred - target)^2 * mask)) / sqrt(sum(target^2 * mask))：
    通道/空间/层四轴按样本求和，最后 .mean() 取 batch 均值并返回 float。
    mask 必须可广播到 pred/tgt（如 build_mask_tensor 的 (1, 2, H, W, Z)，
    1 = 有效）；平方误差不存在正负抵消；分母 clamp(min=1e-12) 防止空掩膜时
    的 0/0。
    """
    diff2 = ((pred - tgt) ** 2 * mask).sum(dim=(1, 2, 3, 4))
    tgt2 = (tgt ** 2 * mask).sum(dim=(1, 2, 3, 4))
    return (diff2.sqrt() / tgt2.sqrt().clamp(min=1e-12)).mean().item()