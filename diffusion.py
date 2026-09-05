# 作者：Vivek Oommen
# 日期：08/01/2024
# 本代码参考了以下 GitHub 仓库：
# [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

"""模块职责：ElucidatedDiffusion——EDM（Karras 等 2022《Elucidating the Design Space
of Diffusion-Based Generative Models》）包装器：围绕传入的 IAFNODiff 骨干实现
预条件网络输出（c_skip/c_out/c_in/c_noise，论文 Table 1 与式 (7)）、训练损失
（log-normal σ 采样 + λ(σ) 加权 + 可选掩膜）与 Heun 采样器。

不负责：模型结构（net 由调用方构造传入）；数据加载与 [0,1] 归一化
（pre_dataset.py）；sigma_data 的换算（pre_config.py——stats 缓存存的是 [0,1]
归一化数据的 pooled std，本类内部把 images*2-1，故 EDM sigma_data = 2.0 ×
stats["sigma"]，surface 为 0.17120）；σ 调度超参的运行时选择（pre_evaluate.py）。

关键约束：
- 输入 images 为 [0,1] 归一化域；采样输出同样反归一化回 [0,1]。
- x_self_cond/self_cond 槽位实际承载外部条件（14/16 通道），沿用了原版
  self-conditioning 的接口名——历史兼容，禁止按名称"修正"。
- 文件中被注释掉的 self-conditioning 语句块是有意禁用的历史接口标记，
  不得删除或恢复。
- 掩膜路径的损失分母是逐样本广播掩膜的有效元素计数（见 forward）。

依赖关系：net 需暴露 self_condition 属性并接受 (x, time, x_self_cond) 三参
调用（IAFNODiff 契约）；仅依赖 torch/tqdm/einops。
"""

from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

# 工具函数

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 张量工具函数

def log(t, eps = 1e-20):
    """先钳制到 eps 再取对数，避免 log(0) 产生 -inf。"""
    return torch.log(t.clamp(min = eps))

# 归一化工具函数

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 主类：ElucidatedDiffusion（EDM 包装器）

class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_size_h,
        image_size_w,
        image_size_z,
        channels = 3,
        num_sample_steps = 32, # 采样步数
        sigma_min = 0.002,     # 噪声水平下界
        sigma_max = 80,        # 噪声水平上界：采样从 σ_max 尺度的纯噪声出发
        sigma_data = 0.5,      # 数据分布标准差；PRE 必须传 2.0×stats["sigma"]（见 pre_config.sigma_data_from_stats）
        rho = 7,               # 采样 schedule 的幂次
        P_mean = -1.2,         # 训练时 log σ 采样分布（log-normal）的均值
        P_std = 1.2,           # 训练时 log σ 采样分布（log-normal）的标准差
        S_churn = 80,          # churn 强度：随数据集调整，见论文 Table 5
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()
        # assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition  # 仅镜像 net 的开关；本类不据此分支，self_cond 一律透传

        self.net = net

        # 网格尺寸：h/w/z 与输入张量 (b, c, h, w, z) 的后三维一致，
        # 须与 net 的 dim_f 相同；z 为 sigma 层数（表层 preset 为 1）

        self.channels = channels
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        self.image_size_z = image_size_z

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # 即论文中的 N（总采样步数）

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # 预条件系数（论文 Table 1）

    def c_skip(self, sigma):
        """跳连权重：σ→0 时趋向 1，输出逼近含噪输入自身的恒等解。"""
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        """输出缩放：把网络原始输出缩放到 D_theta 的量纲。"""
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        """输入缩放：σ 大时压低含噪输入幅度，稳定网络输入分布。"""
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        """噪声水平条件：0.25·log σ，形状 (b,)，交给 net 的 time 槽位。"""
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_images, sigma, self_cond = None, clamp = False):
        """预条件网络输出（论文式 (7)）：D_theta = c_skip·x + c_out·F_theta(c_in·x, c_noise)。

        参数：
        - noised_images：(b, c, h, w, z) 含噪输入。
        - sigma：标量或 (b,)；标量时扩展为逐样本 (b,)，再 rearrange 成
          (b, 1, 1, 1, 1) 与 5D 张量广播。
        - self_cond：外部条件，原样传入 net 的 x_self_cond 槽位。
        - clamp=True 时输出钳制到 [-1, 1]（采样路径使用）。
        """
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            self_cond
        )  # 即论文中的 F_theta（预条件网络原始输出）

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out  # 即论文中的 D_theta：跳连 + 缩放后的网络输出

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # 采样（Heun 采样器）

    def sample_schedule(self, num_sample_steps = None):
        """生成递减 σ 序列（论文式 (5)）：
        σ_i = (σ_max^(1/ρ) + i/(N−1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ，
        末尾追加 σ=0 作为循环终点（该步不执行网络前向）。"""
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.)  # 末尾追加 σ=0，仅作采样循环终点
        return sigmas

    @torch.no_grad()
    def sample(self, self_cond, batch_size = None, num_sample_steps = None, clamp = True):
        """Heun 采样器：从 σ_max 尺度的纯噪声出发逐步去噪，返回 [0,1] 域预测场。

        参数：
        - self_cond：外部条件（PRE 路径为 14 通道归一化条件），每个去噪步原样
          传入网络（x_self_cond 槽位语义见模块 docstring）；batch_size 取其首维。
        - clamp=True：每步网络输出与终值都钳制到 [-1, 1]。

        副作用：@torch.no_grad() 包裹，不建梯度图；噪声取自 torch 全局 RNG
        （PRE 评估通过外层逐窗口种子控制轨迹）。
        """
        batch_size = self_cond.shape[0]
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size_h, self.image_size_w, self.image_size_z)

        # 生成 σ 调度并计算各步 churn 强度 gamma；zip 把 (当前 σ, 下一 σ, gamma) 配成三元组
        sigmas = self.sample_schedule(num_sample_steps)

        # churn 只在 [S_tmin, S_tmax] 区间启用，强度取 min(S_churn/N, √2−1)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # 初始图像即 σ_max 尺度的高斯白噪声
        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # self-conditioning 分支（整体禁用，历史接口，勿恢复）：

        # x_start = self_cond

        # 逐步去噪
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step',disable=True):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))  # 转 python 标量，分支判断与算术不再依赖张量

            eps = self.S_noise * torch.randn(shape, device = self.device)  # 每步新抽噪声并乘 S_noise——churn 的随机性来源

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            # self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp = clamp)
            # ODE 右端项 (x − D_theta(x))/σ：Euler 步沿 σ 减小方向推进
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # Heun 二阶校正：仅在 σ_next != 0（非最后一步）时执行

            if sigma_next != 0:
                # self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)  # 两个斜率取平均（梯形法）

            images = images_next
            # x_start = model_output_next if sigma_next != 0 else model_output

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    @torch.no_grad()
    def sample_using_dpmpp(self, self_cond, batch_size = None, num_sample_steps = None):
        """功能：DPM++ 2M 二阶多步采样器。当前全仓无调用方（PRE 正式评估走
        sample() 的 Heun 路径），保留作备选。

        致谢 Katherine Crowson (https://github.com/crowsonkb) 完成推导：
        https://arxiv.org/abs/2211.01095

        参数：
        - self_cond：外部条件，每个去噪步原样传入网络（与 sample() 的 Heun
          路径一致）；batch_size 取其首维。

        关键转换：在 t = −log σ 域迭代（σ = e^(−t)）；存在上一步 denoised 且
        未到最后一步时做二阶外推（gamma = −1/(2r)），否则回退一阶。
        """
        batch_size = self_cond.shape[0]
        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        # 初始噪声必须覆盖 z 轴，与 (b, c, h, w, z) 的网络输入契约一致
        # （2026-09-05 修复：旧实现为 4 维 (b, c, h, w)，在 3D 骨干必崩）。
        shape = (batch_size, self.channels, self.image_size_h, self.image_size_w, self.image_size_z)
        images  = sigmas[0] * torch.randn(shape, device = device)

        # σ = e^(−t) 与 t = −log σ 互逆：在对数域做外推数值更稳
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item(), self_cond)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            # 二阶多步项：用上一步 denoised 做线性组合校正；首步或 σ_next=0 时回退一阶
            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    # 训练目标

    def loss_weight(self, sigma):
        """EDM 损失权重 λ(σ) = (σ² + σ_data²) / (σ·σ_data)²：平衡各噪声水平的贡献。"""
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        """训练 σ 采样：log σ ~ N(P_mean, P_std)，即 log-normal 分布。"""
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, images, self_cond=None, mask=None):
        """功能：单步 EDM 去噪训练损失（返回标量）。

        参数：
        - images：(b, c, h, w, z)，[0,1] 归一化目标场（PRE：c=2，u/v）；h/w/z/c
          必须与构造时的 image_size_*/channels 一致，否则 assert 失败（fail-fast）。
          内部先映射到 [-1,1]（EDM 工作域）；因此 EDM 的 sigma_data 必须取
          2.0 × [0,1] 数据的 pooled std（见 pre_config.SIGMA_DATA_SCALE）。
        - self_cond：外部条件张量，原样传入 net 的 x_self_cond 槽位；可为 None。
        - mask：可广播到 (b, c, h, w, z) 的掩膜，1=有效海洋、0=陆地；接受
          (1,1,h,w,z)、(1,c,h,w,z)、(b,c,h,w,z) 三种形式。expand_as 只产生
          广播 view，不复制内存。

        关键转换：
        - 加噪：x_noised = x + σ·ε（EDM 中信号衰减系数 α=1，无信号缩放）；
          σ 逐样本采样后经 rearrange 广播到 5D。
        - 掩膜路径：逐样本只在有效元素上取 MSE 均值，分母是该样本广播掩膜的
          有效元素计数（clamp(min=1) 防空掩膜除零）；双变量掩膜与逐 batch
          变化掩膜都按元素精确计数一次，陆地填充值不进入去噪目标。
        - 无掩膜路径：逐样本全元素均值。
        - 最后乘 loss_weight(σ)，再对 batch 取均值。
        """
        batch_size, c, h, w, z = images.shape
        device = images.device

        image_size_h, image_size_w, image_size_z, channels = self.image_size_h, self.image_size_w, self.image_size_z, self.channels
        assert h == image_size_h and w == image_size_w, f'height and width of image must be {image_size_h}, {image_size_w}'
        assert z == image_size_z, f'depth of image must be {image_size_z}, got {z}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)  # [0,1] -> [-1,1]：EDM 工作域（sigma_data 换算见 docstring）

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # EDM 中信号衰减系数 α=1：加噪即 x + σ·ε

        # self-conditioning 训练分支：整体禁用，历史接口标记，勿删除或恢复

        # self_cond = None

        # if self.self_condition and random() < 0.5:
        #     # 源自 Hinton 组的 bit diffusion 论文
        #     with torch.no_grad():
        #         self_cond = self.preconditioned_network_forward(noised_images, sigmas)
        #         self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)

        losses = F.mse_loss(denoised, images, reduction = 'none')

        if exists(mask):
            # 1=有效海洋、0=陆地；分母=该样本广播掩膜的有效元素数（逐样本精确计数一次）
            mask = mask.expand_as(losses)
            losses = (losses * mask).sum(dim = (1, 2, 3, 4))
            denom = mask.sum(dim = (1, 2, 3, 4))
            losses = losses / denom.clamp(min = 1.)
        else:
            losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()