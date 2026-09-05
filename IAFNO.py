# 作者：Yuchi Jiang
# 最近更新：07/27/2026（专为 diffusion 路径设计）
#
# 引用：Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. arXiv preprint arXiv:2111.13587 (2021)

"""模块职责：IAFNO 3D 骨干——正弦时间嵌入、3D Conv patch 化（PatchEmbed）、
频域自适应傅里叶 token 混合器（AFNO）与 patch head 还原；IAFNODiff 是条件 EDM
（diffusion.py）与确定性持续性-残差模型（pre_models.py）共用的骨干。

不负责：扩散噪声调度、损失与采样（diffusion.py）；持续性基线与掩膜损失
（pre_models.py）；数据加载与 [0,1] 归一化（pre_dataset.py）。

关键约束：
- torch.manual_seed(123) 在模块导入时执行：import 本模块会重置 torch 全局
  RNG，import 顺序会改变其后的全局随机流。
- forward 的 x_self_cond 槽位实际承载外部条件（PRE 为 14 通道，静态掩膜臂
  追加 2 通道至 16），是历史接口兼容而非常规 self-conditioning，禁止按名称
  "修正"。
- patch 尺寸必须整除网格尺寸，否则进入非整除轴的零填充/裁剪路径
  （dim vs dim_f）；填充轴写错会使输出与目标/掩膜错位。
- AFNO 的频域计算强制升到 fp32（半精度 FFT 支持受限），结束恢复进入时 dtype。

依赖关系：torch、timm（DropPath）、einops；网格与 patch 由调用方给定——
legacy 湍流路径网格 (64, 66, 32)/patch (2, 2, 2)（真实 y=65 零填充到 66），
PRE 路径 surface 400x441x1 patch (4, 3, 1)、full3d 400x441x30 patch (4, 3, 2)，
逐轴整除、padding 分支不触发。
"""

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath

torch.manual_seed(123)  # 模块级全局种子（项目约定，与 trainer.py 一致；副作用见模块 docstring）


class SinusoidalPosEmb(nn.Module):
    """把 (b,) 标量序列（EDM 传入 c_noise = 0.25·log σ）映射为 (b, dim) 正弦特征：
    频率取 theta^(−i/(half_dim−1)) 几何级数，前半 sin、后半 cos 拼接。
    dim 取 IAFNODiff 的总输入通道数，输出直接进入 time_mlp。"""

    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RMSNorm(nn.Module):
    """沿通道维（dim=1）做 L2 归一化，再乘可学习增益 g 与 sqrt(dim)：
    g 形状 (1, dim, 1, 1, 1)，广播到 (bs, c, x, y, z) 的 5D 特征。"""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class PatchEmbed(nn.Module):
    """3D Conv patch 化：把 channel-last 网格切成不重叠 patch 并映射到 embed_dim。

    布局链路（channel-last -> channel-first -> token 网格）：
        (bs, x, y, z, c) --flatten(4)--> (bs, x, y, z, c)
            --permute--> (bs, c, x, y, z)           # Conv3d 要求 channel-first
            --Conv3d(kernel=stride=patch)--> (bs, embed_dim, x//px, y//py, z//pz)
            --permute--> (bs, x//px, y//py, z//pz, embed_dim)
    历史上接受 6D 输入 (bs,x,y,z,c,nt)，flatten(4) 把时间维并入通道得 c·nt；
    当前调用链（IAFNODiff）已提前把时间维并入通道，flatten(4) 为恒等操作。

    异常 / 前置条件：length 与 patch_size 必须是三元组，且 patch 逐轴整除
    网格，否则 token 网格与 pos_embed/head 的形状契约被破坏。
    """

    def __init__(self, length, patch_size, embed_dim, in_chans):
        super().__init__()
        num_patches = (length[0] // patch_size[0]) * (length[1] // patch_size[1]) * (length[2] // patch_size[2])
        self.length = length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # kernel=stride 的 Conv3d 即不重叠 patch 化：每个空间 patch 产出一个 token，
        # token 网格三轴尺寸为原网格除以 patch
        x = x.flatten(4)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)

        return x

class Mlp(nn.Module):
    """两层 MLP（GELU + Dropout），token 布局 (bs, x//px, y//py, z//pz, embed_dim)
    进出不变；hidden_features 由 Block 固定为 4*embed_dim。"""

    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    """AFNO 块：pre-norm 的频域 token 混合 + MLP，输入输出均为 patch token 网格
    (bs, x//px, y//py, z//pz, embed_dim)，必须在 patch_embed 之后使用。

    double_skip=True（当前唯一用法）时残差基准在 AFNO 支输出后更新一次，
    MLP 支再累加一次（两次残差）；drop_path=0 时 DropPath 退化为 Identity
    （IAFNODiff 构造 Block 时不传 drop_path）。
    """

    def __init__(
            self, embed_dim, hidden_size_factor, num_blocks,
            drop_path=0., norm_layer=nn.LayerNorm, double_skip=True
        ):
        super().__init__()
        hidden_features = embed_dim * 4

        self.filter = AFNO(embed_dim, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(embed_dim, hidden_features, embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual  # double skip：AFNO 支结果先并入残差基准，MLP 支再基于新基准累加
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        return x

class AFNO(nn.Module):
    """AFNO token 混合器：沿 token 网格三个空间轴做实 FFT，在频域对每个频率点的
    通道向量施加分 block 的两层复数线性变换，irfftn 还原后加回输入残差。

    形状与符号：输入/输出 (B, X, Y, Z, C)——B=batch，X/Y/Z 为 patch 网格三轴的
    token 数（原空间尺寸除以 patch），C=embed_dim。

    关键转换：
    - rfftn(dim=(1,2,3), norm="ortho")：只有最后一个被变换的频率轴减半——
      Z -> Z//2+1（Hermitian 对称），X/Y 保持全尺寸；irfftn 用 s=(X,Y,Z) 还原。
    - 通道 C 拆成 num_blocks 组（block_size = C // num_blocks，须整除），复数
      线性在各 block 内独立进行；权重以 [实部, 虚部] 两个实张量存储，复数乘法
      由两组实数 einsum 展开，第一层输出对实/虚部各自过 ReLU（AFNO 对复分量
      独立施加非线性）。
    - mode 截断：仅频率轴前 kept_modes 个模式参与两层变换，其余保持 0；
      hard_thresholding_fraction=1（当前唯一用法）时不发生截断。
    - softshrink(lambd=0.01) 逐元素把 |x|<λ 的频域分量置 0（稀疏化），
      stack([real, imag]) + view_as_complex 重组复数张量后 irfftn 还原。
    """

    def __init__(self, hidden_size, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # 复数权重的前导维 2 = [实部, 虚部]；每层是按 block 分组的两组实数矩阵
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x  # 输入残差备份：频域混合是围绕恒等映射的扰动，最后加回

        dtype = x.dtype   # 记录进入时 dtype：autocast 下输入可能为 fp16
        x = x.float()     # FFT 强制 fp32（半精度 FFT 支持受限）；dtype 转换产生副本
        B, X, Y, Z, C = x.shape

        # rfftn 只压缩最后一个被变换的频率轴：Z -> Z//2+1（Hermitian 对称），X/Y 保持全尺寸
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        # o1/o2 以全零起底：kept_modes 之外的模式保持 0（硬阈值截断的"丢弃"实现）
        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = Z // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # 复数线性层的实数展开：w = w[0] + i·w[1]，(a+bi)·w 的实部 = a·w[0] − b·w[1]、
        # 虚部 = a·w[1] + b·w[0]；einsum '...bi,bio->...bo' 中 b=block 组号（输入与
        # 权重在该轴对齐）、i=组内通道（收缩）、o=组内输出通道，'...' 覆盖 (B,X,Y,Zp)
        o1_real[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # softshrink 把 |x|<lambd 的频域分量置 0（稀疏化）；stack + view_as_complex
        # 以最后一维 [real, imag] 重组复数张量，reshape 回 (B,X,Y,Zp,C)
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        # irfftn 以 s=(X,Y,Z) 指定输出全尺寸，从半谱还原 Z；随后恢复进入时 dtype
        x = torch.fft.irfftn(x, s=(X, Y, Z), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias

class IAFNODiff(nn.Module):
    """扩散与确定性两条路径共用的 IAFNO 骨干。

    通道契约（PRE：目标 2 通道 u/v，条件窗口 14 通道 day-major u/v 交错）：
    - 构造参数 in_chans 只指 noisy target 的通道数；self_condition=True 时
      forward 把 x_self_cond（外部条件，cond_chans 通道）与 x 沿通道轴拼接，
      条件在前、noisy target 在后，patch-embed 输入通道为
      self.in_chans = in_chans + cond_chans。
    - x_self_cond 槽位沿用了原版 self-conditioning 的函数签名，但实际承载的
      是外部条件而非模型自身的历史输出——历史接口兼容，禁止按名称"修正"。
    - cond_chans=None（默认）时取 cond_chans = in_chans，self.in_chans =
      2*in_chans，复现 legacy 湍流路径的通道加倍；self_condition=False 时
      不拼接、忽略 x_self_cond。
    - time 输入为 EDM 的 c_noise = 0.25·log σ（形状 (b,)）；确定性路径传入
      同形式的常数（见 pre_models）。

    空间流程：(bs, c, x, y, z) —— 时间调制 —— channel-last —— （按需零填充）——
    patch 化 —— blocks —— head 还原 —— 裁掉填充 —— (bs, out_chans, x, y, z)。
    """

    def __init__(
            self,
            dim, # 网格（patch 可整除）：例 legacy (64, 66, 32)（含 y 轴填充）；PRE surface (400, 441, 1)、full3d (400, 441, 30)
            patch_size,
            embed_dim,
            num_blocks,
            in_chans,
            out_chans,
            ex_layer,
            nlayer,
            hidden_size_factor,
            dim_f, # 真实数据网格：例 legacy (64, 65, 32)；与 dim 逐轴比较决定是否零填充
            self_condition,
            cond_chans=None, # 外部条件通道数；None -> 取 in_chans（复现 legacy 通道加倍）
            drop_rate=0.,
            drop_path_rate=0., # 未使用：Block 固定 drop_path=0，DropPath 退化为 Identity
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.dim = dim
        self.dim_f = dim_f
        if cond_chans is None:
            cond_chans = in_chans  # legacy 默认：条件通道数 == target 通道数，复现旧的通道加倍
        # 注意：构造参数 in_chans 只是 noisy target 通道；self.in_chans 才是
        # patch-embed 的实际输入通道（target + 条件）
        self.in_chans = in_chans + (cond_chans if self_condition else 0)
        self.out_chans = out_chans
        self.ex_layer = ex_layer
        self.nlayer = nlayer
        self.patch_size = patch_size
        self.self_condition = self_condition
        self.patch_embed = PatchEmbed(dim, patch_size, embed_dim, self.in_chans)
        # 可学习位置嵌入（零初始化），形状与 token 网格一致，按 batch 广播；
        # 基于 dim（填充后网格）计算，与 patch_embed 的输出网格对齐
        self.pos_embed = nn.Parameter(torch.zeros(1, dim[0] // patch_size[0], dim[1] // patch_size[1], dim[2] // patch_size[2], embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, hidden_size_factor, num_blocks)
            for i in range(self.ex_layer)])

        # head 不带 bias：每个 token 直接线性映射到其 patch 内全部像素值
        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1]*self.patch_size[2], bias=False)

        # time_mlp：c_noise 标量 -> 正弦嵌入（self.in_chans 维）-> 两层 MLP
        # （4*self.in_chans 维），输出 chunk 成 FiLM 的 scale/shift 各 2*self.in_chans 维
        sinu_pos_emb = SinusoidalPosEmb(self.in_chans, theta = 10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.in_chans, self.in_chans*4),
            nn.GELU(),
            nn.Linear(self.in_chans*4, self.in_chans*4)
        )
        self.silu = nn.SiLU()
        self.rmsnorm1 = RMSNorm(2*self.in_chans)
        self.rmsnorm2 = RMSNorm(self.in_chans)

        self.upproj = nn.Conv3d(self.in_chans, 2*self.in_chans, 3, padding = 1)
        self.downproj = nn.Conv3d(2*self.in_chans, self.in_chans, 3, padding = 1)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 显式括号固定分支语义：仅"多个 explicit Block 且 nlayer==1"时顺序
        # 堆叠，其余（含 nlayer>=2 或 ex_layer==1）走缩放残差堆叠。
        # 2026-09-05 修复：旧写法 `self.ex_layer!=1 & self.nlayer==1` 因 & 优先
        # 级高于比较，被解析为链式比较"ex_layer != (1 & nlayer) 且 (1 & nlayer)
        # == 1"，等价于"ex_layer != 1 且 nlayer 为奇数"，会把 (ex_layer!=1,
        # nlayer 为奇数>1) 的配置错误送入顺序堆叠支。当前全部既有配置
        # （nlayer=2/4，或冒烟 ex_layer=1,nlayer=1）在新旧条件下都落在 else 支，
        # 行为不变。
        if (self.ex_layer != 1) and (self.nlayer == 1):
            # 顺序堆叠：逐个过 ex_layer 个 Block，不做残差缩放
            for j in range(self.ex_layer):
                x = self.blocks[j](x)
        else:
            # implicit 式堆叠：同一组 Block 重复 nlayer 轮，每轮残差按
            # 1/(nlayer * ex_layer) 缩放，避免深层残差直接累加导致量值发散
            for i in range(self.nlayer):
                for j in range(self.ex_layer):
                    coef = 1/(self.nlayer * self.ex_layer)
                    x = x + self.blocks[j](x) * coef

        return x

    def forward(self, x, time, x_self_cond):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))  # 未提供条件时以全零代替
            x = torch.cat((x_self_cond, x), dim = 1)  # [外部条件 cond_chans 通道, noisy target in_chans 通道]

        # 时间条件调制（FiLM）：先升维到 2*self.in_chans，用 time_mlp 输出的
        # scale/shift 做逐通道仿射调制，再降回 self.in_chans
        x = self.upproj(x)
        x = self.rmsnorm1(x)

        t = self.time_mlp(time)
        t = rearrange(t, 'b c -> b c 1 1 1')  # (b, 4*in_chans) -> (b, 4*in_chans, 1, 1, 1)，沿空间轴广播
        scale_shift = t.chunk(2, dim = 1)
        scale, shift = scale_shift
        x = x * (scale + 1) + shift
        x = self.silu(x)

        x = self.downproj(x)
        x = self.rmsnorm2(x)
        x = self.silu(x)
        
        x = rearrange(x, "bs c x y z -> bs x y z c")  # 转 channel-last，对齐 PatchEmbed 的输入契约

        # 非整除轴的零信息填充：dim_f 为真实网格、dim 为 patch 可整除网格，逐轴
        # 比较，不足处在该轴末尾补 1 个零。legacy 网格 y=65 不能被 patch 2 整除，
        # 补到 66。填充轴必须与 dim/dim_f 的轴序一一对应：补错轴会把零信息混入
        # 错误的空间方向，还原后的裁剪也随之错位。PRE 网格（surface 400x441x1、
        # full3d 400x441x30）与 patch (4,3,1)/(4,3,2) 逐轴整除，以下分支不触发。
        # new_zeros 继承 x 的 dtype/device（autocast 下为 fp16），不引入精度断层。
        if (self.dim_f[0]!=self.dim[0]):        # x 轴不足
            pad = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
            x = torch.cat((x, pad), 1)
        if (self.dim_f[1]!=self.dim[1]):        # y 轴不足
            pad = x.new_zeros(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4])
            x = torch.cat((x, pad), 2)
        if (self.dim_f[2]!=self.dim[2]):        # z 轴不足
            pad = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[4])
            x = torch.cat((x, pad), 3)
        
        x = self.forward_features(x)
        x = self.head(x)  # 每个 token 映射到 out_chans*px*py*pz，随后按 patch 展开还原像素
        # patch 还原：token 网格 (dim//patch 各轴) 展开回像素网格 (dim[0], dim[1], dim[2])
        x = rearrange(
            x,
            "b h w z (p1 p2 p3 c_out) -> b (h p1) (w p2) (z p3) c_out",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
            h=self.dim[0] // self.patch_size[0],
            w=self.dim[1] // self.patch_size[1],
            z=self.dim[2] // self.patch_size[2],
        )
        # 裁掉各轴末尾的零填充（与进入 forward_features 前的填充一一对应），对齐 dim_f
        if (self.dim_f[0]!=self.dim[0]):
            x = x[:, :-1, :, :, :]
        if (self.dim_f[1]!=self.dim[1]):
            x = x[:, :, :-1, :, :]
        if (self.dim_f[2]!=self.dim[2]):
            x = x[:, :, :, :-1, :]

        x = rearrange(x, "bs x y z c -> bs c x y z")
        return x
