# ---------------------------------------------------------------------------------------------
# Author: Yuchi Jiang
# LatestVersionDate: 07/27/2026 (specifically designed for diffusion)
# ---------------------------------------------------------------------------------------------

# Many thanks to all the authors of:
# Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. arXiv preprint arXiv:2111.13587 (2021)

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath

torch.manual_seed(123)

################################################################################################################################

class SinusoidalPosEmb(nn.Module):
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
        # print(emb.shape)
        return emb

class RMSNorm(nn.Module):
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
    def __init__(self, length, patch_size, embed_dim, in_chans):              #####   Length & Patch_size must be 3 dims   #####
        super().__init__()
        num_patches = (length[0] // patch_size[0]) * (length[1] // patch_size[1]) * (length[2] // patch_size[2])
        self.length = length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        ##### make sure an input of shape: (bs x y z c nt) #####

        x = x.flatten(4)                     ##### (bs x y z c*nt)
        x = x.permute(0, 4, 1, 2, 3)         ##### (bs c*nt x y z)
        x = self.proj(x)                     ##### (bs embed_dim x//px y//py z//pz)
        x = x.permute(0, 2, 3, 4, 1)         ##### (bs x//px y//py z//pz embed_dim)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        ##### make sure an input of shape: (bs x//px y//py z//pz embed_dim) #####

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Block(nn.Module):
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

        ##### must after patch_embed #####
        ##### input (bs x//px y//py z//pz embed_dim) #####

        residual = x

        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class AFNO(nn.Module):
    def __init__(self, hidden_size, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, X, Y, Z, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = Z // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

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

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(X, Y, Z), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias

##################################################################################################################

class IAFNODiff(nn.Module):
    def __init__(
            self,
            dim, # 64 66 32
            patch_size,
            embed_dim,
            num_blocks,
            in_chans,
            out_chans,
            ex_layer,
            nlayer,
            hidden_size_factor,
            dim_f, # 64 65 32
            self_condition,
            cond_chans=None, # external-condition channels; None -> same as in_chans (legacy behavior)
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.dim = dim
        self.dim_f = dim_f
        # noisy target (in_chans) is channel-concatenated with the external condition
        # (cond_chans); legacy default cond_chans == in_chans reproduces the old doubling.
        if cond_chans is None:
            cond_chans = in_chans
        self.in_chans = in_chans + (cond_chans if self_condition else 0)
        self.out_chans = out_chans
        self.ex_layer = ex_layer
        self.nlayer = nlayer
        self.patch_size = patch_size
        self.self_condition = self_condition
        self.patch_embed = PatchEmbed(dim, patch_size, embed_dim, self.in_chans)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim[0] // patch_size[0], dim[1] // patch_size[1], dim[2] // patch_size[2], embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, hidden_size_factor, num_blocks)
            for i in range(self.ex_layer)])

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1]*self.patch_size[2], bias=False)

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
        
        if (self.ex_layer!=1 & self.nlayer==1):
            for j in range(self.ex_layer):
                x = self.blocks[j](x)
        else:
            for i in range(self.nlayer):
                for j in range(self.ex_layer):
                    coef = 1/(self.nlayer * self.ex_layer)
                    x = x + self.blocks[j](x) * coef

        return x

    def forward(self, x, time, x_self_cond):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        ##### time embedding process

        x = self.upproj(x)
        x = self.rmsnorm1(x)

        t = self.time_mlp(time)
        t = rearrange(t, 'b c -> b c 1 1 1')
        scale_shift = t.chunk(2, dim = 1)
        scale, shift = scale_shift
        x = x * (scale + 1) + shift
        x = self.silu(x)

        x = self.downproj(x)
        x = self.rmsnorm2(x)
        x = self.silu(x)
        
        x = rearrange(x, "bs c x y z -> bs x y z c")

        ######################

        ##### considering patch size [2,2,2] and input shape [x=64,y=65,z=32], we add a zero-information padding in y-axis to provide a smoother patching
        ##### in other words: dim_f:[64,65,32] --> dim:[64,66,32] 
        if (self.dim_f[0]!=self.dim[0]):
            pad = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
            x = torch.cat((x, pad), 1)
        if (self.dim_f[1]!=self.dim[1]):
            pad = x.new_zeros(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4])
            x = torch.cat((x, pad), 2)
        if (self.dim_f[2]!=self.dim[2]):
            pad = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[4])
            x = torch.cat((x, pad), 3)
        
        x = self.forward_features(x)
        x = self.head(x)
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
        if (self.dim_f[0]!=self.dim[0]):
            x = x[:, :-1, :, :, :]
        if (self.dim_f[1]!=self.dim[1]):
            x = x[:, :, :-1, :, :]
        if (self.dim_f[2]!=self.dim[2]):
            x = x[:, :, :, :-1, :]

        x = rearrange(x, "bs x y z c -> bs c x y z")
        return x
