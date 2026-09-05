"""模块职责：原始 DiAFNO/FNO 代码的共享工具库——checkpoint 读写、MAT/HDF5 数据读取、
归一化器与 Lp/Sobolev 损失，供 trainer.py 等 legacy 路径使用。

不负责：PRE 海流任务的原生 staggered 网格指标（由 pre_metrics.py 承担）；PRE 的统计
缓存与掩膜（由 pre_dataset.py 承担）。

关键约束：各 helper 保留 .cuda() 便捷方法以兼容旧调用，但当前主训练路径不再调用它们；
load_checkpoint 默认 weights_only=True（安全边界），仅对已核实来源的项目 checkpoint
才允许显式传 weights_only=False。

依赖关系：torch、numpy、scipy.io、h5py；被 trainer.py 与 smoke_test.py 导入。
"""

import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location=None, weights_only=True):
    """功能：把 checkpoint 恢复到调用方指定的设备上，并返回完整 checkpoint dict。

    参数：
    - path：checkpoint 文件路径。
    - model：必填，权重写入目标。
    - optimizer/scheduler/scaler：可选；checkpoint 存在对应 *_state_dict 字段时才恢复。
    - map_location：torch.load 的设备映射（None 按张量记录的设备还原）。
    - weights_only=True：安全默认，不反序列化任意 Python 对象，杜绝 pickle 注入。

    返回：
    - checkpoint dict（裸 state_dict，或含 model_state_dict 等字段的完整 dict）。

    关键转换：
    - 裸 state_dict（旧版 torch.save(model.state_dict()) 的产物）经
      checkpoint.get('model_state_dict', checkpoint) 回退分支读取；此形态下
      优化器/调度器/scaler 字段必然缺失，不会被恢复。

    异常 / 前置条件：
    - weights_only=True 加载失败时抛 RuntimeError 并说明原因；只有对已核实来源的
      项目 checkpoint 才允许显式传 weights_only=False（信任加载路径），禁止盲目降级。
    """
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as e:
        if weights_only:
            raise RuntimeError(
                f"failed to load {path} with weights_only=True ({type(e).__name__}: {e}); "
                f"only pass weights_only=False for a verified project checkpoint") from e
        raise
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint

# 数据读取
class MatReader(object):
    """功能：读取 .mat 数据字段，返回 numpy 数组或 torch 张量。

    表示：scipy.io.loadmat 成功 → 旧版 MATLAB（v5/v7，self.old_mat=True，键直接是数组）；
    仅当 scipy 抛 NotImplementedError（判定为 v7.3）时回退 h5py.File → HDF5
    （self.old_mat=False，字段是惰性 Dataset）；其它异常（文件缺失、权限、损坏）
    原样向上传播，不再被吞掉。

    关键转换：
    - HDF5 路径读取后做全轴反转（transpose(range(ndim-1, -1, -1))）：MATLAB 按列主序
      存储，h5py 按行主序读出，各轴顺序整体颠倒；scipy 路径已在内部完成转置，不做此步。

    注意：
    - to_cuda 走 .cuda() 便捷方法（legacy 保留；主训练路径不再调用）。
    """

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except NotImplementedError:
            # 仅 scipy 明确判定"v7.3 必须用 HDF 读取"时才回退 HDF5。
            # 2026-09-05 修复：旧写法为裸 except，会把文件缺失、权限、损坏、
            # 中断（BaseException）等全部误当成"尝试 HDF5"。
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        """功能：按字段名取出数据；HDF5 字段先物化再全轴反转（见类 docstring），
        统一转 float32，可选转 torch/CUDA。返回 numpy 数组或 torch 张量。"""
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# 逐特征高斯归一化
class UnitGaussianNormalizer(object):
    """逐特征高斯归一化：mean/std 沿样本维（轴 0）统计，保留其余各轴的逐特征形状。
    x 可为 (n, d)、(n, T, d) 或 (n, d, T) 等布局，统计结果均按非样本维逐点给出。"""

    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x 允许 (n, d)、(n, T, d) 或 (n, d, T) 布局：统计沿样本维（轴 0）
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # 逐特征统计直接作用
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # mean 与样本同维：按 batch 索引
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # mean 多一维时间轴：先取时间片再按 batch 索引
                mean = self.mean[:,sample_idx]

        # x 为 (batch, n) 或 (T, batch, n)
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# 全标量高斯归一化
class GaussianNormalizer(object):
    """全标量高斯归一化：mean/std 是全体元素的单一标量（区别于 UnitGaussianNormalizer
    的逐特征统计）。eps 防止零方差除零。"""

    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# 线性范围归一化
class RangeNormalizer(object):
    """线性归一化到 [low, high]：min/max 沿样本维（轴 0）统计，encode/decode 走仿射
    变换 a*x+b（a=(high-low)/(max-min)，b=-a*max+high）。"""

    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

# 相对/绝对 Lp 损失
class LpLoss(object):
    """相对/绝对 Lp 损失（FNO 标准损失）。

    - abs：网格离散化范数，h = 1/(N-1)，乘 h^(d/p) 做数值积分近似；
    - rel：逐样本 ||x-y||_p / ||y||_p；__call__ 默认走 rel（相对 L2）；
    - reduction=True 时按 size_average 返回均值/求和；False 返回逐样本向量。
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # 维数与 Lp 阶数必须为正，构造时断言
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # h = 1/(N-1)：按展平后的总网格点数取均匀网格步长，与下方 view() 的
        # 数值积分口径一致（任意维数成立，且对输入转置不敏感）。
        # 2026-09-05 修复：旧实现仅用第 1 维推 h（隐含"全部网格点都在该维"
        # 的单维假设），多维输入下同一数据转置会得到不同结果。
        # 仓库当前只经 __call__ 使用 rel()，abs() 无调用方。
        x = x.reshape(num_examples, -1)
        y = y.reshape(num_examples, -1)
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x - y, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev 范数（HS 范数）损失
# 在 FFT 频域按波数加权比较输出与目标，即同时比较数值导数
class HsLoss(object):
    """Sobolev（HS）范数损失：频域按波数加权（含数值导数项）。

    - k_x/k_y 为 FFT 排布的整数波数网格（0..n//2 与 -n//2..-1 拼接），取绝对值后作
      频率幅值；
    - balanced=False（默认）：单个加权相对范数，weight = sqrt(Σ a_k²·|k|^(2k))；
    - balanced=True：各阶导数项分别求相对范数后平均（除以 k+1）。
    """

    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        # 维数与 Lp 阶数必须为正，构造时断言
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# 简单前馈网络构造器
class DenseNet(torch.nn.Module):
    """简单 MLP 构造器：layers 给出各层宽度，相邻线性层之间可插 BatchNorm 与激活。"""

    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# 模型参数总量
def count_params(model):
    """统计模型参数总量：逐参数元素数求和（不含缓冲区）。"""
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c
