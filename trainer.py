"""模块职责：原始 DiAFNO 湍流扩散模型的 legacy 训练入口，是一份必须先替换占位符才能运行的
模板脚本；不是当前 PRE 海流任务的正式入口（正式入口为 pre_trainer.py）。

不负责：PRE 数据管线、双变量掩膜协议、DDP 分布式、断点续训状态恢复；这些由 pre_*.py 承担。

关键约束：三处占位符运行前必须替换，否则加载失败或静默写错位置——
- np.load('your dataset')：数据路径，布局 (bs, nt, x, y, z, c)，通道为 3 个速度分量；
- info_folder_path：归一化统计缓存目录；
- parent_dir：checkpoint 输出目录。
checkpoint 每 epoch 存 test_Ep{n}.pth，只含 model.state_dict()，无优化器/调度器状态；
loss.dat 写在当前工作目录。AMP 使用旧版 torch.cuda.amp API，且不检测 GradScaler 跳步。

依赖关系：IAFNO.IAFNODiff、diffusion.ElucidatedDiffusion、utilities3（load_checkpoint、
count_params、LpLoss）。数据规模由 trainset_num×count 决定；count 默认被覆盖为 200，
仅为快速测试。
"""

import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
# from YourDataset import YourDataset  # 模板指引：如需自定义数据集，在此导入
from tqdm import tqdm
# 旧版 AMP API：legacy 路径专用；PRE 路径（pre_trainer.py）使用 torch.amp 新 API。
from torch.cuda.amp import autocast, GradScaler
# from torchinfo import summary  # 可选调试：取消注释以打印模型结构摘要
from einops import rearrange
from utilities3 import *
from timeit import default_timer

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff

# 全局种子固定在模块导入期（项目约定），保证下方 80/20 随机划分可复现。
torch.manual_seed(123)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def normalize_prep(data,dim,need_max_min=True):
    """功能：按通道计算 min/max，供逐通道 min-max 归一化使用。

    参数：
    - data：待统计张量；dim=5 时为 (bs, nt, x, y, z, c)。
    - dim：data 维数。5：沿 batch/时间/空间轴 (0,1,2,3) 逐通道统计，结果
      reshape 为 (1,1,1,1,c)；6：跳过轴 4（层维），沿其余轴统计，结果为
      (1,1,1,1,z,c)，用于带独立层维的数据。
    - need_max_min：恒为 True；为 False 时不返回任何值（隐式 None）。

    返回：
    - (max, min)，形状可直接与原张量广播相减/相除。

    前置条件：
    - dim 必须为 5 或 6，否则 min/max 未定义（NameError）。
    """

    if dim == 5:
        # axis= 关键字在 torch 2.4.1（本仓锁定版本，含 +cpu/+cu124）实测可被
        # amin/amax 接受，与 dim= 等价；跨版本若报 TypeError，改回 dim= 即可。
        min = torch.amin(data, axis=(0,1,2,3)).reshape(1,1,1,1,-1)
        max = torch.amax(data, axis=(0,1,2,3)).reshape(1,1,1,1,-1)
    
    if dim == 6:
        min = torch.amin(data, axis=(0,1,2,3,5)).reshape(1,1,1,1,-1,1)
        max = torch.amax(data, axis=(0,1,2,3,5)).reshape(1,1,1,1,-1,1)

    if need_max_min:
        return max, min

# 超参数

batch_size = 4

trainset_num = 20

InferenceWidth = 1  # 前瞻帧数 iw：窗口取第 t..t+iw 帧共 iw+1 帧；参与缓存文件名与模型通道数（in_chans=iw*3）

InitialInterval = 1  # 仅作为归一化缓存文件名的 ii 字段，未参与下方窗口构造

num_epochs = 150

embed_dim = 180

implicit_layer = 4

explicit_layer = 4

sampling_steps = 32

hidden_size_factor = 4

# 设为 .pth 路径则从该 checkpoint 恢复权重；保持 None 从头训练。
checkpoint_path = None

# 数据加载

# 占位符：运行前必须替换为真实数据路径。
data = np.load('your dataset')
data = data[0:trainset_num,...,0:3]
data = torch.from_numpy(data)  # 布局 (bs, nt, x, y, z, c)

data_list = []
# 固定 200 窗（模板"快速测试"上限；2026-09-05 清理：原模板先写
# count = data.shape[1] 再立即被本行覆盖、从未被读取，属死赋值，已删除）。
# 如需全量数据，把 count 改回 data.shape[1]。
count = 200

print('Datasets start preparing.',data.shape)
for j in range(data.shape[0]):
    for i in range(count-InferenceWidth):
        data_list.append(data[j,i:i+InferenceWidth+1,...])  # 第 i 窗覆盖第 t..t+iw 帧（iw+1 帧）

data_set = torch.stack(data_list)

# TensorDataset 元组位：输入=窗口首帧 t，目标=次帧 t+1
full_set = torch.utils.data.TensorDataset(data_set[:,0,...], data_set[:,1,...])
# 80/20 随机划分训练/测试集（受全局种子 123 影响，可复现）
train_dataset, test_dataset = torch.utils.data.random_split(full_set,[int(0.8*len(full_set)),len(full_set)-int(0.8*len(full_set))])

# 归一化准备与 sigma 计算

# 占位符：归一化统计缓存目录，运行前必须替换。
info_folder_path = "max_min_sigma info of your dataset"
# 缓存文件名以超参数编码（ts/c/iw/ii），防止不同数据规模与配置互相污染；删除缓存即触发重算。
target_file = f"ts{trainset_num}_c{count}_iw{InferenceWidth}_ii{InitialInterval}.npy"
file_path = os.path.join(info_folder_path, target_file)

if os.path.exists(file_path):
    info = np.load(file_path)
    print(f"{file_path} is loaded with a shape of {info.shape}")
    # 缓存布局 (3, ...)：[0]=y_max、[1]=y_min（逐通道 min/max）、[2]=sigma 张量
    #（各通道同值，info[2,...,0] 取回首位置还原标量）。
    y_max = torch.from_numpy(info[0,...]).unsqueeze(0).to(device)
    y_min = torch.from_numpy(info[1,...]).unsqueeze(0).to(device)
    sigma = torch.from_numpy(info[2,...,0]).item()
else:
    print('Beginning of normalization & calculating sigma.')
    train_input = []
    for set in train_dataset:
        input, output = set
        train_input.append(input)

    y_train = torch.stack(train_input)

    y_max, y_min = normalize_prep(y_train,5)
    y_train = (y_train - y_min) / (y_max - y_min)
    # sigma = 归一化到 [0,1] 后训练输入的全局标准差（标量），供 EDM 的 sigma_data 使用
    sigma = torch.std(y_train).item()

    y_max = y_max.to(device)
    y_min = y_min.to(device)

    info = torch.cat([y_max,y_min],dim=0)
    sigma_tensor = torch.ones((1,1,1,1,InferenceWidth*3))*sigma
    sigma_tensor = sigma_tensor.to(device)
    info = torch.cat([info, sigma_tensor],dim=0)

    # 缓存以 float32 numpy 落盘，布局 [y_max; y_min; sigma]
    numpy_info = info.cpu().numpy()
    numpy_info=np.float32(numpy_info)
    np.save(file_path,numpy_info)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# 骨干网格 (64, 66, 32)：y 轴由真实 65 零填充到 66，保证 patch_size=2 整除；
# dim_f=(64, 65, 32) 记录真实网格，还原时裁掉填充点
dm_backbone = IAFNODiff(
    dim = (64, 66, 32),
    patch_size = (2, 2, 2),
    embed_dim = embed_dim,
    num_blocks = 1,
    in_chans = InferenceWidth*3,
    out_chans = InferenceWidth*3,
    ex_layer = explicit_layer,
    nlayer = implicit_layer,
    hidden_size_factor = hidden_size_factor,
    dim_f = (64, 65, 32),
    self_condition = True
).to(device).to(torch.float32)

# 通道流：EDM 目标通道数 = iw*3；legacy 路径不传 cond_chans，走 IAFNODiff 的
# 默认倍增（cond_chans=None → 条件通道 = in_chans）

model = ElucidatedDiffusion(dm_backbone,
                                channels = InferenceWidth*3,
                                num_sample_steps = sampling_steps,
                                image_size_h = 64,
                                image_size_w = 65,
                                image_size_z = 32,
                                sigma_data = sigma)

# 优化器与调度器
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

# 余弦退火按优化步数退火：T_max = 总 epoch 数 × 每 epoch batch 数
scheduler = CosineAnnealingLR(optimizer, T_max= num_epochs * len(train_loader) )

# legacy 恢复：checkpoint 只含权重（裸 state_dict 经 load_checkpoint 的
# model_state_dict 缺省回退分支读取），优化器/调度器/scaler 状态与已完成
# epoch 一并丢失，等于带旧权重从头训练。
if checkpoint_path is not None:
    load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, map_location=device)

# 每 epoch 累计：train_loss=EDM 加权去噪 MSE（[-1,1] 图像域）、test_loss=相对 L2
#（[0,1] 归一化域）、real_loss=相对 L2（反归一化物理域）、timecost=epoch 耗时
mse_train = []
mse_test = []
mse_real = []
timecost = []

print('Model start training.')
print('Model Total Params:', count_params(model))
print('With hyperparameters: batchsize:', batch_size, '  implicit_layer:', implicit_layer, '  explicit_layer:', explicit_layer, '  inference_width:', InferenceWidth)
print('embed_dim:', embed_dim, '  hidden_size_factor', hidden_size_factor, '  sampling_steps:', sampling_steps, '  trainset_num:', trainset_num, '  count:', count)
# 相对 L2 损失（LpLoss 默认走 rel 分支）
myloss = LpLoss()
for ep in range(num_epochs):
    model.train()
    train_loss = 0.0
    t1 = default_timer()
    for i, (xx, yy) in enumerate(train_loader):
        
        xx = xx.to(device)
        yy = yy.to(device)

        # 逐通道 min-max 归一化到 [0,1]，再转换为模型布局 (bs, c, x, y, z)
        xx = (xx - y_min) / (y_max - y_min)
        yy = (yy - y_min) / (y_max - y_min)

        xx = rearrange(xx, "bs x y z c -> bs c x y z")
        yy = rearrange(yy, "bs x y z c -> bs c x y z")

        optimizer.zero_grad()
        # EDM 训练损失：images=目标 yy，condition=输入 xx
        with autocast():
            loss = model(yy.to(device), xx.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    mse_train.append(train_loss)
        
    # 测试循环：Heun 采样一帧并与真值比较
    model.eval()
    test_loss = 0.0
    real_loss = 0.0
    with torch.no_grad():
        for j, (xx, yy) in enumerate(test_loader):

            xx = xx.to(device)
            yy = yy.to(device)

            xx = (xx - y_min) / (y_max - y_min)
            yy = (yy - y_min) / (y_max - y_min)

            xx = rearrange(xx, "bs x y z c -> bs c x y z")

            # 采样与损失均在 autocast 内：test_loss 用 [0,1] 归一化域相对 L2，
            # real_loss 用反归一化物理域相对 L2
            with autocast():
                
                pred = model.sample(xx.to(device))
                pred = rearrange(pred, "bs c x y z -> bs x y z c", bs = batch_size)
                loss = myloss(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))

                rpred = pred * (y_max - y_min) + y_min
                ryy = yy * (y_max - y_min) + y_min

                real_loss += myloss(rpred.reshape(rpred.shape[0], -1), ryy.reshape(ryy.shape[0], -1)).item()
            test_loss += loss.item()
    real_loss /= len(test_loader)
    test_loss /= len(test_loader)
    mse_test.append(test_loss)
    mse_real.append(real_loss)

    t2 = default_timer()
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss), 
          'test_loss: {:.4f}'.format(test_loss))
    print('  real loss: ',real_loss)

    # 占位符：checkpoint 输出目录，运行前必须替换。
    parent_dir = "your directory for saving files"
    # 输出目录名以超参数编码，不同配置的运行互不混放
    folder_name = f"BS{batch_size}_EMD{embed_dim}_I{implicit_layer}_E{explicit_layer}_HSF{hidden_size_factor}_S{sampling_steps}_IW{InferenceWidth}_TS{trainset_num}_C{count}"
    folder_path = os.path.join(parent_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 每 epoch 保存一个 test_Ep{n}.pth：仅 model.state_dict()，无优化器/调度器/scaler 状态
    ccount = ep + 1
    pth_name = f"test_Ep{ccount}.pth"
    pth_path = os.path.join(folder_path, pth_name)
    torch.save(model.state_dict(), pth_path)
    
    timecost.append(t2-t1)
    # loss.dat 写在当前工作目录：每行一个 epoch 的 (耗时, train_loss, test_loss, real_loss)
    MSE_save=np.dstack((timecost,mse_train,mse_test,mse_real)).squeeze()
    np.savetxt(f'loss.dat',MSE_save,fmt="%16.7f")
