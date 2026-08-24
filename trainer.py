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
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
# from torchinfo import summary
from einops import rearrange
from utilities3 import *
from timeit import default_timer

from diffusion import ElucidatedDiffusion
from IAFNO import IAFNODiff

torch.manual_seed(123)
import pickle

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def normalize_prep(data,dim,need_max_min=True):

    if dim == 5:
        min = torch.amin(data, axis=(0,1,2,3)).reshape(1,1,1,1,-1)
        max = torch.amax(data, axis=(0,1,2,3)).reshape(1,1,1,1,-1)
    
    if dim == 6:
        min = torch.amin(data, axis=(0,1,2,3,5)).reshape(1,1,1,1,-1,1)
        max = torch.amax(data, axis=(0,1,2,3,5)).reshape(1,1,1,1,-1,1)

    if need_max_min:
        return max, min

########## HYPERPARAMETERS ##########

batch_size = 4

trainset_num = 20

InferenceWidth = 1 ##### iw

InitialInterval = 1 ##### ii

num_epochs = 150

embed_dim = 180

implicit_layer = 4

explicit_layer = 4

sampling_steps = 32

hidden_size_factor = 4

# Set to a .pth file to resume; None preserves training from scratch.
checkpoint_path = None

########## DATA Loader ##########

data = np.load('your dataset')
data = data[0:trainset_num,...,0:3]
data = torch.from_numpy(data) ##### bs nt x y z c

data_list = []
count = data.shape[1]
count = 200                    ###################################### scalable for fast testing ##############################################

print('Datasets start preparing.',data.shape)
for j in range(data.shape[0]):
    for i in range(count-InferenceWidth):
        data_list.append(data[j,i:i+InferenceWidth+1,...])        ##### bs nt x y z c,    interval: t->t+iw+1

data_set = torch.stack(data_list)

full_set = torch.utils.data.TensorDataset(data_set[:,0,...], data_set[:,1,...])
train_dataset, test_dataset = torch.utils.data.random_split(full_set,[int(0.8*len(full_set)),len(full_set)-int(0.8*len(full_set))])

######################## normalize prep && calc sigma ###########################

info_folder_path = "max_min_sigma info of your dataset"
target_file = f"ts{trainset_num}_c{count}_iw{InferenceWidth}_ii{InitialInterval}.npy"
file_path = os.path.join(info_folder_path, target_file)

if os.path.exists(file_path):
    info = np.load(file_path)
    print(f"{file_path} is loaded with a shape of {info.shape}")
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
    sigma = torch.std(y_train).item()

    y_max = y_max.to(device)
    y_min = y_min.to(device)

    info = torch.cat([y_max,y_min],dim=0)
    sigma_tensor = torch.ones((1,1,1,1,InferenceWidth*3))*sigma
    sigma_tensor = sigma_tensor.to(device)
    info = torch.cat([info, sigma_tensor],dim=0)

    numpy_info = info.cpu().numpy()
    numpy_info=np.float32(numpy_info)
    np.save(file_path,numpy_info)

###################################################################

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

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

#######################  channel flow  ##############################

model = ElucidatedDiffusion(dm_backbone,
                                channels = InferenceWidth*3,
                                num_sample_steps = sampling_steps,
                                image_size_h = 64,
                                image_size_w = 65,
                                image_size_z = 32,
                                sigma_data = sigma)

##############################################################

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= num_epochs * len(train_loader) )  # Adjust T_max as needed

if checkpoint_path is not None:
    load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, map_location=device)

mse_train = []
mse_test = []
mse_real = []
timecost = []

print('Model start training.')
print('Model Total Params:', count_params(model))
print('With hyperparameters: batchsize:', batch_size, '  implicit_layer:', implicit_layer, '  explicit_layer:', explicit_layer, '  inference_width:', InferenceWidth)
print('embed_dim:', embed_dim, '  hidden_size_factor', hidden_size_factor, '  sampling_steps:', sampling_steps, '  trainset_num:', trainset_num, '  count:', count)
myloss = LpLoss()
for ep in range(num_epochs):
    model.train()
    train_loss = 0.0
    t1 = default_timer()
    for i, (xx, yy) in enumerate(train_loader):
        
        xx = xx.to(device)
        yy = yy.to(device)

        xx = (xx - y_min) / (y_max - y_min)
        yy = (yy - y_min) / (y_max - y_min)

        xx = rearrange(xx, "bs x y z c -> bs c x y z")
        yy = rearrange(yy, "bs x y z c -> bs c x y z")

        optimizer.zero_grad()
        with autocast():
            loss = model(yy.to(device), xx.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    mse_train.append(train_loss)
        
    # Testing loop
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

    parent_dir = "your directory for saving files"
    folder_name = f"BS{batch_size}_EMD{embed_dim}_I{implicit_layer}_E{explicit_layer}_HSF{hidden_size_factor}_S{sampling_steps}_IW{InferenceWidth}_TS{trainset_num}_C{count}"
    folder_path = os.path.join(parent_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    ccount = ep + 1
    pth_name = f"test_Ep{ccount}.pth"
    pth_path = os.path.join(folder_path, pth_name)
    torch.save(model.state_dict(), pth_path)
    
    timecost.append(t2-t1)
    MSE_save=np.dstack((timecost,mse_train,mse_test,mse_real)).squeeze()
    np.savetxt(f'loss.dat',MSE_save,fmt="%16.7f")
