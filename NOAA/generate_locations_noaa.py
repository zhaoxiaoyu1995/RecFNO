# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_cylinder.py
import pickle
import numpy as np
import h5py
import torch

from utils.utils import generate_locations

index = [i for i in range(1500)]
data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
sst = data['sst'][:]
mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
mask = np.flip(mask, axis=0).copy()
sst[np.isnan(sst)] = 0
print(np.max(sst), np.min(sst))
data = torch.from_numpy(sst.reshape(sst.shape[0], 1, 360, 180)[index, :, :, :]).float().permute(0, 1, 3, 2)
data = torch.flip(data, dims=[2]).squeeze(dim=1)

locations = generate_locations(data.numpy(), observe_num=8, interval=20)
print(locations)
