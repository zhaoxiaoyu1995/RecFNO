# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 22:16
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import h5py
import numpy as np

from utils.visualization import plot_locations

data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
sst = data['sst'][:]
mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
mask = np.flip(mask, axis=0).copy()
sst[np.isnan(sst)] = 0
data = sst.reshape(sst.shape[0], 1, 360, 180)[0, 0, :, :].transpose(1, 0)
data = np.flip(data, axis=[0])

positions = np.array(
    [[43, 49], [125, 302], [64, 119], [22, 196], [101, 278], [146, 144], [167, 174], [0, 228]]
)
plot_locations(positions, data)
