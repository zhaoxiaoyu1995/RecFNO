# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 22:16
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import h5py
import numpy as np

from utils.visualization import plot_locations

f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
data = f['u'][0, 0, :, :]
f.close()

positions = np.array([[199, 90], [31, 178], [0, 97], [0, 16]])
print(positions.tolist())
plot_locations(positions, data)
