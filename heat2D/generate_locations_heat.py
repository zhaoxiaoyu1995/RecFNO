# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_heat.py
import h5py

from utils.utils import generate_locations

f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
data = f['u'][:, 0, :, :]
f.close()
locations = generate_locations(data, observe_num=5000, interval=2)

print(locations)
