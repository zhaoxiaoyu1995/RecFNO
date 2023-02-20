# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 10:57
# @Author  : zhaoxiaoyu
# @File    : generate_locations_cylinder.py
import pickle
import numpy as np

from utils.utils import generate_locations

df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
data = np.squeeze(pickle.load(df), axis=-1)[[i for i in range(4250)], :, :]
locations = generate_locations(data, observe_num=100, interval=2)

print(locations)
