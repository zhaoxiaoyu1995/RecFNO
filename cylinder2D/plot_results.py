# -*- coding: utf-8 -*-
# @Time    : 2022/5/17 22:16
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import pickle
import numpy as np

from utils.visualization import plot_locations

df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
data = pickle.load(df)[250, :, :, 0]
df.close()

positions = np.array(
    [[56, 37], [56, 42], [56, 32], [55, 47], [56, 52], [60, 47], [51, 42], [56, 19], [55, 57], [51, 52], [50, 47],
     [61, 52], [51, 37], [55, 62], [60, 57], [61, 42], [50, 57], [55, 67], [50, 62], [60, 62], [55, 72], [50, 67],
     [56, 27], [60, 67], [51, 32], [55, 77], [50, 72], [60, 72], [55, 82], [50, 77], [60, 77], [61, 37]]
)
plot_locations(positions, data)
