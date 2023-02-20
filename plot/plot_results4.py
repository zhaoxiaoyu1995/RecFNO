# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 0:02
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import pandas as pd
import seaborn as sbs
import numpy as np
import matplotlib.pyplot as plt

sbs.set_style('white')

df = pd.read_csv('./data/darcy_loss.csv')
data = df.to_numpy().T / 100

index = ['Voronoi+FNO Val', 'Voronoi+UNet Val', 'Voronoi+UNet Train', 'Voronoi+FNO Train']
id = [3, 2, 4, 1]

markers = ['*', 'o', '>', '<', '^', 'v', 'd', 's']
for i in id:
    plt.plot(data[0, :] * 100, data[i, :], label=index[i - 1], linewidth=2.0)

# font1 = {'family': 'Times New Roman'}
# plt.legend(prop=font1, loc='upper right')
plt.legend(loc='upper right')

plt.minorticks_on()
plt.tick_params(direction='in', which='major', length=6, width=2, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='minor', length=4, left=True, right=True)
plt.grid(visible=1, linestyle="--", axis='both')

plt.yscale('log')
# plt.ylim(0.3, 6)

# 设置字体：
# fontproperties='Times New Roman', fontsize=12
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.savefig('darcy_loss.pdf', bbox_inches='tight', pad_inches=0)