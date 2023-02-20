# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 0:02
# @Author  : zhaoxiaoyu
# @File    : plot_results.py
import pandas as pd
import seaborn as sbs
import numpy as np
import matplotlib.pyplot as plt

sbs.set_style('white')

df = pd.read_excel('./data/test.xlsx', engine='openpyxl', index_col=0, sheet_name='Sheet1')
data = df.to_numpy()

index = df.index
columns = df.columns
x = np.array([i + 1 for i in range(len(columns))])

markers = ['>', '<', '^', 'v', 'd', '*', 'o', 's']
for i in range(data.shape[0]):
    plt.plot(x, data[i, :], label=index[i], linestyle='-.', marker=markers[i], linewidth=2.0, markersize=8)

# font1 = {'family': 'Times New Roman'}
# plt.legend(prop=font1, loc='upper right')
plt.legend(loc='upper right')

plt.minorticks_on()
plt.tick_params(direction='in', which='major', length=6, width=2, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in', which='minor', length=4, left=True, right=True)
plt.grid(visible=1, linestyle="--", axis='both')

plt.xticks(x, columns)
plt.yscale('log')
# plt.ylim(0.3, 6)

# 设置字体：
# fontproperties='Times New Roman', fontsize=12
plt.xlabel("Number of sensors")
plt.ylabel("MAE(e-3)")
plt.show()
# plt.savefig('heat.pdf', bbox_inches='tight', pad_inches=0)