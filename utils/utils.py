# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 17:41
# @Author  : zhaoxiaoyu
# @File    : utils.py
import torch
import numpy as np


def cre(truth, pre, ord):
    """
    Calculated relative error
    :param truth: (N,)
    :param pre: (N,)
    :param ord: 1, 2
    :return: (N,)
    """
    return torch.linalg.norm((truth - pre).flatten(1), ord, dim=1) / torch.linalg.norm(truth.flatten(1), ord, dim=1)


def generate_locations(data, observe_num=2, interval=2):
    """
    根据数据每个位置方差，生成测点位置。
    :param data: 物理场数据，(N, h, w)
    :param observe_num: 测点数量
    :param interval: 测点之间上下左右最小间隔
    :return: 测点位置，包含observe_num个测点位置的list
    """
    w, h = data.shape[2], data.shape[1]

    # 按照方差大小排序
    data = np.std(data, axis=0)
    argsort_index = np.flipud(np.argsort(data.flatten()))

    raw, col = np.linspace(0, h - 1, h), np.linspace(0, w - 1, w)
    col, raw = np.meshgrid(col, raw)
    col, raw = col.astype(np.int).flatten(), raw.astype(np.int).flatten()

    locations = []
    locations.append([raw[argsort_index[0]], col[argsort_index[0]]])
    for i in range(1, len(argsort_index)):
        if len(locations) < observe_num:
            cur_raw, cur_col = raw[argsort_index[i]], col[argsort_index[i]]
            flag = -1
            for [for_raw, for_col] in locations:
                if abs(for_raw - cur_raw) <= interval and abs(for_col - cur_col) <= interval:
                    flag = 1
            if flag == -1:
                locations.append([raw[argsort_index[i]], col[argsort_index[i]]])
        else:
            break
    return locations


def generate_locations_random(data, observe_num=2, interval=2):
    """
    根据数据每个位置方差，生成测点位置。
    :param data: 物理场数据，(N, h, w)
    :param observe_num: 测点数量
    :param interval: 测点之间上下左右最小间隔
    :return: 测点位置，包含observe_num个测点位置的list
    """
    w, h = data.shape[2], data.shape[1]

    # 按照方差大小排序
    data = np.std(data, axis=0)
    argsort_index = np.flipud(np.argsort(data.flatten()))

    raw, col = np.linspace(0, h - 1, h), np.linspace(0, w - 1, w)
    col, raw = np.meshgrid(col, raw)
    col, raw = col.astype(np.int).flatten(), raw.astype(np.int).flatten()

    locations = []
    locations.append([raw[argsort_index[0]], col[argsort_index[0]]])
    for i in range(1, len(argsort_index)):
        if len(locations) < observe_num:
            cur_raw, cur_col = raw[argsort_index[i]], col[argsort_index[i]]
            flag = -1
            for [for_raw, for_col] in locations:
                if abs(for_raw - cur_raw) <= interval or abs(for_col - cur_col) <= interval:
                    flag = 1
            if flag == -1:
                locations.append([raw[argsort_index[i]], col[argsort_index[i]]])
        else:
            break
    return locations
