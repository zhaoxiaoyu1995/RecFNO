# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 21:41
# @Author  : zhaoxiaoyu
# @File    : mlp.py
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.GELU())
        # linear_layers.append(nn.Dropout(0.1))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.layers(x)


class PolyMLP(nn.Module):
    def __init__(self, layers=[4, 64, 128]):
        super(PolyMLP, self).__init__()
        linear_layers = []
        inject_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                nn.GELU()
            ))
            inject_layers.append(nn.Sequential(
                nn.Linear(layers[0], layers[i + 1]),
                nn.GELU()
            ))
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.ModuleList(linear_layers)
        self.inject_layers = nn.ModuleList(inject_layers)

    def forward(self, x):
        x_in = x
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x) * self.inject_layers[i](x_in)
        return self.layers[-1](x)


if __name__ == '__main__':
    net = PolyMLP([4, 128, 1280, 4800, 21504])
    print(net)
    x = torch.randn(1, 4)
    print(net(x).shape)
