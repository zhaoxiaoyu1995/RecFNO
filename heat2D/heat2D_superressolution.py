# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 8:56
# @Author  : zhaoxiaoyu
# @File    : heat2D_superressolution.py
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(filename)

from model.cnn import CNNRecon
from data.dataset import HeatDataset
from utils.options import parses
import numpy as np
from utils.visualization import plot_results
from model.fno import FNORecon
from data.dataset import HeatObserveDataset
from data.dataset import HeatInterpolDataset
from model.cnn import UNet
from model.fno import VoronoiFNO2d

# Configure the arguments
args = parses()
args.exp = 'recon_cnn_heat_4'
args.epochs = 300
args.batch_size = 8
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


def super_resolution():
    # Path of trained network
    args.snapshot = '/home/ubuntu/zhaoxiaoyu/ARE/field_recon/heat2D/logs/ckpt/recon_fno_heat_36/best_epoch_294_loss_0.00008363.pth'

    # Define data loader
    test_dataset = HeatDataset(index=[5500])
    # test_dataset = HeatObserveDataset(index=[5500])
    # test_dataset = HeatInterpolDataset(index=[5500])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Load trained network
    # net = CNNRecon(sensor_num=36, fc_size=(12, 12)).cuda()
    # net = VoronoiFNO2d(modes1=32, modes2=32, width=32, in_channels=3).cuda()
    # net = UNet(in_channels=3, out_channels=1).cuda()
    net = FNORecon(sensor_num=36, fc_size=(12, 12), out_size=(200 * 2, 200 * 2), modes1=36, modes2=36, width=32).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    for i, (inputs, outputs) in enumerate(test_loader):
        inputs, outputs = inputs.cuda(), outputs.cuda()
        inputs, outputs = inputs.cuda(), F.interpolate(outputs.cuda(), scale_factor=1)
        with torch.no_grad():
            pre = net(inputs)

    plot_results(
        np.array(
            [[28, 28], [28, 56], [28, 84], [28, 112], [28, 140], [28, 168], [56, 28], [56, 56], [56, 84], [56, 112],
             [56, 140], [56, 168], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [112, 28], [112, 56],
             [112, 84], [112, 112], [112, 140], [112, 168], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140],
             [140, 168], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168]]),
        pre[-1, 0, :, :].cpu().numpy() * 50)


if __name__ == '__main__':
    super_resolution()
