# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 8:27
# @Author  : zhaoxiaoyu
# @File    : cylinder2D_superresolution.py
import torch
import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(filename)

from utils.options import parses
from utils.visualization import plot_results
from model.fno import FNORecon
from model.mlp import MLP
from data.dataset import CylinderPodDataset
from data.dataset import CylinderObserveDataset
from data.dataset import CylinderInterpolDataset
from model.cnn import UNet
from model.fno import VoronoiFNO2d

# Configure the arguments
args = parses()
args.exp = 'recon_cnn_cylinder_32'
args.epochs = 300
args.batch_size = 16
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


def super_resolution():
    # Path of trained network
    args.snapshot = '/home/ubuntu/zhaoxiaoyu/ARE/field_recon/cylinder2D/logs/ckpt/recon_voronoiunet_cylinder_2/best_epoch_299_loss_0.00023879.pth'

    # Define data loader
    # test_dataset = CylinderDataset(index=[4500])
    # test_dataset = CylinderObserveDataset(index=[4500])
    test_dataset = CylinderInterpolDataset(index=[4500])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Load trained network
    # net = CNNRecon(sensor_num=2, fc_size=(7, 12)).cuda()
    # net = VoronoiFNO2d(modes1=32, modes2=32, width=32, in_channels=4).cuda()
    net = UNet(in_channels=4, out_channels=1).cuda()
    # net = FNORecon(sensor_num=2, fc_size=(7, 12), out_size=(112 * 1, 192 * 1), modes1=32, modes2=32, width=32).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    for i, (inputs, outputs) in enumerate(test_loader):
        # N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), F.interpolate(outputs.cuda(), scale_factor=1)
        with torch.no_grad():
            pre = net(inputs)

    plot_results(
        np.array([[56, 37], [53, 40]]), pre[-1, 0, :, :].cpu().numpy())


if __name__ == "__main__":
    super_resolution()
