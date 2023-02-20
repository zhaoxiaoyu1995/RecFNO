# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : noaa_fno.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
import numpy as np
import h5py

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(filename)

from field_recon.model.cnn import UNetGappyPOD, UNet
from field_recon.data.dataset import HeatInterpolGappyDataset
from field_recon.utils.misc import save_model, prep_experiment
from field_recon.utils.options import parses
from field_recon.utils.visualization import plot3x1
from field_recon.utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'recon_voronoicnn_heat_4_semi'
args.epochs = 300
args.batch_size = 8
args.ckpt = 'semi_logs/ckpt'
args.tb_path = 'semi_logs/tb'
print(args)
torch.cuda.set_device(args.gpu_id)
cudnn.benchmark = True


class GappyPod():
    def __init__(self, index, batch_size=8):
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][index, :, :, :]
        self.data = (self.data - 308) / 50
        f.close()

        self.pca = PCA(n_components=20)
        self.pca.fit(self.data.reshape(len(index), -1))

        self.positions = np.array([[i, j] for i in range(200) for j in range(200)])
        self.positions_true = np.array(
            [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
             [99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132],
             [132, 165], [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]]
        )

        components = self.pca.components_
        means = self.pca.mean_

        component = components.reshape(-1, 200, 200)
        mean = means.reshape(-1, 200, 200)

        component_mask, mean_mask = [], []
        for i in range(self.positions_true.shape[0]):
            component_mask.append(
                component[:, self.positions_true[i, 0], :][:, self.positions_true[i, 1]].reshape(-1, 1))
            mean_mask.append(mean[:, self.positions_true[i, 0], :][:, self.positions_true[i, 1]].reshape(-1, 1))

        self.component_mask = torch.from_numpy(np.concatenate(component_mask, axis=-1)).float().cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        self.mean_mask = torch.from_numpy(np.concatenate(mean_mask, axis=-1).T).float().cuda()
        self.components_ = torch.from_numpy(components).float().cuda()
        self.mean_ = torch.from_numpy(means).reshape(1, -1).float().cuda()

    def gappy_pod_weight(self, inputs):
        observe = inputs
        observe = (observe.T - self.mean_mask).unsqueeze(dim=1).permute(2, 0, 1)
        component_mask_temp = self.component_mask
        coff_pre = torch.linalg.inv(component_mask_temp @ component_mask_temp.permute(0, 2, 1)) @ component_mask_temp @ observe
        coff_pre = coff_pre.squeeze(dim=-1)
        pseudo = self.inverse_transform(coff_pre)
        pseudo = pseudo.reshape(pseudo.shape[0], 1, 200, 200)
        return pseudo

    def inverse_transform(self, coff):
        return coff @ self.components_ + self.mean_


def test(index):
    # Define data loader
    test_dataset = HeatInterpolGappyDataset(index=index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    gappy_pod = GappyPod(index=[i for i in range(20)])

    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs, gappy_inputs) in enumerate(test_loader):
        N, _, _, _ = inputs.shape
        # inputs, outputs = inputs.cuda(), outputs.cuda()
        inputs, outputs, gappy_inputs = inputs.cuda(), outputs.cuda(), gappy_inputs.cuda()
        with torch.no_grad():
            pre_gappy = gappy_pod.gappy_pod_weight(gappy_inputs)
            pre = pre_gappy
        test_num += N
        test_mae += F.l1_loss(outputs, pre).item() * N
        test_rmse += torch.sum(cre(outputs, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(outputs - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')


if __name__ == '__main__':
    test(index=[i for i in range(8000, 9000)])
    # test(index=[i for i in range(4000, 5000)])
    # test(index=[i for i in range(8000, 9200)])
