# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:08
# @Author  : zhaoxiaoyu
# @File    : cylinder2D_pod_snr.py
import torch
import torch.nn.functional as F
import logging
import os
import sys
from torch.utils.data import DataLoader

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(filename)

from model.mlp import MLP
from data.dataset import CylinderPodSNRDataset, CylinderPodDataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'recon_pod_cylinder_8_snr80'
args.epochs = 300
args.batch_size = 16
print(args)
torch.cuda.set_device(args.gpu_id)


def train():
    # Prepare the experiment environment
    tb_writer = prep_experiment(args)
    # Create figure dir
    args.fig_path = args.exp_path + '/figure'
    os.makedirs(args.fig_path, exist_ok=True)
    args.best_record = {'epoch': -1, 'loss': 1e10}

    # Build neural network
    net = MLP(layers=[8, 64, 64, 64, 25]).cuda()

    # Build data loader
    train_dataset = CylinderPodSNRDataset(pod_index=[i for i in range(3500)], index=[i for i in range(3500)],
                                          n_components=25, snr=True, SNRdB=80)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = CylinderPodSNRDataset(pod_index=[i for i in range(3500)], index=[i for i in range(3500, 4250)],
                                        n_components=25, snr=True, SNRdB=80)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs, _) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            loss = F.l1_loss(outputs.flatten(1), pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record results by tensorboard
            tb_writer.add_scalar('train_loss', loss, i + epoch * len(train_loader))
            train_loss += loss.item() * inputs.shape[0]
            train_num += inputs.shape[0]

        train_loss = train_loss / train_num
        logging.info("Epoch: {}, Avg_loss: {}".format(epoch, train_loss))
        scheduler.step()

        # Validation procedure
        if epoch % args.val_interval == 0:
            net.eval()
            val_loss, val_num, val_mae = 0., 0., 0.
            for i, (inputs, outputs, labels) in enumerate(val_loader):
                inputs, outputs, labels = inputs.cuda(), outputs.cuda(), labels.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                loss = F.l1_loss(outputs, pre)
                pre_maps = val_dataset.inverse_transform(pre)
                mae = F.l1_loss(labels, pre_maps)

                val_loss += loss.item() * inputs.shape[0]
                val_mae += mae.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Record results by tensorboard
            val_loss = val_loss / val_num
            val_mae = val_mae / val_num
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            tb_writer.add_scalar('val_mae', val_mae, epoch)
            logging.info("Epoch: {}, Val_loss: {}, Val_mae: {}".format(epoch, val_loss, val_mae))
            if val_mae < args.best_record['loss']:
                save_model(args, epoch, val_mae, net)
            net.train()

            # Plotting
            if epoch % args.plot_freq == 0:
                plot3x1(labels[-1, 0, :, :].cpu().numpy(), pre_maps[-1, 0, :, :].cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


def test(index):
    # Path of trained network
    args.snapshot = '/home/ubuntu/zhaoxiaoyu/ARE/field_recon/cylinder2D/logs/ckpt/recon_pod_cylinder_8/best_epoch_297_loss_0.00026303.pth'

    # Define data loader
    test_dataset = CylinderPodDataset(pod_index=[i for i in range(3500)], index=index,
                                      n_components=25)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[8, 64, 64, 64, 25]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.eval()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    test_max_ae = 0
    for i, (inputs, outputs, labels) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs, labels = inputs.cuda(), outputs.cuda(), labels.cuda()
        with torch.no_grad():
            pre = net(inputs)
        pre = test_dataset.inverse_transform(pre)
        test_num += N
        test_mae += F.l1_loss(labels, pre).item() * N
        test_rmse += torch.sum(cre(labels, pre, 2))
        test_max_ae += torch.sum(torch.max(torch.abs(labels - pre).flatten(1), dim=1)[0]).item()
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)
    print('test max_ae:', test_max_ae / test_num)

    plot3x1(labels[-1, 0, :, :].cpu().numpy(), pre[-1, 0, :, :].cpu().numpy(), './test.png')


if __name__ == '__main__':
    train()
    # test(index=[i for i in range(0, 3500)])
    # test(index=[i for i in range(3500, 4250)])
    # test(index=[i for i in range(4250, 5000)])
