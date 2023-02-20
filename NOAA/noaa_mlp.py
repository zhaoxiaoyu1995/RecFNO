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

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(filename)

from model.mlp import MLP
from data.dataset import NOAADataset
from utils.misc import save_model, prep_experiment
from utils.options import parses
from utils.visualization import plot3x1
from utils.utils import cre

# Configure the arguments
args = parses()
args.exp = 'recon_mlp_noaa_48'
args.epochs = 300
args.batch_size = 8
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
    net = MLP(layers=[48, 128, 1280, 5400, 180 * 360]).cuda()

    # Build data loader
    train_dataset = NOAADataset(index=[i for i in range(1600)])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataset = NOAADataset(index=[i for i in range(1600, 1750)])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # Build optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(args.epochs):
        # Training procedure
        train_loss, train_num = 0., 0.
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pre = net(inputs)
            pre[:, train_dataset.mask.flatten()] = 0
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
            val_loss, val_num = 0., 0.
            for i, (inputs, outputs) in enumerate(val_loader):
                inputs, outputs = inputs.cuda(), outputs.cuda()
                with torch.no_grad():
                    pre = net(inputs)
                    pre[:, train_dataset.mask.flatten()] = 0
                loss = F.l1_loss(outputs.flatten(1), pre)

                val_loss += loss.item() * inputs.shape[0]
                val_num += inputs.shape[0]

            # Record results by tensorboard
            val_loss = val_loss / val_num
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            logging.info("Epoch: {}, Val_loss: {}".format(epoch, val_loss))
            if val_loss < args.best_record['loss']:
                save_model(args, epoch, val_loss, net)
            net.train()

            # Plotting
            if epoch % args.plot_freq == 0:
                plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(180, 360).cpu().numpy(),
                        file_name=args.fig_path + f'/epoch{epoch}.png')


def test():
    # Path of trained network
    args.snapshot = '/home/ubuntu/zhaoxiaoyu/ARE/field_recon/NOAA/logs/ckpt/recon_mlp_noaa/best_epoch_152_loss_0.34152752.pth'

    # Define data loader
    test_dataset = NOAADataset(index=[i for i in range(1040, 1902)])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Load trained network
    net = MLP(layers=[64, 128, 1280, 5400, 180 * 360]).cuda()
    net.load_state_dict(torch.load(args.snapshot)['state_dict'])
    print('load models: ' + args.snapshot)

    # Test procedure
    net.train()
    test_mae, test_rmse, test_num = 0.0, 0.0, 0.0
    for i, (inputs, outputs) in enumerate(test_loader):
        N, _ = inputs.shape
        inputs, outputs = inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            pre = net(inputs)
            pre[:, test_dataset.mask.flatten()] = 0
        test_num += N
        test_mae += F.l1_loss(outputs.flatten(1), pre).item() * N
        test_rmse += torch.sum(cre(outputs.flatten(1), pre, 2))
    print('test mae:', test_mae / test_num)
    print('test rmse:', test_rmse / test_num)

    plot3x1(outputs[-1, 0, :, :].cpu().numpy(), pre[-1, :].reshape(180, 360).cpu().numpy(), './test.png')


if __name__ == '__main__':
    train()
    # test()
