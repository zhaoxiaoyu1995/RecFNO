# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 15:19
# @Author  : zhaoxiaoyu
# @File    : dataset.py
import pickle
import h5py
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import scipy.io as sio
import os


def awgn(s, SNRdB, L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal
    's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power
    spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if np.isrealobj(s):  # check if input is real/complex object type
        n = np.sqrt(N0 / 2) * np.random.standard_normal(s.shape)  # computed noise
    else:
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(s.shape) + 1j * np.random.standard_normal(s.shape))
    r = s + n  # received signal
    return r


class CylinderSNRDataset(Dataset):
    def __init__(self, index, mean=0, std=1, snr=False, SNRdB=20, test=False):
        """
        The cylinder dataset with noise
        :param index:
        """
        super(CylinderSNRDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        original_data = pickle.load(df)
        # Add AWGN noise
        if snr:
            np.random.seed(123)
            data_snr = np.zeros_like(original_data)
            for i in range(original_data.shape[0]):
                data_snr[i, :, :, :] = awgn(original_data[i, :, :, :].flatten(), SNRdB).reshape(original_data.shape[1:])
            self.data = data_snr
        else:
            self.data = original_data
        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47], [62, 50], [47, 60], [65, 57], [44, 65]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        if test:
            self.data = torch.from_numpy(original_data).float().permute(0, 3, 1, 2)[index, :, :, :]
            self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolSNRDataset(Dataset):
    def __init__(self, index, mean=0, std=1, snr=False, SNRdB=20, test=False):
        """
        圆柱绕流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(CylinderInterpolSNRDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        original_data = pickle.load(df)
        # Add AWGN noise
        if snr:
            np.random.seed(123)
            data_snr = np.zeros_like(original_data)
            for i in range(original_data.shape[0]):
                data_snr[i, :, :, :] = awgn(original_data[i, :, :, :].flatten(), SNRdB).reshape(original_data.shape[1:])
            self.data = data_snr
        else:
            self.data = original_data
        self.data = self.data[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47], [62, 50], [47, 60], [65, 57], [44, 65]])

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()
        if test:
            self.data = torch.from_numpy(original_data).float().permute(0, 3, 1, 2)[index, :, :, :]
            self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderObserveSNRDataset(Dataset):
    def __init__(self, index, mean=0, std=1, snr=False, SNRdB=20, test=False):
        """
        圆柱绕流数据集：输入采用掩码表示
        :param index:
        """
        super(CylinderObserveSNRDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        original_data = pickle.load(df)
        # Add AWGN noise
        if snr:
            np.random.seed(123)
            data_snr = np.zeros_like(original_data)
            for i in range(original_data.shape[0]):
                data_snr[i, :, :, :] = awgn(original_data[i, :, :, :].flatten(), SNRdB).reshape(original_data.shape[1:])
            self.data = data_snr
        else:
            self.data = original_data
        self.data = self.data[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47], [62, 50], [47, 60], [65, 57], [44, 65]])

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w) / w, np.linspace(h - 1, 0, h) / h
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0), torch.from_numpy(y_coor).unsqueeze(dim=0)],
                              dim=0).float()
        if test:
            self.data = torch.from_numpy(original_data).float().permute(0, 3, 1, 2)[index, :, :, :]
            self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]
        # return self.observe[index, :, :, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderPodSNRDataset(Dataset):
    def __init__(self, pod_index, index, n_components=20, mean=0, std=1, snr=False, SNRdB=20, test=False):
        """
        圆柱绕流数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(CylinderPodSNRDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        original_data = pickle.load(df)
        # Add AWGN noise
        if snr:
            np.random.seed(123)
            data_snr = np.zeros_like(original_data)
            for i in range(original_data.shape[0]):
                data_snr[i, :, :, :] = awgn(original_data[i, :, :, :].flatten(), SNRdB).reshape(original_data.shape[1:])
            self.data = data_snr
        else:
            self.data = original_data
        df.close()
        self.pca_data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)[pod_index, :, :, :]
        self.pca_data = (self.pca_data - mean) / std
        # if test:
        #     self.pca_data = torch.from_numpy(original_data).float().permute(0, 3, 1, 2)[pod_index, :, :, :]
        #     self.pca_data = (self.pca_data - mean) / std
        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47], [62, 50], [47, 60], [65, 57], [44, 65]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]
        if test:
            self.data = torch.from_numpy(original_data).float().permute(0, 3, 1, 2)[index, :, :, :]
            self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class CylinderDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集
        :param index:
        """
        super(CylinderDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(CylinderInterpolDataset, self).__init__()

        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = pickle.load(df)[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array([[56, 37], [53, 40], [59, 43], [50, 47], [62, 50], [47, 60], [65, 57], [44, 65]])

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderInterpolGappyDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(CylinderInterpolGappyDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = pickle.load(df)[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array([[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60],
                              [50, 65], [62, 70], [50, 75], [62, 80], [50, 85], [62, 90], [50, 95], [62, 100],
                              [50, 105], [62, 110], [50, 115], [62, 120], [50, 125], [62, 130], [50, 135], [62, 140],
                              [50, 145], [62, 150], [50, 155], [62, 160], [50, 165], [62, 170], [50, 175], [62, 180],
                              [72, 25], [40, 30], [72, 35], [40, 40], [72, 45], [40, 50], [72, 55], [40, 60],
                              [72, 65], [40, 70], [72, 75], [40, 80], [72, 85], [40, 90], [72, 95], [40, 100],
                              [72, 105], [40, 110], [72, 115], [40, 120], [72, 125], [40, 130], [72, 135], [40, 140],
                              [72, 145], [40, 150], [72, 155], [40, 160], [72, 165], [40, 170], [72, 175], [40, 180]])

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()
        self.gappy = torch.from_numpy(sparse_data).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :], self.gappy[
                                                                                                            index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderObserveDataset(Dataset):
    def __init__(self, index, mean=0, std=1):
        """
        圆柱绕流数据集：输入采用掩码表示
        :param index:
        """
        super(CylinderObserveDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = pickle.load(df)[index, :, :, :].transpose(0, 3, 1, 2)
        self.data = (self.data - mean) / std
        df.close()
        positions = np.array([[56, 37], [53, 40]])

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w) / w, np.linspace(h - 1, 0, h) / h
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0), torch.from_numpy(y_coor).unsqueeze(dim=0)],
                              dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]
        # return self.observe[index, :, :, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class CylinderPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=20, mean=0, std=1):
        """
        圆柱绕流数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(CylinderPodDataset, self).__init__()
        self.mean, self.std = mean, std
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.pca_data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[pod_index, :, :, :]
        self.pca_data = (self.pca_data - mean) / std
        df.close()
        df = open('/home/ubuntu/zhaoxiaoyu/data/cylinder/Cy_Taira.pickle', 'rb')
        self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[index, :, :, :]
        self.data = (self.data - mean) / std
        df.close()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array([[56, 37], [53, 40]])

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class DarcyDataset(Dataset):
    def __init__(self, index, mean=0, std=0.01):
        """
        达西渗流数据集
        :param index:
        """
        super(DarcyDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        self.data = torch.from_numpy(f['sol'][index, ::2, ::2]).float().unsqueeze(dim=1)
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[21, 21], [21, 42], [21, 63], [21, 84], [21, 105], [42, 21], [42, 42], [42, 63], [42, 84], [42, 105],
             [63, 21], [63, 42], [63, 63], [63, 84], [63, 105], [84, 21], [84, 42], [84, 63], [84, 84], [84, 105],
             [105, 21], [105, 42], [105, 63], [105, 84], [105, 105]]
        )
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        # f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        # self.data = torch.from_numpy(f['sol'][index, :, :]).float().unsqueeze(dim=1)
        # self.data = (self.data - mean) / std
        # f.close()

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class DarcyInterpolDataset(Dataset):
    def __init__(self, index, mean=0, std=0.01):
        """
        达西渗流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(DarcyInterpolDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        self.data = f['sol'][index, ::2, ::2]
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[21, 21], [21, 42], [21, 63], [21, 84], [21, 105], [42, 21], [42, 42], [42, 63], [42, 84], [42, 105],
             [63, 21], [63, 42], [63, 63], [63, 84], [63, 105], [84, 21], [84, 42], [84, 63], [84, 84], [84, 105],
             [105, 21], [105, 42], [105, 63], [105, 84], [105, 105]]
        )

        _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float().unsqueeze(dim=1)
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0), torch.from_numpy(y_coor).unsqueeze(dim=0)],
                              dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class DarcyInterpolGappyDataset(Dataset):
    def __init__(self, index, mean=0, std=0.01):
        """
        达西渗流数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(DarcyInterpolGappyDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        self.data = f['sol'][index, ::2, ::2]
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[21, 21], [21, 42], [21, 63], [21, 84], [21, 105], [42, 21], [42, 42], [42, 63], [42, 84], [42, 105],
             [63, 21], [63, 42], [63, 63], [63, 84], [63, 105], [84, 21], [84, 42], [84, 63], [84, 84], [84, 105],
             [105, 21], [105, 42], [105, 63], [105, 84], [105, 105]]
        )

        _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float().unsqueeze(dim=1)
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0), torch.from_numpy(y_coor).unsqueeze(dim=0)],
                              dim=0).float()
        self.gappy = torch.from_numpy(sparse_data).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :], self.gappy[
                                                                                                            index, :]

    def __len__(self):
        return self.data.shape[0]


class DarcyObserveDataset(Dataset):
    def __init__(self, index, mean=0, std=0.01):
        """
        达西渗流数据集：输入采用掩码表示
        :param index:
        """
        super(DarcyObserveDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        self.data = np.expand_dims(f['sol'][index, ::2, ::2], axis=1)
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[21, 21], [21, 42], [21, 63], [21, 84], [21, 105], [42, 21], [42, 42], [42, 63], [42, 84], [42, 105],
             [63, 21], [63, 42], [63, 63], [63, 84], [63, 105], [84, 21], [84, 42], [84, 63], [84, 84], [84, 105],
             [105, 21], [105, 42], [105, 63], [105, 84], [105, 105]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0), torch.from_numpy(y_coor).unsqueeze(dim=0)],
                              dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]
        # return self.observe[index, :, :, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class DarcyPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=20, mean=0, std=0.01):
        """
        达西渗流数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(DarcyPodDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/darcy/darcy12000.h5', 'r')
        self.data = np.expand_dims(f['sol'][:, ::2, ::2], axis=1)
        self.data = (self.data - mean) / std
        f.close()

        self.pca_data = torch.from_numpy(self.data[pod_index, :, :]).float()
        self.data = torch.from_numpy(self.data[index, :, :]).float()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array(
            [[21, 21], [21, 42], [21, 63], [21, 84], [21, 105], [42, 21], [42, 42], [42, 63], [42, 84], [42, 105],
             [63, 21], [63, 42], [63, 63], [63, 84], [63, 105], [84, 21], [84, 42], [84, 63], [84, 84], [84, 105],
             [105, 21], [105, 42], [105, 63], [105, 84], [105, 105]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class NOAADataset(Dataset):
    def __init__(self, index, mean=0, std=40):
        """
        NOAA海平面温度数据集
        :param index:
        """
        super(NOAADataset, self).__init__()
        self.mean, self.std = mean, std
        data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
        sst = data['sst'][:]
        self.mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
        self.mask = np.flip(self.mask, axis=0).copy()
        sst[np.isnan(sst)] = 0
        self.data = torch.from_numpy(sst.reshape(sst.shape[0], 1, 360, 180)[index, :, :, :]).float().permute(0, 1, 3, 2)
        self.data = torch.flip(self.data, dims=[2])
        self.data = (self.data - mean) / std

        positions = np.array(
            [[43, 49], [49, 120], [50, 283], [43, 30], [47, 136], [44, 299], [35, 10], [31, 140], [60, 48], [26, 199],
             [60, 247], [47, 152], [125, 302], [60, 265], [56, 10], [31, 162], [50, 359], [65, 118], [25, 36],
             [24, 180], [53, 193], [47, 168], [51, 209], [31, 265], [74, 343], [33, 282], [43, 315], [21, 56],
             [107, 139], [29, 215], [34, 359], [63, 136], [44, 231], [101, 278], [50, 333], [100, 13], [63, 152],
             [60, 299], [125, 256], [124, 77], [59, 315], [67, 90], [124, 240], [109, 121], [25, 318], [91, 262],
             [27, 302], [126, 188], [20, 234], [76, 42], [124, 272], [124, 93], [124, 138], [119, 335], [131, 318],
             [63, 168], [23, 343], [127, 210], [120, 351], [141, 291], [123, 61], [15, 2], [17, 359], [134, 171],
             [126, 154], [66, 68], [19, 18], [93, 359], [120, 0], [113, 43], [15, 73], [131, 40], [15, 128], [90, 246],
             [118, 172], [124, 19], [66, 281], [115, 319], [97, 46], [15, 285], [128, 122], [135, 334], [108, 77],
             [84, 0], [110, 156], [90, 230], [90, 343], [81, 107], [60, 225], [110, 188], [15, 145], [111, 204],
             [75, 327], [90, 214], [145, 144], [147, 310], [142, 187], [111, 224], [89, 198], [145, 213], [136, 350],
             [140, 275], [151, 127], [14, 89], [108, 93], [69, 184], [154, 38], [156, 275], [9, 40], [151, 102],
             [15, 269], [152, 171], [108, 262], [139, 56], [13, 109], [152, 80], [147, 229], [80, 311], [151, 19],
             [137, 0], [15, 161], [141, 245], [89, 182], [69, 200], [82, 281], [157, 259], [93, 123], [108, 246]]
        )
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class NOAAInterpolDataset(Dataset):
    def __init__(self, index, mean=0, std=40):
        """
        NOAA海平面温度数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(NOAAInterpolDataset, self).__init__()
        self.mean, self.std = mean, std
        data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
        sst = data['sst'][:]
        self.mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
        self.mask = np.flip(self.mask, axis=0).copy()
        sst[np.isnan(sst)] = 0
        self.data = torch.from_numpy(sst.reshape(sst.shape[0], 1, 360, 180)[index, :, :, :]).float().permute(0, 1, 3, 2)
        self.data = torch.flip(self.data, dims=[2]).numpy()
        self.data = (self.data - mean) / std

        positions = np.array(
            [[43, 49], [49, 120], [50, 283], [43, 30], [47, 136], [44, 299], [35, 10], [31, 140], [60, 48], [26, 199],
             [60, 247], [47, 152], [125, 302], [60, 265], [56, 10], [31, 162], [50, 359], [65, 118], [25, 36],
             [24, 180], [53, 193], [47, 168], [51, 209], [31, 265], [74, 343], [33, 282], [43, 315], [21, 56],
             [107, 139], [29, 215], [34, 359], [63, 136], [44, 231], [101, 278], [50, 333], [100, 13], [63, 152],
             [60, 299], [125, 256], [124, 77], [59, 315], [67, 90], [124, 240], [109, 121], [25, 318], [91, 262],
             [27, 302], [126, 188], [20, 234], [76, 42], [124, 272], [124, 93], [124, 138], [119, 335], [131, 318],
             [63, 168], [23, 343], [127, 210], [120, 351], [141, 291], [123, 61], [15, 2], [17, 359], [134, 171],
             [126, 154], [66, 68], [19, 18], [93, 359], [120, 0], [113, 43], [15, 73], [131, 40], [15, 128], [90, 246],
             [118, 172], [124, 19], [66, 281], [115, 319], [97, 46], [15, 285], [128, 122], [135, 334], [108, 77],
             [84, 0], [110, 156], [90, 230], [90, 343], [81, 107], [60, 225], [110, 188], [15, 145], [111, 204],
             [75, 327], [90, 214], [145, 144], [147, 310], [142, 187], [111, 224], [89, 198], [145, 213], [136, 350],
             [140, 275], [151, 127], [14, 89], [108, 93], [69, 184], [154, 38], [156, 275], [9, 40], [151, 102],
             [15, 269], [152, 171], [108, 262], [139, 56], [13, 109], [152, 80], [147, 229], [80, 311], [151, 19],
             [137, 0], [15, 161], [141, 245], [89, 182], [69, 200], [82, 281], [157, 259], [93, 123], [108, 246]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.observe_mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.observe_mask, self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class NOAAObserveDataset(Dataset):
    def __init__(self, index, mean=0, std=40):
        """
        NOAA海平面温度数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(NOAAObserveDataset, self).__init__()
        self.mean, self.std = mean, std
        data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
        sst = data['sst'][:]
        self.mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
        self.mask = np.flip(self.mask, axis=0).copy()
        sst[np.isnan(sst)] = 0
        self.data = torch.from_numpy(sst.reshape(sst.shape[0], 1, 360, 180)[index, :, :, :]).float().permute(0, 1, 3, 2)
        self.data = torch.flip(self.data, dims=[2]).numpy()
        self.data = (self.data - mean) / std

        positions = np.array(
            [[43, 49], [49, 120], [50, 283], [43, 30], [47, 136], [44, 299], [35, 10], [31, 140], [60, 48], [26, 199],
             [60, 247], [47, 152], [125, 302], [60, 265], [56, 10], [31, 162], [50, 359], [65, 118], [25, 36],
             [24, 180], [53, 193], [47, 168], [51, 209], [31, 265], [74, 343], [33, 282], [43, 315], [21, 56],
             [107, 139], [29, 215], [34, 359], [63, 136], [44, 231], [101, 278], [50, 333], [100, 13], [63, 152],
             [60, 299], [125, 256], [124, 77], [59, 315], [67, 90], [124, 240], [109, 121], [25, 318], [91, 262],
             [27, 302], [126, 188], [20, 234], [76, 42], [124, 272], [124, 93], [124, 138], [119, 335], [131, 318],
             [63, 168], [23, 343], [127, 210], [120, 351], [141, 291], [123, 61], [15, 2], [17, 359], [134, 171],
             [126, 154], [66, 68], [19, 18], [93, 359], [120, 0], [113, 43], [15, 73], [131, 40], [15, 128], [90, 246],
             [118, 172], [124, 19], [66, 281], [115, 319], [97, 46], [15, 285], [128, 122], [135, 334], [108, 77],
             [84, 0], [110, 156], [90, 230], [90, 343], [81, 107], [60, 225], [110, 188], [15, 145], [111, 204],
             [75, 327], [90, 214], [145, 144], [147, 310], [142, 187], [111, 224], [89, 198], [145, 213], [136, 350],
             [140, 275], [151, 127], [14, 89], [108, 93], [69, 184], [154, 38], [156, 275], [9, 40], [151, 102],
             [15, 269], [152, 171], [108, 262], [139, 56], [13, 109], [152, 80], [147, 229], [80, 311], [151, 19],
             [137, 0], [15, 161], [141, 245], [89, 182], [69, 200], [82, 281], [157, 259], [93, 123], [108, 246]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class NOAAPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=20, mean=0, std=40):
        """
        NOAA海平面温度数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(NOAAPodDataset, self).__init__()
        self.mean, self.std = mean, std
        data = h5py.File('/home/ubuntu/zhaoxiaoyu/data/noaa/sst_weekly.mat')
        sst = data['sst'][:]
        self.mask = np.isnan(sst[0, :]).reshape(360, 180).transpose()
        self.mask = np.flip(self.mask, axis=0).copy()
        sst[np.isnan(sst)] = 0
        self.data = torch.from_numpy(sst.reshape(sst.shape[0], 1, 360, 180)[:, :, :, :]).float().permute(0, 1, 3, 2)
        self.data = torch.flip(self.data, dims=[2])
        self.data = (self.data - mean) / std

        self.pca_data = self.data[pod_index, :, :, :]
        self.data = self.data[index, :, :, :]

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array(
            [[43, 49], [49, 120], [50, 283], [43, 30], [47, 136], [44, 299], [35, 10], [31, 140], [60, 48], [26, 199],
             [60, 247], [47, 152], [125, 302], [60, 265], [56, 10], [31, 162], [50, 359], [65, 118], [25, 36],
             [24, 180], [53, 193], [47, 168], [51, 209], [31, 265], [74, 343], [33, 282], [43, 315], [21, 56],
             [107, 139], [29, 215], [34, 359], [63, 136], [44, 231], [101, 278], [50, 333], [100, 13], [63, 152],
             [60, 299], [125, 256], [124, 77], [59, 315], [67, 90], [124, 240], [109, 121], [25, 318], [91, 262],
             [27, 302], [126, 188], [20, 234], [76, 42], [124, 272], [124, 93], [124, 138], [119, 335], [131, 318],
             [63, 168], [23, 343], [127, 210], [120, 351], [141, 291], [123, 61], [15, 2], [17, 359], [134, 171],
             [126, 154], [66, 68], [19, 18], [93, 359], [120, 0], [113, 43], [15, 73], [131, 40], [15, 128], [90, 246],
             [118, 172], [124, 19], [66, 281], [115, 319], [97, 46], [15, 285], [128, 122], [135, 334], [108, 77],
             [84, 0], [110, 156], [90, 230], [90, 343], [81, 107], [60, 225], [110, 188], [15, 145], [111, 204],
             [75, 327], [90, 214], [145, 144], [147, 310], [142, 187], [111, 224], [89, 198], [145, 213], [136, 350],
             [140, 275], [151, 127], [14, 89], [108, 93], [69, 184], [154, 38], [156, 275], [9, 40], [151, 102],
             [15, 269], [152, 171], [108, 262], [139, 56], [13, 109], [152, 80], [147, 229], [80, 311], [151, 19],
             [137, 0], [15, 161], [141, 245], [89, 182], [69, 200], [82, 281], [157, 259], [93, 123], [108, 246]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


class HeatDataset(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集
        :param index:
        """
        super(HeatDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = torch.from_numpy(f['u'][index, :, :, :]).float()
        self.data = (self.data - mean) / std
        f.close()

        positions = np.array(
            [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
             [99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132],
             [132, 165], [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatDatasetMulti(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集
        :param index:
        """
        super(HeatDatasetMulti, self).__init__()
        self.mean, self.std = mean, std
        data = sio.loadmat('/mnt/jfs/Lyuyanfang/datasets/Temperature/target_50_complex.mat')['data']
        self.data = torch.from_numpy(data)[index, :, :, :].float()
        self.data = (self.data - mean) / std

        positions = np.array(
            [[10, 10], [10, 20], [10, 25], [10, 30], [10, 40], [20, 10], [20, 20], [20, 25], [20, 30], [20, 40],
             [25, 10], [25, 20], [25, 25], [25, 30], [25, 40], [30, 10], [30, 20], [30, 25], [30, 30], [30, 40],
             [40, 10], [40, 20], [40, 25], [40, 30], [40, 40]]
        )
        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatInterpolDataset(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集：输入进行近邻插值转换为泰森多边形形式
        :param index:
        """
        super(HeatInterpolDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()
        positions = np.array(
            [[33, 33], [33, 66], [33, 99], [33, 132], [33, 165], [66, 33], [66, 66], [66, 99], [66, 132], [66, 165],
             [99, 33], [99, 66], [99, 99], [99, 132], [99, 165], [132, 33], [132, 66], [132, 99], [132, 132],
             [132, 165], [165, 33], [165, 66], [165, 99], [165, 132], [165, 165]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        sparse_data = np.concatenate(sparse_data, axis=-1)

        sparse_datas = []
        print("processing data...")
        for i in tqdm(range(sparse_data.shape[0])):
            input_1 = griddata(sparse_locations_ex, sparse_data[i], (x_coor, y_coor), method='nearest')
            sparse_datas.append(np.expand_dims(input_1, axis=0))
        sparse_datas = np.concatenate(sparse_datas, axis=0)
        mask = np.zeros_like(sparse_datas[0, :, :])
        for i in range(positions.shape[0]):
            mask[positions[i, 0], positions[i, 1]] = 1

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_datas).float().unsqueeze(dim=1)
        self.mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.mask, self.coor], dim=0), self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatObserveDataset(Dataset):
    def __init__(self, index, mean=308, std=50):
        """
        热布局数据集：输入采用掩码表示
        :param index:
        """
        super(HeatObserveDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][index, :, :, :]
        self.data = (self.data - mean) / std
        f.close()
        positions = np.array(
            [[28, 28], [28, 56], [28, 84], [28, 112], [28, 140], [28, 168], [56, 28], [56, 56], [56, 84], [56, 112],
             [56, 140], [56, 168], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [112, 28], [112, 56],
             [112, 84], [112, 112], [112, 140], [112, 168], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140],
             [140, 168], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168]]
        )

        _, _, h, w = self.data.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)

        sparse_data = np.zeros_like(self.data)
        for i in range(positions.shape[0]):
            sparse_data[:, 0, positions[i, 0], positions[i, 1]] = self.data[:, 0, positions[i, 0], positions[i, 1]]

        self.data = torch.from_numpy(self.data).float()
        self.observe = torch.from_numpy(sparse_data).float()
        self.coor = torch.cat([torch.from_numpy(x_coor).unsqueeze(dim=0) / w,
                               torch.from_numpy(y_coor).unsqueeze(dim=0) / h], dim=0).float()

    def __getitem__(self, index):
        return torch.cat([self.observe[index, :, :, :], self.coor], dim=0), self.data[index, :]
        # return self.observe[index, :, :, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]


class HeatPodDataset(Dataset):
    def __init__(self, pod_index, index, n_components=25, mean=308, std=50):
        """
        热布局数据集：对输出进行POD，并返回POD系数
        :param pod_index: 进行POD的数据索引
        :param index: 数据索引
        :param n_components: POD模态数量
        """
        super(HeatPodDataset, self).__init__()
        self.mean, self.std = mean, std
        f = h5py.File('/home/ubuntu/zhaoxiaoyu/data/heat/temperature.h5', 'r')
        self.data = f['u'][:, :, :, :]
        self.data = (self.data - mean) / std
        f.close()

        self.pca_data = torch.from_numpy(self.data[pod_index, :, :, :]).float()
        self.data = torch.from_numpy(self.data[index, :, :, :]).float()

        self.pca = PCA(n_components=n_components)
        X_t = self.pca.fit_transform(self.pca_data.reshape(len(pod_index), -1).numpy())
        self.max, self.min = np.max(X_t, axis=0).reshape(1, -1), np.min(X_t, axis=0).reshape(1, -1)

        # 计算POD系数并归一化
        self.coff = self.pca.transform(self.data.reshape(len(index), -1).numpy())
        self.coff = (self.coff - self.min) / (self.max - self.min)

        positions = np.array(
            [[28, 28], [28, 56], [28, 84], [28, 112], [28, 140], [28, 168], [56, 28], [56, 56], [56, 84], [56, 112],
             [56, 140], [56, 168], [84, 28], [84, 56], [84, 84], [84, 112], [84, 140], [84, 168], [112, 28], [112, 56],
             [112, 84], [112, 112], [112, 140], [112, 168], [140, 28], [140, 56], [140, 84], [140, 112], [140, 140],
             [140, 168], [168, 28], [168, 56], [168, 84], [168, 112], [168, 140], [168, 168]]
        )

        sparse_data = []
        for i in range(positions.shape[0]):
            sparse_data.append(self.data[:, 0, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
        self.observe = np.concatenate(sparse_data, axis=-1)
        self.size = self.data.shape[-3:]

    def __getitem__(self, index):
        return self.observe[index, :], self.coff[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def inverse_transform(self, coff):
        inverse_coff = coff.cpu().numpy() * (self.max - self.min) + self.min
        return torch.from_numpy(self.pca.inverse_transform(inverse_coff)).to(coff.device) \
            .float().reshape(coff.shape[0], self.size[0], self.size[1], self.size[2])


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = CylinderInterpolSNRDataset(index=[100], snr=True, SNRdB=10)
    data_loader = DataLoader(dataset, batch_size=1)
    data_iter = iter(data_loader)
    observe, outputs = next(data_iter)
    # plt.figure(figsize=(9.6, 5.6))
    plt.axis('off')
    plt.imshow(outputs[0, 0, :, :].numpy(), cmap='seismic')
    plt.colorbar()
    plt.show()
