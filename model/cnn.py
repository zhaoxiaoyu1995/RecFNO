# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 18:45
# @Author  : zhaoxiaoyu
# @File    : cnn.py.py
import torch
import torch.nn.functional as F
from torch import nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False, dropout=False):
        super(_DecoderBlock, self).__init__()
        if dropout:
            self.decode = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.decode = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
                nn.GELU(),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
            )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn=False):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 32, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(32, 64, bn=bn)
        self.enc3 = _EncoderBlock(64, 128, bn=bn)
        self.enc4 = _EncoderBlock(128, 256, bn=bn, dropout=False)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(256, 512, 256, bn=bn)
        self.dec4 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec3 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec2 = _DecoderBlock(128, 64, 32, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if bn else nn.GroupNorm(32, 64),
            nn.GELU()
        )
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        # dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], mode='bilinear',
                                                  align_corners=True), enc4], 1))
        # dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], mode='bilinear',
                                                  align_corners=True), enc3], 1))
        # dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], mode='bilinear',
                                                  align_corners=True), enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.GELU(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class CNNRecon(nn.Module):
    def __init__(self, sensor_num, fc_size):
        super(CNNRecon, self).__init__()
        self.fc_size = fc_size
        self.fc = nn.Sequential(
            nn.Linear(sensor_num, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, fc_size[0] * fc_size[1]),
        )
        self.embedding = nn.Conv2d(1, 32, kernel_size=1)
        self.decoder1 = DecoderBlock(32, 64, 64)
        self.decoder2 = DecoderBlock(64, 64, 64)
        self.decoder3 = DecoderBlock(64, 64, 64)
        self.decoder4 = DecoderBlock(64, 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        N, _ = x.shape
        x = self.fc(x).reshape(N, 1, self.fc_size[0], self.fc_size[1])
        x = self.embedding(x)

        x = self.decoder1(x)
        # x = self.decoder2(F.interpolate(x, (25, 25), mode='bilinear', align_corners=True))
        x = self.decoder2(x)
        # x = self.decoder3(F.interpolate(x, (45, 90), mode='bilinear', align_corners=True))
        x = self.decoder3(x)
        x = self.decoder4(x)
        return self.final(x)


if __name__ == '__main__':
    model = CNNRecon(sensor_num=4, fc_size=(8, 3))
    x = torch.randn(1, 4)
    with torch.no_grad():
        y = model(x)
    print(y.shape)
