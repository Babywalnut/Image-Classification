# -*- coding:utf-8 -*-
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=output_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x
