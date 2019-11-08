# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=(11, 11),
                stride=4
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, inputs):
        print(inputs.shape)
        output = self.conv1(inputs)
        print(output.shape)
        return inputs
