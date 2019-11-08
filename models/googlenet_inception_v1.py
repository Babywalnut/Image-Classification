# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f
from .utils import BasicCNN


class InceptionV1(nn.Module):
    def __init__(self, in_channels, _1x1, _3x3_reduce, _3x3, _5x5_reduce, _5x5, _pool_proj):
        super().__init__()
        self.branch1 = nn.Conv2d(
            in_channels=in_channels, out_channels=_1x1, kernel_size=(1, 1), stride=1
        )
        self.branch2 = nn.Sequential(
            BasicCNN(in_channels=in_channels, output_channels=_3x3_reduce, kernel_size=(1, 1), stride=1),
            BasicCNN(in_channels=_3x3_reduce, output_channels=_3x3, kernel_size=(3, 3), stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicCNN(in_channels=in_channels, output_channels=_5x5_reduce, kernel_size=(1, 1), stride=1),
            BasicCNN(in_channels=_5x5_reduce, output_channels=_5x5, kernel_size=(5, 5), stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            BasicCNN(in_channels=in_channels, output_channels=_pool_proj, kernel_size=(1, 1), stride=1)
        )

    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        x4 = self.branch4(inputs)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class GoogleNetInceptionV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            BasicCNN(in_channels=3, output_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        # Inception 3
        self.inception_a3 = InceptionV1(in_channels=64, _1x1=64, _3x3_reduce=96, _3x3=128, _5x5_reduce=16, _5x5=32,
                                        _pool_proj=32)
        self.inception_b3 = InceptionV1(in_channels=256, _1x1=128, _3x3_reduce=128, _3x3=192, _5x5_reduce=32, _5x5=96,
                                        _pool_proj=64)

        # Inception 4
        self.inception_a4 = InceptionV1(in_channels=480, _1x1=192, _3x3_reduce=96, _3x3=208, _5x5_reduce=16, _5x5=48,
                                        _pool_proj=64)
        self.inception_b4 = InceptionV1(in_channels=512, _1x1=160, _3x3_reduce=112, _3x3=224, _5x5_reduce=24, _5x5=64,
                                        _pool_proj=64)
        self.inception_c4 = InceptionV1(in_channels=512, _1x1=128, _3x3_reduce=128, _3x3=256, _5x5_reduce=24, _5x5=64,
                                        _pool_proj=64)
        self.inception_d4 = InceptionV1(in_channels=512, _1x1=112, _3x3_reduce=144, _3x3=288, _5x5_reduce=32, _5x5=64,
                                        _pool_proj=64)
        self.inception_e4 = InceptionV1(in_channels=528, _1x1=256, _3x3_reduce=160, _3x3=320, _5x5_reduce=32, _5x5=128,
                                        _pool_proj=128)

        # Inception 5
        self.inception_a5 = InceptionV1(in_channels=832, _1x1=256, _3x3_reduce=160, _3x3=320, _5x5_reduce=32, _5x5=128,
                                        _pool_proj=128)
        self.inception_b5 = InceptionV1(in_channels=832, _1x1=384, _3x3_reduce=192, _3x3=384, _5x5_reduce=48, _5x5=128,
                                        _pool_proj=128)

        # fully connection
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, 100)
        )

    def forward(self, inputs):
        # Stem Layer
        x = self.stem(inputs)

        # Inception 3 Layer
        x = self.inception_a3(x)
        x = self.inception_b3(x)
        x = f.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Inception 4 Layer
        x = self.inception_a4(x)
        x = self.inception_b4(x)
        x = self.inception_c4(x)
        x = self.inception_d4(x)
        x = self.inception_e4(x)
        x = f.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Inception 5 Layer
        x = self.inception_a5(x)
        x = self.inception_b5(x)

        x = f.avg_pool2d(x, kernel_size=int(x.shape[-1]), stride=1)

        # fully connection
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
