# -*- coding:utf-8 -*-
import torch.nn as nn


cfg = {
    'VGG11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_type, batch_norm=True, num_classes=100):
        super().__init__()
        self.vgg_type = vgg_type
        self.batch_norm = batch_norm
        self.cnn = self.cnn_layers()
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, inputs):
        cnn_output = self.cnn(inputs)
        x = cnn_output.view(cnn_output.size(0), -1)
        output = self.fc(x)
        return output

    def cnn_layers(self):
        layers = []

        input_channel = 3
        for layer in cfg[self.vgg_type]:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, layer, kernel_size=(3, 3), padding=1)]

            if self.batch_norm:
                layers += [nn.BatchNorm2d(layer)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = layer

        return nn.Sequential(*layers)


