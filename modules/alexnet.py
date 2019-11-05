# -*- coding:utf-8 -*-
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Convolution Network Network 1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolution Network Network 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolution Network Network 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            # Convolution Network Network 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, inputs):
        output = self.features(inputs)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output


def train(arguments):
    # Loading Train Dataset & DataLoader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.CIFAR10(root=arguments.data_path,
                                           train=True, download=True,
                                           transform=transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=arguments.batch_size, shuffle=True)
    classes = dataset.classes

    # AlexNet model
    model = AlexNet(num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    # Training...
    for _ in range(arguments.epochs):
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            label = label.to(device)
            print(image.shape)
            print(label.shape)
            exit()

            pred = model(image)
            loss = criterion(pred, label)

            # optimizer & backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(loss.item())


if __name__ == '__main__':
    args = get_args()
    train(args)