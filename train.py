# -*- coding:utf-8 -*-
import os
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--summary_dir', type=str, default='./summary')
    parser.add_argument('--model', type=str,
                        default='VGG11',
                        choices=['AlexNet', 'VGG11', 'VGG19', 'GoogleNet'])

    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--writer_step', type=int, default=50)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, argument):
        self.argument = argument
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(
            logdir=os.path.join(argument.summary_dir, argument.model)
        )

    def start(self):
        train_data_loader = self.train_data_loader()
        eval_data_loader = self.eval_data_loader()

        model = self.get_model()
        model.to(device)
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=self.argument.learning_rate,
            momentum=0.9, weight_decay=5e-4
        )

        global_step = 0
        for epoch in range(self.argument.epochs):
            for i, (train_date, eval_data) in enumerate(zip(train_data_loader, eval_data_loader)):
                train_loss, train_acc = self.get_loss_accuracy(date=train_date, model=model)
                eval_loss, eval_acc = self.get_loss_accuracy(date=eval_data, model=model)

                # Printing console
                if i % self.argument.print_step == 0:
                    print('[Epoch] {0:2d} \t '
                          '[Iter] {1:2d} \t '
                          '[Train] loss : {2:.4f} \t acc : {3:4f} \t '
                          '[Eval] loss: {4:.4f} \t acc: {5:.4f}'.
                          format(epoch, i, train_loss.item(), train_acc.item(), eval_loss.item(), eval_acc.item()))

                # Writer Summary
                if i % self.argument.writer_step == 0:
                    self.writer.add_scalar('train/loss', train_loss.item(), global_step)
                    self.writer.add_scalar('train/accuracy', train_acc.item(), global_step)
                    self.writer.add_scalar('eval/loss', eval_loss.item(), global_step)
                    self.writer.add_scalar('eval/accuracy', eval_acc.item(), global_step)

                # optimizer & backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                global_step += 1

    def get_loss_accuracy(self, date, model):
        image, label = date
        image = image.to(device)
        label = label.to(device)
        pred = model(image)

        # calculation loss
        loss = self.criterion(input=pred, target=label)

        # calculation accuracy
        pred = pred.argmax(dim=-1)
        eq = torch.eq(pred, label)
        acc = eq.to(torch.float32).mean()
        return loss, acc

    def train_data_loader(self):
        # Train dataset & data loader
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.argument.data_path,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.ToTensor()
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.argument.batch_size,
            shuffle=True,
            num_workers=self.argument.num_workers
        )
        return train_loader

    def eval_data_loader(self):
        # Eval dataset & data loader
        eval_dataset = torchvision.datasets.CIFAR100(
            root=self.argument.data_path,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=self.argument.batch_size,
            shuffle=False,
            num_workers=self.argument.num_workers
        )
        return eval_loader

    @staticmethod
    def get_accuracy(pred, real):
        pred = pred.argmax(dim=-1)
        eq = torch.eq(pred, real)
        acc = eq.to(torch.float32).mean()
        return acc

    def get_model(self):
        # Get modeling
        model_name = self.argument.model
        if model_name == 'AlexNet':
            from models.alexnet import AlexNet
            return AlexNet()

        elif model_name == 'VGG11':
            from models.vgg import VGG
            return VGG(vgg_type='VGG11')

        elif model_name == 'VGG13':
            from models.vgg import VGG
            return VGG(vgg_type='VGG13')

        elif model_name == 'VGG16':
            from models.vgg import VGG
            return VGG(vgg_type='VGG16')

        elif model_name == 'VGG19':
            from models.vgg import VGG
            return VGG(vgg_type='VGG19')

        else:
            raise NotImplemented()


if __name__ == '__main__':
    args = get_args()
    Trainer(args).start()
