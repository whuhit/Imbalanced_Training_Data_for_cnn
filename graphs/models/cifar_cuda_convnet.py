#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 10:05
# @Author  : yangqiang
# @File    : cifar_cuda_convnet.py
import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(8 * 8 * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)  # num_classes
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.dropout(self.layer1(x))
        out = self.dropout(self.layer2(out))
        out = out.reshape(out.size(0), -1)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return self.softmax(out)


if __name__ == '__main__':
    net = Net()
    x = torch.randn(8, 3, 32, 32)
    y = net(x)
    pred = y.max(1, keepdim=True)[1]
    print(y.size())
# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)