import torch
import torch.nn as nn
import torch.nn.functional as F
from models.orig_hyper import get_hyper
from models.orig_resnet import ResnetBlock
import numpy as np


class PrimaryNetwork(nn.Module):
    MOD_SIZES = np.array([2304, 2304, 2304, 2304, 2304, 2304, 4608, 9216, 9216, 9216, 9216, 9216, 18432,
                          36864, 36864, 36864, 36864, 36864])
    SHAPES = [(16, 16, 3, 3), (16, 16, 3, 3), (16, 16, 3, 3), (16, 16, 3, 3), (16, 16, 3, 3), (16, 16, 3, 3),
              (32, 16, 3, 3), (32, 32, 3, 3), (32, 32, 3, 3), (32, 32, 3, 3), (32, 32, 3, 3), (32, 32, 3, 3),
              (64, 32, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3), (64, 64, 3, 3)]

    IN_DIMS = [3 * 32 * 32, 64 * 16 * 16, 64 * 16 * 16, 64 * 16 * 16, 128 * 8 * 8, 128 * 8 * 8, 256 * 4 * 4,
               256 * 4 * 4, 512 * 4, 512]

    PLANES = [16, 32, 64]

    def __init__(self, num_classes, device, regularize, std, dropout, architecture = "A"):
        super(PrimaryNetwork, self).__init__()

        self.device = device

        self.hyper_size = np.sum(
            [self.MOD_SIZES[i * 2: i * 2 + 2] for i in range(len(regularize.reshape(9))) if regularize.reshape(9)[i]])
        self.hyper_net = get_hyper(device, self.hyper_size, dropout, architecture)
        self.hyper_net.to(device)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.std = std
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter_size = [[64, 64], [64, 64], [64, 128], [128, 128], [128, 256], [256, 256], [256, 512], [512, 512]]
        self.strid = [1, 2, 2]
        self.res_net = nn.ModuleList()
        self.regularize = regularize
        self.in_planes = 16
        self.final = nn.Linear(64, num_classes)

        for i in range(3):
            self._make_layer(ResnetBlock, self.PLANES[i], 3, regularize[i], self.strid[i])

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, regularize, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for j in range(len(strides)):
            self.res_net.append(block(self.in_planes, planes, strides[j], regularize[j]))
            self.in_planes = planes * block.expansion

    def forward(self, x, std, keep_weights=False):

        noise = self.hyper_net(std).view(-1)
        index = 0
        curr = 0
        x = F.relu(self.bn1(self.conv1(x)))
        #        x = F.relu(self.bn1(
        #            F.conv2d(x / math.sqrt(3 * 3 * 3), noise[curr:curr + self.MOD_SIZES[index]].reshape((16, 3, 3, 3)),
        #                     stride=1, padding=1)))
        for i in range(9):
            if self.regularize.reshape(9)[i]:
                w1 = noise[curr:curr + self.MOD_SIZES[index]].reshape(self.SHAPES[i * 2])
                curr += self.MOD_SIZES[index]
                index += 1
                w2 = noise[curr:curr + self.MOD_SIZES[index]].reshape(self.SHAPES[i * 2 + 1])
                curr += self.MOD_SIZES[index]
                index += 1
                w1.to(self.device)
                w2.to(self.device)
            else:
                w1 = None
                w2 = None
                index += 2
            x = (self.res_net[i](x, w1, w2))
        x = self.global_avg(x)
        # x = F.linear((x.view(-1, 64) / math.sqrt(64)), noise[curr:curr + 64 * self.num_classes].reshape((self.num_classes, 64)))
        x = self.final(x.view(-1, 64))

        return x, noise[:int(self.hyper_size)]

    def get_size_ratio(self, std):

        weights = self.hyper_net(std, batch=32)
        return torch.count_nonzero(weights[0, :int(self.hyper_size)]) / weights[0, :int(self.hyper_size)].numel()