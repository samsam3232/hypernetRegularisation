import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.hyperNetwork import HyperNetwork
import math
from models.resnetModule import ResnetBlock

#BLOCKS = {'CIFAR': ResnetCifarBlock, 'REG': ResnetBlock}
SIZES_CIFAR = {18: [3, 3, 3], 32: [5, 5, 5], 44: [7, 7, 7]}
SIZES_REG = {18: [2, 2, 2, 2], 32: [3, 4, 6, 4], 101: [3, 4, 23, 3]}
SIZES = {'CIFAR': SIZES_CIFAR, 'REG': SIZES_REG}

class PrimaryNetwork(nn.Module):

    FILTER_SIZE_REG = [64, 128, 256, 512]
    FILTER_SIZE_CIFAR = [16, 32, 64]
    IN_PLANES = {'CIFAR': 16, 'REG': 64}
    FILTER_SIZE = {'CIFAR': FILTER_SIZE_CIFAR, 'REG': FILTER_SIZE_REG}
    STRIDE_REG = [1, 2, 2, 2]
    STRIDE_CIFAR = [1, 2, 2, 2]
    STRIDE = {'CIFAR': STRIDE_CIFAR, 'REG': STRIDE_REG}

    def __init__(self, block, num_blocks, num_classes, type, device, regularize, options ={}):

        super(PrimaryNetwork, self).__init__()

        self.device = device
        self.type = type
        self.do_dropout = False
        self.in_planes = self.IN_PLANES[self.type]
        self.res_net = nn.ModuleList()
        self.num_classes = num_classes
        self.lay_shapes = list()
        self.mod_sizes = list()
        self.regularize = regularize

        if type == 'CIFAR':
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.lay_shapes = [(16, 3, 3, 3)]
            self.mod_sizes = [16*3*3*3]
            self.final = nn.Linear(64, num_classes)
            self.final_shape = 64
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.lay_shapes = [(64, 3, 7, 7)]
            self.mod_sizes = [64*3*7*7]
            self.final = nn.Linear(512, num_classes)
            self.final_shape = 512

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        self.num_blocks = num_blocks

        for i in range(len(self.FILTER_SIZE[self.type])):
            self._make_layer(block, self.FILTER_SIZE[self.type][i], self.num_blocks[i], self.STRIDE[self.type][i], self.regularize[i])

        if type == 'CIFAR':
            self.mod_sizes.append(64 * num_classes)
            self.lay_shapes.append((num_classes, 64))
        else:
            self.mod_sizes.append(512 * num_classes)
            self.lay_shapes.append((num_classes, 512))

        if "dropout" in options:
            self.dropout = nn.Dropout(options["dropout"])
        else:
            self.dropout = nn.Dropout(0.0)

        self.hyper_net = HyperNetwork(device, torch.sum(torch.tensor(self.mod_sizes)), options)

        self.weight_init()


    def _make_layer(self, block, planes, num_blocks, stride, regularize):

        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            self.res_net.append(block(self.in_planes, planes, stride, regularize))
            self.lay_shapes.append((planes, self.in_planes, 3, 3))
            self.lay_shapes.append((planes, planes, 3, 3))
            self.mod_sizes.append(planes * self.in_planes * 3 * 3)
            self.mod_sizes.append(planes * planes * 3 * 3)
            self.in_planes = planes * block.expansion

    def replace_layer(self, index):

        self.res_net[index] = ResnetBlock(self.lay_shapes[index+1][1],self.lay_shapes[index+1][0],self.res_net[index].stride1, True)

    def forward(self, x):

        index = 0
        curr = 0
        noise = self.hyper_net()
        x = F.relu(self.bn1(self.conv1(x)))
        index = 1
        curr = 1
        for i in range(len(self.res_net)):
            # if i != 15 and i != 17:
            w1 = Parameter(noise[curr:curr + self.mod_sizes[index]].reshape(self.lay_shapes[index]))
            curr += self.mod_sizes[index]
            index += 1
            w2 = Parameter(noise[curr:curr + self.mod_sizes[index]].reshape(self.lay_shapes[index]))
            curr += self.mod_sizes[index]
            index += 1
            w1.to(self.device)
            w2.to(self.device)
            x = (self.res_net[i](x, w1, w2) / self.get_dims(x.shape))

        x = self.global_avg(x)
        x = self.final(x.view(-1,self.final_shape))

        return x, noise

    def insert_dropout(self):

        added = 0
        for i in range(len(self.res_net)):
            if i % 2 == 1:
                self.hyper_deconv.insert(i + added, self.dropout)
                added += 1

    def get_dims(self, shape):

        dim = 1
        for i in shape[1:]:
            dim *= i
        return math.sqrt(torch.tensor(dim).to(self.device))

    def set_regularize(self, index, value):

        self.regularize[index] = value
        return 1

    def set_dropout(self, value):

        self.do_dropout = value
        return 1

    def set_hyper_dropout(self, value):

        self.hyper_net.do_dropout = value
        return 1

    def set_hyper_relu(self, value):

        self.hyper_net.do_relu = value
        return 1

    def weight_init(self):

        for m in self.modules():
            if (isinstance(m, nn.ConvTranspose2d)) or (isinstance(m, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



def get_resnet(size, type, num_classes, device, regularize, options = {}):

    return PrimaryNetwork(BLOCKS[type], SIZES[type][size], num_classes, type, device, regularize, options)

#primary = get_resnet(18, 'CIFAR', 100, 'cpu', [False, False, True], {"dropout" : 0.25, "relu": True})
#print('Hey')