import torch.nn as nn
import torch
import torch.nn.functional as F


class ElementWiseLayer(nn.Module):

    def __init__(self, dim, device):
        super(ElementWiseLayer, self).__init__()
        self.vec = torch.randn(dim).to(device)

    def forward(self, input):
        return torch.mul(self.vec, input)

class HyperNetwork(nn.Module):

    def __init__(self, device, size_tot, options):

        super(HyperNetwork, self).__init__()

        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
        self.do_dropout = False
        self.do_relu = False
        self.fc1 = nn.Linear(15 * 15 * 3, 30 * 30 * 3)
        curr_shape = (30,30)
        curr_size = 7 * ((curr_shape[-2] - 1) *2 + 1) * ((curr_shape[-2] - 1) *2 + 1)
        curr_shape = (((curr_shape[-2] - 1) *2 + 1), ((curr_shape[-2] - 1) *2 + 1))
        curr_channels = 7

        i = 0
        while curr_size < size_tot:
            if i % 2 == 0:
                self.hyper_deconv.append(nn.ConvTranspose2d(curr_channels, curr_channels, 3, stride=2, padding=1))
                curr_shape = (((curr_shape[-2] - 1) * 2 + 1), ((curr_shape[-2] - 1) * 2 + 1))
                curr_size = curr_channels * curr_shape[0] * curr_shape[1]
            else:
                self.hyper_deconv.append(nn.ConvTranspose2d(curr_channels, min(curr_channels*2, 30), 3, stride=2, padding=1))
                curr_channels = min(curr_channels*2, 30)
                curr_shape = (((curr_shape[-2] - 1) * 2 + 1), ((curr_shape[-2] - 1) * 2 + 1))
                curr_size = curr_channels * curr_shape[0] * curr_shape[1]
            i += 1

        if "dropout_hyper" in options:
            self.do_dropout = True
            self.dropout = nn.Dropout(options["dropout_hyper"])
        else:
            self.dropout = nn.Dropout(0.0)
        if "relu" in options:
            self.do_relu = True
        self.final = ElementWiseLayer(curr_size, device)

        self.device = device
        self.weight_init()

    def forward(self, std = 1):

        noise = torch.rand(15*15*3).to(self.device) * std
        noise.requires_grad = True
        input = self.fc1(noise)
        input = input.view((-1, 3, 30, 30))
        for i in range(len(self.hyper_deconv)):
            input = self.hyper_deconv[i](input)
            if i % 2 == 0 and self.do_dropout:
                input = self.dropout(input)

        if self.do_relu:
            input = self.final(F.relu(input.view(-1)))

        return input.view(-1)

    def weight_init(self):

        for m in self.modules():
            if (isinstance(m, nn.ConvTranspose2d)) or (isinstance(m, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')