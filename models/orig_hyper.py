import torch.nn as nn
import torch
import torch.nn.functional as F


class ElementWiseLayer(nn.Module):

    def __init__(self, dim, device):
        super(ElementWiseLayer, self).__init__()
        self.vec = torch.randn(dim).to(device)

    def forward(self, x):
        return torch.mul(self.vec, x)


class HyperNetwork(nn.Module):

    def __init__(self, device, size_tot, dropout):

        super(HyperNetwork, self).__init__()
        self.device = device
        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
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

        self.final = ElementWiseLayer(curr_size, device)

    def forward(self, std, batch = 1):

        noise = torch.rand(batch, 15*15*3).to(self.device) * std
        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = inp.view((batch, 3, 30, 30))
        for i in range(len(self.hyper_deconv)):
            inp = self.hyper_deconv[i](inp)

        inp = self.final(F.relu(inp.view(batch, -1)))
        return inp.view(-1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def set_dropout(self, new_dropout):

        self.dropout = nn.Dropout(new_dropout)