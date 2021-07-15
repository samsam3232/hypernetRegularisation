import torch.nn as nn
import torch
import torch.nn.functional as F

RANDOM_FUNCS = {"normal": torch.normal, "uniform": torch.rand}

class ElementWiseLayer(nn.Module):

    def __init__(self, dim, device):
        super(ElementWiseLayer, self).__init__()
        self.vec = torch.randn(dim).to(device)

    def forward(self, x):
        return torch.mul(self.vec, x)


class HyperNetworkA(nn.Module):

    def __init__(self, device, size_tot, dropout, random_type = "normal"):

        super(HyperNetworkA, self).__init__()
        self.device = device
        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
        self.fc1 = nn.Linear(15 * 15 * 3, 30 * 30 * 3)
        self.dropout = nn.Dropout(dropout)
        self.random_type = random_type

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

    def forward(self, std, mean, batch = 1):

        if self.random_type == "uniform":
            noise = RANDOM_FUNCS[self.random_type](batch, 15*15*3).to(self.device) * std
        elif self.random_type == "normal":
            noise = RANDOM_FUNCS[self.random_type](mean, std, (batch, 15*15*3)).to(self.device)

        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = inp.view((batch, 3, 30, 30))
        inp = self.dropout(inp)
        for i in range(len(self.hyper_deconv)):
            inp = self.hyper_deconv[i](inp)

        inp = self.dropout(inp)
        inp = self.final(F.relu(inp.view(batch, -1)))
        return inp.view(batch, -1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class HyperNetworkB(nn.Module):

    def __init__(self, device, size_tot, dropout, random_type = "normal"):

        super(HyperNetworkB, self).__init__()
        self.device = device
        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
        self.fc1 = nn.Linear(15 * 15 * 3, 30 * 30 * 3)
        self.fc2 = nn.Linear(30 * 30 *3, 60 * 60 * 6)
        self.fc3 = nn.Linear(60 * 60 * 6, 60 * 60 * 3)
        self.dropout = nn.Dropout(dropout)
        self.random_type = random_type

        curr_shape = (60,60)
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

    def forward(self, std, mean, batch = 1):

        if self.random_type == "uniform":
            noise = RANDOM_FUNCS[self.random_type](batch, 15*15*3).to(self.device) * std
        elif self.random_type == "normal":
            noise = RANDOM_FUNCS[self.random_type](mean, std, (batch, 15*15*3)).to(self.device)

        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = self.dropout(inp)
        inp = self.fc2(inp)
        inp = self.fc3(inp)
        inp = self.dropout(inp)
        inp = inp.view((batch, 3, 60, 60))
        for i in range(len(self.hyper_deconv)):
            inp = self.hyper_deconv[i](inp)

        inp = self.dropout(inp)
        inp = self.final(F.relu(inp.view(batch, -1)))
        return inp.view(batch, -1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class HyperNetworkC(nn.Module):

    def __init__(self, device, size_tot, dropout, random_type = "normal"):

        super(HyperNetworkC, self).__init__()
        self.device = device
        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
        self.fc1 = nn.Linear(15 * 15 * 3, 15 * 15 * 3)
        self.fc2 = nn.Linear(15 * 15 * 3, 15 * 15 * 3)
        self.fc3 = nn.Linear(15 * 15 * 3, 15 * 15 * 3)
        self.fc4 = nn.Linear(15 * 15 * 3, 30 * 30 * 3)
        self.dropout = nn.Dropout(dropout)
        self.random_type = random_type

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

    def forward(self, std, mean, batch=1):

        if self.random_type == "uniform":
            noise = RANDOM_FUNCS[self.random_type](batch, 15 * 15 * 3).to(self.device) * std
        elif self.random_type == "normal":
            noise = RANDOM_FUNCS[self.random_type](mean, std, (batch, 15 * 15 * 3)).to(self.device)

        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = self.fc2(inp)
        inp = self.dropout(inp)
        inp = self.fc3(inp)
        inp = self.fc4(inp)
        inp = self.dropout(inp)

        inp = inp.view((batch, 3, 30, 30))
        for i in range(len(self.hyper_deconv)):
            inp = self.hyper_deconv[i](inp)

        inp = self.dropout(inp)
        inp = self.final(F.relu(inp.view(batch, -1)))
        return inp.view(batch, -1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class HyperNetworkD(nn.Module):

    def __init__(self, device, size_tot, dropout, random_type = "normal"):

        super(HyperNetworkD, self).__init__()
        self.device = device
        self.hyper_deconv = nn.ModuleList()
        self.hyper_deconv.append(nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1))
        self.fc1 = nn.Linear(15 * 15 * 3, 15 * 15 * 3)
        self.fc2 = nn.Linear(15 * 15 * 3, 7 * 7 * 3)
        self.fc3 = nn.Linear(7 * 7 * 3, 15 * 15 * 3)
        self.fc4 = nn.Linear(15 * 15 * 3, 7 * 7 * 3)
        self.fc5 = nn.Linear(7 * 7 * 3, 50 * 50 * 3)
        self.dropout = nn.Dropout(dropout)
        self.random_type = random_type

        curr_shape = (50,50)
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

    def forward(self, std, mean, batch=1):

        if self.random_type == "uniform":
            noise = RANDOM_FUNCS[self.random_type](batch, 15 * 15 * 3).to(self.device) * std
        elif self.random_type == "normal":
            noise = RANDOM_FUNCS[self.random_type](mean, std, (batch, 15 * 15 * 3)).to(self.device)

        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = self.fc2(inp)
        inp = self.dropout(inp)
        inp = self.fc3(inp)
        inp = self.fc4(inp)
        inp = self.fc5(inp)
        inp = inp.view((batch, 3, 50, 50))
        inp = self.dropout(inp)

        for i in range(len(self.hyper_deconv)):
            inp = self.hyper_deconv[i](inp)

        inp = self.dropout(inp)
        inp = self.final(F.relu(inp.view(batch, -1)))
        return inp.view(batch, -1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def get_hyper(device, size_tot, dropout, random_type="normal", architecture = "A"):

    if architecture == "B":
        return HyperNetworkB(device, size_tot, dropout, random_type=random_type)
    elif architecture == "C":
        return HyperNetworkC(device, size_tot, dropout, random_type=random_type)
    elif architecture == "D":
        return HyperNetworkD(device, size_tot, dropout, random_type=random_type)
    else:
        return HyperNetworkA(device, size_tot, dropout, random_type=random_type)