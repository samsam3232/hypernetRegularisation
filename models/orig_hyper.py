import torch.nn as nn
import torch


class ElementWiseLayer(nn.Module):

    def __init__(self, dim, device):
        super(ElementWiseLayer, self).__init__()
        self.vec = torch.randn(dim).to(device)

    def forward(self, input):
        return torch.mul(self.vec, input)


class HyperNetwork(nn.Module):

    def __init__(self, device, dropout):

        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(15 * 15 * 3, 30 * 30 * 3)
        self.device = device
        self.dc1 = nn.ConvTranspose2d(3, 7, 3, stride=2, padding=1)
        self.dc2 = nn.ConvTranspose2d(7, 7, 3, stride=2, padding=1)
        self.dc3 = nn.ConvTranspose2d(7, 14, 1, stride=1, padding=1)
        self.dc4 = nn.ConvTranspose2d(14, 23, 1, stride=1, padding=1)
        self.neg = ElementWiseLayer(293687, device)
        self.dropout = nn.Dropout(dropout)
        self.weight_init()

    def forward(self, std):

        noise = torch.rand(15 * 15 * 3).to(self.device) * std
        noise.requires_grad = True
        inp = self.fc1(noise)
        inp = inp.view((-1, 3, 30, 30))
        inp = self.dropout(inp)
        inp = self.dc1(inp)
        inp = self.dc2(inp)
        inp = self.dropout(inp)
        inp = self.dc3(inp)
        output = self.dc4(inp)
#        output = self.neg(F.relu(output.view(-1)))
        return output.view(-1)

    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def set_dropout(self, new_dropout):

        self.dropout = nn.Dropout(new_dropout)