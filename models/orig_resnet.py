import torch.nn as nn
import torch.nn.functional as F


class IdentityLayer(nn.Module):

    def forward(self, x):
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes=None, planes=None, stride=1, regularize=False):
        super(ResnetBlock, self).__init__()
        if not regularize:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1, self.conv2 = None, None

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride1 = stride

        self.reslayer = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        else:
            self.reslayer = IdentityLayer()

    def forward(self, x, conv1_w, conv2_w):

        residual = self.reslayer(x)
        if self.conv1 is not None:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
            out = self.bn2(F.conv2d(out, conv2_w, padding=1))
        out += residual

        out = F.relu(out)

        return out
