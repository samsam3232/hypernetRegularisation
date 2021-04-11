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
            self.reslayer = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
        else:
            self.reslayer = IdentityLayer()

        self.weight_init()

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

    def weight_init(self):

        for m in self.modules():
            if (isinstance(m, nn.ConvTranspose2d)) or (isinstance(m, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



# class ResnetBlock(nn.Module):
#
#     def __init__(self, in_planes=16, planes=16, stride = 1):
#
#         super(ResnetBlock,self).__init__()
#
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#
#         if stride == 2:
#             self.stride1 = 2
#             self.reslayer = nn.Conv2d(in_channels=self.in_planes, out_channels=self.planes, stride=2, kernel_size=1, bias=False)
#         else:
#             self.stride1 = 1
#             self.reslayer = IdentityLayer()
#
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.weight_init()
#
#
#     def forward(self, x, conv1_w, conv2_w, regularize):
#
#         residual = self.reslayer(x)
#         if not regularize:
#             out = F.relu(self.bn1(self.conv1(x)))
#             out = self.bn2(self.conv2(out))
#         else:
#             out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
#             out = self.bn2(F.conv2d(out, conv2_w, padding=1))
#
#         out += residual
#
#         out = F.relu(out)
#
#         return out
#
#     def weight_init(self):
#
#         for m in self.modules():
#             if (isinstance(m, nn.ConvTranspose2d)) or (isinstance(m, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def regularize(self):
#
#         self.conv1, self.conv2 = None, None