import torch.nn as nn
import torch


class resblock(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(resblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.shotcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.shotcut(x)
        out = x1+x
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn1.parameters():
                i.requires_grad = False
        padding = 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, affine=True)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_of_layers=8):
        super(ResNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_ch, out_channels=64,
                                kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-1):
            layers.append(Bottleneck(64, 64))
        layers.append(nn.Conv2d(in_channels=64, out_channels=out_ch,
                             kernel_size=1, padding=0,
                             bias=False))
        self.resnet = nn.Sequential(*layers)


    def forward(self, x):
        out = self.resnet(x)
        return out
