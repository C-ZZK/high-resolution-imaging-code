import torch.nn as nn
import torch


class resblock(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  #3, 5, 7
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )  #conv1

        self.act = nn.ReLU(inplace=True)
        # self.shotcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x1 = self.conv(x)
        # x = self.shotcut(x)
        out = x1+x
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),  #3, 5, 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  #3, 5, 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res1 = resblock(128, 128)
        self.res2 = resblock(128, 128)
        self.res3 = resblock(128, 128)
        self.res4 = resblock(128, 128)
        self.conv3 = nn.Conv2d(128, out_ch, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.res1(x1)
        x1 = self.res2(x1)
        x1 = self.res3(x1)
        x1 = self.res4(x1)
        x1 = self.conv3(x1)
        out = x1
        return out
