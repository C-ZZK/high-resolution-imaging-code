import torch
import torch.nn as nn
import math


class _Residual_Block(nn.Module):
    def __init__(self,c):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm2d(c)
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(c)
        # self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()

        self.conv_input1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_input2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.residual1 = self.make_layer(_Residual_Block, 64, 4)
        self.residual2 = self.make_layer(_Residual_Block, 64, 4)
        self.residual = self.make_layer(_Residual_Block, 128, 16)

        self.conv_mid = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(128)
        # self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        # self.upscale4x = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PixelShuffle(2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PixelShuffle(2),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        self.conv_output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, c, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(c))
        return nn.Sequential(*layers)

    def forward(self, x1,x2):
        out1 = self.relu1(self.conv_input1(x1))
        out1 = self.residual1(out1)
        out2 = self.relu2(self.conv_input2(x2))
        out2 = self.residual2(out2)
        out = torch.cat([out1,out2],dim=1)
        # residual = out
        out = self.residual(out)
        # out = self.conv_mid(out) ###
        out = self.bn_mid(self.conv_mid(out)) ###
        # out = torch.add(out, residual)
        # out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

