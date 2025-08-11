import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_input_channels=1, num_output_channels=1,
                 feature_scale=1, upsample_mode='bilinear', norm_layer=nn.BatchNorm2d,
                 need_sigmoid=False, need_relu=False):
        super(UNet, self).__init__()

        self.need_sigmoid = need_sigmoid
        self.need_relu = need_relu

        # Encoder
        self.conv1a = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5a = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6a = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv7a = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.final_conv = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

        # Normalization layers
        self.norm1 = norm_layer(64)
        self.norm2 = norm_layer(128)
        self.norm3 = norm_layer(256)
        self.norm4 = norm_layer(256)
        self.norm5 = norm_layer(256)
        self.norm6 = norm_layer(128)
        self.norm7 = norm_layer(64)

    def forward(self, x):
        # Encoder
        x1 = F.tanh(self.norm1(self.conv1a(x)))
        x1 = F.tanh(self.norm1(self.conv1b(x1)))
        x1_pool = self.pool1(x1)

        x2 = F.tanh(self.norm2(self.conv2a(x1_pool)))
        x2 = F.tanh(self.norm2(self.conv2b(x2)))
        x2_pool = self.pool2(x2)

        x3 = F.tanh(self.norm3(self.conv3a(x2_pool)))
        x3 = F.tanh(self.norm3(self.conv3b(x3)))
        x3_pool = self.pool3(x3)

        x4 = F.tanh(self.norm4(self.conv4a(x3_pool)))
        x4 = F.tanh(self.norm4(self.conv4b(x4)))

        # Decoder
        x5 = self.upconv1(x4)
        x5 = torch.cat((x5, x3), dim=1)
        x5 = F.tanh(self.norm5(self.conv5a(x5)))
        x5 = F.tanh(self.norm5(self.conv5b(x5)))

        x6 = self.upconv2(x5)
        x6 = torch.cat((x6, x2), dim=1)
        x6 = F.tanh(self.norm6(self.conv6a(x6)))
        x6 = F.tanh(self.norm6(self.conv6b(x6)))

        x7 = self.upconv3(x6)
        x7 = torch.cat((x7, x1), dim=1)
        x7 = F.tanh(self.norm7(self.conv7a(x7)))
        x7 = F.tanh(self.norm7(self.conv7b(x7)))

        output = self.final_conv(x7)
        # if self.need_sigmoid:
        #     output = torch.sigmoid(output)
        # elif self.need_relu:
        #     output = F.relu(output)
        return output
