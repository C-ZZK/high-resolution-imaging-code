import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import time
from model.unetm import UNet
from model.resnet import ResNet
from model.srresnet import NetG
from model.DnCNN import DnCNN
import torch.nn.functional as F
from utils.mmd import MMDLoss



def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def total_variation_loss(image):
    # ---------------------------------------------------------------
    # shift one pixel and get difference (for both x and y direction)
    # ---------------------------------------------------------------
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

    return loss


class InpTrainer(nn.Module):
    def __init__(self, opt):
        super(InpTrainer, self).__init__()
        self.net = nn.DataParallel(UNet(num_input_channels=1, num_output_channels=1,
                       feature_scale=1, upsample_mode='bilinear',
                                        norm_layer=nn.BatchNorm2d, need_sigmoid=False,need_relu=False)).cuda()
        # self.net = nn.DataParallel(ResNet(2, 1)).cuda()
        # self.net = nn.DataParallel(NetG()).cuda()
        # self.net = nn.DataParallel(DnCNN(image_channels=1)).cuda()

        self.opt = opt
        self.low = nn.AvgPool2d(6,1)
        self.lossTV = TVLoss()
        self.lossL1 = nn.L1Loss()
        self.lossMSE = nn.MSELoss()
        self.relu = nn.ReLU()
        self.lossmmd = MMDLoss()

        # self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
        # print(self.net)


    def train_onebatch2(self, data,ref,target):
        self.lr_scheduler.step()
        outimg = self.net(data)
        cropsize = 128
        batch_size = 20
      #经过特征变化后的靶区数据
        target1 = self.net(target)
        target1 = target1.view(batch_size,cropsize*cropsize)
        target2 = target.view(batch_size,cropsize*cropsize)
        data_out = outimg.view(batch_size,cropsize*cropsize)
        
        loss1 = self.lossMSE(outimg, ref)
        loss2 = self.lossmmd(data_out,target1)
        loss3 = self.lossTV(outimg)
        
        l1_regularization = 0
        # l1_regularization L1约束
        for param in self.net.parameters():
              l1_regularization += torch.sum(torch.abs(param))
        loss = loss1 + 0.0001*loss2+ 0*loss3

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test_onebatch(self, data):
        output = self.net(data)
        return output

    def test_onebatch2(self, image, data2, vmax1, vmean1, vmax2, vmean2):
        output = self.net(image, data2)

        outlow = self.pad(output)
        outlow = self.low(outlow)
        aa = torch.log(self.relu(output * vmax1 + vmean1) + 0.001) / 2
        outhigh = aa[:, :, 1:, :] - aa[:, :, :-1, :]
        outhigh = (outhigh - vmean2) / vmax2
        lossL = self.lossMSE(outlow, image)
        lossH = self.lossMSE(outhigh, data2[:, :, :-1, :])
        loss = lossL + lossH

        output = output.detach().cpu().numpy()

        return output, loss.detach().cpu().numpy()


    def save_net(self, save_path=None, **kwargs):
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = './checkpointsfor_mar/Unet_'# + '_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path = save_path + ".pth"
        torch.save(self.net.state_dict(), save_path)
        return save_path
        


    def load_net(self, save_path):
        state_dict = torch.load(save_path)
        self.net.load_state_dict(state_dict)
        return self


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
