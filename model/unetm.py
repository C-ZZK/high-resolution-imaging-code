import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.io import savemat
from .common import *


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class UNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''

    def __init__(self, num_input_channels=3, num_output_channels=3,
                 feature_scale=4, more_layers=0, concat_x=False,
                 upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_relu=True, need_bias=True):
        super(UNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        filters = [64, 128, 256, 512]
        # filters = [64, 128, 256]
        # filters = [128, 128, 128, 128, 128]
        filters = [x // self.feature_scale for x in filters]

        # self.start1 = conv(num_input_channels, 64, 3, bias=need_bias, pad=1)
        # self.start2 = unetConv2(num_input_channels, 64, norm_layer, need_bias, pad)

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels,
                               norm_layer, need_bias, pad)

        self.convdata2 = unetConv2(64, 64,norm_layer, need_bias, pad)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer,
                              need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer,
                              need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[2] - num_input_channels, norm_layer,
                              need_bias, pad)


        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer,
                         need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt=True) for i in
                             range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)
        self.BatchNormal = nn.BatchNorm3d(64)
        self.lastconv = unetConv2(filters[0], 32, norm_layer, need_bias, pad)
        self.final = conv(64, num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        if need_relu:
            self.final = nn.Sequential(self.final, nn.ReLU())

        self.convdata1=unetConv2(num_input_channels, filters[0],
                               norm_layer, need_bias, pad)
        self.convdata2 = unetConv2(filters[0], filters[0],
                                   norm_layer, need_bias, pad)
        self.outdata1=conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

    def forward(self, data1):
        unloader = transforms.ToPILImage()

        in64 = self.start(data1)

        # a = in64[0,0]
        # image = a.cpu().clone()  # clone the tensor
        # image = unloader(image)
        # image.save('example.jpg')
        # b = data1[0]
        # image = b.cpu().clone()  # clone the tensor
        # image_b = unloader(image)
        # image_b.save('example_data.jpg')

        down1 = self.down1(in64)
        down2 = self.down2(down1)

        # c = down1[0,0]
        # image = c.cpu().clone()  # clone the tensor
        # image_c = unloader(image)
        # image_c.save('example_down1.jpg')
        # d = down2[0, 0]
        # image = d.cpu().clone()  # clone the tensor
        # image_d = unloader(image)
        # image_d.save('example_down2.jpg')
        down3 = self.down3(down2)

        # up_ = down2

        up_ = down3
        up3 = self.up3(up_, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)
        # up0 = self.lastconv(up1)

        result = self.final(up1)#+data1
        # result1 = result.view(20,128,128)
        # result1 = result1.cpu().detach().numpy()
        # print(np.max(result1))
        # result = self.BatchNormal(result1)
        # e = up2[0, 0]
        # image = e.cpu().clone()  # clone the tensor
        # image_e = unloader(image)
        # image_e.save('example_up2.jpg')
        # f = up1[0, 0]
        # image = f.cpu().clone()  # clone the tensor
        # image_f = unloader(image)
        # image_f.save('example_up1.jpg')
        # g = result[0, 0]
        # image = g.cpu().clone()  # clone the tensor
        # image_g = unloader(image)
        # image_g.save('example_result.jpg')
        # file_name = 'ref.mat'
        # # file_name1= 'data.mat'
        # # file_name2 ='kernel1.mat'
        # savemat(file_name, {'ref': data1})
        return result


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(), )
        else:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv = unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv = unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        unloader = transforms.ToPILImage()
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2
        a = torch.cat([in1_up, inputs2_], 1)
        # h = a[0, 0]
        # image = h.cpu().clone()  # clone the tensor
        # image_h = unloader(image)
        # image_h.save('example_cat.jpg')
        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output
