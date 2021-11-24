import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class MYFCN(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=11k2Q0bvRQgQbT6-jYWeh6nmAsWlSCY3f',  # NOQA
            path=cls.pretrained_model,
            md5='d3eb467a80e7da0468a20dfcbc13e6c8',
        )

    def __init__(self, in_channels=2, n_class=21):
        super(MYFCN, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=2)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        # self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        # self.relu4_1 = nn.ReLU(inplace=True)
        # self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_2 = nn.ReLU(inplace=True)
        # self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu4_3 = nn.ReLU(inplace=True)
        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_2 = nn.ReLU(inplace=True)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.relu5_3 = nn.ReLU(inplace=True)
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        # self.fc6 = nn.Conv2d(256, 512, 7)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(256, 256, 2)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.2)

        self.score_fr = nn.Conv2d(256, n_class, 1)
        self.upscore = nn.ConvTranspose2d(256, n_class, 8, stride=8, bias=False) #Hout = (Hin - 1)*stride - 2*padding + kernel + outputpadding n_class to 256

        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        # print("starting_forward:", h.size())
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        # print("after conv1:", h.size())

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        # print("after conv2:", h.size())

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        # print("after conv3:", h.size())

        # h = self.relu4_1(self.conv4_1(h))
        # h = self.relu4_2(self.conv4_2(h))
        # h = self.relu4_3(self.conv4_3(h))
        # h = self.pool4(h)
        # print("after conv4:", h.size())

        # h = self.relu5_1(self.conv5_1(h))
        # h = self.relu5_2(self.conv5_2(h))
        # h = self.relu5_3(self.conv5_3(h))
        # h = self.pool5(h)
        # print("after conv5:", h.size())

        # h = self.relu6(self.fc6(h))
        # h = self.drop6(h)
        # print("after layr6:", h.size())

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        # print("after layr7:", h.size())
    
        # h = self.score_fr(h)
        # print("after sc_fr:", h.size())

        h = self.upscore(h)
        # print("after trans:", h.size())
        # h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            # self.conv4_1, self.relu4_1,
            # self.conv4_2, self.relu4_2,
            # self.conv4_3, self.relu4_3,
            # self.pool4,
            # self.conv5_1, self.relu5_1,
            # self.conv5_2, self.relu5_2,
            # self.conv5_3, self.relu5_3,
            # self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

class MYFCN2(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=11k2Q0bvRQgQbT6-jYWeh6nmAsWlSCY3f',  # NOQA
            path=cls.pretrained_model,
            md5='d3eb467a80e7da0468a20dfcbc13e6c8',
        )

    def __init__(self, in_channels=2, n_class=10):
        super(MYFCN2, self).__init__()
        # convTranspose1
        self.convTrans1 = nn.ConvTranspose2d(in_channels, 4, 13, padding=100, stride=5)
        self.relu1 = nn.ReLU(inplace=True)

        self.convTrans2 = nn.ConvTranspose2d(4, 8, 10, padding=0, stride=1)
        self.relu2 = nn.ReLU(inplace=True)

        #self.linear3 = nn.Linear(8, 8)

        # conv2
        self.conv4 = nn.Conv2d(8, 16, 10, padding=0, stride=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(16, 10, 65, padding=0, stride=1)
        self.relu5 = nn.ReLU(inplace=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        h = self.relu1(self.convTrans1(h))
        h = self.relu2(self.convTrans2(h))

        #h = self.linear3(h)

        h = self.relu4(self.conv4(h))
        h = self.relu5(self.conv5(h))

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            # self.conv4_1, self.relu4_1,
            # self.conv4_2, self.relu4_2,
            # self.conv4_3, self.relu4_3,
            # self.pool4,
            # self.conv5_1, self.relu5_1,
            # self.conv5_2, self.relu5_2,
            # self.conv5_3, self.relu5_3,
            # self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

class MYFCN3(nn.Module):
    def __init__(self, in_channels=2,n_class=10):
        super(MYFCN3, self).__init__()
        # convTranspose1
        self.convtrans1 = nn.ConvTranspose2d(in_channels, in_channels, 63, padding=31, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.convtrans2 = nn.ConvTranspose2d(in_channels, 4, 7, padding=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.convtrans3 = nn.ConvTranspose2d(4, 6, 7, padding=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.convtrans4 = nn.ConvTranspose2d(6, 10, 7, padding=3, stride=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.convtrans5 = nn.ConvTranspose2d(10, 16, 7, padding=3, stride=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.convtrans6 = nn.ConvTranspose2d(16, 26, 7, padding=3, stride=1)
        self.relu6 = nn.ReLU(inplace=True)

        self.convtrans7 = nn.ConvTranspose2d(26, 42, 7, padding=3, stride=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.convtrans8 = nn.ConvTranspose2d(42, 68, 7, padding=3, stride=1)
        self.relu8 = nn.ReLU(inplace=True)

        self.convtrans9 = nn.ConvTranspose2d(68, 42, 7, padding=3, stride=1)
        self.relu9 = nn.ReLU(inplace=True)

        self.convtrans10 = nn.ConvTranspose2d(42, 16, 7, padding=3, stride=1)
        self.relu10 = nn.ReLU(inplace=True)

        self.convtrans11 = nn.ConvTranspose2d(16, n_class, 7, padding=3, stride=1)
        self.tanh11 = nn.Tanh()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        h = self.relu1(self.convtrans1(h))
        h = self.relu2(self.convtrans2(h))
        h = self.relu3(self.convtrans3(h))
        h = self.relu4(self.convtrans4(h))
        h = self.relu5(self.convtrans5(h))
        h = self.relu6(self.convtrans6(h))
        h = self.relu7(self.convtrans7(h))
        h = self.relu8(self.convtrans8(h))

        h = self.relu9(self.convtrans9(h))
        h = self.relu10(self.convtrans10(h))
        h = self.tanh11(self.convtrans11(h))

        return h


class MYFCN4(nn.Module):
    def __init__(self, in_channels=2,n_class=10):
        super(MYFCN4, self).__init__()
        # convTranspose1
        self.convtrans1 = nn.ConvTranspose2d(in_channels, in_channels, 63, padding=31, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.convtrans2 = nn.ConvTranspose2d(in_channels, 4, 7, padding=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.convtrans3 = nn.ConvTranspose2d(4, 6, 7, padding=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.convtrans4 = nn.ConvTranspose2d(6, 10, 7, padding=3, stride=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.convtrans5 = nn.ConvTranspose2d(10, 16, 7, padding=3, stride=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.convtrans6 = nn.ConvTranspose2d(16, 26, 7, padding=3, stride=1)
        self.relu6 = nn.ReLU(inplace=True)

        self.convtrans7 = nn.ConvTranspose2d(26, 42, 7, padding=3, stride=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.convtrans8 = nn.ConvTranspose2d(42, 68, 7, padding=3, stride=1)
        self.relu8 = nn.ReLU(inplace=True)

        self.convtrans9 = nn.ConvTranspose2d(68, 42, 7, padding=3, stride=1)
        self.relu9 = nn.ReLU(inplace=True)

        self.convtrans10 = nn.ConvTranspose2d(42, 16, 7, padding=3, stride=1)
        self.relu10 = nn.ReLU(inplace=True)

        self.convtrans11 = nn.ConvTranspose2d(16, in_channels, 7, padding=3, stride=1)
        self.sig11 = nn.Sigmoid()

        ##=======fcn========##
        self.conv12 = nn.Conv2d(in_channels + 1, 64, 3, padding=1, stride=2)    #1/2
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(64, 128, 3, padding=1, stride=2)    #1/4
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(128, 256, 3, padding=1, stride=2)   #1/8
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(256, 512, 3, padding=1, stride=2)   #1/16
        self.relu15 = nn.ReLU(inplace=True)

        self.convtrans16 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)   #1/8
        self.relu16 = nn.ReLU(inplace=True)

        self.convtrans17 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1)   #1/4
        self.relu17 = nn.ReLU(inplace=True)

        self.convtrans18 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)   #1/2
        self.relu18 = nn.ReLU(inplace=True)

        self.convtrans19 = nn.ConvTranspose2d(64, n_class, 3, padding=1, stride=2, output_padding=1)   #1/2
        self.tanh19 = nn.Tanh()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, mask):
        #print("x:",x.size())
        
        h = x
        h = self.relu1(self.convtrans1(h))
        h = self.relu2(self.convtrans2(h))
        h = self.relu3(self.convtrans3(h))
        h = self.relu4(self.convtrans4(h))
        h = self.relu5(self.convtrans5(h))
        h = self.relu6(self.convtrans6(h))
        h = self.relu7(self.convtrans7(h))
        h = self.relu8(self.convtrans8(h))

        h = self.relu9(self.convtrans9(h))
        h = self.relu10(self.convtrans10(h))
        h = self.sig11(self.convtrans11(h))

        ##======== cat mask =========##

        h = torch.cat((h, mask), dim=1)

        ##===========fcn=============##
        h = self.relu12(self.conv12(h))
        h = self.relu13(self.conv13(h))
        h = self.relu14(self.conv14(h))
        h = self.relu15(self.conv15(h))

        h = self.relu16(self.convtrans16(h))
        h = self.relu17(self.convtrans17(h))
        h = self.relu18(self.convtrans18(h))
        h = self.tanh19(self.convtrans19(h))


        return h

