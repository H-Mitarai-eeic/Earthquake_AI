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

    def __init__(self, n_class=21):
        super(MYFCN2, self).__init__()
        # conv1 256*256
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2d(2, 8, 5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(8, 8, 5, padding=2)
        self.relu8 = nn.ReLU(inplace=True)
        # self.conv9 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu9 = nn.ReLU(inplace=True)
        # self.conv10 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu10 = nn.ReLU(inplace=True)
        # self.conv11 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu11 = nn.ReLU(inplace=True)
        # self.conv12 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu12 = nn.ReLU(inplace=True)
        # self.conv13 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu13 = nn.ReLU(inplace=True)
        # self.conv14 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu14 = nn.ReLU(inplace=True)
        # self.conv15 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu15 = nn.ReLU(inplace=True)
        # self.conv16 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu16 = nn.ReLU(inplace=True)
        # self.conv17 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu17 = nn.ReLU(inplace=True)
        # self.conv18 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu18 = nn.ReLU(inplace=True)
        # self.conv19 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu19 = nn.ReLU(inplace=True)
        # self.conv20 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu20 = nn.ReLU(inplace=True)
        # self.conv21 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu21 = nn.ReLU(inplace=True)
        # self.conv22 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu22 = nn.ReLU(inplace=True)
        # self.conv23 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu23 = nn.ReLU(inplace=True)
        # self.conv24 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu24 = nn.ReLU(inplace=True)
        # self.conv25 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu25 = nn.ReLU(inplace=True)
        # self.conv26 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu26 = nn.ReLU(inplace=True)
        # self.conv27 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu27 = nn.ReLU(inplace=True)
        # self.conv28 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu28 = nn.ReLU(inplace=True)
        # self.conv29 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu29 = nn.ReLU(inplace=True)
        # self.conv30 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu30 = nn.ReLU(inplace=True)
        # self.conv31 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu31 = nn.ReLU(inplace=True)
        # self.conv32 = nn.Conv2d(8, 8, 5, padding=2)
        # self.relu32 = nn.ReLU(inplace=True)
        

        # fc7
        # self.fc7 = nn.Conv2d(256, 256, 2)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d(p=0.2)

        self.score_fr = nn.Conv2d(8, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 4, stride=4, bias=False) #Hout = (Hin - 1)*stride - 2*padding + kernel + outputpadding n_class to 256

        self._initialize_weights()

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
        
        h = x
        h = self.pool1(h)
        h = self.relu1(self.conv1(h))
        h = self.relu2(self.conv2(h))
        h = self.pool1(h)
        h = self.relu3(self.conv3(h))
        h = self.relu4(self.conv4(h))
        h = self.relu5(self.conv5(h))
        h = self.relu6(self.conv6(h))
        h = self.relu7(self.conv7(h))
        h = self.relu8(self.conv8(h))
        # h = self.relu9(self.conv9(h))
        # h = self.relu10(self.conv10(h))
        # h = self.relu11(self.conv11(h))
        # h = self.relu12(self.conv12(h))
        # h = self.relu13(self.conv13(h))
        # h = self.relu14(self.conv14(h))
        # h = self.relu15(self.conv15(h))
        # h = self.relu16(self.conv16(h))
        # h = self.relu17(self.conv17(h))
        # h = self.relu18(self.conv18(h))
        # h = self.relu19(self.conv19(h))
        # h = self.relu20(self.conv20(h))
        # h = self.relu21(self.conv21(h))
        # h = self.relu22(self.conv22(h))
        # h = self.relu23(self.conv23(h))
        # h = self.relu24(self.conv24(h))
        # h = self.relu25(self.conv25(h))
        # h = self.relu26(self.conv26(h))
        # h = self.relu27(self.conv27(h))
        # h = self.relu28(self.conv28(h))
        # h = self.relu29(self.conv29(h))
        # h = self.relu30(self.conv30(h))
        # h = self.relu31(self.conv31(h))
        # h = self.relu32(self.conv32(h))
        h = self.score_fr(h)
        h = self.upscore(h)
        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            # self.conv1_1, self.relu1_1,
            # self.conv1_2, self.relu1_2,
            # self.pool1,
            # self.conv2_1, self.relu2_1,
            # self.conv2_2, self.relu2_2,
            # self.pool2,
            # self.conv3_1, self.relu3_1,
            # self.conv3_2, self.relu3_2,
            # self.conv3_3, self.relu3_3,
            # self.pool3,
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