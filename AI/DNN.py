import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
len_data = 64
one_pool_datasize = int(2*len_data*len_data/4)
two_pool_datasize = int(2*len_data*len_data/16)

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


class DNN(nn.Module):

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
        super(DNN, self).__init__()
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.layer1 = nn.Linear(
            2*len_data*len_data, 4096)
        self.layer2 = nn.Linear(4096,
                                n_class*len_data*len_data)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
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
        batch_size = len(x)
        # h = self.pool(h)
        # h = self.pool(h)
        h = h.view(-1, 2*len_data*len_data)
        h = self.relu(self.layer1(h))
        h = self.layer2(h)
        h = h.view(batch_size, 10, len_data, len_data)
        return h