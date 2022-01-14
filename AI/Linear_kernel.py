from audioop import bias
import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn
len_data = 64
in_channels = 2


class Linear(nn.Module):

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
        super(Linear, self).__init__()
        self.linear = nn.Linear(2*len_data*len_data, n_class*len_data*len_data)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(len_data*2 + 1, len_data*2 + 1), padding=(int(len_data), int(len_data)), bias=False)



    def forward(self, x):
        h = x
        h = self.conv(h)
        batch_size = len(x)
        h = h.view(-1, 2*len_data*len_data)
        h = self.linear(h)
        h = h.view(batch_size, 10, len_data, len_data)
        return h