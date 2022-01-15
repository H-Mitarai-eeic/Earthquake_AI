import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn
len_data = 64

class Linear(nn.Module):

    def __init__(self, n_class=21):
        super(Linear, self).__init__()
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear = nn.Linear(2*len_data*len_data, n_class*len_data*len_data)

    def forward(self, x):
        h = x
        batch_size = len(x)
        # h = self.pool1(h)
        # h = self.pool1(h)
        # h = h.view(-1, int(2*len_data*len_data/16))
        h = h.view(-1, 2*len_data*len_data)
        h = self.linear(h)
        h = h.view(batch_size, 10, len_data, len_data)
        return h