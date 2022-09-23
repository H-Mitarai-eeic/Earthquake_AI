import os.path as osp

import numpy as np
import torch
import torch.nn as nn

len_data = 64


class Linear(nn.Module):
    def __init__(self, n_class=10, dim=2):
        super(Linear, self).__init__()
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear = nn.Linear(
            dim * len_data * len_data, n_class * len_data * len_data
        )
        self.n_class = n_class
        self.dim = dim

    def forward(self, x):
        h = x
        batch_size = len(x)
        # h = self.pool1(h)
        # h = self.pool1(h)
        # h = h.view(-1, int(2*len_data*len_data/16))
        h = h.view(-1, self.dim * len_data * len_data)
        h = self.linear(h)
        h = h.view(batch_size, self.n_class, len_data, len_data)
        return h
