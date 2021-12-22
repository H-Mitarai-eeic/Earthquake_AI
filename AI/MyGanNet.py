import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn


class MYFCN4gan(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mesh_size=(64, 64, 10)):
        super(MYFCN4gan, self).__init__()
        # convTranspose1
        self.mesh_size = mesh_size
        self.in_channels = in_channels

        self.fc0 = nn.Linear(in_channels*mesh_size[0]*mesh_size[1]*mesh_size[2], out_channels*mesh_size[0]*mesh_size[1])

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        h = h.view(len(x), self.in_channels*self.mesh_size[0]*self.mesh_size[1]*self.mesh_size[2])
        h = self.fc0(h)
        h = h.view(len(x), 1, 1, 64, 64)
        return h


class MyDiscriminator(nn.Module):
    def __init__(self, in_channels=1, mesh_size=(64, 64, 10)):
        super(MyDiscriminator, self).__init__()

        self.mesh_size = mesh_size

        self.fc0 = nn.Linear(mesh_size[0]*mesh_size[1]*(mesh_size[2] + 1), 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        
            h = x
            h = h.view(len(x), self.mesh_size[0]*self.mesh_size[1]*(self.mesh_size[2] + 1))
            h = self.sigmoid(self.fc0(h))

            return h