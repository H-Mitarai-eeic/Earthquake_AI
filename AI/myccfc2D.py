import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=2, mesh_size=(64, 64)):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size
        self.in_channels = in_channels
        #self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]+1, mesh_size[0]+1), padding=(int(mesh_size[1]/2), int(mesh_size[0]/2)), bias=False)

        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * in_channels, 64*64, bias=False)



    def forward(self, x):
        #print("x:",x.size())
        
        h = x

        #h = self.conv0(h)
        #h = self.conv1(h)

        h = self.relu0(self.conv0(h))
        h = self.relu1(self.conv1(h))

        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.in_channels)
        h = self.fc0(h)

        h = h.view(len(x), self.mesh_size[1], self.mesh_size[0])

        return h