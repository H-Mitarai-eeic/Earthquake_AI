import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=1, mesh_size=(64, 64, 64)):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size

        self.conv0 = nn.Conv3d(in_channels, 1, kernel_size=(mesh_size[2]+1, mesh_size[1]+1, mesh_size[0]+1), padding=(int(mesh_size[2]/2), int(mesh_size[1]/2), int(mesh_size[0]/2)), bias=True)

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * mesh_size[2], 64*64, bias=True)

        #self.fc1 = nn.Linear(256, 64**2, bias=True)
        #self.relu0 = nn.ReLU6(inplace=True)


    def forward(self, x):
        #print("x:",x.size())
        
        h = x

        h = self.conv0(h)

        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.mesh_size[2])
        h = self.fc0(h)

        h = h.view(len(x), self.mesh_size[0], self.mesh_size[1])

        return h