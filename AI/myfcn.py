import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=1, mesh_size=(64, 64, 64)):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size
        
        self.conv1 = nn.Conv3d(in_channels, 1, 5, stride=1,padding=2,bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(1, 1, 5, stride=1,padding=2,bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(1, 1, 5, stride=1,padding=2,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv3d(1, 1, 5, stride=1,padding=2,bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv3d(1, 1, 5, stride=1,padding=2,bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * mesh_size[2], 64**2, bias=False)
        #self.relu0 = nn.ReLU6(inplace=True)


    def forward(self, x):
        #print("x:",x.size())
        
        h = x

        h = self.relu1(self.conv1(h))
        h = self.relu2(self.conv2(h))
        h = self.relu3(self.conv3(h))
        h = self.relu4(self.conv4(h))
        h = self.relu5(self.conv5(h))

        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.mesh_size[2])
        
        h = self.fc0(h)
        h = h.view(len(x), self.mesh_size[0], self.mesh_size[1])

        return h

