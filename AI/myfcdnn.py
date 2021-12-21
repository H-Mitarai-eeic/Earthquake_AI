import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=1, mesh_size=(64, 64, 64)):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * mesh_size[2], 256, bias=True)

        self.fc1 = nn.Linear(256, 128, bias=True)

        self.fc2 = nn.Linear(128, 64, bias=True)

        self.fc3 = nn.Linear(64, 128, bias=True)

        self.fc4 = nn.Linear(128, 64**2, bias=False)
        #self.relu0 = nn.ReLU6(inplace=True)


    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.mesh_size[2])
        
        h = self.fc0(h)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.fc4(h)

        h = h.view(len(x), self.mesh_size[0], self.mesh_size[1])

        return h