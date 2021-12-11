import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=1, mesh_size=(64, 64, 64)):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * mesh_size[2], mesh_size[0]/2 * mesh_size[1]/2 * mesh_size[2]/2, bias=False)

        self.fc1 = nn.Linear(mesh_size[0]/2 * mesh_size[1]/2 * mesh_size[2]/2, mesh_size[0]/4 * mesh_size[1]/4 * mesh_size[2]/4, bias=False)

        self.fc2 = nn.Linear(mesh_size[0]/4 * mesh_size[1]/4 * mesh_size[2]/4, mesh_size[0]/4 * mesh_size[1]/4 * 3, bias=False)

        self.fc3 = nn.Linear(mesh_size[0]/4 * mesh_size[1]/4 * 3, mesh_size[0]/2 * mesh_size[1]/2 * 2, bias=False)

        self.fc4 = nn.Linear(mesh_size[0]/2 * mesh_size[1]/2 * 2, 64**2, bias=False)
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

