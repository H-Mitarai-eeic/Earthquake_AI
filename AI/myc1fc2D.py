import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=2, mesh_size=(64, 64), ratio=0.5, dropout_flag=False, activation_flag=False, kernel_size=129):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size
        self.in_channels = in_channels
        self.ratio = ratio
        self.dropout_flag = dropout_flag
        self.activation_flag = activation_flag
        self.kernel_size=kernel_size
        #self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]+1, mesh_size[0]+1), padding=(int(mesh_size[1]/2), int(mesh_size[0]/2)), bias=False)

        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=int((kernel_size - 1)/2), bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.dropout0 = nn.Dropout2d(p=ratio)

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * in_channels, 64*64, bias=False)



    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        if self.activation_flag == False:
            h = self.conv0(h)
            if self.dropout_flag == True:
                h = self.dropout0(h) / (1 - self.ratio)

        elif self.activation_flag == True:
            h = self.relu0(self.conv0(h))
            if self.dropout_flag == True:
                h = self.dropout0(h) / (1 - self.ratio)

        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.in_channels)
        h = self.fc0(h)

        h = h.view(len(x), self.mesh_size[1], self.mesh_size[0])

        return h