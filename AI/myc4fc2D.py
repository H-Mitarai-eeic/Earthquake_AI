import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=2, mesh_size=(64, 64), ratio=0.5, dropout_flag=False, activation_flag=False):
        super(MYFCN, self).__init__()
        self.mesh_size = mesh_size
        self.in_channels = in_channels
        self.ratio = ratio
        self.dropout_flag = dropout_flag
        self.activation_flag = activation_flag
        #self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]+1, mesh_size[0]+1), padding=(int(mesh_size[1]/2), int(mesh_size[0]/2)), bias=False)

        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.dropout0 = nn.Dropout2d(p=ratio)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout2d(p=ratio)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout2d(p=ratio)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(mesh_size[1]*2 + 1, mesh_size[0]*2 + 1), padding=(int(mesh_size[1]), int(mesh_size[0])), bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout2d(p=ratio)

        self.fc0 = nn.Linear(mesh_size[0] * mesh_size[1] * in_channels, 64*64, bias=False)



    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        if self.activation_flag == False:
            h = self.conv0(h)
            if self.dropout_flag == True:
                h = self.dropout0(h) / (1 - self.ratio)
            h = self.conv1(h)
            if self.dropout_flag == True:
                h = self.dropout1(h) / (1 - self.ratio)
            h = self.conv2(h)
            if self.dropout_flag == True:
                h = self.dropout2(h) / (1 - self.ratio)
            h = self.conv3(h)
            if self.dropout_flag == True:
                h = self.dropout3(h) / (1 - self.ratio)



        elif self.activation_flag == True:
            h = self.relu0(self.conv0(h))
            if self.dropout_flag == True:
                h = self.dropout0(h) / (1 - self.ratio)
            h = self.relu1(self.conv1(h))
            if self.dropout_flag == True:
                h = self.dropout1(h) / (1 - self.ratio)
            h = self.relu2(self.conv2(h))
            if self.dropout_flag == True:
                h = self.dropout2(h) / (1 - self.ratio)
            h = self.relu3(self.conv3(h))
            if self.dropout_flag == True:
                h = self.dropout3(h) / (1 - self.ratio)

        
        h = h.view(-1, self.mesh_size[0] * self.mesh_size[1] * self.in_channels)
        h = self.fc0(h)

        h = h.view(len(x), self.mesh_size[1], self.mesh_size[0])

        return h