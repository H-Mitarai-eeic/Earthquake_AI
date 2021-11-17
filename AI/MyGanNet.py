import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn


class MYFCN4gan(nn.Module):
    def __init__(self, in_channels=2, n_class=1):
        super(MYFCN4gan, self).__init__()
        # convTranspose1
        """
            self.conv1 = nn.Conv2d(in_channels, 8, 13, padding=0, stride=3)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(8, 32, 10, padding=0, stride=1)
            self.relu2 = nn.ReLU(inplace=True)

            #self.linear3 = nn.Linear(1296, 1296)

            # conv2
            self.convTrans4 = nn.ConvTranspose2d(32, 16, 10, stride=2)
            self.relu4 = nn.ReLU(inplace=True)

            self.convTrans5 = nn.ConvTranspose2d(16, n_class, 14, padding=0, stride=2)
            self.relu5 = nn.ReLU(inplace=True)
        """

        """
            self.conv1 = nn.Conv2d(in_channels, 8, 10, padding=0, stride=3)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(8, 64, 7, padding=0, stride=6)
            self.relu2 = nn.ReLU(inplace=True)

            self.linear1 = nn.Linear(576, 256)
            self.relu3 = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(256, 128)
            self.relu4 = nn.ReLU(inplace=True)

            # conv2
            self.convTrans1 = nn.ConvTranspose2d(2, 1, 8, stride=8)
            self.relu5 = nn.ReLU(inplace=True)
        """
        self.in_channels = in_channels
        self.n_class = n_class

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.linear1 = nn.Linear(16*16*self.in_channels, 32*32)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(32*32, 64*64*self.n_class)
        self.relu2 = nn.ReLU(inplace=True)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        """
            h = self.relu1(self.conv1(h))
            h = self.relu2(self.conv2(h))

            h = h.view(-1, 576)
            h = self.relu3(self.linear1(h))
            h = self.relu4(self.linear2(h))
            h = h.view(len(x), 2, 8, 8)

            h = self.relu5(self.convTrans1(h))
        """
        h = self.pool1(h)
        h = self.pool2(h)
        h = h.view(-1, 16*16*self.in_channels)
        h = self.relu1(self.linear1(h))
        h = self.relu2(self.linear2(h))
        h = h.view(len(x), 1, 64, 64)
        return h


class MyDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(MyDiscriminator, self).__init__()
        #self.ngpu = ngpu
        #self.nc = 1
        #self.ndf = 64
        self.in_channels = in_channels
        """
            self.main = nn.Sequential(

                #nc = ndf = 1
                # input is (nc) x 64 x 64
                nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
       
        )
        """
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear1 = nn.Linear(16*16*self.in_channels, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        h = x
        #print("h: ", h.size())
        h = self.pool1(h)
        #print("h: ", h.size())
        h = self.pool2(h)
        #print("h: ", h.size())
        h = h.view(-1, 16*16*self.in_channels)
        #print("h: ", h.size())
        h = self.relu1(self.linear1(h))
        #print("h: ", h.size())
        h = self.linear2(h)
        #print("h: ", h.size())
        h = h.view(len(x), 1, 1)
        h = torch.sigmoid(h)

        return h