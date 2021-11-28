import numpy as np
import torch
import torch.nn as nn


class MYFCN(nn.Module):
    def __init__(self, in_channels=3, mesh_size=64):
        super(MYFCN, self).__init__()

        self.fc0 = nn.Linear(in_channels, 64*64)

        self.fc1 = nn.Linear(in_channels, 16)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(16, 32)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(32, 64)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(64, 128)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc5 = nn.Linear(128, 256)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc6 = nn.Linear(256, 512)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc7 = nn.Linear(512, 1024)
        self.relu7 = nn.ReLU(inplace=True)

        self.fc8 = nn.Linear(1024, 2048)
        self.relu8 = nn.ReLU(inplace=True)

        self.fc9 = nn.Linear(2048, 4096)
        #self.relu9 = nn.ReLU(inplace=True)

    def forward(self, x):
        #print("x:",x.size())
        
        h = x
        """
        h = self.relu1(self.fc1(h))
        h = self.relu2(self.fc2(h))
        h = self.relu3(self.fc3(h))
        h = self.relu4(self.fc4(h))
        h = self.relu5(self.fc5(h))
        h = self.relu6(self.fc6(h))
        h = self.relu7(self.fc7(h))
        h = self.relu8(self.fc8(h))
        h = self.fc9(h)
        """
        h = self.fc0(h)

        h = h.view(len(x), 64, 64)
        return h

