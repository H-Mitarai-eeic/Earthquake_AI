
import fcn
import numpy as np
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, mask):
        outputs = outputs * mask
        targets = targets * mask

        loss = ((outputs - targets) * (outputs - targets)).mean(dim=(0,1,2,3))
        return loss
