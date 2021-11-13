
import fcn
import numpy as np
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, mask):
        _, predicted = torch.max(outputs, 1)
        #predicted.requires_grad = True
        #predicted.dtype=float
        #outputs.requires_grad = True
        #loss = ((outputs - targets)**4).mean()

        total = 0
        loss_sum = 0
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                for k in range(len(predicted[i][j])):
                    if mask[j][k] != 0:
                        label = labels[i][j][k]
                        predic = predicted[i][j][k]
                        diff = predic - label
                        if diff < 0:
                            loss_sum += -diff
                        else:
                            loss_sum += diff
                        total += 1 

        loss = loss_sum.to(torch.float16) / float(total)
        loss = loss.to(torch.float16)
        loss.requires_grad = True
        return loss
