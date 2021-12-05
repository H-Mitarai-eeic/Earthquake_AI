import fcn
import numpy as np
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        """
        pooling層のkernel_sizeとstrideは、最初に指定してください。
        """

        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride)
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride)


    def forward(self, outputs, targets, mask=None, weight=(0.5, 0.0, 0.5), exponent=2):
        """
        =======引数=======
        outputs: 予測結果のテンソル
        targets: 実際のデータのテンソル
        mask: 観測地点のあるところを 1 、ないところを 0 としたマスク。テンソル。サイズはoutputsやtargetsと同じにすること
        weight: maxpoolして誤差とるのと, avgpoolして誤差とるのの重みづけのタプル。weight[0] がmaxの重み, weight[1] がavgの重み
        """
        if type(mask) != type(None):
            outputs = outputs * mask
            targets = targets * mask
            N = mask.sum(dim=(0,1,2))
        else:
            N = 64*64

        loss_maxpool = ((self.maxpool(outputs) - self.maxpool(targets)).pow(exponent)).sum(dim=(0,1,2))
        loss_avgpool = ((self.avgpool(outputs) - self.avgpool(targets)).pow(exponent)).sum(dim=(0,1,2))
        loss_nonpool = ((outputs - targets).pow(exponent)).sum(dim=(0,1,2))
        #loss = ((outputs - targets) * (outputs - targets)).mean(dim=(0,1,2,3))
        loss = weight[0] * loss_maxpool + weight[1] * loss_avgpool + weight[2] * loss_nonpool
        return loss / N