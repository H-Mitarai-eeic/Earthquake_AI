
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


    def forward(self, outputs, targets, mask=None, weight=(0.5, 0.5)):
        """
        =======引数=======
        outputs: 予測結果のテンソル
        targets: 実際のデータのテンソル
        mask: 観測地点のあるところを 1 、ないところを 0 としたマスク。テンソル。サイズはoutputsやtargetsと同じにすること
        weight: maxpoolして誤差とるのと, avgpoolして誤差とるのの重みづけのタプル。weight[0] がmaxの重み, weight[1] がavgの重み
        """
        if mask != None:
            outputs = outputs * mask
            targets = targets * mask
            N = mask.sum(dim=(0,1,2,3))
        else:
            N = 4096

        loss_maxpool = ((maxpool(outputs) - maxpool(targets)).pow(2)).sum(dim=(0,1,2,3))
        loss_avgpool = ((avgpool(outputs) - avgpool(targets)).pow(2)).sum(dim=(0,1,2,3))
        #loss = ((outputs - targets) * (outputs - targets)).mean(dim=(0,1,2,3))
        loss = weight[0] * loss_maxpool + weight[1] * loss_avgpool
        return loss / N
