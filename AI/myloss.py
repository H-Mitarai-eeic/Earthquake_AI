
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
    


class MyLoss2(nn.Module):
    def __init__(self, kernel_size=2, stride=None, gpu=-1):
        super().__init__()
        """
        pooling層のkernel_sizeとstrideは、最初に指定してください。
        2021/12/03: コード変更に伴い、init部分は使っていません
        """

        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride)
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride)

        self.gpu = gpu


    def forward(self, outputs, targets, mask=None, weight=(0.1,)*10, exponent=2):
        """
        =======引数=======
        outputs: 予測結果のテンソル
        targets: 実際のデータのテンソル
        mask: 観測地点のあるところを 1 、ないところを 0 としたマスク。テンソル。サイズはoutputsやtargetsと同じにすること
        weight: 誤差を足すときの重みづけのタプル。weight[0] が震度0が観測された地点に対する誤差の重み。他も同様。
        exponent: 何乗誤差にするか。現状、exponent[0]しか使ってない。今後の機能拡張のためtupleになっている。

        =========返り値=========
        学習に使うのはlossだけ。他はtestとかに使う用
        """
        #outputs = outputs * mask
        #targets = targets * mask
        #N = mask.sum(dim=(0,1,2))


        IntensityMask = torch.zeros(10, len(targets), len(targets[0]), len(targets[0][0]))  #ある震度階級が観測された場所だけ1になる
        Loss4eachIintensity = torch.zeros(10)   #各震度ごとの誤差
        Class_N = [0 for i in range(10)]    #震度 i が観測された地点数
        loss = torch.tensor(0.)     #返り値の誤差

        if self.gpu >= 0:
            device = 'cuda:' + str(self.gpu)
            IntensityMask = IntensityMask.to(device)
            Loss4eachIintensity = Loss4eachIintensity.to(device)
            loss = loss.to(device)

        for i in range(10):
            for B in range(len(targets)):
                for Y in range(len(targets[B])):
                    for X in range(len(targets[B][Y])):
                        if targets[B][Y][X].item() == i and mask[B][Y][X].item() == 1:
                            IntensityMask[i][B][Y][X] = 1
                            Class_N[i] += 1
                        else:
                            IntensityMask[i][B][Y][X] = 0

        for i in range(10):
            if Class_N[i] > 0:
                #lossに加算する
                #まず上のfor文で用意したマスクを掛ける
                masked_outputs = outputs * IntensityMask[i]
                masked_targets = targets * IntensityMask[i]
                #震度 i が（実際に）観測された場所でののexponent[0]乗誤差の和を、その震度 (i) が観測された地点数で割る
                #観測される場所のが少ない大きな震度と、震度0のように広い範囲で観測される震度が平等に評価される。気がする。
                Loss4eachIintensity[i] = ((masked_outputs - masked_targets).pow(exponent)).sum(dim=(0,1,2)) / Class_N[i]
                loss += weight[i] * Loss4eachIintensity[i]

        #返り値　学習に使うのはlossだけ。他はtestとかに使う用
        return loss, Loss4eachIintensity, Class_N
        #return loss

class MyLoss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, mask=None, weight=(0.51,0.49), exponent=1):
        """
        =======引数=======
        outputs: 予測結果のテンソル
        targets: 実際のデータのテンソル
        mask: 観測地点のあるところを 1 、ないところを 0 としたマスク。テンソル。サイズはoutputsやtargetsと同じにすること
        weight: maxpoolして誤差とるのと, avgpoolして誤差とるのの重みづけのタプル。weight[0] がmaxの重み, weight[1] がavgの重み
        exponent: 奇数
        """
        if type(mask) != type(None):
            outputs = outputs * mask
            targets = targets * mask
            N = mask.sum(dim=(0,1,2))
        else:
            N = 64*64

        loss_abs = (outputs - targets).pow(exponent).abs().sum(dim=(0,1,2))
        loss_signed = (targets - outputs).pow(exponent).sum(dim=(0,1,2))
        loss = weight[0] * loss_abs + weight[1] * loss_signed
        return loss / N 