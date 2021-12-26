import torch
import torch.nn as nn

def Calc_Error(outputs, targets, mask=None):
    if type(mask) != type(None):
        outputs = outputs * mask
        targets = targets * mask
        N = mask.sum(dim=(0,1,2))
    else:
        N = 64*64
    error = outputs - targets
    E_error = error.sum(dim=(0,1,2)) / N
    Var_error = (error - E_error).pow(2).sum(dim=(0,1,2)) / N
    return E_error, Var_error