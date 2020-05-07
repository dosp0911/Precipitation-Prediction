import torch

def MAE(y, y_):
    return torch.abs(y - y_)