import torch

def MAE(y, y_):
    return torch.abs(y - y_)

def categorical_mse_2Dloss(classes, y_pred, y):
    """
    :param classes: (N,H,W) classes index : int
    :param y_pred: (N,C,H,W) float values
    :param y: (N,H,W) target float values
    :return: mean((y_pred[classes] - y) ** 2)
    """
    return torch.mean((y_pred[classes] - y)**2)