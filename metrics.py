import torch

def MAE(y, y_):
    assert y.size() == y_.size()
    return (y - y_).abs().mean()


def categorical_mse_2d_loss(y_pred, classes, y):
    """
    :param classes: (N,H,W) classes index : int
    :param y_pred: (N,C,H,W) float values
    :param y: (N,H,W) target float values
    :return: mean((y_pred[classes] - y) ** 2)
    """
    y_pred = y_pred.gather(1, classes.unsqueeze(1)).squeeze()

    assert y_pred.size() == y.size()
    return torch.nn.functional.mse_loss(y_pred, y)


class Categorical_MSE_2D_Loss(torch.nn.Module):
    def __init__(self):
        super(Categorical_MSE_2D_Loss, self).__init__()

    def forward(self, y_pred, classes, y):
        return categorical_mse_2d_loss(y_pred, classes, y)