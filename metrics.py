import torch
from sklearn.metrics import f1_score
import numpy as np

def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''

    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]

    # 실제값이 0.1 이상인 픽셀의 위치 확인
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1

    # 실제 값에 결측값이 없는 픽셀의 위치 확인
    IsNotMissing = y_true >= 0

    # mae 계산
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))

    # f1_score 계산 위해, 실제값에 결측값이 없는 픽셀에 대해 1과 0으로 값 변환
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)

    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)

    # f1_score 계산
    f_score = f1_score(y_true, y_pred)

    # f1_score가 0일 나올 경우를 대비하여 소량의 값 (1e-07) 추가
    return mae / (f_score + 1e-07)

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


def categorical_mae_2d_loss(y_pred, classes, y):
    """
    :param classes: (N,H,W) classes index : int
    :param y_pred: (N,C,H,W) float values
    :param y: (N,H,W) target float values
    :return: mean((y_pred[classes] - y) ** 2)
    """
    y_pred = y_pred.gather(1, classes.unsqueeze(1)).squeeze()

    assert y_pred.size() == y.size()
    return torch.nn.functional.l1_loss(y_pred, y)


class Categorical_MAE_2D_Loss(torch.nn.Module):
    def __init__(self):
        super(Categorical_MAE_2D_Loss, self).__init__()

    def forward(self, y_pred, classes, y):
        return categorical_mae_2d_loss(y_pred, classes, y)


class Categorical_MSE_2D_Loss(torch.nn.Module):
    def __init__(self):
        super(Categorical_MSE_2D_Loss, self).__init__()

    def forward(self, y_pred, classes, y):
        return categorical_mse_2d_loss(y_pred, classes, y)