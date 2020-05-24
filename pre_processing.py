import cv2
import numpy as np
import util

import matplotlib.pyplot as plt
import seaborn as sns


def handle_outliers(arr):
    """
    convert outlier values(-9999, nan) into 0
    :param arr: numpy arr
    :return: numpy arr converted
    """

    arr[np.isnan(arr)] = 0
    arr[arr < -1000] = 0
    return arr


def equalize_hist(arr):
    """
    :param arr: must be uint [C,H,W]
    :return: arr[C,H,W]
    """
    for i, a in enumerate(arr):
        arr[i] = cv2.equalizeHist(a.astype(np.uint8))

    return arr


def random_flip_both(x, y, t, axes=(0, 1)):
    """
        input shape:[H,W,C]
    """
    a = (axes, axes[0], axes[1], 0)
    r = a[np.random.randint(0, 4)]
    if r != 0:
        return np.flip(x, axis=r).copy(), np.flip(y, axis=r).copy(), np.flip(t, axis=r).copy()
    else:
        return x, y, t


def random_rot90_both(x, y, t, axes=(0, 1)):
    """
        input shape:[H,W,C]
    """
    r = np.random.randint(0, 4)
    return np.rot90(x, r, axes=axes).copy(), np.rot90(y, r, axes=axes).copy(), np.rot90(t, r, axes=axes).copy()


if __name__ == '__main__':
    a = np.random.rand(3,12,40,40)
    b = np.random.rand(3,40,40)
    c = np.random.randint(0,3,(3,40,40))

    random_rot90_both(a,b,c)