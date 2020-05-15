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


def is_outlier(arr):
    """
    check whether there is a outlier in array
    """
    if (np.isnan(arr).sum() > 0) or ((arr < 0).sum() > 0):
        return True
    else:
        return False


def equalize_hist(arr):
    """
    :param arr: must be uint [C,H,W]
    :return: arr[C,H,W]
    """
    for i, a in enumerate(arr):
        arr[i] = cv2.equalizeHist(a.astype(np.uint8))

    return arr

def morphology_ex(arr):
    return cv2.morphologyEx(arr)


if __name__ == '__main__':
    f_list = util.get_file_names_in_folder('B:\\preciptation\\train-002', 'npy')
    arr = np.load(f_list[0])
    print(np.shape(arr[..., 0]))
    plt.imshow(arr[..., 0])
    # a = equalize_hist(arr[..., 0], )
    # cv2.imshow('hist', a)
    img = arr[..., 6]
    e = cv2.equalizeHist(img.astype(np.uint8))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.imshow(e, cmap='gray')
    plt.subplot(133)
    plt.imshow(arr[...,14], cmap='gray')