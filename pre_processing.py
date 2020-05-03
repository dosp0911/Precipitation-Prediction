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
    :param arr: must be uint
    :return:
    """
    return cv2.equalizeHist(arr.astype(np.uint8))


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