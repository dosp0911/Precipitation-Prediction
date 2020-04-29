import util
from torchvision.datasets import DatasetFolder as DatasetFolder
from tqdm.notebook import tqdm
import multiprocessing

import numpy as np
import pandas as pd

import os
from pathlib import Path


DATA_PATH = 'D:\\forecasting'

def file_loader(path):
    return np.load(path)

dataset_folder = DatasetFolder(DATA_PATH, file_loader, extensions='.npy')
s = next(iter(dataset_folder))
print(s)