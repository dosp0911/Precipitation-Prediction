from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import util
import numpy as np
import cv2
from pre_processing import is_outlier


class GpmDataset(Dataset):
    def __init__(self, f_paths, t_f=None):
        super(GpmDataset, self).__init__()
        # self.f_path = f_path
        self.t_f = t_f
        self.file_list = f_paths

    def __getitem__(self, item):
        # gpm_data = (40, 40, 15)
        while True:
            gpm_data = util.load_npy_files(str(self.file_list[item]))
            assert gpm_data.shape == (40, 40, 15)
            if not is_outlier(gpm_data):
                break
        precipitation = torch.from_numpy(gpm_data[..., -1])
        types = torch.from_numpy((gpm_data[..., 9] // 100).astype(int))
        # remove 9th: types, 10,11,12,13th: (LONG/LATITUDE), 14th:precipitation(target data)
        gpm_data = np.delete(gpm_data, (9,10,11,12,13,14), axis=2)
        gpm_data = self.t_f(gpm_data)

        return gpm_data, precipitation, types

    def __len__(self):
        return len(self.file_list)


class TestGpmDataset(Dataset):
    def __init__(self, f_paths, t_f=None):
        super(TestGpmDataset, self).__init__()
        # self.f_path = f_path
        self.t_f = t_f
        self.file_list = f_paths

    def __getitem__(self, item):
        # test gpm_data = (40, 40, 14)
        gpm_data = util.load_npy_files(str(self.file_list[item]))
        assert gpm_data.shape == (40, 40, 14)

        types = torch.from_numpy((gpm_data[..., 9] // 100).astype(int))
        # remove 9th: types, 12,13th: (DPR LONG/LATITUDE)
        gpm_data = np.delete(gpm_data, (9,10,11,12,13), axis=2)

        gpm_data = self.t_f(gpm_data)

        return gpm_data, types

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':

    # DATA_PATH = 'D:\\forecasting\\train-002'
    DATA_PATH = 'B:\\preciptation\\train-002'
    f_paths = util.get_file_names_in_folder(DATA_PATH, 'npy')
    npy = util.load_npy_files(f_paths[:100])
    for i in npy:
        print(is_outlier(i))

