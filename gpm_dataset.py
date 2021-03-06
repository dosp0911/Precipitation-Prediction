from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import util
import numpy as np
import cv2
from pre_processing import handle_outliers, equalize_hist

class GpmDataset(Dataset):
    def __init__(self, f_paths, t_f=None):
        super(GpmDataset, self).__init__()
        # self.f_path = f_path
        self.t_f = t_f
        self.file_list = f_paths

    def __getitem__(self, item):
        # gpm_data = (40, 40, 15)
        gpm_data = util.load_npy_files(str(self.file_list[item]))
        assert gpm_data.shape == (40, 40, 15)
        precipitation = torch.from_numpy(gpm_data[..., -1])
        types = torch.from_numpy((gpm_data[..., 9] // 100).astype(int))
        # remove 9th: types, 12,13th: (DPR LONG/LATITUDE), 14th:precipitation(target data)
        gpm_data = np.delete(gpm_data, (9,12,13,14), axis=2)
        gpm_data = handle_outliers(gpm_data)
        # opecv histogram equalization , morphologyEx
        # gpm_data[..., :9] = equalize_hist(gpm_data[..., :9]) # apply equalize histogram on only image channels
        # pre-processing gpm data ( normalize, brightness, noisy, to_tensor.. so on)
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
        gpm_data = np.delete(gpm_data, (9,12,13), axis=2)
        gpm_data = handle_outliers(gpm_data)
        # opecv histogram equalization , morphologyEx
        # gpm_data[..., :9] = equalize_hist(gpm_data[..., :9]) # apply equalize histogram on only image channels
        # pre-processing gpm data ( normalize, brightness, noisy, to_tensor.. so on)
        gpm_data = self.t_f(gpm_data)

        return gpm_data, types

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    # except types, DPR LONG/LATITUDE, Precipitation
    avg_mean = [1.96452e+02, 1.39343e+02, 2.16165e+02, 1.68963e+02, 2.38547e+02, 2.32316e+02, 1.91321e+02, 2.63215e+02,
                2.44779e+02,
                0.0, 0.0]
    avg_std = [45.087, 72.679, 36.286, 58.972, 29.526, 27.983, 47.61, 23.429, 30.96,
               180.0, 180.0]

    t = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=avg_mean,
                              std=avg_std)])

    split_ratio = 0.2
    n_batch = 64
    DATA_PATH = 'D:\\forecasting\\train-002'
    f_paths = util.get_file_names_in_folder(DATA_PATH, 'npy')
    train_paths, val_paths = train_test_split(f_paths, test_size=split_ratio)

    train_dataset = GpmDataset(train_paths, t)
    val_dataset = GpmDataset(val_paths, t)
    train_dataloader = DataLoader(train_dataset, n_batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, n_batch, shuffle=True)

    train_len = len(train_dataloader)
    val_len = len(val_dataloader)

    sample = next(iter(train_dataset))
    print(sample[0].size(), sample[1].size(), sample[2].size() )