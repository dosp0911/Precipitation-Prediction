from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import util
import numpy as np
import cv2
from pre_processing import handle_outliers, equalize_hist

class GpmDataset(Dataset):
    def __init__(self, f_path, t_f=None, extension='npy'):
        super(GpmDataset, self).__init__()
        # self.f_path = f_path
        self.t_f = t_f
        self.file_list = util.get_file_names_in_folder(f_path, extension)

    def __getitem__(self, item):
        # gpm_data = (40, 40, 15)
        gpm_data = util.load_npy_files(str(self.file_list[item]))
        # remove 12,13th columns (DPR LONG/LATITUDE)
        gpm_data = np.delete(gpm_data, (12,13), axis=2)
        gpm_data = handle_outliers(gpm_data)
        # opecv histogram equalization , morphologyEx
        gpm_data = equalize_hist(gpm_data) # apply equalize histogram or not?
        # pre-processing gpm data ( normalize, brightness, noisy, to_tensor.. so on)
        gpm_data = self.t_f(gpm_data)

        return gpm_data

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    # except types, DPR LONG/LATITUDE, Precipitation
    avg_mean = [1.96452e+02, 1.39343e+02, 2.16165e+02, 1.68963e+02, 2.38547e+02, 2.32316e+02, 1.91321e+02, 2.63215e+02, 2.44779e+02,
                 0.0, 0.0]
    avg_std = [45.087, 72.679, 36.286, 58.972, 29.526, 27.983, 47.61 , 23.429, 30.96,
               180.0, 180.0]

    t= transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=avg_mean,
                              std=avg_std)])
