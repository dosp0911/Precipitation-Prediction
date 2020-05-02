from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import util
import numpy as np
import cv2

class gpm_dataset(Dataset):
    def __init__(self, f_path, t_f=None, extension='npy'):
        super(gpm_dataset, self).__init__()
        # self.f_path = f_path
        self.t_f = t_f
        self.file_list = util.get_file_names_in_folder(f_path, extension)

    def __getitem__(self, item):
        # gpm_data = (40, 40, 15)
        gpm_data = util.load_npy_files(str(self.file_list[item]))
        # remove 12,13th columns (DPR LONG/LATITUDE)
        gpm_data = np.delete(gpm_data, (12,13), axis=2)
        # pre-processing gpm data ( normalize, brightness, noisy, to_tensor.. so on)
        # opecv histogram equalization , morphologyEx
        gpm_data = self.t_f(gpm_data)

    def __len__(self):
        return len(self.file_list)

transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    cv2.equalizeHist()
    cv2.imshow()
    cv2.morphologyEx()
    cv2.imshow()