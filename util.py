

import pandas as pd
import shutil

import pathlib
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from multiprocessing import Pool
from pathlib import Path


def csv_file_load(f_p, index_col=False ):
    if f_p.exists():
        return pd.read_csv(f_p, index_col=index_col)
    else:
        raise FileExistsError(f'{f_p} no exist!')



def move_files_to_class_folders(f_names, classes, root_f):
    root_path = pathlib.Path(str(root_f))
    if not root_path.exists():
        raise FileExistsError(f'{root_f} does not exist')
        
    class_dirs = pd.unique(classes).astype('str')
    for d in class_dirs:
        class_dir = root_path / d
        if not class_dir.exists():
            class_dir.mkdir()
        
    for file, c_ in tqdm(zip(f_names, classes.astype('str'))):
        shutil.copy(str(root_path / file), str(root_path / c_))
        
    print('Done.')



def print_model_memory_size(model):
    total_ = 0
    for k, v in model.state_dict().items():
        print(f'name:{k} size:{v.size()} dtype:{v.dtype}')
        total_ += v.numel()
    print(f'Model size : {total_*4} byte -> {total_*4/1024**2} MiB')




def get_pixel_value_frequencies(img_arr, dtype=int):
  '''
    img_arr = (N,H,W) or (H,W) 
    dtype = pixel values 
    counts unique pixel values of images
  '''
  arr = np.reshape(img_arr, -1)
  uvals = np.unique(arr).astype(dtype)
  uvals_dic = {}
  
  for u in list(uvals):
    uvals_dic[u] = np.sum(arr==u)
  return uvals_dic




def get_weights_ratio_over_frequnecies(freq):
  '''
    # [2,3,4,5] -> [1/2, 1/3, 1/4, 1/5]
  '''
  return list(map(lambda x: 1/x, freq))




def save_model(model, optim, save_path, epoch, loss):
  torch.save({
        # 'model' : model,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss' : loss,
        'optim_state_dict': optim.state_dict()
    }, save_path)
  print(f'model saved \n {save_path}')


def load_model(path, model, map_location=None):
  '''
    args:
      path : location to load model
      model : model variable
      map_location : device to load model
    return:
      model loaded weights from saved model
  '''
  load_model = torch.load(path, map_location=map_location)
  model.load_state_dict(load_model['model_state_dict'])
  return model





def display_imgs(imgs, figsize, cols=2, title='img'):
  '''
      imgs : (N, H, W) or (N, C, H, W)
  '''
  plt.figure(figsize=figsize) 
  if np.ndim(imgs) == 2:
    plt.imshow(imgs)

  num = (len(imgs) // cols) + 1
  for i in range(len(imgs)):
    plt.subplot(num, cols, i+1)
    plt.title(f'{i}th {title}')
    if np.ndim(imgs) == 3:
      plt.imshow(imgs[i], cmap='gray')
    elif np.ndim(imgs) == 4:
      plt.imshow(np.transpose(imgs[i], (2,3,0)))




def display_weights_of_model(model):
  l_p = sum(1 for x in model.parameters())
  fg, axes = plt.subplots(l_p//5+1, 5, figsize=(15,15))
  fg.tight_layout()

  #torch.nn.utils.parameters_to_vector
  for i,(n, p) in enumerate(model.named_parameters()):
    ax = axes[i//5,i%5]
    ax.set_title(n)
    sns.distplot(p.detach().numpy(), ax=ax)




def display_trained_mask(output, title='trained'):
  """
    output : (N,C,H,W) display N trained masks 
  """
  output = torch.argmax(output, dim=1)
  plt.figure(figsize=(15,2))
  for i in range(len(output)):
    plt.subplot(len(output)//2, 2, i+1)
    plt.title(f'{i}th {title} mask')
    plt.imshow(output[i], cmap='gray')




def get_class_weights_by_pixel_frequencies(classId, EncodedPixels, img_size):
  '''
     img_size must be 1 dimension. (H*W)
  '''
  p_counts = np.zeros(len(classId.unique())+1)
  
  # counts total pixels of training image dataset
  for c, e in zip(classId, EncodedPixels):
    rlc = np.asarray(e.split(' '))
    cls_pixels = sum(rlc[1::2].astype(int))
    p_counts[c] += cls_pixels
    p_counts[0] += img_size - cls_pixels 

  p_counts /= img_size
  
  return get_weights_ratio_over_frequnecies(p_counts)



class class2d_to_onehot(nn.Module):
  def __init__(self, classes):
    '''
    args:
      classes: [0,1,2,3... labels] labels must be integer
      It will add channles of the number of labels to target 
    '''
    super(class2d_to_onehot, self).__init__()
    self.classes = torch.tensor(classes).unique()
    
  def forward(self, target):
    '''
      args: 
        target: (N,H,W), (H,W)
      return:
        (N,H,W)->(N,C,H,W)
        (H,W)->(C,H,W)
    ''' 
    ndims = len(target.size())

    assert ndims == 2 or ndims == 3

    if ndims == 2:
      cls_stacks = torch.stack([(target==c).type(torch.float32) for c in self.classes], dim=0)
    elif ndims == 3:
      cls_stacks = torch.stack([(target==c).type(torch.float32) for c in self.classes], dim=1)

    return cls_stacks
  



def get_file_names_in_folder(path, extension):
    '''
    :param path: directory path
    extension = "exe", "jpg"..
    :return: file name list
    '''
    p = Path(path)

    if not p.is_dir():
        raise ValueError(f'{path} does not exist or is not directory.')

    return list(p.glob(f'*.{extension}'))

def load_npy_files(paths):
    '''
    :param path: npy files n paths
    :return: (n, npy dim)
    '''
    if isinstance(paths, list):
        np_stacks = [np.load(str(p)) for p in tqdm(paths)]
        return np_stacks
    elif isinstance(paths, str):
        return np.load(paths)



if __name__ == '__main__':
    l = get_file_names_in_folder('C:\\Users\\DSKIM\\Google 드라이브\\DACON\\강수량측정\\train-002', "npy")
    p_num = 4
    p = Pool(p_num)
    idx = np.linspace(0, len(l)/20, p_num + 1).astype(int)

    # i_1, i_2, i_3 = int(len(l) * 0.25), int(len(l) * 0.5), int(len(l) * 0.75)

    part_of_files = [l[idx[i]:idx[i + 1]] for i in range(p_num)]

    result = p.map(load_npy_files, part_of_files)
    npy_arr = np.concatenate(result, axis=0).reshape(-1, 1600, 15)
    print(np.shape(npy_arr))