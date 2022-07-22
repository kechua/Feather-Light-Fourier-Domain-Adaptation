import glob
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from . dataloader import CustomPictDataset
from . dataloader_functions import change_2d
from . mask import get_image_sum


def prepare_dict(td):
    data_loader = DataLoader(td, batch_size=4, shuffle=True, drop_last=True)
    counter = 0
    values_sum = 0
    phase_sum = 0

    for (idx, batch) in enumerate(data_loader):  # Print the 'text' data of the batch
        for i in range(len(batch['image'])):
            values_sum, phase_sum = get_image_sum(batch['image'][i], values_sum, phase_sum)
            counter += 1

    return values_sum, phase_sum, counter


def preproc_stage(save=True):
    td = CustomPictDataset(None, None, None, load_dir='/raid/data/DA_BrainDataset/Back')

    td.domain_preproc('/raid/data/DA_BrainDataset/philips_15', 'philips_15')
    philips_sum_amp_15, _, philips_counter_15 = prepare_dict(td)

    td.domain_preproc('/raid/data/DA_BrainDataset/siemens3', 'siemens3')
    siemens_sum_amp_3, _, siemens_counter_3 = prepare_dict(td)

    philips_sum_amp_15 = philips_sum_amp_15[0]
    siemens_sum_amp_3 = siemens_sum_amp_3[0]

    philips_sum_amp_15 /= philips_counter_15
    siemens_sum_amp_3 /= siemens_counter_3

    diff = abs(philips_sum_amp_15 - siemens_sum_amp_3)
    if save:
        with open(f'results/diff_values.pickle', 'wb') as f:
            pickle.dump(diff, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'results/phil_sum_amp_15', 'wb') as f:
            pickle.dump(philips_sum_amp_15, f, protocol=pickle.HIGHEST_PROTOCOL)

    return diff, philips_sum_amp_15

def data_preparing(img, file_id, df, itt = 'z'):
    img_list = []
    roll_dict = {'z': 0, 'y': 1, 'x': 2}
    print(img.shape)
    img = np.rollaxis(img, roll_dict[itt])
    print(img.shape, '_____')
    dim_size, _, _ = img.shape

    for pos in range(dim_size):
        res_img = img[pos, ...]
        im = (change_2d(res_img, file_id, df, with_perc=True) * 255).astype(np.uint8)
        # img_list.append(im/im.max())
        img_list.append(im)


    return img_list


def create_df(old_csv_name):
    old_df = pd.read_csv(old_csv_name)
    old_df.set_index('id', inplace=True)
    return old_df[['x', 'y', 'z']]

def create_path_dict(dir_files1, dir_files2, dir_mask):
    files1 = glob.glob(f'{dir_files1}/*.nii.gz')
    files2 = glob.glob(f'{dir_files2}/*.nii.gz')
    mask = glob.glob(f'{dir_mask}/*.nii.gz')
    result_dict = {}
    files1 = glob.glob(f'{dir_files1}/*ge_15*.nii.gz')
    print(files1)
    print(dir_files1)
    for filename1 in files1:
        short_name1 = os.path.basename(filename1)
        elem_id = short_name1[:short_name1.find('_')]
        filename2 = next(x for x in files2 if elem_id in x)
        filename_mask = next(x for x in mask if elem_id in x)
        result_dict[elem_id] = (filename1, filename2, filename_mask)

    return result_dict

def create_path_dict_on_mask(dir_files1, dir_files2, dir_mask, mask):
    files1 = glob.glob(f'{dir_files1}/*.nii.gz')
    files2 = glob.glob(f'{dir_files2}/*.nii.gz')
    mask = glob.glob(f'{dir_mask}/*.nii.gz')
    result_dict = {}

    for filename1 in [x for x in files1 if mask in x]:
        short_name1 = os.path.basename(filename1)
        elem_id = short_name1[:short_name1.find('_')]
        filename2 = next(x for x in files2 if elem_id in x)
        filename_mask = next(x for x in mask if elem_id in x)
        result_dict[elem_id] = (filename1, filename2, filename_mask)

    return result_dict


