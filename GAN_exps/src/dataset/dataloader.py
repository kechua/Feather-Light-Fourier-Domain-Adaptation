import os
import pickle

import numpy as np
import pandas as pd
import torch
# from distlib._backport import shutil
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
# import dataloader_functions as loaderFunc


from dataloader_functions import files_preprocessing, get_data_analyse, data_preparing, create_dataframe_from_scalefile, \
    find_max_content, create_dataframe_from_path


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


class CustomPictDataset(Dataset):
    ''' Custom class dataset for Calgary-Campinas Public Brain MR Dataset

        Attributes:
            df - datafraem with loaded scale table which consist next columns: ['name', 'mask', 'x', 'y', 'z']
            mark_dict - inform dict with scale information
            elem_dict - inform dict with context information
            self.amount - variable describing the number of pictures with information
            transform - torch transformation
            domain_choosen - boolean value which determines whether the domain has been selected
            data - dataframe with paths to image and mask

    '''

    def __init__(self, dataset_path, domain_mask, path_to_csv, mask_tail='_ss', save_dir='',
                 load_dir='', transform=transforms.Compose([transforms.Resize([int(300), int(300)]),
                                                            transforms.PILToTensor(),
                                                            ThresholdTransform(thr_255=240)])):
        """
        For first launch

        :param dataset_path: Absolute path to dataset images, for ex - 'Dataset/Original'.
        :param domain_mask: Absolute path to dataset mask,  for ex - 'Dataset/Mask'.
        :param path_to_csv: Absolute path to rescale csv file, for ex - './meta.csv'
        :param mask_tail: String parameter describing the tail by which the mask differs, '_ss' - default
        :param save_dir: Directory in which save all class value attributes (optional)
        :param load_dir: Directory from which Dataset will be loaded. IMPORTANT, if this parameter is specified,
        class will be loaded from it, without the preprocessing stage
        :param transform: Parameter with tensor transformation
        """

        if not load_dir:
            dataset_files = files_preprocessing(dataset_path[0], dataset_path[1], domain_mask, mask_tail)
            self.df = create_dataframe_from_scalefile(dataset_files, path_to_csv)
            self.mark_dict, self.elem_dict, self.file_res, self.amount = get_data_analyse(dataset_files)

            if save_dir:
                self.create_back(save_dir)
        else:
            self.__load_back(load_dir)
            self.mark_dict, self.elem_dict = {}, {}
            self.amount = find_max_content(self.file_res)

        self.transform = transform

        self.domain_choosen = False
        self.data = None

    def get_data_statistic(self):
        """
        :return: two dictionaries with statistics
        """
        return self.elem_dict, self.mark_dict

    def __load_back(self, load_dir):
        """
        This function download class important variables from directory

        :param load_dir: the directory from which the download takes place
        :return: None
        """

        with open(f'{load_dir}/file_inf.pickle', 'rb') as f:
            self.file_res = pickle.load(f)

        with open(f'{load_dir}/mark_inf.pickle', 'rb') as f:
            self.mark_dict = pickle.load(f)

        with open(f'{load_dir}/elem_inf.pickle', 'rb') as f:
            self.elem_dict = pickle.load(f)

        self.df = pd.read_csv(f'{load_dir}/scale_table.csv')
        self.df.set_index('id', inplace=True)

    def create_back(self, save_dir, rewrite_flag=False):
        """
        This function save class important variables in to directory

        :param save_dir: the directory from which the download takes place
        :param rewrite_flag: should this function overwrite files if there is such a directory
        :return: None
        """

        if os.path.exists(save_dir):
            print(f'{save_dir} already exist.')

            if rewrite_flag:
                print('Rewriting directory')
                # shutil.rmtree(save_dir)

            else:
                print('Stopping')
                return

        os.mkdir(save_dir)

        self.df.reset_index().to_csv(f'{save_dir}/scale_table.csv', index=False)

        with open(f'{save_dir}/file_inf.pickle', 'wb') as f:
            pickle.dump(self.file_res, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{save_dir}/mark_inf.pickle', 'wb') as f:
            pickle.dump(self.mark_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{save_dir}/elem_inf.pickle', 'wb') as f:
            pickle.dump(self.elem_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def domain_preproc(self, path, domain_name, rewrite_flag=False):
        """

        :param path: Path to which all images of the given domain should be saved
        :param domain_name: domain name
        :param rewrite_flag: should this function overwrite files if there is such a directory
        :return:
        """
        if os.path.exists(path):
            print('This directory already exist')

            if rewrite_flag:
                print('Rewriting directory')
                # shutil.rmtree(path)
                data_preparing(path, self.file_res, self.amount, domain_name, self.df)
            else:
                print('Using existed directory.')
        else:
            data_preparing(path, self.file_res, self.amount, domain_name, self.df)

        self.data = create_dataframe_from_path(path)
        self.domain_choosen = True

    def __len__(self):
        if self.domain_choosen:
            return len(self.data)
        else:
            print('You forget to choose current domain')

    def __getitem__(self, idx):
        if self.domain_choosen:
            pict, mask = self.data.iloc[idx]
            brain = self.transform(Image.open(pict).convert('L'))
            mask = self.transform(Image.open(mask).convert('L'))

            return {'image': brain, 'mask': mask}
        else:
            print('You forget to choose current domain')


if __name__ == '__main__':
    dataset_path = ('Dataset/Original', 'Dataset/Mask')
    domain_mask = ['philips_15', 'philips_3', 'siemens_15']
    path_to_csv = './meta.csv'
    TD = CustomPictDataset(None, None, None, load_dir='Back')
    print(TD.df)
    TD.domain_preproc('./siemens_3', 'siemens_3')
    DL_DS = DataLoader(TD, batch_size=2, shuffle=False, drop_last=True)
    print(TD.data)

    pos = 0
    for (idx, batch) in enumerate(DL_DS):  # Print the 'text' data of the batch
        if idx == 0:
            image, mask = batch['image'], batch['mask']
            print(image)
            print(mask)
