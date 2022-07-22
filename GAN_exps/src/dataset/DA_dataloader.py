import os
import numpy as np
import nibabel as nib
from PIL import Image
import glob
import pandas as pd
import json
import shutil
import pickle5 as pickle

from torch.utils.data import Dataset
from torchvision.transforms import transforms



def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


def images_with_masks(path_dict, path, domain_name, save_df=None, lim=172, save=True, itt='x'):
    if not check_path(path):
        return None

    df = pd.DataFrame(columns=['img', 'mask', 'file_id', 'domain'])
    roll_dict = {'z': 0, 'y': 1, 'x': 2}
    for file_id, (file, mask, domain) in path_dict.items():
        if domain_name not in file:
            continue
        current_file_dir = file[file.rfind('/') + 1: file.find('.')]

        os.mkdir(f'{path}/train/{current_file_dir}')
        os.mkdir(f'{path}/mask/{current_file_dir}')

        img = np.array(nib.load(file).dataobj)
        img = np.rollaxis(img, roll_dict[itt])
        img = img[:lim, ...]

        img_mask = np.array(nib.load(mask).dataobj)
        img_mask = np.rollaxis(img_mask, roll_dict[itt])
        img_mask = img_mask[:lim, :, :]
        z, _, _ = img.shape

        for i in range(z):
            im = Image.fromarray(scale_mri(img[i, ...]) * 255).convert('L')
            im_mask = Image.fromarray(scale_mri(img_mask[i, ...]) * 255).convert('L')

            im.save(f'{path}/train/{current_file_dir}/lvl{i}.png')
            im_mask.save(f'{path}/mask/{current_file_dir}/lvl{i}.png')
            data = {'img': f'{path}/train/{current_file_dir}/lvl{i}.png',
                    'mask': f'{path}/mask/{current_file_dir}/lvl{i}.png',
                    'file_id': file_id, 'domain': domain}
            df = df.append(data, ignore_index=True)

        if save_df:
            df.to_csv(f'{path}/df_save.csv', index=False)

    return df


def check_path(path):
    if not os.path.exists(f'{path}'):
        os.mkdir(f'{path}')
        os.mkdir(f'{path}/train')
        os.mkdir(f'{path}/mask')
        return True

    else:
        shutil.rmtree(path)
        check_path(path)
        print('Directory was deleted')
        return True


def create_path_dict_on_mask(dir_files, mask_list):
    dir_files1, dir_mask = dir_files
    files1 = glob.glob(f'{dir_files1}/*.nii.gz')
    mask_files = glob.glob(f'{dir_mask}/*.nii.gz')
    result_dict = {}
    for mask in mask_list:
        for filename1 in [x for x in files1 if mask in x]:
            short_name1 = os.path.basename(filename1)
            elem_id = short_name1[:short_name1.find('_')]
            filename_mask = next(x for x in mask_files if elem_id in x)
            result_dict[elem_id] = (filename1, filename_mask, mask)

    return result_dict


class CustomPictDataset(Dataset):
    def __init__(self, dataset_path, domain_mask, save_dir=None, itt_value='z',
                 load_dir=None, transform=transforms.Compose([transforms.Resize([int(256), int(256)]),
                                                              transforms.PILToTensor()])):
        self.domain_choosen = True
        self.df = None
        #         self.path_dict = create_path_dict_on_mask(dataset_path, domain_mask)
        #         self.transform = transform
        if not load_dir:
            self.path_dict = create_path_dict_on_mask(dataset_path, domain_mask)
            if save_dir:
                self.create_back(save_dir)
        else:
            self.load_back(load_dir)

        self.transform = transform

    def load_back(self, load_dir):
        with open(f'{load_dir}/file_inf.pickle', 'rb') as f:
            self.path_dict = pickle.load(f)

    def create_back(self, save_dir, rewrite_flag=False):

        if os.path.exists(save_dir):
            print(f'{save_dir} already exist.')

            if rewrite_flag:
                print('Rewriting directory')
                shutil.rmtree(save_dir)
            else:
                print('Stopping')
                return

        os.mkdir(save_dir)

        with open(f'{save_dir}/file_inf.pickle', 'wb') as f:
            pickle.dump(self.path_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def domain_preproc(self, path, domain_name, save_df=None, load_df=None, itt_value='x', rewrite_flag=False):
        if os.path.exists(path):
            print('This directory already exist')

            if rewrite_flag:
                print('Rewriting directory')
                shutil.rmtree(path)
                self.df = images_with_masks(self.path_dict, path, domain_name, save_df=save_df, itt=itt_value)
            else:
                print('Using existed directory.')
                if not os.path.exists(f'{path}/df_save.csv'):
                    raise ValueError(f'No df backup file in {path}. Change flag and try again')

                self.df = pd.read_csv(f'{path}/df_save.csv')
        else:
            self.df = images_with_masks(self.path_dict, path, domain_name, save_df=save_df, itt=itt_value)

    def __len__(self):
        if self.domain_choosen:
            return len(self.df)
        else:
            print('You forget to choose current domain')

    def split(self, train_path, val_path, test_path=None):
        print('!')
        with open(train_path, 'r') as tr:
            json_train = json.loads(tr.read())
        with open(val_path, 'r') as vl:
            json_val = json.loads(vl.read())

        if test_path:
            with open(test_path, 'r') as tst:
                json_test = json.loads(tst.read())

            self.df_test = self.df[self.df.file_id.isin(json_test)]

        self.df_val = self.df[self.df.file_id.isin(json_val)]

        self.df_train = self.df[self.df.file_id.isin(json_train)]

    def set_current(self, current_val, back=False):
        data_dict = {'val': self.df_val, 'test': self.df_test, 'train': self.df_train}
        if self.back:
            self.df = self.tmp
            return
        else:
            self.tmp = self.df
            self.df = data_dict[current_val]

    def __getitem__(self, idx):
        if self.domain_choosen:
            pict, mask, _, _ = self.df.iloc[idx]
            brain = self.transform(Image.open(pict).convert('L'))
            brain = brain / brain.max()
            mask = self.transform(Image.open(mask).convert('L'))
            mask = (mask > 200).to(mask.dtype)
            return {'image': brain, 'mask': mask}
        else:
            print('You forget to choose current domain')


if __name__ == '__main__':
    dataset_path = ('images', 'masks')
    domain_mask = ['philips_15', 'philips_3', 'siemens_3', 'siemens_15', 'ge_3', 'ge_15']
    TD = CustomPictDataset(dataset_path, domain_mask, save_dir='BackUp', itt_value='x')
    TD.domain_preproc('./siemens3', 'siemens_3', save_df=True, rewrite_flag=True)
    print('siemens3 finished!')
    TD.domain_preproc('./siemens15', 'siemens_15', save_df=True, rewrite_flag=True)
    print('siemens15 finished!')
    TD.domain_preproc('./ge3', 'ge_3', save_df=True, rewrite_flag=True)
    print('ge3 finished!')
    TD.domain_preproc('./ge15', 'ge_15', save_df=True, rewrite_flag=True)
    print('ge15 finished!')
    TD.domain_preproc('./philips3', 'philips_3', save_df=True, rewrite_flag=True)
    print('philips3 finished!')
    TD.domain_preproc('./philips15', 'philips_15', save_df=True, rewrite_flag=True)

