import os
import numpy as np
import nibabel as nib
from PIL import Image
import glob
import csv
import pandas as pd
from collections import defaultdict

# from tqdm.notebook import tqdm
from dpipe.im.shape_ops import zoom


def resample(num_slices, amount):
    return np.linspace(0, num_slices - 1, amount).astype(int)


def find_max_content(values):
    max_val = 0
    for row in values:
        _, st, en = row
        max_val = en - st if en - st > max_val else max_val
    return max_val


##Better with regexp, but NO
def files_preprocessing(main_directory, mask_directory, domain_list, mask_tail=None):
    result_files = list()
    dataset_brain_images = [file for file in glob.iglob(f'{main_directory}/*.nii.gz')]

    for file_path in dataset_brain_images:
        file_in_domain = False

        for domain in domain_list:
            if domain in file_path:
                file_in_domain = True
                break

        if not file_in_domain:
            continue

        if mask_directory:
            file_name = file_path[file_path.rfind('/') + 1: file_path.find('.')]
            file_name += mask_tail
            mask_path = f'{mask_directory}/{file_name}.nii.gz'
            result_files.append((file_path, mask_path))

        else:
            result_files.append(file_path)

    return result_files


def get_data_analyse(files):
    mark_dict = {'philips': {'M': set(), 'F': set()}, 'siemens': {'M': set(), 'F': set()},
                 'ge': {'M': set(), 'F': set()}}
    mark = ('philips', 'siemens', 'ge')
    data = list()
    sex = ('M', 'F')
    max_levels = -1
    x_dict, y_dict, z_dict = defaultdict(int), defaultdict(int), defaultdict(int)

    for file_path, mask_path in files:
        st, en = -1, -1
        img = nib.load(file_path)

        if mask_path:
            mask = nib.load(mask_path)

        z, y, x = img.shape

        z_dict[z] += 1
        y_dict[y] += 1
        x_dict[x] += 1

        # Fix hardcode
        img = np.array(img.dataobj)
        if mask_path:
            mask = np.array(mask.dataobj)

            for level in range(z):
                bit_context_amount = mask[level][:][:].sum()
                # 5 for example
                if bit_context_amount > 5 and st == -1:
                    st = level
                elif bit_context_amount > 5:
                    en = level
        else:
            st, en = 0, z

        max_levels = en - st if en - st > max_levels else max_levels

        for device in mark:
            if device in file_path:
                break

        for gender in sex:
            if gender in file_path:
                break

        mark_dict[device][gender].add((file_path, st, en, en - st, (en - st) / z))

        file_res = (file_path, mask_path) if mask_path else (file_path,)
        data.append((file_res, st, en))

    elem_dict = {'x': x_dict, 'y': y_dict, 'z': z_dict}
    return mark_dict, elem_dict, data, max_levels


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


def scale_factor(pict_id, df, new_vowel=(1, 0.95, 0.95)):
    old_voxel_spacing = np.array([df['x'].loc[pict_id], df['y'].loc[pict_id], df['z'].loc[pict_id]])
    scale_factor = old_voxel_spacing / new_vowel
    return np.nan_to_num(scale_factor, nan=1)


def change_2d(x, pict_id, df, spacing_order=3, with_perc=False):
    pict = zoom(x, scale_factor(pict_id, df)[1:], order=spacing_order)
    return scale_mri(pict) if with_perc else pict


def save_images_with_masks(values, path, height, df):
    ((file_name, mask), st, en) = values

    current_file_dir = file_name[file_name.rfind('/') + 1: file_name.find('.')]
    os.mkdir(f'{path}/train/{current_file_dir}')
    os.mkdir(f'{path}/mask/{current_file_dir}')

    img = nib.load(file_name)
    img = np.array(img.dataobj)
    img = img[st:en, :, :]
    z, _, _ = img.shape

    img_mask = np.array(nib.load(mask).dataobj)
    img_mask = img_mask[st:en, :, :]

    resample_positions = resample(z, height - z)
    # Insert copies of elements
    for pos in resample_positions:
        img = np.insert(img, pos, img[pos][:][:], axis=0)
        img_mask = np.insert(img_mask, pos, img_mask[pos][:][:], axis=0)

    z, _, _ = img.shape
    for i in range(z):
        res_im = img[i][:][:]
        res_mask = img_mask[i][:][:]

        file_id = file_name[file_name.rfind('/') + 1: file_name.find('_')]

        im = Image.fromarray((change_2d(res_im, file_id, df, with_perc=True) * 255).astype(np.uint8))

        im_mask = Image.fromarray((change_2d(res_mask, file_id, df, with_perc=True) * 255).astype(np.uint8))

        im.save(f'{path}/train/{current_file_dir}/lvl{i}.png')
        im_mask.save(f'{path}/mask/{current_file_dir}/lvl{i}.png')


def data_preparing(path, dataset_info, height, domain_name, df, with_masks=True):
    if not os.path.exists(f'{path}'):
        os.mkdir(f'{path}')
        os.mkdir(f'{path}/train')
        if with_masks:
            os.mkdir(f'{path}/mask')

    else:
        print('Directory with this name already created')
        return

    for values in dataset_info:

        if with_masks:
            ((file_name, _), _, _) = values

            if domain_name in file_name:
                save_images_with_masks(values, path, height, df),

        else:
            file_name = values

            # if domain_name in file_name:
            #     save_only_images(file_name, path, height, df, final_size)


def create_dataframe_from_scalefile(dataset_files, old_csv_name):
    old_df = pd.read_csv(old_csv_name)
    old_df.set_index('id', inplace=True)
    old_df = old_df[['x', 'y', 'z']]
    column_names = ['name'] if len(dataset_files[0]) != 2 else ['name', 'mask']

    df = pd.DataFrame.from_records(dataset_files, columns=column_names)
    # Need fix
    df['id'] = df['name'].str[17: 23]
    df = df.set_index('id').join(old_df)

    return df


def create_dataframe_from_path(path):
    files_brain = glob.glob(f'{path}/' + 'train/**/*.png', recursive=True)
    files_mask = glob.glob(f'{path}/' + 'mask/**/*.png', recursive=True)
    data = [(brain, mask) for brain, mask in zip(files_brain, files_mask)]
    return pd.DataFrame.from_records(data, columns=['image', 'mask'])


# A little bit strange, but according to upper code it's ok
def create_csv_table(path, csv_name):
    files_brain = glob.glob(f'{path}/' + 'train/**/*.png', recursive=True)
    files_mask = glob.glob(f'{path}/' + 'mask/**/*.png', recursive=True)
    with open(f'{csv_name}.csv', 'w', newline='') as csvfile:
        fieldnames = ['brain', 'mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for brain, mask in zip(files_brain, files_mask):
            writer.writerow({'brain': brain, 'mask': mask})




dataset_path = ('Dataset/Original', 'Dataset/Mask')
domain_mask = ['philips_15', 'philips_3', 'siemens_3']

## dataset_path - tuple of path to target and mask
## domain_mask - tuple, which consist name of domains to analyse
## mask_tail - tail of the mask TODO - make it parameter RegExp
## path_to_csv - path to csv with scaling
