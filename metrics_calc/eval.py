import json
import os.path

import yaml
from itertools import zip_longest
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap
from dpipe.predict import divisible_shape, add_extract_dims
from dpipe.batch_iter import unpack_args
from dpipe.torch import inference_step
from torchvision import transforms

from dpipe.im.metrics import dice_score

from functions.dataloader_functions import scale_mri
from functions.PredictVgg import predict_vgg
from functions.mask import *
from functions.preprocessing import *
import nibabel as nib

import numpy as np
from PIL import Image
import surface_distance.metrics as surf_dc

from models.model_ver2 import UNet2D


def sdice(a, b, spacing, tolerance=1):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def get_params(root):
    with open(os.path.join(root, "configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    params = {'train_data': configs['paths']['data']['train_data'],
              'dataset_table_path': configs['paths']['dataset_table'],
              'log_dir': configs['paths']['log_dir']}

    for param in params.keys():
        params[param] = os.path.join(root, params[param])

    params.update({'n_channels': int(configs['data_parameters']['n_channels']),
                   'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size'])})

    params.update({'init_features': int(configs['model_parameters']['UNet']['init_features']),
                   'depth': int(configs['model_parameters']['UNet']['depth'])})

    params.update({'num_init_features': int(configs['model_parameters']['DenseNet']['num_init_features']),
                   'growth_rate': int(configs['model_parameters']['DenseNet']['growth_rate']),
                   'block_config': tuple(
                       map(int, configs['model_parameters']['DenseNet']['block_config'].split(', ')))})

    params.update({'blocks': tuple(map(int, configs['model_parameters']['ResNet']['blocks'].split(', ')))})

    params.update({'fourier_layer': configs['model_parameters']['Fourier']['fourier_layer']})

    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})


def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper


SPATIAL_DIMS = (-3, -2, -1)


def get_pred(x, threshold=0.5):
    return x > threshold


def resize_mask(mask, size):
    result_list = np.zeros(shape=(size[0], size[1], 172))

    t = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    for pos, mask_2d in enumerate(np.rollaxis(mask, 2)):
        img = (t(Image.fromarray((mask_2d * 255).astype(np.uint8))).detach().numpy()).astype(np.float32)
        result_list[..., pos] = img

    return result_list


def save_result(array, mask, dir_name):
    for pos, (ar, mk) in enumerate(zip(np.rollaxis(array, 2), np.rollaxis(mask, 2))):
        ar = ar[np.newaxis, ...]
        mk = mk[np.newaxis, ...]

        ar = (ar * 255).astype(np.uint8)
        mk = (mk * 255).astype(np.uint8)

        diff = np.concatenate((ar, mk, np.zeros_like(ar).astype(np.uint8)), axis=0)
        diff = np.moveaxis(diff, 0, -1)

        im = Image.fromarray(diff)
        im = im.convert("RGB")
        im.save(f'{dir_name}/lvl{pos}.png')


###RESIZE APPROACH
def basic_eval(net_list, path, mask, donor, gan_file, resize_flag=True):
    net, net_own = net_list

    target = scale_mri(np.array(nib.load(path).dataobj).astype(np.float32)[..., :172])
    donor = scale_mri(np.array(nib.load(donor).dataobj).astype(np.float32)[..., :172])
    mask = scale_mri(np.array(nib.load(mask).dataobj).astype(np.float32)[..., :172])
    size_z, size_y, _ = target.shape

    if gan_file[0] is not None:
        gan_style = scale_mri(np.array(nib.load(gan_file[0]).dataobj).astype(np.float32))
        gan_style = resize_mask(gan_style, (size_z, size_y))
        gan_style = gan_style.astype(np.float32)

    if gan_file[1] is not None:
        gan_cycle = scale_mri(np.array(nib.load(gan_file[1]).dataobj).astype(np.float32))
        gan_cycle = resize_mask(gan_cycle, (size_z, size_y))
        gan_cycle = gan_cycle.astype(np.float32)

    # predict
    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=net, activation=torch.sigmoid)

    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict_own(image):
        return inference_step(image, architecture=net_own, activation=torch.sigmoid)

    sdice_metric = lambda x, y: sdice(get_pred(x), get_pred(y), (1, 1, 1), 1)
    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))

    result = predict_own(target)
    current_domain_sdice = sdice_metric(result, mask)
    current_domain_dice = dice_metric(result, mask)
    print(current_domain_dice, current_domain_sdice, 'current')

    result = predict(target)
    baseline_sdice = sdice_metric(result, mask)
    baseline_dice = dice_metric(result, mask)

    print(baseline_sdice, baseline_dice, 'baseline')

    styled = predict_vgg(target, donor, resize_flag).astype(np.float32)

    styled = np.clip(styled, 0, 1)
    result = predict(styled)
    styled_sdice = sdice_metric(result, mask)
    styled_dice = dice_metric(result, mask)
    print(styled_sdice, styled_dice, 'vgg16')

    if gan_file[0] is not None:
        result = predict(gan_style)
        gan_sdice_style = sdice_metric(result, mask)
        gan_dice_style = dice_metric(result, mask)
        print(gan_sdice_style, gan_dice_style, 'styleGan')

    if gan_file[1] is not None:
        result = predict(gan_cycle)
        gan_sdice_cycle = sdice_metric(result, mask)
        gan_dice_cycle = dice_metric(result, mask)
        print(gan_sdice_cycle, gan_dice_cycle, 'cycle')

    return (current_domain_sdice, current_domain_dice), (baseline_sdice, baseline_dice), (styled_sdice, styled_dice), \
           (gan_sdice_style, gan_dice_style), (gan_sdice_cycle, gan_dice_cycle),


def create_df(image_path, mask_path):
    result_data = []
    all_images = glob.glob(image_path + '*')
    all_masks = glob.glob(mask_path + '*')
    for image in all_images:
        im = os.path.basename(image)
        im_id = im[im.rfind('/') + 1: im.find('_')]
        domain_st = im.find('_') + 1  # find start
        div_dom = im.find('_', domain_st + 1)  # find _ between domain name and number

        domain = im[domain_st: im.find('_', div_dom + 1)]

        mask = next(x for x in all_masks if im_id in x)

        result_data.append((im_id, image, mask, domain))
    df = pd.DataFrame(result_data, columns=['id', 'image_path', 'mask_path', 'domain'])
    df.set_index('id', inplace=True)
    return df


def find_gan_file(path, dir_name, id):
    gan_path = path + '/' + dir_name
    files = glob.glob(f'{gan_path}/*.nii.gz')
    file = next(x for x in files if id in x)
    return file


def draw_one(img_3d, pos, name, mask=None):
    img = img_3d[..., pos]
    sdice_metric = lambda x, y: sdice(get_pred(x), get_pred(y), (1, 1), 1)
    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))

    im = Image.fromarray(np.uint8(img * 255)).resize((256, 256))
    im.save(name + '.jpg')
    if type(mask) == np.ndarray:
        print('__________________________________________')
        print(name, round(sdice_metric(img, mask), 3), round(dice_metric(img, mask), 3))
        print('__________________________________________')


def calculate_current():
    main_path = '/raid/data/DA_BrainDataset/'
    gan_path_style = main_path + 'predictions/StyleGAN'
    gan_path_cycle = main_path + 'predictions/cyclegan'

    df = create_df(main_path + 'images/', main_path + 'masks/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)

    weight = 'weights/model_1.pth'
    net.load_state_dict(torch.load(weight, map_location=device))

    # predict
    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=net, activation=torch.sigmoid)

    sdice_metric = lambda x, y: sdice(get_pred(x), get_pred(y), (1, 1, 1), 1)
    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))

    name = 'sie3_to_ge15'

    id = 'CC0276'
    # id = 'CC0156'
    row = df.loc[id]
    path, mask = row.image_path, row.mask_path
    donor = main_path + 'images/' + 'CC0226_siemens_3_64_F.nii.gz'
    # donor = main_path + 'images/' +'CC0276_ge_15_52_M.nii.gz'
    gan_file_style = find_gan_file(gan_path_style, 'ge15_to_sm3', id)
    gan_file_cycle = find_gan_file(gan_path_cycle, 'ge15_to_sm3', id)

    target = scale_mri(np.array(nib.load(path).dataobj).astype(np.float32)[..., :172])
    _, _, z = target.shape
    donor = scale_mri(np.array(nib.load(donor).dataobj).astype(np.float32)[..., :172])
    mask = scale_mri(np.array(nib.load(mask).dataobj).astype(np.float32)[..., :172])

    gan_style = scale_mri(np.array(nib.load(gan_file_style).dataobj).astype(np.float32))
    gan_cycle = scale_mri(np.array(nib.load(gan_file_cycle).dataobj).astype(np.float32))

    result = predict(target)
    baseline_sdice = sdice_metric(result, mask)
    baseline_dice = dice_metric(result, mask)

    size_z, size_y, _ = target.shape
    gan_style = resize_mask(gan_style, (size_z, size_y))
    gan_style = gan_style.astype(np.float32)

    gan_cycle = resize_mask(gan_cycle, (size_z, size_y))
    gan_cycle = gan_cycle.astype(np.float32)

    # predict
    @slicewise  # 3D -> 2D iteratively
    @add_extract_dims(2)  # 2D -> (4D -> predict -> 4D) -> 2D
    @divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
    def predict(image):
        return inference_step(image, architecture=net, activation=torch.sigmoid)

    sdice_metric = lambda x, y: sdice(get_pred(x), get_pred(y), (1, 1, 1), 1)
    dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))

    # draw_one(mask, 86, 'mask_original', mask[..., 86])
    draw_one(mask, 26, 'mask_original_26', mask[..., 26])
    draw_one(mask, 56, 'mask_original_56', mask[..., 56])
    draw_one(mask, 86, 'mask_original_86', mask[..., 86])
    draw_one(mask, 116, 'mask_original_116', mask[..., 116])
    draw_one(mask, 146, 'mask_original_146', mask[..., 146])
    # save_result(result, mask,'ge_15_result')
    draw_one(target, 26, 'target_baseline_26')
    draw_one(target, 56, 'target_baseline_56')
    draw_one(target, 86, 'target_baseline_86')
    draw_one(target, 116, 'target_baseline_116')
    draw_one(target, 146, 'target_baseline_146')

    result = predict(target)
    baseline_sdice = sdice_metric(result, mask)
    baseline_dice = dice_metric(result, mask)
    result = result > 0.5
    draw_one(result, 26, 'mask_baseline_26', mask[..., 26])
    draw_one(result, 56, 'mask_baseline_56', mask[..., 56])
    draw_one(result, 86, 'mask_baseline_86', mask[..., 86])
    draw_one(result, 116, 'mask_baseline_116', mask[..., 116])
    draw_one(result, 146, 'mask_baseline_146', mask[..., 146])

    print(baseline_sdice, baseline_dice, 'baseline')
    # save_result(result, mask,'ge_15_result')

    styled = predict_vgg(target, donor, False).astype(np.float32)

    styled = np.clip(styled, 0, 1)

    draw_one(styled, 26, 'styled_basic_26', mask[..., 26])
    draw_one(styled, 56, 'styled_basic_56', mask[..., 56])
    draw_one(styled, 86, 'styled_basic_86', mask[..., 86])
    draw_one(styled, 116, 'styled_basic_116', mask[..., 116])
    draw_one(styled, 146, 'styled_basic_146', mask[..., 146])

    result = predict(styled)
    styled_sdice = sdice_metric(result, mask)
    styled_dice = dice_metric(result, mask)
    print(styled_sdice, styled_dice, 'vgg16')
    result = result > 0.5
    draw_one(result, 26, 'styled_mask_26', mask[..., 26])
    draw_one(result, 56, 'styled_mask_56', mask[..., 56])
    draw_one(result, 86, 'styled_mask_86', mask[..., 86])
    draw_one(result, 116, 'styled_mask_116', mask[..., 116])
    draw_one(result, 146, 'styled_mask_146', mask[..., 146])

    draw_one(gan_style, 26, 'gan_style_basic_26', mask[..., 26])
    draw_one(gan_style, 56, 'gan_style_basic_56', mask[..., 56])
    draw_one(gan_style, 86, 'gan_style_basic_86', mask[..., 86])
    draw_one(gan_style, 116, 'gan_style_basic_116', mask[..., 116])
    draw_one(gan_style, 146, 'gan_style_basic_146', mask[..., 146])

    result = predict(gan_style)
    result = result > 0.5
    draw_one(result, 86, 'gan_style_mask', mask[..., 86])

    draw_one(result, 26, 'gan_style_mask_26', mask[..., 26])
    draw_one(result, 56, 'gan_style_mask_56', mask[..., 56])
    draw_one(result, 86, 'gan_style_mask_86', mask[..., 86])
    draw_one(result, 116, 'gan_style_mask_116', mask[..., 116])
    draw_one(result, 146, 'gan_style_mask_146', mask[..., 146])

    gan_sdice_style = sdice_metric(result, mask)
    gan_dice_style = dice_metric(result, mask)
    print(gan_sdice_style, gan_dice_style, 'styleGan')

    draw_one(gan_cycle, 86, 'gan_cycle_basic')

    draw_one(gan_cycle, 26, 'gan_cycle_basic_26', mask[..., 26])
    draw_one(gan_cycle, 56, 'gan_cycle_basic_56', mask[..., 56])
    draw_one(gan_cycle, 86, 'gan_cycle_basic_86', mask[..., 86])
    draw_one(gan_cycle, 116, 'gan_cycle_basic_116', mask[..., 116])
    draw_one(gan_cycle, 146, 'gan_cycle_basic_146', mask[..., 146])

    result = predict(gan_cycle)
    result = result > 0.5
    draw_one(result, 26, 'gan_cycle_mask_26', mask[..., 26])
    draw_one(result, 56, 'gan_cycle_mask_56', mask[..., 56])
    draw_one(result, 86, 'gan_cycle_mask_86', mask[..., 86])
    draw_one(result, 116, 'gan_cycle_mask_116', mask[..., 116])
    draw_one(result, 146, 'gan_cycle_mask_146', mask[..., 146])

    gan_sdice_cycle = sdice_metric(result, mask)
    gan_dice_cycle = dice_metric(result, mask)
    print(gan_sdice_cycle, gan_dice_cycle, 'cycle')


def test():
    # TODO make paths from script

    with open(os.path.join('.', "eval_configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)


    main_path = configs['paths']['main_path']
    gan_path_style = configs['paths']['gan_path']
    gan_path_cycle = configs['paths']['gan_path_cycle']
    name_values = configs['list_values']['domain_names']

    df = create_df(main_path + 'images/', main_path + 'masks/')

    gan_values = configs['list_values']['gan_values']

    weights_names_predict = configs['list_values']['weights_names_predict']
    weigths_names_own = configs['list_values']['weigths_names_own']
    data_directories = configs['list_values']['data_directories']

    result = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
    net_own = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to(device)
    for name, (test_domain, learned_domain), weight_p, weight_o, gan_dir in zip_longest(name_values, data_directories,
                                                                  weights_names_predict, weigths_names_own, gan_values):
        net.load_state_dict(torch.load(weight_p, map_location=device))
        net_own.load_state_dict(torch.load(weight_o, map_location=device))

        df_donor = df[df.domain == test_domain]
        with open(main_path + 'split_settings/' + name + '/test_t_ids.json', 'r') as tr:
            json_train = json.loads(tr.read())

        with open(main_path + 'split_settings/' + name + '/val_t_ids.json', 'r') as tr:
            json_val = json.loads(tr.read())

        for json_file, is_column_val in zip([json_val, json_train], [True, False]):
            for file_id in json_file:
                gan_file_style = find_gan_file(gan_path_style, gan_dir, file_id)
                gan_file_cycle = find_gan_file(gan_path_cycle, gan_dir, file_id)
                row = df.loc[file_id]
                path, mask = row.image_path, row.mask_path
                path_donor = df_donor.sample().image_path.values[0]

                short_name_path = os.path.basename(path)
                short_name_donor = os.path.basename(path_donor)

                current, baseline, styled, st_gan, cy_gan = basic_eval([net, net_own], path, mask, path_donor,
                                                                       [gan_file_style, gan_file_cycle])

                result.append((file_id, short_name_path, short_name_donor, current[0], current[1], baseline[0], styled[0],
                               baseline[1], styled[1],
                               st_gan[0], st_gan[1], cy_gan[0], cy_gan[1], is_column_val))

                df_result = pd.DataFrame(result, columns=['id', 'target', 'donor', 'current_sdice', 'current dice',
                                                          'baseline_sdice', 'styled_sdice', 'baseline_dice',
                                                          'styled_dice', 'StyleGan_sdice', 'StyleGan_dice',
                                                          'CycleGan_sdice',
                                                          'CycleGan dice', 'is_val'])
            df_result['mean_baseline'] = df_result.baseline_sdice.mean()
            df_result['mean_styled'] = df_result.styled_sdice.mean()
            df_result['mean_GANstyle'] = df_result.StyleGan_sdice.mean()
            df_result['mean_CYCLEstyle'] = df_result.CycleGan_sdice.mean()
            df_result['mean_current'] = df_result.current_sdice.mean()

            print(df_result)
            if not os.path.exists('csv_files'):
                os.mkdir('csv_files')

            df_result.to_csv(f'csv_files/{name}_val_{is_column_val}.csv', index=False)  # To Delete
            result.clear()


def count_mean():
    df = pd.read_csv('csv_files/sie3_to_ge15_val_True_resize_True.csv')
    print(f'mean, baseline - {df.baseline_sdice.mean()}')
    print(f'mean, styled -  {df.styled_sdice.mean()}')


def test_csv():
    with open(os.path.join('.', "eval_configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    print(configs['paths'])
    print(configs['list_values'])


if __name__ == '__main__':
    # net = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16).to('cuda')
    # net.load_state_dict(torch.load('weights/model_0.pth', map_location='cuda'))
    # path = '/raid/data/DA_BrainDataset/images/CC0240_ge_15_60_F.nii.gz'
    # donor = '/raid/data/DA_BrainDataset/images/CC0240_ge_15_60_F.nii.gz'
    # mask = '/raid/data/DA_BrainDataset/masks/CC0200_siemens_3_37_F.nii.gz'
    # basic_eval(net, path, mask, path)
    # count_mean()
    # test()
    test()
    # test_csv()