from collections import defaultdict

import numpy as np

from torch.utils.data import DataLoader

from DA_dataloader import CustomPictDataset


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    #     mask = (radius == dist_from_center) or (dist_from_center < (radius + 1))
    mask = np.logical_or(radius == dist_from_center, dist_from_center < (radius + 1))
    return mask

def get_image_sum(image, values_amp, values_phase):
    fourier = np.fft.fft2(np.asarray(image))
    image_amp, image_phase = np.absolute(fourier), np.angle(fourier)
    values_amp += image_amp
    values_phase += image_phase
    return values_amp, values_phase


def image_get_inform(image, values_amp, values_phase, debug = True):
    fourier = np.fft.fft2(np.asarray(image))

    image_amp, image_phase = np.absolute(fourier), np.angle(fourier)
    width = fourier.shape[0]
    height = fourier.shape[1]

    shift = min(width, height) - 1
    for _ in range(2):
        for r in range(min(width // 2, height // 2)):
            mask = create_circular_mask(width, height, radius=r)
            ampl_vals = image_amp[mask].flatten()

            for val in ampl_vals[ampl_vals != 0]:
                res_val = int(abs(val))
                numb_len = len(str(res_val))

                if numb_len > 2:
                    res_val = res_val // 10 ** (numb_len - 2) * 10 ** (numb_len - 2)

                if res_val in values_amp[r + shift].keys():
                    values_amp[r + shift][res_val] += 1
                else:
                    values_amp[r + shift][res_val] = 1

            phase_vals = image_phase[mask].flatten()
            if debug:
                print(phase_vals)
            for val in phase_vals[phase_vals != 0]:
                res_val = int(abs(val/ np.pi * 10000))


                if res_val in values_phase[r + shift].keys():
                    values_phase[r + shift][res_val] += 1
                else:
                    values_phase[r + shift][res_val] = 1

            shift = shift - 2 if shift > 0 else 0

        fourier = np.fft.fftshift(fourier)
        image_amp, image_phase = np.absolute(fourier), np.angle(fourier)
        shift = 0

    return values_amp, values_phase


def prepare_dict(td, one_batch=True):
    data_loader = DataLoader(td, batch_size=4, shuffle=False, drop_last=True)
    values_amp = {r: {} for r in range(300)}
    values_phase = {r: {} for r in range(300)}

    mask_amp = {r: {} for r in range(300)}
    mask_phase = {r: {} for r in range(300)}

    for (idx, batch) in enumerate(data_loader):  # Print the 'text' data of the batch
        for i in range(len(batch[0])):
            values_amp, values_phase = image_get_inform(batch[0][i], values_amp, values_phase)

        for i in range(len(batch[1])):
            mask_amp, mask_phase = image_get_inform(batch[1][i], mask_amp, mask_phase)

        if one_batch:
            break

    return values_amp, values_phase


def get_statistic(domain_dict1, domain_dict2):
    if domain_dict1:
        for key in domain_dict1.keys():
            print('For key - ', key)
            siem = domain_dict1[key].keys()
            phil = domain_dict2[key].keys()
            intersect = siem & phil
            print('Intersection: ')
            for w in intersect:
                print('Value - ', w, '  ', domain_dict1[key][w], ' - ', domain_dict2[key][w])
            if siem ^ phil:
                print('Difference: ')
                s_vals = siem - phil
                for w in s_vals:
                    print('Value - ', w, '   ', domain_dict1[key][w], ' - siemens')

                p_vals = phil - siem
                for w in p_vals:
                    print('Value - ', w, '    ', domain_dict2[key][w], ' -  philips')
            else:
                print('Where is no difference!')
    else:
        print('Dictionary is empty')

    def get_value_list(value_dict, x_lim=10 ** 7, y_lim=30000):
        x_vals, y_vals = [], []
        dropped = 0
        for x, y in value_dict.items():
            if x > x_lim or y > y_lim:
                dropped += 1
                continue
            x_vals.append(x)
            y_vals.append(y)

        return x_vals, y_vals, dropped

    def get_dict(value_dict, rad_en, rad_st=0):
        # siemens_amp, siemens_phase
        dict_siem = defaultdict(int)
        for i in range(rad_st, rad_en):
            for key in value_dict[i].keys():
                dict_siem[key] += value_dict[i][key]
        return dict_siem

if __name__ == '__main__':
    dataset_path = ('Dataset/Original', 'Dataset/Mask')
    domain_mask = ['philips_3', 'siemens_3']
    path_to_csv = './meta.csv'

    TD = CustomPictDataset(dataset_path, domain_mask, path_to_csv)
    TD.domain_preproc('./philips_15', 'philips_15')

    philips_amp, philips_phase = prepare_dict(TD)
    TD.domain_preproc('./siemens_15', 'siemens_15')

    siemens_amp, siemens_phase = prepare_dict(TD)
    get_statistic(philips_amp, siemens_amp)
