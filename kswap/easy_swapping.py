import os.path
import numpy as np
import torch
import json
from os.path import join as jp
from kswap.module.unet import UNet2D
import random

class MRIs:

    def __init__(self, img_style, img_smntcs):

        """
        This class is for *source* -> *target* style swap

        1) img_style might be either a list of the 'style' slices or a single style slice
        In case of >1 style slices, their spectra are averaged
        2) the same input images size is expected
        """

        # CUDA_DEVICE -- to re-write in a smarter way 
        self.CUDA_DEVICE = 3
        torch.cuda.set_device(self.CUDA_DEVICE)
        
        if type(img_style) != list:
            img_style = [img_style, ]

        self.img_style = img_style
        self.rft_img_style = [torch.fft.fftshift(torch.fft.fft2(torch.from_numpy(el).cuda(self.CUDA_DEVICE)),
                                                 dim=(-2, -1)) for el in self.img_style]

        self.img_smntcs = img_smntcs
        self.rft_img_smntcs = torch.fft.fftshift(torch.fft.fft2(torch.from_numpy(self.img_smntcs).cuda(self.CUDA_DEVICE),
                                                                dim=(-2, -1)))  # should we really do cloning here?

        self.h, self.w = img_smntcs.shape
        self.size = min([self.h, self.w])
        self.centre = [self.h // 2, self.w // 2]


    def _swap_low_freq(self, beta, with_phase=True):
        beta = int(beta * self.size)
        mask = torch.zeros([self.h, self.w]).cuda(self.CUDA_DEVICE)
        mask[self.centre[0] - beta:self.centre[0] + beta, self.centre[1] - beta:self.centre[1] + beta] = 1
        return self.swap_freq(mask, with_phase)

    def swap_low_freq_circle(self, beta, with_phase=False):
        beta = beta * self.size
        x, y = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2)
        mask = torch.from_numpy(dist_from_center < beta).long().cuda(self.CUDA_DEVICE)
        return self.swap_freq(mask, with_phase)

    def swap_freq(self, mask, with_phase):
        """
        we average over the source scans (in case their quantity is > 1)
        """
        if with_phase:
            avg_rft_img_style = self.average_style(with_phase=True)  # averaging the style images
            new_rft_img_smntcs = (1 - mask) * self.rft_img_smntcs + mask * avg_rft_img_style
        else:
            rft_img_style_amp = self.average_style(with_phase=False)
            rft_img_smntcs_amp, rft_img_smntcs_pha = self.rft_img_smntcs.abs(), self.rft_img_smntcs.angle()
            rft_img_smntcs_amp = (1 - mask) * rft_img_smntcs_amp + mask * rft_img_style_amp
            new_rft_img_smntcs = torch.cos(rft_img_smntcs_pha) * rft_img_smntcs_amp + \
                                 1j * torch.sin(rft_img_smntcs_pha) * rft_img_smntcs_amp

        return self.back_to_pic(new_rft_img_smntcs)

    def back_to_pic(self, new_rft_img_smntcs):

        new_rft_img_smntcs = torch.fft.ifftshift(new_rft_img_smntcs, dim=(-2, -1))
        img_complex = torch.fft.ifft2(new_rft_img_smntcs)
        img_mixed = img_complex.abs()
        img_mixed = img_mixed / img_mixed.max()
        return img_mixed, img_complex

    def average_style(self, with_phase=True):
        if with_phase:
            # average the whole signal
            return torch.mean(torch.stack(self.rft_img_style), dim=0)
        else:
            # average the amplitude
            return torch.mean(torch.stack([el.abs() for el in self.rft_img_style]), dim=0)


class DA_experiment(MRIs):

    def __init__(self, sourceID, targetID, dataset, predict, sdice_metric,
                 source_test_ids = None, target_test_ids = None,
                 mode=None, closest_scans_mode = None, n_scans = 7, shift=0):
        """
        This class is for testing Domain Adaptation (DA) between 2 pairs.

        Init structure:
        1. necessary enteties for operating with the data
        2. loading closeness dicts
        3. loading the model
        4. Loading the IDs
        """

        # Some necessary enteties for operating with the data
        self.CUDA_DEVICE = 3
        torch.cuda.set_device(self.CUDA_DEVICE)
        self.id2dom = {0: 'sm15', 1: 'sm3', 2: 'ge15', 3: 'ge3', 4: 'ph15', 5: 'ph3'}
        self.shift = shift
        self.n_scans = n_scans
        self.closest_scans_mode = closest_scans_mode

        self.dataset = dataset
        self.predict = predict
        self.sdice_metric = sdice_metric

        self.pair2exp_num = self.pair2exp_num_func()
        self.split_id = self.pair2exp_num[(sourceID, targetID)]

        # Loading closeness dicts
        closest_scans_base_path = '/mnt/nfs_storage/name/exps/closest_scans'
        if closest_scans_mode is not None:
            if mode == 'val':
                self.closest_scans_path = jp(closest_scans_base_path, 'closest_scans_val_' +
                                             str(self.split_id) + '_' + closest_scans_mode + '.json')
            elif mode == 'test':
                self.closest_scans_path = jp(closest_scans_base_path, 'closest_scans_test_' +
                                             str(self.split_id) + '_' + closest_scans_mode + '.json')
            else:
                raise ValueError
            with open(self.closest_scans_path) as fp:
                self.sim_scores_dict = json.load(fp) # [str(self.split_id)]

        # Loading the model
        self.base = '/mnt/nfs_storage/name/exps/baseline_w_dice'
        self.source_path = jp(self.base, 'experiment_' + str(sourceID))
        self.source_model_path = jp(self.source_path, 'model.pth')
        self.source_model = UNet2D(n_chans_in=1, n_chans_out=1, n_filters_init=16)
        self.source_model.cuda(self.CUDA_DEVICE)
        self.source_model.load_state_dict(torch.load(self.source_model_path))

        self.path_split = os.path.join('/mnt/nfs_storage/name/exps/split/experiment_' + str(self.split_id))

        # Loading the IDs
        if mode is None:
            self.source_test_ids = source_test_ids
            self.target_test_ids = target_test_ids
        else:
            self.source_test_ids_path = jp(self.path_split, 'train_s_ids.json')
            if mode == 'val':
                self.target_test_ids_path = jp(self.path_split, 'val_t_ids.json')
            elif mode == 'test':
                self.target_test_ids_path = jp(self.path_split, 'test_t_ids.json')
            else:
                raise ValueError
            self.source_test_ids = self.read_json(self.source_test_ids_path)
            self.target_test_ids = self.read_json(self.target_test_ids_path)

    def exp(self, method, **kwargs):
        res = {}
        for t_id in self.target_test_ids:
            res[t_id] = self.get_score_3d(t_id, method, **kwargs)
        return res

    def get_score_3d(self, t_id, method, **kwargs):
        _, t_mask = self.get_scan(t_id)
        t_scans_new = self.style_transfer(t_id, method, **kwargs)
        preds = []
        for t_scan_new in t_scans_new:
            pred = self.predict(t_scan_new, net=self.source_model)
            preds.append(pred)
        pred = np.mean(preds, axis=0)
        sdice_pred = self.sdice_metric(pred, t_mask)
        return sdice_pred

    def style_transfer(self, t_id, method, **kwargs):
        random.seed(42) # in case we pick source slices randomly
        n = self.n_scans
        t_scan, _ = self.get_scan(t_id)
        z_len = t_scan.shape[-1]
        processed_scans = {}

        for i in range(n):
            processed_scans[i] = []
        for i in range(z_len):
            for j in range(n):
                if self.closest_scans_mode is None:
                    position = 0
                    s_id = random.choice(self.source_test_ids)
                else:
                    s_id, position, _ = self.sim_scores_dict[t_id][str(i)][j]

                s_scan, s_mask = self.get_scan(s_id)
                super().__init__(img_style=s_scan[:, :, i+position], img_smntcs=t_scan[:, :, i])
                slc_mixed, _ = method(**kwargs)
                processed_scans[j].append(slc_mixed)

        processed_scans_list = []
        for i in range(n):
            processed_scans_list.append(torch.stack(processed_scans[i], dim=2).cpu().numpy())

        return processed_scans_list

    def get_scan(self, t_id):
        t = self.dataset.load_image(t_id)
        t_mask = self.dataset.load_segm(t_id)
        return t, t_mask

    def pair2exp_num_func(self):
        pair2exp_num = {}
        i = 0
        for d1 in range(6):
            for d2 in range(6):
                if d1 != d2:
                    pair2exp_num[(d1, d2)] = i
                    i += 1
        return pair2exp_num

    def read_json(self, addr):

        f = open(addr)
        data = json.load(f)
        f.close()
        return data
