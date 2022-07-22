import numpy as np
import scipy

from dpipe.dataset.segmentation import SegmentationFromCSV
from dpipe.dataset.wrappers import Proxy
from dpipe.im.shape_ops import zoom


class CC359(SegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())

    def load_image(self, i, step=1):
        return np.float32(super().load_image(i)[0][:,:,::step])  # 4D -> 3D

    def load_images_decomposed(self, i, mask_side = 40):
        """
        image = image1 + image2;
        return (image1, image2)
        """

        img = self.load_image (i)
        cp1, cp2 = img.copy(), img.copy()

        c1,c2 = img.shape [1:]
        mask = np.full(img.shape, True, dtype=bool)
        mask[:, c1 - mask_side:c1 + mask_side, c2 - mask_side:c2 + mask_side] = False

        cp1[~mask] = 0
        cp2[mask] = 0

        return cp1, cp2

    def load_segm(self, i):
        return np.float32(super().load_segm(i))  # already 3D

    def load_image_slice(self, i, shift):
        volume = super().load_image(i)[0]  # 4D -> 3D
        n = volume.shape[-1] // 2 + shift + 15
        return np.float32(scipy.ndimage.rotate(volume[:,:,n], 90))

    def load_segm_slice(self, i, shift):
        volume = super().load_segm(i)  # already 3D
        n = volume.shape[-1] // 2 + shift + 15
        return np.float32(scipy.ndimage.rotate(volume[:,:,n], 90))

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]

    def load_domain_label_number_binary_setup(self, i, domains):
        """Assigns '1' to the domain of the largest index; '0' to another one
        Domains may be either (index1, index2) or (sample_scan1_id, sample_scan2_id) """

        if type(domains[0]) != int:
            # the fold numbers of the corresponding 2 samples
            doms = (self.load_domain_label_number (domains[0]), self.load_domain_label_number (domains[1]))
        else:
            doms = domains
        largest_domain = max(doms)
        domain_id = self.df['fold'].loc[i]
        if domain_id == largest_domain:
            return 1
        elif domain_id in doms:  # error otherwise
            return 0


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)

    def load_image_slice(self, i, shift):
        return self._change_2d(self._shadowed.load_image_slice(i, shift), i)

    def load_segm_slice(self, i, shift):
        return np.float32(self._change_2d(self._shadowed.load_segm_slice(i, shift), i) >= .5)

    def load_images_decomposed(self, i, mask_side = 40):
        return self._change_2d_2_imgs(self._shadowed.load_images_decomposed(i, mask_side = 40), i)


class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=3):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def _change_2d(self, x, i):
        return zoom(x, self._scale_factor(i)[:-1], order=self.order)

    def _change_2d_2_imgs (self, imgs, i):
        return zoom(imgs[0], self._scale_factor(i)[:-1], order=self.order),\
               zoom(imgs[1], self._scale_factor(i)[:-1], order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


def scale_mri_multiple_imgs(*args, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    processed_imgs = []
    for image in args:
        image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
        image -= np.min(image)
        image /= np.max(image)
        processed_imgs.append(np.float32(image))
    return processed_imgs