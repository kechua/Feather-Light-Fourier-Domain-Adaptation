import numpy as np
from dpipe.dataset.segmentation import SegmentationFromCSV
from dpipe.dataset.wrappers import Proxy


class CC359(SegmentationFromCSV):
    def __init__(self, data_path, low: int = None, high: int = None, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())
        self.low, self.high = low, high

    def load_image(self, i, step=1):
        if self.low is None:
            return np.float32(super().load_image(i)[0][:, :, ::step])  # 4D -> 3D
        else:
            return np.float32(super().load_image(i)[0][:, :, self.low:self.high:step])

    def load_segm(self, i, step=1):
        if self.low is None:
            return np.float32(super().load_segm(i)[:, :, ::step])  # already 3D
        else:
            return np.float32(super().load_segm(i)[:, :, self.low:self.high:step])

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i, step=1):
        return self._change(self._shadowed.load_image(i), i)[:, :, ::step]

    def load_segm(self, i, step=1):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)[:, :, ::step]


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)
