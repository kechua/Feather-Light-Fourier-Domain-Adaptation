import numpy as np
import torch
import pandas as pd
from os.path import join as jp

from dpipe.im.metrics import dice_score
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.predict import add_extract_dims, divisible_shape
from dpipe.torch import inference_step
from kswap.dataset.cc359 import CC359, scale_mri
from kswap.utils import choose_root
from kswap.utils import get_pred, sdice
from kswap.batch_iter import slicewise_naive_swap

data_path = choose_root(
    '/home/name/raid/cc359',
    '/raid/name/cc359',
)

img = 'images'
msk = 'masks'
meta = 'meta.csv'
path_meta = jp(data_path, meta)
meta = pd.read_csv(path_meta, index_col='id')

id2dom = {0: 'sm15', 1: 'sm3', 2: 'ge15', 3: 'ge3', 4: 'ph15', 5: 'ph3'}
pairs = [(1, 2), (3, 5), (5, 2), (0, 5), (5, 0), (2, 4)]

# DATASET
# if `voxel_spacing[i]` is `None` when `i`-th dimension will be used without scaling
low = 0
high = 172

preprocessed_dataset = apply(CC359(data_path, low, high), load_image=scale_mri)
dataset = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)

# PREDICTION
SPATIAL_DIMS = (-3, -2, -1)
@slicewise_naive_swap  # 3D -> 2D iteratively
@add_extract_dims(2)
@divisible_shape(divisor=[8] * 2, padding_values=np.min, axis=SPATIAL_DIMS[1:])
def predict(image, net):
    return inference_step(image, architecture=net, activation=torch.sigmoid)

# METRIC
sdice_tolerance = 1
sdice_metric = lambda x, y: sdice(get_pred(x), get_pred(y), [1, 1, 1], sdice_tolerance)
dice_metric = lambda x, y: dice_score(get_pred(x), get_pred(y))
