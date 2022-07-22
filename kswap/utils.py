import os
from dpipe.torch import load_model_state
import torch
import random
from pathlib import Path
import numpy as np
import surface_distance.metrics as surf_dc
from dpipe.io import PathLike
from dpipe.torch.functional import weighted_cross_entropy_with_logits, dice_loss


def get_pred(x, threshold=0.5):
    return x > threshold

def sdice(a, b, spacing, tolerance):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)

def load_model_state_fold_wise(architecture, baseline_exp_path, n_folds=6, modify_state_fn=None, n_first_exclude=0):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1]) + n_first_exclude
    path_to_pretrained_model = os.path.join(baseline_exp_path, f'experiment_{n_val // (n_folds - 1)}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model, modify_state_fn=modify_state_fn)

def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)

def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')

def ce_plus_dice(pred: torch.Tensor, target: torch.Tensor):
    return 0.4 * weighted_cross_entropy_with_logits(pred, target) + 0.6 * dice_loss(torch.sigmoid(pred), target)
