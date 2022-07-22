import torch
import surface_distance.metrics as surf_dc

class SDiceMetric(object):
    def __init__(self, n_classes=1, weighted=False):
        self.n_classes = n_classes
        self.weighted = weighted

    def __call__(self, pred, target):
        pred = pred.detach().squeeze().cpu().numpy().astype(bool)
        target = target.detach().squeeze().cpu().numpy().astype(bool)
        surface_distances = surf_dc.compute_surface_distances(pred, target, (0.95, 0.95, 1))
        return 1 - surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1)

