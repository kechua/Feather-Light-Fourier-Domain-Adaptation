import torch
import surface_distance.metrics as surf_dc


def sdice(a, b, spacing, tolerance):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)

def dice_loss(pred, target, debug = False):
    epsilon = 1e-6
    inter = torch.dot(pred.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(pred) + torch.sum(target)

    if sets_sum.item() == 0:
        sets_sum = 2 * inter

    return (2 * inter + epsilon) / (sets_sum + epsilon)