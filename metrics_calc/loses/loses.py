import torch
import surface_distance.metrics as surf_dc


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predicted_masks, masks):
        return self.loss(predicted_masks, masks)



class SoloClassDiceLoss(torch.nn.Module):
    def __init__(self, n_classes=1):
        super(SoloClassDiceLoss, self).__init__()

        self.softmax = torch.nn.Softmax(dim=1)
        self.n_classes = n_classes

    def forward(self, pred, target):
        # print(pred.byte())
        epsilon = 1e-6
        inter = torch.dot(pred.reshape(-1), target.reshape(-1))
        print(torch.sum(pred), torch.sum(target))
        sets_sum = torch.sum(pred) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return 1 - (2 * inter + epsilon) / (sets_sum + epsilon)



class CombinedLoss(torch.nn.Module):
    def __init__(self, losses, coefficients=[0.4, 0.6]):
        super(CombinedLoss, self).__init__()

        self.losses = losses
        self.coefficients = coefficients

    def forward(self, predicted_masks, masks):
        loss = 0.0
        for loss_function, coefficient in zip(self.losses, self.coefficients):
            loss += coefficient * loss_function(predicted_masks, masks)

        return loss
