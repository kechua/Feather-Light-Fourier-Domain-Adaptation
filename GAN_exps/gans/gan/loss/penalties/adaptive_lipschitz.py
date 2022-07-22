from typing import List

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.gan.loss.penalties.penalty import DiscriminatorPenalty


class AdaptiveLipschitzPenalty(DiscriminatorPenalty):

    def __init__(self, weight: float, learning_rate: float = 0.001):
        self.weight = weight
        self.lr = learning_rate

    def __call__(self, Dx: Tensor, x: List[Tensor]) -> Loss:

        gradients = torch.autograd.grad(outputs=Dx,
                                        inputs=x,
                                        grad_outputs=torch.ones(Dx.size(), device=Dx.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients: Tensor = gradients.view(gradients.size(0), -1)
        gradient_penalty_value = (gradients.norm(2, dim=1) - 1).mean()
        res = self.weight * gradient_penalty_value
        self.weight += self.lr * gradient_penalty_value.item()
        self.weight = max(0, self.weight)
        return Loss(res)
