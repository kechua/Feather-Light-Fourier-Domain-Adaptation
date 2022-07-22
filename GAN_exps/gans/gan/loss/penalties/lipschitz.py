from typing import List

import torch
from torch import Tensor

from gan.loss.penalties.penalty import GradientDiscriminatorPenalty, ApproxGradientDiscriminatorPenalty
from gan.loss.loss_base import Loss


class LipschitzPenalty(GradientDiscriminatorPenalty):

    def __init__(self, weight: float, mix):
        super().__init__(mix)
        self.weight = weight

    def _compute(self, gradients: List[Tensor]) -> Loss:
        gradients_cat = torch.cat([g.view(g.size(0), -1) for g in gradients], dim=1)
        # gradients: Tensor = gradients.view((gradients.size(0), -1))
        gradient_penalty_value = ((gradients_cat.norm(2, dim=1) - 1)**2).mean()
        return Loss(self.weight * gradient_penalty_value)


class ApproxLipschitzPenalty(ApproxGradientDiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def _compute(self, delta: Tensor) -> Loss:

        gradient_penalty_value = delta.relu().norm(2, dim=1).pow(2).mean()

        return Loss(self.weight * gradient_penalty_value)
