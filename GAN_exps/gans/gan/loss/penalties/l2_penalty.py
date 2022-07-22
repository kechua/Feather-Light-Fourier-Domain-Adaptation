from typing import List

from torch import Tensor

from framework.Loss import Loss
from framework.gan.loss.penalties.penalty import DiscriminatorPenalty


class L2Penalty(DiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, dx: Tensor, x: List[Tensor]) -> Loss:

        penalty_value = dx.abs().pow(1.5).mean()
        return Loss(self.weight * penalty_value)
