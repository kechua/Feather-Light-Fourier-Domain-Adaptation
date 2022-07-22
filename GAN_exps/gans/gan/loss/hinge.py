from typing import List

from torch import Tensor

from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss


class HingeLoss(GANLoss):

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        return Loss(-dgz.mean())

    def _discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = (1 - d_real).relu().mean() + (1 + d_fake).relu().mean()
        return Loss(-discriminator_loss)
