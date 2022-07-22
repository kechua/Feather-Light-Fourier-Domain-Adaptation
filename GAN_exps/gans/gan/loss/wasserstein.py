from torch import Tensor

from gan.discriminator import Discriminator
from gan.loss.base import GANLoss
from gan.loss.penalties.lipschitz import LipschitzPenalty
from gan.loss.penalties.penalty import default_mix
from gan.loss.loss_base import Loss


class WassersteinLoss(GANLoss):

    def __init__(self, discriminator: Discriminator, mix=default_mix, penalty_weight: float = 1):
        super().__init__(discriminator)
        self.add_penalty(LipschitzPenalty(penalty_weight, mix))

    def _generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return Loss(-dgz.mean())

    def _discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = (d_real).mean() - d_fake.mean()

        return Loss(discriminator_loss)
