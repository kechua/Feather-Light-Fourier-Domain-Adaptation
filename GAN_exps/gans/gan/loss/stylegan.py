from typing import List, Callable, Iterator
import numpy
from torch import Tensor, nn
from torch.nn import functional as F
from gan.discriminator import Discriminator
from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss

from gan.loss.penalties.style_gan_penalty import StyleDiscriminatorPenalty, PenaltyWithCounter


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


class StyleGANLoss(GANLoss):
    def __init__(self, discriminator: Discriminator, r1=10, d_reg_every=16):
        super().__init__(discriminator=discriminator)
        penalty = StyleDiscriminatorPenalty(r1 * d_reg_every / 2)
        # penalty = StyleDiscriminatorImagePenalty(r1 * d_reg_every / 2)
        penalty_counter = PenaltyWithCounter(penalty, lambda x: x % d_reg_every == 0)
        self.add_penalties([penalty_counter])

    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:
        return Loss(-d_logistic_loss(dx, dy))

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        return Loss(g_nonsaturating_loss(dgz))


class StyleGANLossWithoutPenalty(GANLoss):
    def __init__(self, discriminator: Discriminator):
        super().__init__(discriminator=discriminator)

    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:
        return Loss(-d_logistic_loss(dx, dy))

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        return Loss(g_nonsaturating_loss(dgz))