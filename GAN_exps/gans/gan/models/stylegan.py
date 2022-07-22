import torch
from torch import nn, Tensor

from typing import List, Dict, Callable, Tuple, Optional, TypeVar, Generic, Any

from torch import Tensor, nn
from torch.nn import functional as F
from gan.loss.penalties.style_gan_penalty import StyleGeneratorPenalty, PenaltyWithCounter
from gan.loss.stylegan import StyleGANLoss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from gan.models.base import ConditionalGANModel, requires_grad, GANModel, name_to_gan_loss
from gan.generator import Generator
from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


GeneratorClass = TypeVar('GeneratorClass', bound=Generator)


class StyleGanModel(GANModel, Generic[GeneratorClass]):

    def __init__(self, generator: GeneratorClass, loss: StyleGANLoss, lr: Tuple[float, float] = (0.001, 0.0015)):
        self.generator = generator
        self.loss = loss
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr[0], lr[1], min_betas=(0, 0.792), max_betas=(0, 0.932))

        self.g_reg_every = 5
        self.path_regularize = 2

        self.gen_penalty = PenaltyWithCounter(
            StyleGeneratorPenalty(self.path_regularize * self.g_reg_every),
            lambda i: i % self.g_reg_every == 0
        )

    def generator_loss_with_penalty(self, real: List[Tensor], fake: List[Tensor], latent: List[Tensor]) -> Loss:
        return self.generator_loss(real, fake) + self.gen_penalty(fake[0], latent)


class CondStyleGanModel(StyleGanModel, ConditionalGANModel):

    def __init__(self, generator: GeneratorClass, loss: StyleGANLoss, lr: Tuple[float, float] = (0.001, 0.0015)):
        super(StyleGanModel, self).__init__(generator, loss, lr)

    def generator_loss_with_penalty(self, real: List[Tensor], fake: List[Tensor], condition: List[Tensor]) -> Loss:
        return self.generator_loss(real, fake, condition) + self.gen_penalty(fake[0], condition)


