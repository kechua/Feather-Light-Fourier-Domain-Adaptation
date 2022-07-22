from typing import List

import torch
from torch import Tensor, nn

from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss


class DCGANLoss(GANLoss):

    __criterion = nn.BCELoss()

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        batch_size = dgz.size(0)
        nc = dgz.size(1)

        real_labels = torch.full((batch_size, nc, ), 1, device=dgz.device)
        errG = self.__criterion(dgz.view(batch_size, nc).sigmoid(), real_labels)
        return Loss(errG)

    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:

        batch_size = dx.size(0)
        nc = dx.size(1)

        real_labels = torch.full((batch_size, nc, ), 1, device=dx.device)
        err_real = self.__criterion(dx.view(batch_size, nc).sigmoid(), real_labels)

        fake_labels = torch.full((batch_size, nc, ), 0, device=dx.device)
        err_fake = self.__criterion(dy.view(batch_size, nc).sigmoid(), fake_labels)

        return Loss(-(err_fake + err_real))

