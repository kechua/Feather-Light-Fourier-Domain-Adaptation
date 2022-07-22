from abc import ABC, abstractmethod
from typing import List, Callable, Iterator
import numpy
from torch import Tensor, nn
from gan.discriminator import Discriminator
from gan.loss.loss_base import Loss
from gan.loss.penalties.penalty import DiscriminatorPenalty


class GANLoss(ABC):

    def __init__(self, discriminator: Discriminator):
        self.__penalties: List[DiscriminatorPenalty] = []
        self.discriminator = discriminator

    def discriminator_loss_with_penalty(self,
                                   x: List[Tensor],
                                   y: List[Tensor]) -> Loss:
        x_detach = [xi.detach() for xi in x]
        y_detach = [yi.detach() for yi in y]

        Dx = self.discriminator.forward(*x_detach)
        Dy = self.discriminator.forward(*y_detach)

        loss_sum: Loss = self._discriminator_loss(Dx, Dy)

        for pen in self.get_penalties():
            loss_sum = loss_sum - pen.__call__(self.discriminator, Dx, Dy, x_detach, y_detach)

        return loss_sum

    def discriminator_loss_as_is(self,
                                   x: List[Tensor],
                                   y: List[Tensor]) -> Loss:

        Dx = self.discriminator.forward(*x)
        Dy = self.discriminator.forward(*y)

        return self._discriminator_loss(Dx, Dy)

    def generator_loss(self, real: List[Tensor], fake: List[Tensor]) -> Loss:
        return self._generator_loss(self.discriminator.forward(*fake), real, fake)

    def parameters(self) -> Iterator[Tensor]:
        return self.discriminator.parameters()

    @abstractmethod
    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss: pass

    @abstractmethod
    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss: pass

    def add_penalty(self, pen: DiscriminatorPenalty):
        self.__penalties.append(pen)
        return self

    def add_penalties(self, pens: List[DiscriminatorPenalty]) -> None:
        self.__penalties.extend(pens)

    def get_penalties(self) -> List[DiscriminatorPenalty]:
        return self.__penalties

    def add_generator_loss(self, loss: nn.Module, weight=1.0):
        return self.__add__(
            GANLossObject(
                lambda dx, dy: Loss.ZERO(),
                lambda dgz, real, fake: Loss(loss(fake[0], real[0].detach()) * weight),
                self.discriminator
            )
        )

    def __add__(self, other):

        discriminator = self.discriminator if self.discriminator is not None else other.discriminator

        obj = GANLossObject(
            lambda dx, dy: self._discriminator_loss(dx, dy) + other._discriminator_loss(dx, dy),
            lambda dgz, real, fake: self._generator_loss(dgz, real, fake) + other._generator_loss(dgz, real, fake),
            discriminator
        )

        obj.add_penalties(self.__penalties)
        obj.add_penalties(other.get_penalties())
        return obj

    def __mul__(self, weight: float):

        obj = GANLossObject(
            lambda dx, dy: self._discriminator_loss(dx, dy) * weight,
            lambda dgz, real, fake: self._generator_loss(dgz, real, fake) * weight,
            self.discriminator
        )

        obj.add_penalties(self.__penalties)

        return obj


class GANLossObject(GANLoss):

    def __init__(self,
                 _discriminator_loss: Callable[[Tensor, Tensor], Loss],
                 _generator_loss: Callable[[Tensor, List[Tensor], List[Tensor]], Loss],
                 discriminator: Discriminator):
        super().__init__(discriminator)
        self.d_loss = _discriminator_loss
        self.g_loss = _generator_loss

    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:
        return self.d_loss(dx, dy)

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        return self.g_loss(dgz, real, fake)







