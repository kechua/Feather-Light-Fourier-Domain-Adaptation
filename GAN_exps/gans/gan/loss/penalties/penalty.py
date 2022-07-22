import torch
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from torch import Tensor
import numpy as np

from gan.discriminator import Discriminator
from gan.loss.loss_base import Loss


class DiscriminatorPenalty(ABC):

    @abstractmethod
    def __call__(
            self,
            discriminator: Discriminator,
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss: pass

def default_mix(x: Tensor, y: Tensor):
    eps = torch.rand([x.shape[0]] + [1]*(len(x.shape)-1), device=x.device)
    x0: Tensor = (x * eps + y * (1 - eps))
    return x0


class GradientDiscriminatorPenalty(DiscriminatorPenalty):
    def __init__(self, mix=default_mix):
        self.mix = mix

    @abstractmethod
    def _compute(self, grad: Tuple[Tensor]) -> Loss: pass

    def gradient_point(self, x: List[Tensor], y: List[Tensor]) -> List[Tensor]:
        x0: List[Tensor] = [self.mix(xi, yi).detach().requires_grad_(True) for xi, yi in zip(x, y)]
        return x0

    def __call__(
            self,
            discriminator: Discriminator,
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss:

        x0 = self.gradient_point(x, y)
        dx0: Tensor = discriminator.forward(*x0)

        grads: Tuple[Tensor] = torch.autograd.grad(
                                   outputs=dx0.sum(),
                                   inputs=x0,
                                   # grad_outputs=torch.ones(dx0.shape, device=dx0.device),
                                   create_graph=True,
                                   only_inputs=True)

        return self._compute(grads)


class ApproxGradientDiscriminatorPenalty(DiscriminatorPenalty):

    @abstractmethod
    def _compute(self, delta: Tensor) -> Loss: pass

    def __call__(
            self,
            discriminator: Discriminator,
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss:

        n = x[0].shape[0]

        if len(x) > 1:
            x = torch.cat(*x, dim=1)
            y = torch.cat(*y, dim=1)
        else:
            x = x[0]
            y = y[0]

        norm = (x - y).view(n, -1).norm(2, dim=1).detach().view(n, 1)
        delta = (dx - dy).abs() - norm

        return self._compute(delta)

