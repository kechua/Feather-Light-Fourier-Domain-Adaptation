import math

from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.loss.loss_base import Loss
from gan.loss.penalties.penalty import GradientDiscriminatorPenalty, DiscriminatorPenalty
from torch import Tensor
import torch
from typing import List, Callable, Tuple, TypeVar


PenClass = TypeVar('PenClass')
class PenaltyWithCounter:
    def __init__(self, penalty: PenClass, cond: Callable[[int], bool], preproc=None):
        self.CLLLeT4uK = -1
        self.penalty = penalty
        self.cond = cond
        self.preproc = preproc

    def __call__(self, *args, **kwargs):
        self.CLLLeT4uK += 1
        if self.cond(self.CLLLeT4uK):
            if self.preproc:
                return self.penalty(*self.preproc(*args, **kwargs))
            else:
                return self.penalty(*args, **kwargs)
        else:
            return Loss.ZERO()


class StyleDiscriminatorPenalty(GradientDiscriminatorPenalty):
    def __init__(self, weight):
        self.weight = weight
        super().__init__(lambda x, y: x)

    def _compute(self, grads: List[Tensor]) -> Loss:
        batch = grads[0].shape[0]
        reshaped = [grad.reshape(batch, -1) for grad in grads]
        grads_cat = torch.cat(reshaped, dim=1)
        return Loss(grads_cat.pow(2).sum(dim=1).mean() * self.weight)


class StyleDiscriminatorImagePenalty(GradientDiscriminatorPenalty):
    def __init__(self, weight):
        self.weight = weight
        super().__init__(lambda x, y: x)

    def _compute(self, grads: List[Tensor]) -> Loss:
        batch = grads[0].shape[0]
        reshaped = grads[0].reshape(batch, -1)
        return Loss(reshaped.pow(2).sum(dim=1).mean() * self.weight)


class StyleDiscriminatorCondPenalty(GradientDiscriminatorPenalty):
    def __init__(self, weight):
        self.weight = weight
        super().__init__(lambda x, y: x)

    def _compute(self, grads: List[Tensor]) -> Loss:
        batch = grads[0].shape[0]
        reshaped = grads[1].reshape(batch, -1)
        return Loss(reshaped.pow(2).sum(dim=1).mean() * self.weight)

class GradientPenalty:

    def __init__(self, out_aggregate: Callable[[Tensor], Tensor]):
        self.aggregate = out_aggregate

    def _compute(self, grads: Tuple[Tensor]) -> Loss: pass

    def __call__(
            self,
            fx: Tensor,
            x: List[Tensor]) -> Loss:

        grads: Tuple[Tensor] = torch.autograd.grad(
                                   outputs=self.aggregate(fx),
                                   inputs=x,
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)

        return self._compute(grads)


class StyleGeneratorPenalty(GradientPenalty):

    def __init__(self, weight):
        super().__init__(lambda img: (img * torch.randn_like(img)).sum() / math.sqrt(img.shape[2] * img.shape[3]))
        self.weight = weight
        self.mean_path_length = 0
        self.decay = 0.01

    def _compute(self, grads: List[Tensor]) -> Loss:

        path_lengths = 0

        for i in range(len(grads)):
            B = grads[i].shape[0]
            path_lengths = path_lengths + grads[i].pow(2).reshape(B, -1).mean(1)

        path_lengths = path_lengths.sqrt()

        mean_path_length = self.mean_path_length + self.decay * (path_lengths.mean() - self.mean_path_length)
        self.mean_path_length = mean_path_length.detach()

        return Loss((path_lengths - self.mean_path_length).pow(2).mean() * self.weight)




