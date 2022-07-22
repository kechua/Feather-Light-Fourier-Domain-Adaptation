from typing import List
from torch import Tensor
from gan.loss.loss_base import Loss
from gan.loss.penalties.style_gan_penalty import GradientPenalty
import torch
import math


class DecoderPenalty(GradientPenalty):

    def __init__(self, weight):
        super().__init__(lambda img: img.sum() / (img.shape[2] * img.shape[3]))
        self.weight = weight

    def _compute(self, grads: List[Tensor]) -> Loss:
        batch = grads[0].shape[0]
        reshaped = grads[0].reshape(batch, -1)
        return Loss(reshaped.pow(2).sum(dim=1).mean() * self.weight)


class EncoderPenalty(GradientPenalty):

    def __init__(self, weight):
        super().__init__(lambda latent: latent.sum() / latent.shape[2])
        self.weight = weight

    def _compute(self, grads: List[Tensor]) -> Loss:
        batch = grads[0].shape[0]
        reshaped = grads[0].reshape(batch, -1)
        return Loss(reshaped.pow(2).sum(dim=1).mean() * self.weight)
