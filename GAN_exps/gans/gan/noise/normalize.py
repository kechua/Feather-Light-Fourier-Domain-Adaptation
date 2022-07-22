from abc import ABC

from torch import Tensor

from framework.gan.noise.Noise import Noise


class Normalization(Noise):

    def __init__(self, noise: Noise):
        self.noise = noise

    def size(self) -> int:
        return self.noise.size()

    def sample(self, n: int) -> Tensor:
        data = self.noise.sample(n)
        return data / data.norm(p=2, dim=1, keepdim=True)
