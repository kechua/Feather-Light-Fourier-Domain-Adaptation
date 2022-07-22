import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor


class Noise(ABC):
    @abstractmethod
    def sample(self, n: int) -> Tensor: pass

    @abstractmethod
    def size(self) -> int: pass


class ProgressiveNoise():
    def __init__(self, numberofsteps: int = 7):
        self.numberofsteps = numberofsteps

    def forward(self, z, cond):
        res_dict = {}
        res_dict["input"] = cond
        res_dict["style1"] = z
        res_dict["style2"] = z
        res_dict["style3"] = z
        res_dict["style4"] = z
        res_dict["style5"] = z
        res_dict["style6"] = z
        res_dict["style7"] = z

        for i in range(self.numberofsteps + 1):
            size = 4 * 2 ** i
            res_dict[f"noise{i}"] = torch.randn(z.shape[0], 1, size, size)

        return res_dict
