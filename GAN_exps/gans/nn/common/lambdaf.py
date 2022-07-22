from typing import Callable, List, Tuple
import torch
from torch import nn, Tensor


class MySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class LambdaF(nn.Module):
    def __init__(self, module: List[nn.Module], f: Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
        super().__init__()
        self.module = nn.ModuleList(module)
        self.f = f

    def forward(self, *input: torch.Tensor):
        return self.f(*input)
