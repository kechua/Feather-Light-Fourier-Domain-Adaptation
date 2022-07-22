from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar, Generic
import torch
from torch import nn
from torch import Tensor


class Discriminator(nn.Module, ABC):

    @abstractmethod
    def forward(self, *x: Tensor) -> Tensor: pass


class ConditionalDiscriminator(Discriminator):

    @abstractmethod
    def forward(self, x: Tensor, *condition: Tensor) -> Tensor: pass










