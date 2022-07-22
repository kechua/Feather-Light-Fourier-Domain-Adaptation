from abc import ABC, abstractmethod
from typing import Optional, List
import torch
from torch import Tensor, nn

from nn.progressiya.base import TLT


class Generator(nn.Module, ABC):

    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def forward(self, *noise: TLT) -> Tensor: pass


class ConditionalGenerator(Generator):

    def __init__(self):
        super(ConditionalGenerator, self).__init__()

    @abstractmethod
    def forward(self, condition: TLT, *noize: TLT) -> Tensor: pass


