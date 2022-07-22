from abc import ABC, abstractmethod
from typing import Optional, Union, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Loss:

    @staticmethod
    def ZERO():
        return Loss(0)

    def __init__(self, tensor: Union[Tensor, float]):
        self.__tensor = tensor
        assert isinstance(tensor, int) or isinstance(tensor, float) or tensor.numel() == 1

    def __add__(self, other):
        return Loss(self.__tensor + other.to_tensor())

    def __sub__(self, other):
        return Loss(self.__tensor - other.to_tensor())

    def __mul__(self, weight: Union[float,  Tensor]):
        return Loss(self.__tensor * weight)

    def __truediv__(self, weight: float):
        return Loss(self.__tensor / weight)

    def cuda(self):
        return Loss(self.__tensor.cuda())

    def minimize(self) -> None:
        return self.__tensor.backward()

    def minimize_step(self, *optimizers: Optimizer, retain_graph=False) -> None:

        if self.__tensor == 0 or (not isinstance(self.__tensor, Tensor)):
            print("ZERO brule_loss value")
            return

        for opt in optimizers:
            opt.zero_grad()

        self.__tensor.backward(retain_graph=retain_graph)

        for opt in optimizers:
            opt.step()

    def maximize(self) -> None:
        return self.__tensor.backward(-torch.ones_like(self.__tensor))

    def maximize_step(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad()
        self.maximize()
        optimizer.step()

    def item(self) -> float:
        if isinstance(self.__tensor, Tensor):
            return self.__tensor.item()
        else:
            return self.__tensor

    def to_tensor(self) -> Tensor:
        return self.__tensor

    def is_sero(self) -> bool:
        return self.__tensor == 0 or (not isinstance(self.__tensor, Tensor))


class LossModule(ABC):

    @abstractmethod
    def forward(self, *tensor: Tensor) -> Loss: pass



