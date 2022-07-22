from typing import Callable, Generic, TypeVar, Iterable, Dict, Tuple, Any
from torch import nn, Tensor, optim
from gan.loss.loss_base import Loss


class CycleGAN:
    def __init__(self,
                 g_12: Callable[[Dict[str, Any]], Dict[str, Any]],
                 g_21: Callable[[Dict[str, Any]], Dict[str, Any]],
                 loss_1: Callable[[Dict[str, Tuple[Any, Any]]], Loss],
                 loss_2: Callable[[Dict[str, Tuple[Any, Any]]], Loss],
                 *optimizers: optim.optimizer.Optimizer):

        self.g_12 = g_12
        self.g_21 = g_21
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.opt = optimizers

    def zero_grad(self):
        for o in self.opt:
            o.zero_grad()

    def loss_forward(self, input1: Dict[str, Any]) -> Loss:

        pred: Dict[str, Any] = self.g_21(self.g_12(input1))
        join: Dict[str, Tuple[Any, Any]] = {(pred[k], input1[k]) for k in input1.keys()}

        return self.loss_1(join)

    def loss_backward(self, input2: Dict[str, Any]) -> Loss:

        pred: Dict[str, Any] = self.g_12(self.g_21(input2))
        join: Dict[str, Tuple[Any, Any]] = {(pred[k], input2[k]) for k in input2.keys()}

        return self.loss_2(join)

    def train(self, input1: Dict[str, Any], input2: Dict[str, Any]):

        self.zero_grad()
        self.loss_forward(input1).minimize_step(*self.opt)

        self.zero_grad()
        self.loss_backward(input2).minimize_step(*self.opt)
