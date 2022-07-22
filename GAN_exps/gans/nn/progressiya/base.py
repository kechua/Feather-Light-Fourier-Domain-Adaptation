from abc import ABC, abstractmethod
from typing import List, Callable, TypeVar, Generic, Optional, Type, Union, Tuple, Dict, Any, Set
import torch
import typing
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

TLT = TypeVar("TLT", Tensor, List[Tensor])

class TensorCollector(ABC, Generic[TLT]):
    @abstractmethod
    def append(self, t: Tensor) -> None:
        pass

    @abstractmethod
    def result(self) -> TLT:
        pass


class ListCollector(TensorCollector[List[Tensor]]):

    def __init__(self):
        self.data = []

    def result(self) -> List[Tensor]:
        out = self.data
        self.data = []
        return out

    def append(self, t: Tensor) -> None:
        self.data.append(t)


class ReverseListCollector(ListCollector):

    def result(self) -> List[Tensor]:
        self.data.reverse()
        out = self.data
        self.data = []
        return out


class LastElementCollector(TensorCollector[Tensor]):

    def __init__(self):
        self.data: Optional[Tensor] = None

    def result(self) -> Tensor:
        out = self.data
        self.data = None
        return out

    def append(self, t: Tensor) -> None:
        self.data = t


class StateInjector(ABC):
    @abstractmethod
    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]): pass


class InjectByName(StateInjector):

    def __init__(self, name):
        self.name = name

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        kw[self.name] = state
        return args, kw


class InjectLast(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (*args, state), kw


class InjectHead(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (state, *args), kw


class Progressive(nn.ModuleList, Generic[TLT]):
    def __init__(self,
                 blocks: List[nn.Module],
                 state_injector: StateInjector = InjectHead(),
                 collector_class: Type[TensorCollector[TLT]] = ListCollector
                 ):
        super(Progressive, self).__init__(blocks)
        self.collector_class = collector_class
        self.injector = state_injector

    def get_args_i(self, i: int, *args: List[Tensor], **kw: List[Tensor]):

        kw_i: Dict[str, Tensor] = dict((k, kw[k][i]) for k in kw.keys())
        args_i: Tuple[Tensor, ...] = tuple(args[k][i] for k in range(len(args)))

        return args_i, kw_i

    def process(self, start: int, state: Tensor, collector: TensorCollector[TLT], *args: List[Tensor], **kw: List[Tensor]) -> TLT:
        i = start

        while i < self.__len__():
            args_i_s, kw_i_s = self.get_args_i(i, *args, **kw)
            args_i_s, kw_i_s = self.injector.inject(state, args_i_s, kw_i_s)
            out = self[i](*args_i_s, **kw_i_s)
            state = out
            collector.append(out)
            i += 1

        return collector.result()

    def forward(self, state: Tensor, *args: List[Tensor], **kw: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()
        collector.append(state)

        return self.process(0, state, collector, *args, **kw)


class ProgressiveWithoutState(Progressive[TLT]):

    def forward(self, *args: List[Tensor], **kw: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()

        args_i_s, kw_i_s = self.get_args_i(0, *args, **kw)
        state = self[0](*args_i_s, **kw_i_s)
        collector.append(state)

        return self.process(1, state, collector, *args, **kw)


class ProgressiveWithStateInit(nn.Module, Generic[TLT]):

    def __init__(self,
                 initial: nn.Module,
                 progressive: Progressive[TLT]):
        super().__init__()
        self.initial = initial
        self.progressive = progressive

    def forward(self, *args: List[Tensor], **kw: List[Tensor]) -> TLT:

        kw_i: Dict[str, Tensor] = dict((k, kw[k][0]) for k in kw.keys())
        args_i: Tuple[Tensor, ...] = tuple(args[k][0] for k in range(len(args)))
        state = self.initial(*args_i, **kw_i)

        kw_tail: Dict[str, List[Tensor]] = dict((k, kw[k][1:]) for k in kw.keys())
        args_tail: Tuple[List[Tensor], ...] = tuple(args[k][1:] for k in range(len(args)))

        return self.progressive.forward(state, *args_tail, **kw_tail)


class ElementwiseModuleList(nn.Module, Generic[TLT]):
    def __init__(self,
                 blocks: List[nn.Module],
                 collector_class: Type[TensorCollector[TLT]] = ListCollector):
        super(ElementwiseModuleList, self).__init__()
        self.model_list = nn.ModuleList(blocks)
        self.collector_class = collector_class

    def forward(self, input: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()
        i = 0
        while i < len(input):
            x = self.model_list[i](input[i])
            collector.append(x)
            i += 1
        return collector.result()


