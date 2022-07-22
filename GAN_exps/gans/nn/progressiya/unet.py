from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Callable, TypeVar, Generic, Optional, Type, Tuple, Dict, Set, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base import Progressive, ReverseListCollector, LastElementCollector, \
    ElementwiseModuleList, StateInjector, ProgressiveWithoutState, InjectLast, TLT


class InjectCatHead(StateInjector):

    def __init__(self, dim=1):
        self.dim = dim

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        B, C, D, D  = state.shape
        input = args[0] if args[0].shape[-1] == D else args[0][:, :, 0:D, 0:D]
        return (torch.cat([input, state], dim=self.dim),), kw


class UNet2(nn.Sequential):

    def __init__(self,
                 down_blocks: List[nn.Module],
                 up_blocks: List[nn.Module]):
        up_blocks.reverse()

        super().__init__(
            Progressive[List[Tensor]](down_blocks, collector_class=ReverseListCollector),
            ProgressiveWithoutState[Tensor](up_blocks, state_injector=InjectCatHead(1), collector_class=LastElementCollector),
        )


class UNet4(nn.Sequential):

    def __init__(self, down_blocks: List[nn.Module], middle_block: List[nn.Module], up_blocks: List[nn.Module],
                 final_blocks: List[nn.Module]):
        middle_block.reverse()
        up_blocks.reverse()
        final_blocks.reverse()

        super().__init__(
            Progressive[List[Tensor]](down_blocks, collector_class=ReverseListCollector),
            ElementwiseModuleList[List[Tensor]](middle_block),
            ProgressiveWithoutState[List[Tensor]](up_blocks, state_injector=InjectCatHead(1)),
            ProgressiveWithoutState[Tensor](final_blocks, state_injector=InjectLast(), collector_class=LastElementCollector)
        )


class InputFilter(nn.Module, ABC):
    @abstractmethod
    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        pass


class InputFilterAll(InputFilter):
    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        return args, kw


class InputFilterName(InputFilter):
    def __init__(self, names: Set[str]):
        super().__init__()
        self.names = names

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_kw = {i: kw[i] for i in self.names}
        return args, filtered_kw


class CopyKwToArgs(InputFilter):
    def __init__(self, names: Set[str]):
        super().__init__()
        self.names = names

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_args = tuple(kw[i] for i in self.names)
        return args + filtered_args, kw


class InputFilterHorisontal(InputFilter):
    def __init__(self, names: Set[str], indices: List[int]):
        super().__init__()
        self.names = names
        self.indices = indices

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_args = tuple(args[i] for i in self.indices)
        filtered_kw = {i: kw[i] for i in self.names}
        return filtered_args, filtered_kw


class InputFilterVertical(InputFilter):
    def __init__(self, indices: List[int]):
        super().__init__()
        self.indices = indices

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_args = tuple([a[j] for j in self.indices] for a in args)
        filtered_kw = {i: [kw[i][j] for j in self.indices] for i in kw.keys()}
        return filtered_args, filtered_kw


class ZapomniKak(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class ProgressiveSequential(nn.Module):

    def __init__(self, *modules: Union[nn.Module, InputFilter, ZapomniKak], return_keys: List[str] = None):
        super().__init__()
        self.ops = nn.ModuleList([*modules])
        self.layer_dict = nn.ModuleDict()
        self.return_keys = return_keys

    def forward(self, *args: TLT, **kw: TLT):
        out, slovar = None, {**kw}
        model_args, model_kw = args, slovar

        i = 0
        for model in self.ops:

            if isinstance(model, InputFilter):
                model_args, model_kw = model.filter(model_args, model_kw)

            elif isinstance(model, ZapomniKak):
                slovar[model.name] = out

            elif isinstance(model, nn.Module):

                out = model(*model_args, **model_kw)
                model_args, model_kw = args, slovar

                i += 1

            else:
                raise Exception("wrong type in modules:", model)

        if self.return_keys:
            return tuple([out] + [slovar[k] for k in self.return_keys])

        return out






