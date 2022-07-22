import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Type, Optional


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        ConvClass: Type[nn.Conv2d] = nn.Conv2d,
        activate: Optional[nn.Module] = nn.ReLU(),
        Norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        downsample=False,
        bias=True
    ):
        layers = []

        if downsample:
            stride = 2
            self.padding = 1
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            ConvClass(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and (activate is None),
            )
        )

        if Norm:
            layers.append(Norm(out_channel))

        if activate:
            layers.append(activate)

        super().__init__(*layers)


class TransposeConv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        ConvClass: Type[nn.ConvTranspose2d] = nn.ConvTranspose2d,
        activate: nn.Module = nn.ReLU(),
        Norm: Type[nn.Module] = nn.InstanceNorm2d,
        padding=1,
    ):
        layers = []

        stride = 2
        self.padding = padding

        layers.append(
            ConvClass(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride
            )
        )

        if Norm:
            layers.append(Norm(out_channel))

        if activate:
            layers.append(activate)

        super().__init__(*layers)