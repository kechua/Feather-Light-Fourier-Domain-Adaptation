import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Type


class LinearBlock(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        LinearClass: Type[nn.Linear] = nn.Linear,
        activate: nn.Module = nn.ReLU(),
        bias=True
    ):
        layers = []

        layers.append(
            LinearClass(
                in_channel,
                out_channel,
                bias=bias,
            )
        )

        if activate:
            layers.append(activate)

        super().__init__(*layers)
