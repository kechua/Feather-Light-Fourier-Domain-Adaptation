from typing import Optional

import torch
from torch import nn


class SkipPlus(nn.Module):
    def __init__(self, layer: nn.Module, upsample: Optional[nn.Module] = nn.Upsample(scale_factor=2)):
        super().__init__()

        self.layer = layer
        self.upsample = upsample
        self.bias = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, input, skip=None):
        out = self.layer(input)

        if skip is not None:
            if self.upsample is not None:
                skip = self.upsample(skip)
            out = out + self.w * skip + self.bias

        return out