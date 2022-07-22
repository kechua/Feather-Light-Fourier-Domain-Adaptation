import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from gan.discriminator import Discriminator
from nn.common.view import View
from stylegan2.model import EqualLinear, EqualConv2d, Blur, ScaledLeakyReLU
from stylegan2.op import fused_leaky_relu, FusedLeakyReLU


class PosLinear(EqualLinear):

    def __init__(self, in_dim, out_dim,  bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__(in_dim, out_dim,  bias=bias, bias_init=bias_init, lr_mul=lr_mul, activation=activation)
        self.scale = 1 / in_dim

    def forward(self, input: Tensor):

        if self.activation:
            out = F.linear(input, torch.abs(self.weight) * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, torch.abs(self.weight) * self.scale, bias=self.bias * self.lr_mul
            )

        return out


class PosConv2d(EqualConv2d):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)
        self.scale = 1 / (in_channel * kernel_size ** 2)

    def forward(self, input: Tensor):

        out = F.conv2d(
            input,
            torch.abs(self.weight) * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class PosConv2dWithActivation(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            PosConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class PosDownSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = PosConv2dWithActivation(in_channel, in_channel, 3,  activate=False)
        self.conv2 = PosConv2dWithActivation(in_channel, out_channel, 3, downsample=True,  activate=False)

        self.skip = PosConv2dWithActivation(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input: Tensor):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class PosDiscriminator(Discriminator):
    def __init__(self, image_size: int, nc: int = 3, ndf: int = 64):
        super(PosDiscriminator, self).__init__()

        layers = [
            EqualConv2d(nc, ndf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        tmp_size = image_size
        tmp_nc = ndf
        nc_next = -1
        while tmp_size > 4:
            tmp_size = tmp_size // 2
            nc_next = min(256, tmp_nc * 2)
            layers += [
                PosDownSampleBlock(tmp_nc, nc_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            tmp_nc = nc_next

        layers += [
            View(-1),
            PosLinear(nc_next * 4 * 4, nc_next, activation=False),
            nn.LeakyReLU(0.2, inplace=True),
            PosLinear(nc_next, 1, activation=False)
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
