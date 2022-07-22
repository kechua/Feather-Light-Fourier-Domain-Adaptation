from torch import nn, Tensor
from enum import Enum
from typing import Dict, List, Callable, TypeVar, Generic


class PaddingType(Enum):
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    NONE = 'zero'


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 use_spectral_norm=False):
        super(ResidualBlock, self).__init__()

        self.padding2module: Dict[PaddingType, List] = {
            PaddingType.REFLECT: [nn.ReflectionPad2d(1)],
            PaddingType.REPLICATE: [nn.ReplicationPad2d(1)],
            PaddingType.NONE: []
        }

        if use_spectral_norm:
            self.spectral_norm = nn.utils.spectral_norm
        else:
            self.spectral_norm = lambda x: x

        self.conv_block = self.build_conv_block(dim, dim, padding_type, norm_layer, activation, use_dropout)
        self.shortcut = self.spectral_norm(nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False))
        self.out = nn.Sequential(
            norm_layer(dim),
            activation
        )

    def build_conv_block(self,
                         dim_in: int,
                         dim_out: int,
                         padding_type: PaddingType,
                         norm_layer: nn.Module,
                         activation: nn.Module,
                         use_dropout: bool):

        conv_block = []
        p = 0 if padding_type != PaddingType.NONE else 1
        padding_module = self.padding2module[padding_type]

        conv_block += padding_module

        conv_block += [self.spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=False)),
                       norm_layer(dim_out),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += padding_module

        conv_block += [self.spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=False))]

        return nn.Sequential(*conv_block)

    def forward(self, x: Tensor):
        res = self.shortcut(x) + self.conv_block(x)
        return self.out(res)


class PooledResidualBlock(ResidualBlock):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_pool: int,
                 pool: nn.Module,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 use_spectral_norm=False):
        super(PooledResidualBlock, self).__init__(dim_in, padding_type, norm_layer, activation, use_dropout, use_spectral_norm)

        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, activation, use_dropout)
        self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=False)
        self.out = nn.Sequential(
            norm_layer(dim_pool),
            activation
        )
        self.pool = pool

    def forward(self, x: Tensor):
        res = self.pool(self.shortcut(x)) + self.pool(self.conv_block(x))
        return self.out(res)


class Down2xResidualBlock(PooledResidualBlock):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 use_spectral_norm=False):
        super(Down2xResidualBlock, self).__init__(
            dim_in,
            dim_out,
            dim_out,
            nn.AvgPool2d(kernel_size=2, stride=2),
            padding_type,
            norm_layer,
            activation,
            use_dropout,
            use_spectral_norm)


class Up2xResidualBlock(PooledResidualBlock):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 use_spectral_norm=False):
        super(Up2xResidualBlock, self).__init__(
            dim_in,
            dim_out * 4,
            dim_out,
            nn.PixelShuffle(upscale_factor=2),
            padding_type,
            norm_layer,
            activation,
            use_dropout,
            use_spectral_norm)


class Up4xResidualBlock(PooledResidualBlock):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 use_spectral_norm=False):
        super(Up4xResidualBlock, self).__init__(
            dim_in,
            dim_out * 16,
            dim_out,
            nn.PixelShuffle(upscale_factor=4),
            padding_type,
            norm_layer,
            activation,
            use_dropout,
            use_spectral_norm)


class ResidualNet(nn.Module):

    def __init__(self,
                 dim: int,
                 n_blocks: int,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 padding_type: PaddingType = PaddingType.REFLECT):
        super(ResidualNet, self).__init__()

        model_list = []
        for i in range(n_blocks):
            model_list += [ResidualBlock(dim, padding_type=padding_type, norm_layer=norm_layer)]

        self.model = nn.Sequential(*model_list)

    def forward(self, x: Tensor):
        return self.model(x)
