import torch
from torch import nn, Tensor
import numpy as np

from gan.generator import Generator
from nn.resnet.down_block import DownBlock
from nn.resnet.residual import PaddingType, ResidualNet
from nn.resnet.up_block import UpBlock


class ResidualGenerator(Generator):

    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 ngf: int = 32,
                 n_down: int = 3,
                 n_residual_blocks: int = 9,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 padding_type: PaddingType = PaddingType.REFLECT):
        super(ResidualGenerator, self).__init__()

        self.n_down = n_down
        max_ngf = 256

        self.model_downsample = DownBlock(input_nc, min(ngf * (2 ** n_down), max_ngf), ngf, n_down, norm_layer)

        self.model_resnet = ResidualNet(min(ngf * (2 ** n_down), max_ngf), n_residual_blocks, norm_layer, padding_type)

        self.model_upsample = UpBlock(2 * min(ngf * (2 ** n_down), max_ngf), output_nc, ngf, n_down, norm_layer)

    def forward(self, img: Tensor) -> Tensor:

        assert img.shape[-1] == img.shape[-2]

        downsample = self.model_downsample(img)
        resnet = self.model_resnet(downsample)
        upsample = self.model_upsample(torch.cat([resnet, downsample], dim=1))

        return upsample


