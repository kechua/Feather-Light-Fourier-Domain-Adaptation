from functools import reduce

from torch import nn, Tensor

from framework.gan.image2image.unet_generator import UNetGenerator
from framework.segmentation.Mask import Mask


class MaskToImageGenerator(nn.Module):

    def __init__(self, base_generator: UNetGenerator):
        super(MaskToImageGenerator).__init__()

        self.unet = base_generator

    def forward(self, mask: Mask) -> Tensor:

        z = self.unet.forward(mask.tensor)

        gs = z.split(3, dim=1)
        ss = mask.tensor.split(1, dim=1)

        segments = [gs[i] * ss[i] for i in range(len(ss))]

        return reduce(lambda a, b: a + b, segments)
