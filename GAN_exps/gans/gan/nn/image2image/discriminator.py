import functools

import torch
from torch import nn, Tensor
import numpy as np

from gan.conditional import ConditionalDiscriminator


class Discriminator(ConditionalDiscriminator):
    def __init__(self, ndf, nc, img_size):
        super(Discriminator, self).__init__()

        down_times = int(np.log2(img_size))
        assert 2**down_times == img_size

        max_nc = 256

        self.main = nn.Sequential(
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True)
        )

        size = int(img_size / 2)
        ndf_tmp = ndf

        for di in range(down_times):
            if size == 2:
                break
            nc_out = min(2 * ndf_tmp, max_nc)
            self.main.add_module("disc_down_" + str(size), nn.Sequential(
                spectral_norm_init(nn.Conv2d(ndf_tmp, nc_out, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nc_out),
                spectral_norm_init(nn.Conv2d(nc_out, nc_out, 3, stride=1, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nc_out)
            ))
            size = int(size / 2)
            ndf_tmp = min(ndf_tmp * 2, max_nc)

        out = nn.Sequential(
            View(-1, 2 * 2 * ndf_tmp),
            spectral_norm_init(nn.Linear(2 * 2 * ndf_tmp, ndf_tmp)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_init(nn.Linear(ndf_tmp, 1)),
            # nn.Tanh()
        )

        self.main.add_module("out", out)

    def forward(self, image: Tensor, *mask: Tensor) -> Tensor:
        # cond = ProbabilityMeasureFabric(image.shape[-1]).from_channels(mask[0]).toImage(image.shape[-1])
        cond = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3]).cuda()
        return self.main(torch.cat((image, cond), dim=1))
