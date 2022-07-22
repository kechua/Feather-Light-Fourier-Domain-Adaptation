import math
from torch import Tensor, nn
from gan.generator import Generator as G
from nn.common.attention import SelfAttention2d
from nn.common.view import View
from nn.resnet.residual import Up2xResidualBlock, PaddingType


class DCGenerator(G):

    def __init__(self, noise_size: int, image_size: int, ngf=64):
        super(DCGenerator, self).__init__()
        n_up = int(math.log2(image_size / 4))
        assert 4 * (2 ** n_up) == image_size
        nc = 3

        layers = [
            nn.Linear(noise_size, noise_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(noise_size, noise_size),
            nn.LeakyReLU(0.2, True),
            View(-1, noise_size, 1, 1),
            nn.ConvTranspose2d(noise_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
        ]

        nc_l_next = -1
        for l in range(n_up):

            nc_l = max(ngf, (ngf * 8) // 2**l)
            nc_l_next = max(ngf, nc_l // 2)

            layers += [
                nn.ConvTranspose2d(nc_l, nc_l_next, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nc_l_next),
                nn.LeakyReLU(0.2, True),
            ]

            # if l == 2:
            #     layers += [SelfAttention2d(nc_l_next)]

        layers += [
            nn.Conv2d(nc_l_next, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*layers)

    def _device(self):
        return next(self.main.parameters()).device

    def forward(self, noise: Tensor) -> Tensor:
        return self.main(noise)


class ResDCGenerator(G):

    def __init__(self, noise_size: int, image_size: int, ngf=32):
        super(ResDCGenerator, self).__init__()
        n_up = int(math.log2(image_size / 4))
        assert 4 * (2 ** n_up) == image_size
        nc = 3

        layers = [
            nn.Linear(noise_size, noise_size, bias=False),
            View(-1, noise_size, 1, 1),
            nn.ConvTranspose2d(noise_size, ngf * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        nc_l_next = -1
        for l in range(n_up):

            nc_l = max(ngf, (ngf * 8) // 2**l)
            nc_l_next = max(ngf, nc_l // 2)

            layers += [
                Up2xResidualBlock(nc_l, nc_l_next, PaddingType.REFLECT, nn.InstanceNorm2d, use_spectral_norm=False)
            ]

            if l == 2:
                layers += [SelfAttention2d(nc_l_next)]

        layers += [
            nn.Conv2d(nc_l_next, nc, 3, 1, 1, bias=False)
        ]

        self.main = nn.Sequential(*layers)

    def _device(self):
        return next(self.main.parameters()).device

    def forward(self, noise: Tensor) -> Tensor:
        return self.main(noise)

