from torch import nn, Tensor

from framework.nn.modules.unet.Down2xConv2d import Down2xConv2d
from framework.nn.modules.unet.Up2xConv2d import Up4xConv2d, Up2xConv2d
from framework.nn.modules.common.View import View
from framework.nn.modules.unet.unet_extra import UNetExtra
from framework.gan.conditional import ConditionalGenerator as CG
from framework.gan.noise import Noise


class UNetGenerator(CG):

    def __init__(self,
                 noise: Noise,
                 image_size,
                 in_channels,
                 out_channels,
                 gen_size=32,
                 n_down=5,
                 nc_max=256):
        super(UNetGenerator, self).__init__(noise)

        middle_data_size = int(image_size * 2 ** (-n_down))
        middle_nc = min(
            int(gen_size * 2 ** n_down),
            nc_max
        )

        assert (middle_data_size >= 4)
        assert (middle_data_size % 4 == 0)

        def down_block_factory(index: int) -> nn.Module:
            mult = 2 ** index
            in_size = min(int(gen_size * mult), nc_max)
            out_size = min(2 * in_size, nc_max)
            return Down2xConv2d(in_size, out_size)

        def up_block_factory(index: int) -> nn.Module:
            mult = 2 ** (n_down - index)
            in_size = min(int(gen_size * mult), nc_max)
            out_size = min(3 * in_size if index > 0 else 2 * in_size, 2 * nc_max)
            return Up2xConv2d(out_size, in_size)

        self.down_last_to_noise = nn.Sequential(
            View(-1, middle_nc * middle_data_size**2),
            spectral_norm_init(nn.Linear(middle_nc * middle_data_size**2, noise.size())),
            nn.ReLU(inplace=True)
        )

        self.noise_up_modules = nn.Sequential(
            spectral_norm_init(nn.Linear(2 * noise.size(), 2 * noise.size())),
            nn.ReLU(True),
            spectral_norm_init(nn.Linear(2 * noise.size(), 2 * noise.size())),
            nn.ReLU(True),
            View(-1, 2 * noise.size(), 1, 1),
            Up4xConv2d(2 * noise.size(), middle_nc)
        )

        up_size = 4

        while up_size < middle_data_size:
            up_size *= 2
            self.noise_up_modules.add_module(
                "up_noize" + str(up_size),
                Up2xConv2d(middle_nc, middle_nc)
            )

        self.unet = UNetExtra(
            n_down,
            in_block=nn.Sequential(
                spectral_norm_init(nn.Conv2d(in_channels, gen_size, 3, stride=1, padding=1)),
                nn.BatchNorm2d(gen_size),
                nn.ReLU(inplace=True)
            ),
            out_block=nn.Sequential(
                spectral_norm_init(nn.Conv2d(2 * gen_size, gen_size, 3, padding=1)),
                nn.BatchNorm2d(gen_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(gen_size, out_channels, 3, 1, 1, bias=True)
            ),
            middle_block=self.down_last_to_noise,
            middle_block_extra=self.noise_up_modules,
            down_block=down_block_factory,
            up_block=up_block_factory
        )

    def forward(self, condition: Tensor, noise: Tensor) -> Tensor:

        return self.unet.forward(condition, noise)
