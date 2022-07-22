import torch
from torch import nn, Tensor
from gan.discriminator import Discriminator as D
from model import EqualConv2d
from nn.common.attention import SelfAttention2d
from nn.common.view import View
from nn.common.positive import PosConv2d, PosLinear


class DCDiscriminator(D):
    def __init__(self, image_size: int, nc: int = 3, nc_out: int = 1, ndf: int = 32):
        super(DCDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        tmp_size = image_size // 2
        tmp_nc = ndf
        while tmp_size > 2:
            tmp_size = tmp_size // 2
            nc_next = min(256, tmp_nc * 2)
            layers += [
                nn.Conv2d(tmp_nc, nc_next, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(nc_next, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if tmp_size == 32:
                layers += [SelfAttention2d(nc_next)]
            tmp_nc = nc_next

        self.main = nn.Sequential(*layers)

        self.linear = nn.Linear(ndf * 8 * 2 * 2, nc_out)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )


class PosDCDiscriminator(D):
    def __init__(self, image_size: int, nc: int = 3, nc_out: int = 1, ndf: int = 64):
        super(PosDCDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf, affine=True),
        ]

        tmp_size = image_size // 2
        tmp_nc = ndf
        nc_next = -1
        while tmp_size > 2:
            tmp_size = tmp_size // 2
            nc_next = min(256, tmp_nc * 2)
            layers += [
                PosConv2d(tmp_nc, nc_next, 4, 2, 1),
                nn.CELU(0.2, inplace=True),
                nn.InstanceNorm2d(nc_next, affine=True),
            ]
            # if tmp_size == 32:
            #     layers += [SelfAttention2d(nc_next)]
            tmp_nc = nc_next

        self.main = nn.Sequential(*layers)

        self.linear = PosLinear(nc_next * 2 * 2, nc_out)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )


class ConvICNN128(D):
    def __init__(self, channels=3):
        super(ConvICNN128, self).__init__()

        self.first_linear = nn.Sequential(
            EqualConv2d(channels, 64, kernel_size=3, padding=1, bias=True),
        )

        self.first_squared = nn.Sequential(
            EqualConv2d(channels, 64, kernel_size=3, padding=1, bias=True),
        )

        self.convex = nn.Sequential(
            nn.LeakyReLU(0.2),
            PosConv2d(64, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            PosConv2d(128, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            PosConv2d(128, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            PosConv2d(128, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            PosConv2d(128, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            PosConv2d(128, 128, kernel_size=3, stride=2, bias=True, padding=1),
            nn.LeakyReLU(0.2),
            View(-1, 128 * 4 * 4),
            nn.LeakyReLU(0.2),
            PosLinear(128 * 4 * 4, 128, activation=False),
            nn.LeakyReLU(0.2),
            PosLinear(128, 1, activation=False)
        ).cuda()

    def forward(self, input):
        output = self.first_linear(input) + self.first_squared(input) ** 2
        output = self.convex(output)
        return output

    def push(self, input):
        return torch.autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True, grad_outputs=torch.ones(input.size()[0]).cuda().float()
        )[0]

    def convexify(self):
        for layer in self.convex:
            # if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
            try:
                layer.weight.data.clamp_(0)
            except:
                pass



