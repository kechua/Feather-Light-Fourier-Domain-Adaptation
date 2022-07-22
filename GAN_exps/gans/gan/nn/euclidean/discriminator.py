from torch import nn, Tensor
from gan.discriminator import Discriminator as D
from nn.common.positive import PosLinear


class EDiscriminator(D):
    def __init__(self, dim=2, ndf=64):
        super(EDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(dim, ndf),
            nn.LeakyReLU(0.2, True),
            PosLinear(ndf, 2 * ndf, activation=None),
            nn.LeakyReLU(0.2, True),
            PosLinear(2 * ndf, 2 * ndf, activation=None),
            nn.LeakyReLU(0.2, True),
            PosLinear(2 * ndf, 1, activation=None),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
