from torch import nn, Tensor

from gan.generator import Generator as G


class EGenerator(G):

    def _device(self):
        return next(self.main.parameters()).device

    def __init__(self, size):
        super(EGenerator, self).__init__()
        n_out = 2
        ngf = 64
        self.main = nn.Sequential(
            nn.Linear(size, ngf),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ngf,  2 * ngf),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2 * ngf, n_out)
        )

    def forward(self, x) -> Tensor:
        return self.main(x)
