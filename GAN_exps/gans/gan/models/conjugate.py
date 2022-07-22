from torch import Tensor, nn
from torch.nn import init
from torch import optim
from gan.loss.penalties.conjugate import ConjugateGANLoss, ConjugateGANLoss2
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from gan.generator import Generator
from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss


class ConjugateGANModel:

    def __init__(self, generator: Generator, loss: ConjugateGANLoss2):
        self.generator = generator
        self.loss = loss

        self.g_opt = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.792))
        self.d_opt = optim.Adam(loss.parameters(), lr=0.001, betas=(0.5, 0.932))

    def generator_loss(self, noise: Tensor) -> Loss:
        fake = self.generator.forward(noise)
        return self.loss.generator_loss(fake)

    def forward(self, noise: Tensor):
        fake = self.generator.forward(noise)
        return fake
        # return self.brule_loss.d_grad(fake)

    def train_disc(self, real: Tensor, noise: Tensor):
        fake = self.generator.forward(noise)
        loss = self.loss.discriminator_loss(fake.detach(), real)
        loss.minimize_step(self.d_opt)

        return loss.item()

    def train_gen(self, noise: Tensor):
        fake = self.generator.forward(noise)
        loss = self.loss.generator_loss(fake)
        loss.minimize_step(self.g_opt)

        return loss.item()
