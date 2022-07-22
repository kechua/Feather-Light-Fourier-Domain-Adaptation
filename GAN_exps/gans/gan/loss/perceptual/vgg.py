from torch import Tensor, nn

from gan.loss.base import GANLoss


class VggGeneratorLoss(GANLoss, nn.Module):

    def __init__(self, depth: int = 20, weight: float = 1):
        super(VggGeneratorLoss, self).__init__()
        self.vgg = Vgg16(depth).to(ParallelConfig.MAIN_DEVICE)
        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.vgg = nn.DataParallel(self.vgg, ParallelConfig.GPU_IDS)

        self.criterion = nn.L1Loss()
        self.weight = weight

    def forward(self, x: Tensor, y: Tensor) -> Loss:

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        return Loss(
            self.criterion(x_vgg, y_vgg.detach()) * self.weight
        )

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return self.forward(fake, real)

    def discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        return Loss.ZERO()
