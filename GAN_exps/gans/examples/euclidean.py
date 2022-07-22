from __future__ import print_function
import random

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from gan.models.conjugate import ConjugateGANModel
from gan.nn.euclidean.discriminator import EDiscriminator
from gan.nn.euclidean import EGenerator
from gan.loss.penalties.conjugate import ConjugateGANLoss2
from gan.noise.normal import NormalNoise

batch_size = 64
noise_size = 2

device = torch.device("cuda:1")

noise = NormalNoise(noise_size, device)
netG = EGenerator(noise_size).to(device)
netD_1 = EDiscriminator().to(device)
netD_2 = EDiscriminator().to(device)

lr = 0.001
betas = (0.5, 0.999)

gan_model = ConjugateGANModel(
    netG,
    ConjugateGANLoss2(netD_1, netD_2, 10),
)


n = 5000

xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
ys = torch.cat((xs.cos(), xs.sin()), dim=1)


plt.scatter(ys[:, 0].view(n).numpy(), ys[:, 1].view(n).numpy())

z = noise.sample(3 * batch_size)
fake = netG.forward(z)
plt.scatter(fake[:, 0].cpu().view(3 * batch_size).detach().numpy(),
            fake[:, 1].cpu().view(3 * batch_size).detach().numpy())
plt.show()


print("Starting Training Loop...")


def gen_batch() -> Tensor:
    i = random.randint(0, n - batch_size)
    j = i + batch_size
    return ys[i:j, :]


for iter in range(0, 9000):

    data = gen_batch().to(device)
    z = noise.sample(batch_size)
    loss_d = gan_model.train_disc(data, z)

    loss_g = 0
    if iter % 5 == 0 and iter > 100:
        loss_g = gan_model.train_gen(z)

    if iter % 300 == 0:
        # print(gan_model.brule_loss.get_penalties()[1].weight)
        print(str(loss_d) + ", g = " + str(loss_g))
        fake = gan_model.forward(z)
        plt.scatter(fake[:, 0].cpu().view(batch_size).detach().numpy(),
                    fake[:, 1].cpu().view(batch_size).detach().numpy())
        plt.show()

