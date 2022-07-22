import random
import time
import os, sys

sys.path.append(os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../stylegan2'))

from typing import List
import torch.utils.data
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from gan.nn.stylegan.discriminator import Discriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator
from gan.noise.stylegan import mixing_noise
from optim.accumulator import Accumulator
from parameters.path import Paths


def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


manualSeed = 71
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 16
image_size = 256
noise_size = 512

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

test_sample_z = torch.randn(8, noise_size, device=device)
Celeba.batch_size = batch_size

generator = Generator(FromStyleConditionalGenerator(image_size, noise_size), n_mlp=8)
discriminator = Discriminator(image_size)

starting_model_number = 290000
weights = torch.load(
    f'{Paths.default.models()}/celeba_gan_256_{str(starting_model_number).zfill(6)}.pt',
    map_location="cpu"
)

discriminator.load_state_dict(weights['d'])
generator.load_state_dict(weights['g'])

generator = generator.cuda()
discriminator = discriminator.cuda()

gan_model = StyleGanModel(generator, StyleGANLoss(discriminator), (0.001, 0.0015))
gan_accumulator = Accumulator(generator, decay=0.99, write_every=100)

writer = SummaryWriter(f"{Paths.default.board()}/celeba{int(time.time())}")

print("Starting Training Loop...")
starting_model_number = 0

for i in range(300000):

    print(i)

    real_img = next(LazyLoader.celeba().loader).to(device)

    noise: List[Tensor] = mixing_noise(batch_size, noise_size, 0.9, device)
    fake, latent = generator.forward(noise, return_latents=True)

    gan_model.discriminator_train([real_img], [fake.detach()])
    gan_model.generator_loss_with_penalty([real_img], [fake], latent).minimize_step(gan_model.optimizer.opt_min)
    # gan_model.generator_loss([real_img], [fake]).minimize_step(gan_model.optimizer.opt_min)

    gan_accumulator.step(i)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():
            fake_test, _ = generator.forward([test_sample_z])
            send_images_to_tensorboard(writer, fake_test, "FAKE", i)


    if i % 10000 == 0 and i > 0:
        gan_accumulator.write_to(generator)
        torch.save(
            {
                'g': generator.state_dict(),
                'd': discriminator.state_dict()
            },
            f'{Paths.default.models()}/celeba_gan_256_{str(i + starting_model_number).zfill(6)}.pt',
        )
