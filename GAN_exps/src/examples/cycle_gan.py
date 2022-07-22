import argparse
import sys, os
import time
import json

sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../../'))
sys.path.append(os.path.join(sys.path[0], '../../gans/'))
sys.path.append(os.path.join(sys.path[0], '../../src/'))
from itertools import chain

from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor, nn

from src.examples.cycle_funcs import iteration_loop, tuda_cuda
from src.examples.autoencoder_penalty import DecoderPenalty, EncoderPenalty
from optim.accumulator import Accumulator
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from src.examples.style_progressive import StyleDisc, StyleTransform
from gan.loss.loss_base import Loss
from gan.loss.perceptual.psp import PSPLoss
from gan.nn.stylegan.generator import Decoder, Generator, FromStyleConditionalGenerator, ImageToImage
from gan.nn.stylegan.discriminator import Discriminator
from gan.nn.stylegan.style_encoder import GradualStyleEncoder, StyleEncoder
from parameters.run import RuntimeParameters

import torch
from dataset.lazy_loader import LazyLoader
from parameters.path import Paths


def jointed_loader(loader1, loader2):
    while True:
        yield next(loader1)
        yield next(loader2)

def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[
        RuntimeParameters(),
    ]
)
args = parser.parse_args()
for k in vars(args):
    print(f"{k}: {vars(args)[k]}")

starting_model_number = 000000

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


writer = SummaryWriter(f"{Paths.default.board()}/CycleGAN_DA_ph15->ge15_{int(time.time())}")

# weights = torch.load(
#     f'{Paths.default.models()}/StyleGAN_DA_Celeba_{str(starting_model_number).zfill(6)}.pt',
#     map_location="cpu"
# )

style_enc_tuda = StyleEncoder(style_dim=512, count=14).cuda() #GradualStyleEncoder(50, 3, mode="ir", style_count=14).cuda()

style_enc_obratno = StyleEncoder(style_dim=512, count=14).cuda() #GradualStyleEncoder(50, 3, mode="ir", style_count=14).cuda()
# style_enc.load_state_dict(weights['enc'])

disc_tuda = Discriminator(size=256).cuda()
disc_obratno = Discriminator(size=256).cuda()

generator_tuda = ImageToImage(gen=FromStyleConditionalGenerator(256,512), image_channels=3).cuda()

generator_obratno = ImageToImage(gen=FromStyleConditionalGenerator(256,512), image_channels=3).cuda()

gan_model_tuda = StyleGanModel(generator_tuda, StyleGANLoss(disc_tuda), (0.001/2, 0.0015/2))
gan_model_obratno = StyleGanModel(generator_obratno, StyleGANLoss(disc_obratno), (0.001/2, 0.0015/2))

loader = jointed_loader(LazyLoader.domain_adaptation_ge15().loader_train_inf, #LazyLoader.domain_adaptation_philips15().loader_train_inf,
                        LazyLoader.domain_adaptation_philips15().loader_train_inf)

rec_loss = PSPLoss(id_lambda=0).cuda()

style_tuda_opt = Adam(style_enc_tuda.parameters(), lr=0.001)
style_obratno_opt = Adam(style_enc_obratno.parameters(), lr=0.001)


gan_tuda_accumulator = Accumulator(gan_model_tuda.generator, decay=0.99, write_every=100)
gan_obratno_accumulator = Accumulator(gan_model_obratno.generator, decay=0.99, write_every=100)

for i in range(60000):
    coefs = json.load(open(os.path.join(sys.path[0], "../parameters/cycle.json")))

    batch_x = next(loader)
    batch_y = next(loader)

    init_image_x = batch_x['image'].to(device).repeat(1, 3, 1, 1)
    init_image_y = batch_y['image'].to(device).repeat(1, 3, 1, 1)

    #tuda
    iteration_loop(
        batch_x=batch_x, batch_y=batch_y,
        style_enc=style_enc_tuda, gan_model=gan_model_tuda,
        style_opt=style_tuda_opt, device=device
    )

   #obratno
    iteration_loop(
        batch_x=batch_y, batch_y=batch_x,
        style_enc=style_enc_obratno, gan_model=gan_model_obratno,
        style_opt=style_obratno_opt, device=device
    )

    #tuda_obratno
    tuda_obratno_loss = tuda_cuda(
        batch_x,
        style_enc_tuda, style_enc_obratno,
        gan_model_tuda, gan_model_obratno,
        style_tuda_opt, style_obratno_opt,
        device, coefs["L1_tuda"]
    )

    #obratno_tuda
    obratno_tuda_loss = tuda_cuda(
        batch_y,
        style_enc_obratno, style_enc_tuda,
        gan_model_obratno, gan_model_tuda,
        style_obratno_opt, style_tuda_opt,
        device, coefs["L1_cuda"]
    )

    gan_tuda_accumulator.step(i)
    gan_obratno_accumulator.step(i)

    if i % 10 == 0:
        writer.add_scalar("tuda_obratno_loss", tuda_obratno_loss, i)
        writer.add_scalar("obratno_tuda_loss", obratno_tuda_loss, i)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():
            batch_x = next(loader)
            batch_y = next(loader)
            image_x = batch_x['image'].to(device).repeat(1, 3, 1, 1)
            image_y = batch_y['image'].to(device).repeat(1, 3, 1, 1)

            send_images_to_tensorboard(writer, image_x, "X", i)

            style_vec_tuda = style_enc_tuda(image_x)
            style_vec_list_tuda = [style_vec_tuda[:, k] for k in range(style_vec_tuda.shape[1])]
            image_y_fake, _ = gan_model_tuda.generator.forward(
                cond=image_x, styles=style_vec_list_tuda
            )


            send_images_to_tensorboard(writer, image_y_fake, "X -> Y", i)

            send_images_to_tensorboard(writer, image_y, "Y", i)

            style_vec_obratno = style_enc_obratno(image_y_fake)
            style_vec_list_obratno = [style_vec_obratno[:, k] for k in range(style_vec_obratno.shape[1])]
            image_x_fake, _ = gan_model_tuda.generator.forward(
                cond=image_y_fake, styles=style_vec_list_obratno
            )

            send_images_to_tensorboard(writer, image_x_fake, " X -> Y -> X", i)


    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'st_enc_tuda': style_enc_tuda.state_dict(),
                'st_enc_obratno': style_enc_obratno.state_dict(),
                'dec_tuda': disc_tuda.state_dict(),
                'dec_obratno': disc_obratno.state_dict(),
                'gen_tuda': generator_tuda.state_dict(),
                'gen_obratno': generator_obratno.state_dict()
            },
            f'{Paths.default.models()}/CycleGAN_DA_ph15->ge15_{str(i + starting_model_number).zfill(6)}.pt',
        )
