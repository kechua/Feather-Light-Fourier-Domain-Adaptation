from torch import nn, optim
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from gan.gan_model import stylegan2, StyleGen2Wrapper
from optim.min_max import MinMaxParameters, MinMaxOptimizer
import argparse
import math
import torch
from torchvision import utils

from stylegan2_pytorch.dataset import MultiResolutionDataset
from stylegan2_pytorch.model import ConvLayer, ResBlock, EqualLinear, Generator


class InvGenarator(nn.Module):

    def __init__(self):
        super().__init__()

        code_dim = 512
        blur_kernel = [1, 3, 3, 1]

        self.progression = nn.Sequential(
                ConvLayer(3, 64, 1),
                ResBlock(64, 128, blur_kernel),  # 128
                ResBlock(128, 256, blur_kernel),  # 64
                ResBlock(256, code_dim, blur_kernel),  # 32
                ResBlock(code_dim, code_dim, blur_kernel),  # 16
                ResBlock(code_dim, code_dim, blur_kernel),  # 8
                ResBlock(code_dim, code_dim, blur_kernel),  # 4
        )

        self.style1 = nn.Sequential(
            EqualLinear(code_dim * 16, code_dim, activation='fused_lrelu', lr_mul=1),
            EqualLinear(code_dim, code_dim, activation='fused_lrelu', lr_mul=1),
            EqualLinear(code_dim, code_dim, activation='fused_lrelu', lr_mul=0.1)
        )

        self.style2 = nn.Sequential(
            EqualLinear(code_dim * 16, code_dim, activation='fused_lrelu', lr_mul=1),
            EqualLinear(code_dim, code_dim, activation='fused_lrelu', lr_mul=1),
            EqualLinear(code_dim, code_dim, activation='fused_lrelu', lr_mul=0.1)
        )

    def forward(self, img):
        conv = self.progression(img).view(img.shape[0], -1)
        return self.style1(conv), self.style2(conv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the image')
    parser.add_argument('path', type=str, help='path to checkpoint file')

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    batch = 8

    args = parser.parse_args()
    size = 256

    noise = []

    gan_model = stylegan2("/home/name/stylegan2/stylegan2-pytorch/checkpoint/790000.pt", "wasserstein", 0.0001)
    netG: StyleGen2Wrapper = gan_model.generator
    netG.inject_index = 3
    netG.return_latents = True

    generator: Generator = Generator(256, 512, 8, channel_multiplier=2)
    generator.load_state_dict(netG.gen.state_dict())
    generator = StyleGen2Wrapper(generator).cuda()
    generator.inject_index = 3
    generator.return_latents = True

    g_inv: InvGenarator = InvGenarator().cuda()
    opt_inv = optim.Adam(g_inv.parameters(),
                         lr=0.002,
                         betas=(0.5, 0.999))
    opt_inv.add_param_group({
        "lr": 0.001,
        "params": netG.parameters()
    })

    for iter in range(100000):

        with torch.no_grad():
            z1 = torch.randn(batch, 512, device=device)
            z2 = torch.randn(batch, 512, device=device)
            generator.input_is_latent = False
            image, latent = generator.forward(z1, z2)

        style1, style2 = g_inv(image)
        netG.input_is_latent = True
        image_pred, _ = netG.forward(style1, style2)
        g_inv.zero_grad()
        loss = nn.L1Loss()(image_pred, image) #+ (style2 - latent[:, -1]).abs().mean()
        loss.backward()
        opt_inv.step()

        if iter % 10 == 0:
            print(iter, loss.item())
            # utils.save_image(
            #     image_pred, f'sample_{iter}.png', nrow=batch // 2, normalize=True, range=(-1, 1)
            # )




