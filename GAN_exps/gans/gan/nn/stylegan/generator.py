import itertools
import math
import random
import time
from typing import List
from stylegan2_bk.model import Generator as StyleGenerator2
import torch
from torch import Tensor, nn
from nn.common.view import View
from gan.nn.stylegan.components import ScaledConvTranspose2d, ModulatedConv2d, PixelNorm, EqualLinear, StyledConv, \
    ToRGB, ConditionInjection, ConstantInput, ConvLayer
from nn.progressiya.base import Progressive, ProgressiveWithoutState, InjectByName, LastElementCollector
from nn.progressiya.unet import ProgressiveSequential, ZapomniKak, InputFilterName, InputFilterVertical, CopyKwToArgs
from gan.generator import Generator as G
from gan.generator import ConditionalGenerator as CG


class NoiseToStyle(nn.Module):

    def __init__(self, style_dim, n_mlp, lr_mlp, n_latent):
        super().__init__()

        self.n_latent = n_latent

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self, z, inject_index):

        styles = [self.style(zi) for zi in z]

        if len(styles) < 2:
            inject_index = self.n_latent
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return [latent[:, i] for i in range(self.n_latent)]


class FromStyleConditionalGenerator(CG):
    def __init__(
        self,
        size,
        style_dim,
        channel_multiplier=1,
        blur_kernel=[1, 3, 3, 1],
        style_multiplayer=1
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.style_multiplayer = style_multiplayer

        self.channels = {
            4: 512,
            8: 512,
            16: 256 * channel_multiplier,
            32: 256 * channel_multiplier,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        conv1 = StyledConv(
            ModulatedConv2d(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel),
            ConditionInjection(self.channels[4])
        )
        to_rgb1 = ToRGB(self.channels[4], style_dim)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        convs = []
        to_rgbs = []

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            convs.append(
                StyledConv(
                    ModulatedConv2d(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel),
                    ConditionInjection(out_channel)
                )
            )

            for _ in range((style_multiplayer - 1) * 2 + 1):
                convs.append(
                    StyledConv(
                        ModulatedConv2d(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel),
                        ConditionInjection(out_channel)
                    )
                )

            to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        # print( self.n_latent)

        self.progression = ProgressiveSequential(
            CopyKwToArgs({"init"}),
            InputFilterName({'noise', 'style'}),
            Progressive[List[Tensor]]([conv1] + convs, InjectByName("input")),
            ZapomniKak("input"),
            InputFilterName({'input', 'style'}),
            InputFilterVertical(list(range(1, len(convs) + 2 + (style_multiplayer - 1) * 2, 2 * style_multiplayer))),
            ProgressiveWithoutState[Tensor]([to_rgb1] + to_rgbs, InjectByName("skip"), LastElementCollector)
        )


    def forward(
        self,
        init: Tensor,
        styles: List[Tensor],
        condition: List[Tensor]
    ):
        image = self.progression.forward(init=init, noise=condition, style=styles)
        return image


class Generator(G):

    def __init__(self, gen: FromStyleConditionalGenerator, n_mlp=5, lr_mlp=0.01):
        super().__init__()
        self.gen = gen
        self.z_to_style = NoiseToStyle(gen.style_dim,  n_mlp, lr_mlp, gen.n_latent)
        self.input = ConstantInput(gen.channels[4])
        self.style_multiplayer = gen.style_multiplayer

        for i in range(self.gen.n_latent * self.style_multiplayer + 1):
            self.register_parameter(f"w_{i}", nn.Parameter(torch.zeros(1)))

    def make_noise(self, B, device):

        ws = [
            getattr(self, f'w_{i}') for i in range(self.gen.n_latent * self.style_multiplayer + 1)
        ]

        noise = [torch.randn(B, self.gen.channels[4], 4, 4, device=device)]

        for i in range(0, self.gen.n_latent // 2):
            for j in range(self.style_multiplayer):
                noise.append(torch.randn(B, self.gen.channels[2 ** (i + 3)], 8 * (2 ** i), 8 * (2 ** i), device=device))
                noise.append(torch.randn(B, self.gen.channels[2 ** (i + 3)], 8 * (2 ** i), 8 * (2 ** i), device=device))

        return [n * w for n, w in zip(noise, ws)]

    def forward(
        self,
        z: List[Tensor],
        inject_index=None,
        return_latents=False
    ):

        latent = self.z_to_style.forward(z, inject_index)
        condition = self.make_noise(z[0].shape[0], z[0].device)
        image = self.gen(self.input(latent[0]), latent, condition)

        res_latent = latent if return_latents else None

        return image, res_latent


class ImageToImage(nn.Module):

    def __init__(self, gen: FromStyleConditionalGenerator, image_channels: int):
        super().__init__()

        self.gen: FromStyleConditionalGenerator = gen

        # self.init_cov = ConvLayer(heatmap_channels, self.gen.channels[256], kernel_size=1)

        self.noise = [
            ConvLayer(image_channels, self.gen.channels[gen.size], kernel_size=1),
            ConvLayer(self.gen.channels[gen.size], self.gen.channels[gen.size], 3, downsample=False),
        ]
        tmp_size = gen.size
        while tmp_size > 4:
            self.noise.append(
                ConvLayer(self.gen.channels[tmp_size], self.gen.channels[tmp_size//2], 3, downsample=True)
            )
            tmp_size = tmp_size // 2

        self.noise = Progressive(self.noise)

    def forward(self, cond: Tensor, styles: List[Tensor], return_latents=False, inject_index=None):

        latent = styles
        condition = self.noise(cond)[-1:1:-1]
        condition = list(itertools.chain(*zip(condition, condition)))[1:]
        image = self.gen(condition[0], latent, condition)

        res_latent = latent if return_latents else None

        return image, res_latent


class Decoder(nn.Module):
    def __init__(self, gen: nn.Module):
        super().__init__()
        self.gen = gen

    def forward(self, latent: List[Tensor]):

        condition = self.gen.make_noise(latent[0].shape[0], latent[0].device)
        image = self.gen.gen(self.gen.input(latent[0]), latent, condition)

        return image

class HeatmapToImage(nn.Module):

    def __init__(self, gen: FromStyleConditionalGenerator, z_to_style: NoiseToStyle, heatmap_channels: int):
        super().__init__()

        self.gen: FromStyleConditionalGenerator = gen
        self.z_to_style: NoiseToStyle = z_to_style

        # self.init_cov = ConvLayer(heatmap_channels, self.gen.channels[256], kernel_size=1)

        self.noise = [
            ConvLayer(heatmap_channels, self.gen.channels[gen.size], kernel_size=1),
            ConvLayer(self.gen.channels[gen.size], self.gen.channels[gen.size], 3, downsample=False),
        ]
        tmp_size = gen.size
        while tmp_size > 4:
            self.noise.append(
                ConvLayer(self.gen.channels[tmp_size], self.gen.channels[tmp_size//2], 3, downsample=True)
            )
            tmp_size = tmp_size // 2

        self.noise = Progressive(self.noise)

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False, inject_index=None):

        latent = self.z_to_style.forward(z, inject_index)
        condition = self.noise(cond)[-1:1:-1]
        condition = list(itertools.chain(*zip(condition, condition)))[1:]
        image = self.gen(condition[0], latent, condition)

        res_latent = latent if return_latents else None

        return image, res_latent


class HeatmapAndStyleToImage(nn.Module):

    def __init__(self, gen: HeatmapToImage):
        super().__init__()
        self.gen = gen

    def forward(self, cond: Tensor, latent: Tensor):

        condition = self.gen.noise(cond)[-1:1:-1]
        condition = list(itertools.chain(*zip(condition, condition)))[1:]
        latent_list = [latent[:, k] for k in range(latent.shape[1])]
        image = self.gen.gen(condition[0], latent_list, condition)

        return image


class CondGen7(nn.Module):

    def __init__(self, gen: StyleGenerator2, heatmap_channels: int, cond_mult: float = 10):
        super().__init__()

        self.cond_mult = cond_mult

        self.gen: Generator = gen

        self.init_cov = ConvLayer(heatmap_channels, self.gen.channels[256], kernel_size=1)

        self.noise = nn.ModuleList([
            ConvLayer(self.gen.channels[256], self.gen.channels[256], 3, downsample=False),
            ConvLayer(self.gen.channels[256], self.gen.channels[128], 3, downsample=True),
            ConvLayer(self.gen.channels[128], self.gen.channels[64], 3, downsample=True),
            ConvLayer(self.gen.channels[64], self.gen.channels[32], 3, downsample=True),
            ConvLayer(self.gen.channels[32], self.gen.channels[16], 3, downsample=True),
            ConvLayer(self.gen.channels[16], self.gen.channels[8], 3, downsample=True),
            ConvLayer(self.gen.channels[8], self.gen.channels[4], 3, downsample=True)
        ])
        self.inject_index = 2

    def make_noise(self, heatmap: Tensor):
        x = self.init_cov(heatmap)

        noise_down_list = []
        for i in self.noise:
            x = i.forward(x)
            noise_down_list.append(x)
            noise_down_list.append(x)

        return noise_down_list[-2::-1]

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):
        noise = self.make_noise(cond)
        return self.gen(z, condition=noise[0], noise=noise, return_latents=return_latents,
                        inject_index=self.inject_index)


class ConditionalDecoder(nn.Module):
    def __init__(self, gen: nn.Module):
        super().__init__()
        self.gen = gen

    def forward(self, cond: Tensor, latent: Tensor):
        noise = self.gen.make_noise(cond)
        latent = [latent[:, 0], latent[:, 1]]
        return self.gen.gen(latent, condition=noise[0], noise=noise, input_is_latent=True, inject_index=self.gen.inject_index)[0]

