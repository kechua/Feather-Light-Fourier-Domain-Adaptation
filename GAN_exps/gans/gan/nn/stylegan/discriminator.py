import math
import torch
from torch import Tensor, nn
from gan.discriminator import Discriminator as D
from gan.nn.stylegan.components import ConvLayer, EqualLinear, ResBlock


def cat_std_dev(x: Tensor, stddev_group, stddev_feat):
    batch, channel, height, width = x.shape
    n_group = min(batch, stddev_group)

    stddev = x.view(
        n_group, -1, stddev_feat, channel // stddev_feat, height, width
    )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(n_group, 1, height, width)

    return torch.cat([x, stddev], 1)


class Discriminator(D):

    def __init__(self, size,  input_nc=3, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: int(256 * channel_multiplier),
            32: int(256 * channel_multiplier),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }
        self.channels = channels
        convs = [ConvLayer(input_nc, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input: Tensor):
        out = self.convs(input)

        out = cat_std_dev(out, self.stddev_group, self.stddev_feat)

        out = self.final_conv(out)

        out = out.view(input.shape[0], -1)
        out = self.final_linear(out)

        return out


class CondBinaryDiscriminator(Discriminator):
    def __init__(self, size, input_nc=4):
        super().__init__(size=size, input_nc=input_nc)
        self.size = size
        self.input_nc = input_nc

    def forward(self, input: Tensor, cond: Tensor):
        B = input.shape[0]
        cond_matrix = cond.view(B, 1, 1, 1) * torch.ones_like(input[:, 0, ...]).unsqueeze(dim=1) # (B, C, 512, 512)
        input = torch.cat((input, cond_matrix), dim=1)
        return super().forward(input)


class ConditionalDiscriminator(D):
    def __init__(self, size, heatmap_channels: int, channel_multiplier=1, blur_kernel=[1, 3, 3, 1], cond_mult=1.0):
        super().__init__()

        self.cond_mult = cond_mult

        channels = {
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

        self.channels = channels
        self.init_conv = ConvLayer(3, channels[size], 1)
        self.init_conv_cond = ConvLayer(heatmap_channels, channels[size], 1)

        self.log_size = int(math.log(size, 2))
        in_channel = 2 * channels[size]

        self.main = nn.ModuleList([])

        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.main.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

        self.inject_index = 2

    def forward(self, input: Tensor, cond: Tensor):

        out = self.init_conv(input)
        out_cond = self.init_conv_cond(cond)

        out = torch.cat([out, out_cond], dim=1)

        for i in range(len(self.main)):
            out = self.main[i](out)

        out = cat_std_dev(out, self.stddev_group, self.stddev_feat)

        out = self.final_conv(out)

        out = out.view(input.shape[0], -1)
        out = self.final_linear(out)

        return out
