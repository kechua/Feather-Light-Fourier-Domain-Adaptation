from collections import namedtuple

import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
from gan.nn.stylegan.components import EqualLinear, EqualConv2d
from nn.common.view import View
from nn.resnet.ir_se import make_res_blocks, bottleneck_IR, bottleneck_IR_SE


class StyleEncoder(nn.Module):
    def __init__(self, style_dim, count):
        super(StyleEncoder, self).__init__()
        self.model = [
            EqualConv2d(3, 16, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1),
            EqualLinear(256 * 4 * 4, style_dim * 2, activation="fused_lrelu"),
            EqualLinear(style_dim * 2, style_dim * count),
            View(count, style_dim)
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers, input_nc, mode='ir', style_count = 18, style_multiplayer=1):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [25, 50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = make_res_blocks(num_layers)

        unit_module = {
            'ir': bottleneck_IR,
            'ir_se': bottleneck_IR_SE
        }

        self.input_layer = nn.Sequential(nn.Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module[mode](bottleneck.in_channel,
                                                 bottleneck.depth,
                                                 bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = style_count * style_multiplayer
        self.coarse_ind = 3 * style_multiplayer
        self.middle_ind = 7 * style_multiplayer

        spatial_num = [16] * self.coarse_ind + \
                      [32] * (self.middle_ind - self.coarse_ind) + \
                      [64] * (self.style_count - self.middle_ind)

        for i in range(self.style_count):
            self.styles.append(GradualStyleBlock(512, 512, spatial_num[i]))

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out