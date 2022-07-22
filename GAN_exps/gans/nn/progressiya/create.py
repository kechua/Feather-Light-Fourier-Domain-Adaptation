import torch
import torch.nn as nn

from nn.common.view import View
from nn.common.conv_block import Conv2dBlock, TransposeConv2dBlock
from nn.common.linear_block import LinearBlock
from nn.progressiya.unet import UNet4, UNet2
from nn.resnet.skip_plus import SkipPlus


def unet4_256():

    return UNet4(
        down_blocks=[
            nn.Sequential(Conv2dBlock(3, 64, 1), Conv2dBlock(64, 64, 4, downsample=True)),  # 128
            # Conv2dBlock(64, 64, 4, downsample=True),  # 64
            Conv2dBlock(64, 128, 4, downsample=True),  # 64
            Conv2dBlock(128, 256, 4, downsample=True),  # 32
            Conv2dBlock(256, 256, 4, downsample=True),  # 16
            Conv2dBlock(256, 512, 4, downsample=True),  # 8
            Conv2dBlock(512, 512, 4, downsample=True),  # 4
            nn.Sequential(View(-1), LinearBlock(512 * 4 * 4, 512))
        ],
        middle_block=[
            nn.Sequential(Conv2dBlock(3, 64, 1), Conv2dBlock(64, 64, 3)),
            Conv2dBlock(64, 64, 3),
            # Conv2dBlock(64, 64, 3),
            Conv2dBlock(128, 128, 3),
            Conv2dBlock(256, 256, 3),
            Conv2dBlock(256, 256, 3),
            Conv2dBlock(512, 512, 3),
            Conv2dBlock(512, 512, 3),
            LinearBlock(512, 512)
        ],
        up_blocks=[
            Conv2dBlock(128, 64, 3),
            # TransposeConv2dBlock(128, 64, 4),
            TransposeConv2dBlock(128, 64, 4),
            TransposeConv2dBlock(256, 64, 4),
            TransposeConv2dBlock(512, 128, 4),
            TransposeConv2dBlock(512, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            nn.Sequential(LinearBlock(512, 256 * 4 * 4), View(256, 4, 4))
        ],
        final_blocks=[
            SkipPlus(Conv2dBlock(64, 1, 3, activate=None), upsample=None),
            SkipPlus(Conv2dBlock(64, 1, 3, activate=None)),
            # SkipPlus(Conv2dBlock(64, 1, 3)),
            SkipPlus(Conv2dBlock(64, 1, 3)),
            SkipPlus(Conv2dBlock(128, 1, 3)),
            SkipPlus(Conv2dBlock(256, 1, 3)),
            SkipPlus(Conv2dBlock(256, 1, 3)),
            SkipPlus(Conv2dBlock(256, 1, 3)),
            Conv2dBlock(256, 1, 3),
        ])


def unet2_256():

    return UNet2(
        down_blocks=[
            nn.Sequential(Conv2dBlock(3, 64, 1), Conv2dBlock(64, 64, 4, downsample=True)),  # 128
            Conv2dBlock(64, 128, 4, downsample=True),  # 64
            Conv2dBlock(128, 256, 4, downsample=True),  # 32
            Conv2dBlock(256, 256, 4, downsample=True),  # 16
            Conv2dBlock(256, 512, 4, downsample=True),  # 8
            Conv2dBlock(512, 512, 4, downsample=True),  # 4
            nn.Sequential(View(-1), LinearBlock(512 * 4 * 4, 512), LinearBlock(512, 512))
        ],
        up_blocks=[
            nn.Sequential(Conv2dBlock(67, 64, 3), Conv2dBlock(64, 1, 3, activate=None)),
            TransposeConv2dBlock(128, 64, 4),
            TransposeConv2dBlock(256, 64, 4),
            TransposeConv2dBlock(512, 128, 4),
            TransposeConv2dBlock(512, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            nn.Sequential(LinearBlock(512, 256 * 4 * 4), View(256, 4, 4))
        ])


def unet2_200():

    return UNet2(
        down_blocks=[
            nn.Sequential(Conv2dBlock(1, 64, 1), Conv2dBlock(64, 64, 4, downsample=True)),  # 100
            Conv2dBlock(64, 128, 4, downsample=True),  # 50
            Conv2dBlock(128, 256, 4, downsample=True),  # 25
            Conv2dBlock(256, 256, 4, downsample=True),  # 12
            Conv2dBlock(256, 512, 4, downsample=True),  # 6
            Conv2dBlock(512, 512, 4, downsample=True),  # 3
            nn.Sequential(View(-1), LinearBlock(512 * 3 * 3, 512), LinearBlock(512, 512))
        ],
        up_blocks=[
            nn.Sequential(Conv2dBlock(65, 64, 3), Conv2dBlock(64, 1, 3, activate=None)),
            TransposeConv2dBlock(128, 64, 4),
            TransposeConv2dBlock(256, 64, 4),
            nn.Sequential(TransposeConv2dBlock(512, 128, 4), nn.ReplicationPad2d(1)),
            TransposeConv2dBlock(512, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            TransposeConv2dBlock(512 + 256, 256, 4),
            nn.Sequential(LinearBlock(512, 256 * 3 * 3), View(256, 3, 3))
        ])


if __name__ == "__main__":
    net = unet2_200()
    res = net.forward(torch.randn(4, 1, 200, 200))

    print(res.shape)