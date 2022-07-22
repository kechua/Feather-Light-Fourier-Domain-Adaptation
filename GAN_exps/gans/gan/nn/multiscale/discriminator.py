from typing import List

from torch import nn, Tensor


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, discriminators: List[nn.Module]):
        super(MultiscaleDiscriminator, self).__init__()
        self.discriminators = discriminators
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input: Tensor):
        result = []
        input_downsampled = input

        for D in self.discriminators:
            result.append(D(input_downsampled))
            input_downsampled = self.downsample(input_downsampled)

        return result
