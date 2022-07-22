from torch import nn
from torch import Tensor


class DownBlock(nn.Module):

    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 intermediate_nc: int = 64,
                 n_down: int = 3,
                 norm_layer=nn.BatchNorm2d):
        super(DownBlock, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_down = n_down
        max_nc = min(512, output_nc)
        activation = nn.LeakyReLU(0.2, True)

        model_list = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(input_nc, intermediate_nc, kernel_size=5, padding=0),
            norm_layer(intermediate_nc),
            activation
        ]

        for i in range(n_down):
            mult = 2 ** i
            tmp_out_nc = min(intermediate_nc * mult * 2, max_nc) if i < (n_down - 1) else output_nc
            model_list += [
                nn.Conv2d(min(intermediate_nc * mult, max_nc), tmp_out_nc, kernel_size=3, stride=2, padding=1),
                norm_layer(tmp_out_nc),
                activation
            ]

        self.model = nn.Sequential(*model_list)

    def forward(self, x: Tensor):
        assert x.shape[1] == self.input_nc
        res: Tensor = self.model(x)
        assert res.shape[1] == self.output_nc

        return res
