from torch import nn
from torch import Tensor


class UpBlock(nn.Module):

    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 intermediate_nc: int = 64,
                 n_up: int = 3,
                 norm_layer=nn.BatchNorm2d):
        super(UpBlock, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        max_nc = 1024
        activation = nn.ReLU(True)

        model_list = []
        for i in range(n_up):
            mult = 2 ** (n_up - i)
            input_ngf = min(intermediate_nc * mult, max_nc)
            out_ngf = min((intermediate_nc * mult // 2), max_nc)
            if i == 0:
                input_ngf = input_nc
            model_list += [
                nn.ConvTranspose2d(input_ngf, out_ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(out_ngf),
                activation
            ]

        model_list += [
            nn.ReflectionPad2d(2),
            nn.Conv2d(intermediate_nc, output_nc, kernel_size=5),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model_list)

    def forward(self, x: Tensor):
        assert x.shape[1] == self.input_nc
        res: Tensor = self.model(x)
        assert res.shape[1] == self.output_nc

        return res
