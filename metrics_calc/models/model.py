from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d
import torch
import numpy as np

class LinearFourier2d(torch.nn.Module):
    def __init__(self, image_size, log):
        super(LinearFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        torch.nn.init.ones_(self.fourier_filter)

    def forward(self, x):
        w = torch.nn.ReLU()(self.fourier_filter.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = torch.exp(w * torch.log(1 + init_spectrum)) - 1
        else:
            spectrum = w * init_spectrum

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf


class GeneralFourier2d(torch.nn.Module):
    def __init__(self, image_size, log):
        super(GeneralFourier2d, self).__init__()

        self.log = log

        c, h, w = image_size
        self.register_parameter(name='W1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        # self.W1 = torch.Tensor(np.load('UNet_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('UNet_log_weights.npy')).squeeze(0)

        # self.W1 = torch.Tensor(np.load('DenseNetSegmentation_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('DenseNetSegmentation_log_weights.npy')).squeeze(0)

        # self.W1 = torch.Tensor(np.load('ResNetSegmentation_weights.npy')).squeeze(0)
        # self.W1 = torch.Tensor(np.load('ResNetSegmentation_log_weights.npy')).squeeze(0)

        self.register_parameter(name='B1', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='W2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))
        self.register_parameter(name='B2', param=torch.nn.Parameter(torch.empty(c, h, w // 2 + 1)))

        torch.nn.init.ones_(self.W1)
        torch.nn.init.zeros_(self.B1)
        torch.nn.init.ones_(self.W2)
        torch.nn.init.zeros_(self.B2)

        # activation functions
        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.ReLU6()
        # self.activation = torch.nn.Softplus()
        # self.activation = torch.nn.Tanh()
        # self.activation = lambda x: x * torch.nn.Sigmoid()(x)  # Swish (beta = 1.0)
        # self.activation = lambda x: x * torch.nn.Tanh()(torch.nn.Softplus()(x))  # Mish

    def forward(self, x):
        w1 = torch.nn.ReLU()(self.W1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        w2 = torch.nn.ReLU()(self.W2.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b1 = torch.nn.ReLU()(self.B1.repeat(x.shape[0], 1, 1, 1).to(x.device))
        b2 = torch.nn.ReLU()(self.B2.repeat(x.shape[0], 1, 1, 1).to(x.device))

        rft_x = torch.rfft(x, signal_ndim=3, normalized=True, onesided=True)
        init_spectrum = torch.sqrt(torch.pow(rft_x[..., 0], 2) + torch.pow(rft_x[..., 1], 2))

        if self.log:
            spectrum = w2 * self.activation(w1 * torch.log(1 + init_spectrum) + b1) + b2
        else:
            spectrum = w2 * self.activation(w1 * init_spectrum + b1) + b2

        irf = torch.irfft(torch.stack([rft_x[..., 0] * spectrum / (init_spectrum + 1e-16),
                                       rft_x[..., 1] * spectrum / (init_spectrum + 1e-16)], dim=-1),
                          signal_ndim=3, normalized=True, onesided=True, signal_sizes=x.shape[1:])

        return irf


class double_conv(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class down_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(down_step, self).__init__()

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(self.pool(x))


class up_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(up_step, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, from_up_step, from_down_step):
        upsampled = self.up(from_up_step)
        x = torch.cat([from_down_step, upsampled], dim=1)
        return self.conv(x)


class out_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(out_conv, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, init_features, depth, image_size, fourier_params=None):
        super(UNet, self).__init__()

        self.features = init_features
        self.depth = depth

        self.fourier_params = fourier_params
        if fourier_params is not None:
            if self.fourier_params['fourier_layer'] == 'linear':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'linear_log':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=True)
            if self.fourier_params['fourier_layer'] == 'general':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'general_log':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=True)

        self.down_path = torch.nn.ModuleList()
        self.down_path.append(double_conv(n_channels, self.features, self.features))
        for i in range(1, self.depth):
            self.down_path.append(down_step(self.features, 2 * self.features))
            self.features *= 2

        self.up_path = torch.nn.ModuleList()
        for i in range(1, self.depth):
            self.up_path.append(up_step(self.features, self.features // 2))
            self.features //= 2
        self.out_conv = out_conv(self.features, n_classes)

    def forward_down(self, input):
        downs = [input]
        for down_step in self.down_path:
            downs.append(down_step(downs[-1]))

        return downs

    def forward_up(self, downs):
        current_up = downs[-1]
        for i, up_step in enumerate(self.up_path):
            current_up = up_step(current_up, downs[-2 - i])

        return current_up

    def forward(self, x):
        if self.fourier_params is not None:
            x = self.fl(x)

        downs = self.forward_down(x)
        up = self.forward_up(downs)

        return self.out_conv(up)


class DenseLayer(torch.nn.Sequential):
    def __init__(self, num_input_features, growth_rate, batch_size):
        super(DenseLayer, self).__init__()

        self.dense_layer = torch.nn.Sequential(torch.nn.BatchNorm2d(num_input_features),
                                               torch.nn.ReLU(inplace=True),
                                               torch.nn.Conv2d(num_input_features, batch_size * growth_rate,
                                                               kernel_size=1, stride=1, bias=False),
                                               torch.nn.BatchNorm2d(batch_size * growth_rate),
                                               torch.nn.ReLU(inplace=True),
                                               torch.nn.Conv2d(batch_size * growth_rate, growth_rate,
                                                               kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.dense_layer(x)], dim=1)


class DenseBlock(torch.nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, batch_size):
        super(DenseBlock, self).__init__()

        self.dense_block = torch.nn.ModuleList()
        for i in range(num_layers):
            self.dense_block.append(DenseLayer(num_input_features + i * growth_rate, growth_rate, batch_size))

    def forward(self, x):
        for layer in self.dense_block:
            x = layer(x)
        return x


class Transition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()

        self.transition_layer = torch.nn.Sequential(torch.nn.BatchNorm2d(num_input_features),
                                                    torch.nn.ReLU(inplace=True),
                                                    torch.nn.Conv2d(num_input_features, num_output_features,
                                                                    kernel_size=1, stride=1, bias=False),
                                                    torch.nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition_layer(x)


class DecoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderLayer, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(self.up(x))


class DenseNetSegmentation(torch.nn.Module):
    """Densenet Segmentation model class, based on
       "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    """

    def __init__(self, n_channels, n_classes, init_features, growth_rate, block_config,
                 image_size, batch_size, fourier_params=None):
        super(DenseNetSegmentation, self).__init__()

        self.fourier_params = fourier_params
        if fourier_params is not None:
            if self.fourier_params['fourier_layer'] == 'linear':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'linear_log':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=True)
            if self.fourier_params['fourier_layer'] == 'general':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'general_log':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=True)

        self.init_conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(init_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = [init_features]
        self.encoder = torch.nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            self.encoder.append(DenseBlock(num_layers=num_layers, num_input_features=num_features[-1],
                                           growth_rate=growth_rate, batch_size=batch_size))

            num_features.append(num_features[-1] + num_layers * growth_rate)
            if i != len(block_config) - 1:
                self.encoder.append(
                    Transition(num_input_features=num_features[-1], num_output_features=num_features[-1] // 2))
                num_features[-1] = num_features[-1] // 2

        self.decoder = torch.nn.ModuleList()
        for i in range(1, len(block_config) + 1):
            self.decoder.append(DecoderLayer(num_features[-i], num_features[-i - 1]))

        self.out_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(num_features[0], num_features[0], kernel_size=2, stride=2),
            torch.nn.Conv2d(num_features[0], n_classes, kernel_size=1, stride=1, padding=0))

        # initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.fourier_params is not None:
            x = self.fl(x)

        x = self.init_conv(x)

        for layer in self.encoder:
            x = layer(x)

        for layer in self.decoder:
            x = layer(x)

        return self.out_conv(x)


class ResLayer(torch.nn.Module):
    def __init__(self, num_input_features, num_features, downsample):
        super(ResLayer, self).__init__()

        self.downsample = downsample

        stride = 1
        if self.downsample is not None:
            stride = 2
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(num_input_features, num_features,
                                                          kernel_size=3, stride=stride, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(num_features))

        self.relu = torch.nn.ReLU(inplace=True)

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(num_features, num_features,
                                                          kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(num_features))

    def forward(self, x):
        residual = x

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual

        return self.relu(x)


class ResBlock(torch.nn.Sequential):
    def __init__(self, num_layers, num_input_features, num_features, downsampling):
        super(ResBlock, self).__init__()

        self.downsample = None
        if downsampling:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(num_input_features, num_features,
                                                                  kernel_size=1, stride=2, bias=False),
                                                  torch.nn.BatchNorm2d(num_features))

        self.res_block = torch.nn.ModuleList()
        for i in range(num_layers):
            self.res_block.append(ResLayer(num_input_features, num_features, self.downsample))
            num_input_features = num_features
            self.downsample = None

    def forward(self, x):
        for layer in self.res_block:
            x = layer(x)
        return x


class ResNetSegmentation(torch.nn.Module):
    """Residual Segmentation model class, based on
       "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    """

    def __init__(self, n_channels, n_classes, blocks, image_size, fourier_params=None):
        super(ResNetSegmentation, self).__init__()

        self.fourier_params = fourier_params
        if fourier_params is not None:
            if self.fourier_params['fourier_layer'] == 'linear':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'linear_log':
                self.fl = LinearFourier2d((n_channels, image_size[0], image_size[1]), log=True)
            if self.fourier_params['fourier_layer'] == 'general':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=False)
            if self.fourier_params['fourier_layer'] == 'general_log':
                self.fl = GeneralFourier2d((n_channels, image_size[0], image_size[1]), log=True)

        num_features = [64, 64, 128, 256, 512]
        self.init_conv = torch.nn.Sequential(torch.nn.Conv2d(n_channels, num_features[0],
                                                             kernel_size=7, stride=2, padding=3, bias=False),
                                             torch.nn.BatchNorm2d(num_features[0]),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.encoder = torch.nn.ModuleList()
        for i, num_layers in enumerate(blocks):
            if i == 0:
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=num_features[i], num_features=num_features[i + 1],
                                             downsampling=False))
            else:
                self.encoder.append(ResBlock(num_layers=num_layers,
                                             num_input_features=num_features[i], num_features=num_features[i + 1],
                                             downsampling=True))

        self.decoder = torch.nn.ModuleList()
        for i in range(1, len(blocks) + 1):
            self.decoder.append(DecoderLayer(num_features[-i], num_features[-i - 1]))

        self.out_conv = torch.nn.Sequential(torch.nn.ConvTranspose2d(num_features[0], num_features[0],
                                                                     kernel_size=2, stride=2),
                                            torch.nn.Conv2d(num_features[0], n_classes,
                                                            kernel_size=1, stride=1, padding=0))

        # initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.fourier_params is not None:
            x = self.fl(x)

        x = self.init_conv(x)

        for layer in self.encoder:
            x = layer(x)

        for layer in self.decoder:
            x = layer(x)

        return self.out_conv(x)