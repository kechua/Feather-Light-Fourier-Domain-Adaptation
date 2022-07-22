from typing import List, Dict, Callable, Tuple, Optional, TypeVar, Generic, Any, Union
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init

from gan.generator import Generator
from gan.loss.base import GANLoss
from gan.loss.hinge import HingeLoss
from gan.loss.loss_base import Loss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def gan_weights_init(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GANModel:

    def __init__(self,
                 generator: Generator,
                 loss: GANLoss,
                 lr: Tuple[float, float] = (0.0002, 0.0002),
                 do_init_ws=False):
        self.generator = generator
        self.loss = loss
        if do_init_ws:
            self.generator.apply(gan_weights_init)
            self.loss.discriminator.apply(gan_weights_init)
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr[0], lr[1])

    def generator_loss(self, real: List[Tensor], fake: List[Tensor]) -> Loss:
        requires_grad(self.loss.discriminator, False)
        return self.loss.generator_loss(real, fake)

    def discriminator_train(self, real: List[Tensor], fake: List[Tensor]):
        requires_grad(self.loss.discriminator, True)
        self.loss.discriminator_loss_with_penalty(real, fake).maximize_step(
            self.optimizer.opt_max
        )

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.loss.parameters())

    def state_dict(self):
        return {'g': self.generator.state_dict(), 'd': self.loss.discriminator.state_dict()}


class ConditionalGANModel(GANModel):

    def generator_loss(self, real: List[Tensor], fake: List[Tensor], condition: List[Tensor]) -> Loss:
        return super(ConditionalGANModel, self).generator_loss(real + condition, fake + condition)

    def discriminator_train(self, real: List[Tensor], fake: List[Tensor], condition: List[Tensor]):
        super(ConditionalGANModel, self).discriminator_train(real + condition, fake + condition)


name_to_gan_loss = {
    "hinge": lambda net_d: HingeLoss(net_d),
    "wasserstein": lambda net_d: WassersteinLoss(net_d, penalty_weight=10),
    "vanilla": lambda net_d: DCGANLoss(net_d)
}
