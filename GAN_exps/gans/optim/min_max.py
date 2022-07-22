from typing import Iterator
from torch import optim, Tensor

from gan.loss.loss_base import Loss


class MinMaxLoss:
    def __init__(self, min_loss: Loss, max_loss: Loss):
        self.min_loss = min_loss
        self.max_loss = max_loss

    def add_min_loss(self, loss: Loss):
        self.min_loss = self.min_loss + loss
        return self

    def add_max_loss(self, loss: Loss):
        self.max_loss += loss
        return self

    def minimize_step(self, optimizer):
        optimizer.opt_max.zero_grad()
        self.max_loss.maximize()
        optimizer.opt_max.step()

        optimizer.opt_min.zero_grad()
        self.min_loss.minimize()
        optimizer.opt_min.step()

    def mul(self, w: float):
        return MinMaxLoss(self.min_loss * w, self.max_loss * w)


class MinMaxParameters:
    def __init__(self,
                 min_parameters: Iterator[Tensor],
                 max_parameters: Iterator[Tensor]):
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters


class MinMaxOptimizer:

    def __init__(self,
                 parameters: MinMaxParameters,
                 min_learning_rate: float,
                 max_learning_rate: float,
                 min_betas=(0.5, 0.999),
                 max_betas=(0.5, 0.999)
                 ):

        self.opt_max = optim.Adam(parameters.max_parameters,
                                  lr=max_learning_rate,
                                  betas=max_betas)
        self.opt_min = optim.Adam(parameters.min_parameters,
                                  lr=min_learning_rate,
                                  betas=min_betas)

    def train_step(self, loss: MinMaxLoss):

        self.opt_max.zero_grad()
        loss.max_loss.maximize()
        self.opt_max.step()

        self.opt_min.zero_grad()
        loss.min_loss.minimize()
        self.opt_min.step()

    def update_lr(self, scale: float):
        for i in self.opt_max.param_groups:
            i['lr'] *= scale

        for i in self.opt_min.param_groups:
            i['lr'] *= scale

    def add_param_group(self, params, lr):
        if params[0] is not None:
            self.opt_min.add_param_group(
                {
                    'params': params[0],
                    'lr': lr[0]
                }
            )

        if params[1] is not None:
            self.opt_max.add_param_group(
                {
                    'params': params[1],
                    'lr': lr[1]
                }
            )

        return self



