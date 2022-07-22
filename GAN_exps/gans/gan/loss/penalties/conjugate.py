from itertools import chain
from typing import List, Tuple

import torch

from gan.discriminator import Discriminator
from gan.generator import Generator
from torch import Tensor, nn

from gan.loss.penalties.penalty import default_mix
from gan.loss.loss_base import Loss


class ConjugateGANLoss:

    def __init__(
            self,
            Dx: Discriminator,
            Tyx: Generator,
            pen_weight: float = 100):

        self.Dx = Dx
        self.Tyx = Tyx
        self.pen_weight = pen_weight
        self.mix = default_mix
        print("pen_w: ", pen_weight)

    def parameters(self):
        return chain(self.Dx.parameters(), self.Tyx.parameters())

    def gradient_point(self, x: List[Tensor], y: List[Tensor]) -> List[Tensor]:
        x0: List[Tensor] = [
            self.mix(xi, yi).detach().requires_grad_(True) for xi, yi in zip(x, y)
        ]
        return x0

    def d_grad(self, x0: Tensor, create_graph=True) -> Tensor:

        if not x0.requires_grad:
            x0.requires_grad_(True)

        dx0: Tensor = self.Dx.forward(x0)

        res, = torch.autograd.grad(outputs=dx0,
                                   inputs=x0,
                                   grad_outputs=torch.ones(dx0.shape, device=dx0.device),
                                   create_graph=create_graph,
                                   only_inputs=True)

        return res

    def product(self, x: Tensor, y: Tensor):
        n = x[0].shape[0]
        return (x * y).view(n, -1).sum(1).mean()

    def transport_loss(self, y: Tensor):
        L1 = nn.L1Loss()
        y = y.detach()
        ty: Tensor = self.Tyx(y)
        y_pred = self.d_grad(ty)
        return Loss(self.product(ty, y) - self.Dx(ty).mean() - L1(y_pred, y) * self.pen_weight)

    def discriminator_loss(self, x: Tensor, y: Tensor):
        L1 = nn.L1Loss()
        L2 = nn.MSELoss()

        x = x.detach()
        y = y.detach()

        tyx: Tensor = self.Tyx(y).detach()
        loss: Tensor = self.Dx(x).mean() + self.product(tyx, y) - self.Dx(tyx).mean()

        x_pred = self.Tyx(self.d_grad(x))
        y_pred = self.d_grad(self.Tyx(y))

        pen = L2(y_pred, y) + L2(x_pred, x)
        # grad_pen = self.d_grad(x).pow(2).view(x.shape[0], -1).sum(1).mean(0)

        return Loss(loss + pen * self.pen_weight)

    def generator_loss(self, x: Tensor):

        L1 = nn.L1Loss()
        tx: Tensor = self.d_grad(x, False).detach()

        return Loss(L1(x, tx))


class ConjugateGANLoss2:

    def __init__(self,
                 Dx: Discriminator,
                 Dy: Discriminator,
                 pen_weight: float = 100):

        self.Dx = Dx
        self.Dy = Dy
        self.pen_weight = pen_weight
        print("pen_w: ", pen_weight)

    def parameters(self):
        return chain(self.Dx.parameters(), self.Dy.parameters())

    def Txy(self, x: Tensor, create_graph=True) -> Tensor:

        if not x.requires_grad:
            x.requires_grad = True

        dx: Tensor = self.Dx.forward(x)

        res, = torch.autograd.grad(outputs=dx,
                                   inputs=x,
                                   grad_outputs=torch.ones(dx.shape, device=dx.device),
                                   create_graph=create_graph,
                                   only_inputs=True)

        return res

    def Tyx(self, y: Tensor, create_graph=True) -> Tensor:

        if not y.requires_grad:
            y.requires_grad = True

        dy: Tensor = self.Dy.forward(y)

        res, = torch.autograd.grad(outputs=dy,
                                   inputs=y,
                                   grad_outputs=torch.ones(dy.shape, device=y.device),
                                   create_graph=create_graph,
                                   only_inputs=True)

        return res

    def discriminator_loss(self, x: Tensor, y: Tensor):
        L1 = nn.L1Loss()
        L2 = nn.MSELoss()

        x = x.detach()
        y = y.detach()

        loss: Tensor = self.Dx(x).mean() + self.Dy(y).mean()

        x_pred = self.Tyx(self.Txy(x))
        y_pred = self.Txy(self.Tyx(y))

        pen = L2(y_pred, y) + L2(x_pred, x)
        # grad_pen = self.d_grad(x).pow(2).view(x.shape[0], -1).sum(1).mean(0)
        # grad_pen = self.Txy(x).pow(2).view(x.shape[0], -1).sum(1).mean() + self.Tyx(y).pow(2).view(x.shape[0], -1).sum(1).mean()
        # print("brule_loss", brule_loss.item())
        # print("pen", pen.item())

        return Loss(loss + pen * self.pen_weight)

    def generator_loss(self, x: Tensor):

        L1 = nn.L1Loss()
        y: Tensor = self.Txy(x, False).detach()

        return Loss(L1(x, y))


