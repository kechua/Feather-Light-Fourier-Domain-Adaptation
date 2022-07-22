import torch
from torch import Tensor
from torch import nn


class Grad(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 create_graph=True):

        super().__init__()
        self.net = net
        self.create_graph = create_graph

    def forward(self, x: Tensor):

        if not x.requires_grad:
            x.requires_grad_(True)

        dx: Tensor = self.net(x)

        res = torch.autograd.grad(
            outputs=dx,
            inputs=x,
            grad_outputs=torch.ones(dx.shape, device=dx.device),
            create_graph=self.create_graph,
            only_inputs=True)

        return res[0]
