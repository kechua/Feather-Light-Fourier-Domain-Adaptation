from typing import Dict
from torch import Tensor
import torch
from torch import nn


class Accumulator(nn.Module):

    def __init__(self, module: nn.Module, accumulate_every: int = 1, write_every: int = 100, decay=0.99):
        super().__init__()

        self.accumulate_every = accumulate_every
        self.write_every = write_every

        self.storage_model_params: Dict[str, Tensor] = dict(module.named_parameters())
        self.module = module
        self.decay = decay

        for k in self.storage_model_params.keys():
            self.storage_model_params[k] = self.storage_model_params[k].detach().cpu().clone()

    def accumulate(self, model: nn.Module, i: int):

        if i % self.accumulate_every == 0:

            params = dict(model.named_parameters())

            for k in params.keys():
                self.storage_model_params[k].data.mul_(self.decay)
                self.storage_model_params[k].data += (1 - self.decay) * params[k].data.cpu()

    def forward(self, *args, **kw):
        return self.module.forward(*args, **kw)

    def write_to(self, model: nn.Module):
        params = dict(model.named_parameters())
        for k in params.keys():
            params[k].data = self.storage_model_params[k].data.clone().cuda()

    def step(self, i: int):
        self.accumulate(self.module, i)
        if i % self.write_every == 0:
            self.write_to(self.module)


