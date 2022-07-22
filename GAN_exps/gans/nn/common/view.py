from torch import nn, Tensor


class View(nn.Module):

    def __init__(self, *dims: int):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, inputs: Tensor):
        return inputs.contiguous().view(inputs.shape[0], *self.dims)

    def extra_repr(self):
        return ', '.join(str(dim) for dim in self.dims)
