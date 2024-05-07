from neuroscribe.tensor import Tensor

from .module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor.create(x, requires_grad=self._training, device=self._device)
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor.create(x, requires_grad=self._training, device=self._device)
        return x.leaky_relu(self.negative_slope)
