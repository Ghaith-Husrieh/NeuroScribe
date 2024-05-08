import neuroscribe.nn.functional as F
from neuroscribe.tensor import Tensor

from .module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.relu(input)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.leaky_relu(input, self.negative_slope)
