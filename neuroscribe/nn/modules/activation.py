import neuroscribe.nn.functional as F
from neuroscribe.core._tensor_lib._tensor import Tensor

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


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.tanh(input)


class HardTanh(Module):
    def __init__(self, min=-1, max=1):
        super().__init__()
        self.min = min
        self.max = max
        assert self.max > self.min

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.hardtanh(input, self.min, self.max)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.sigmoid(input)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.softmax(input)


class Softmin(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)
        return F.softmin(input)
