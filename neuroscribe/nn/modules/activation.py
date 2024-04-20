from neuroscribe.tensor import Tensor

from .module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor.create(x, requires_grad=self._training, device=self._device)
        return x.relu()
