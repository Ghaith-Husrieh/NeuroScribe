from neuroscribe.tensor import Tensor

from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features) if bias else None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor.create(x, requires_grad=self._training, device=self._device)

        if self.bias is not None:
            return x @ self.weight.T + self.bias
        return x @ self.weight.T
