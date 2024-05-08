import math

import neuroscribe.nn.functional as F
from neuroscribe.tensor import Tensor

from .. import init
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.empty((in_features, out_features))
        self.bias = Tensor.empty(out_features) if bias else None
        self.init_parameters()

    def init_parameters(self):
        init.kaiming_uniform(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = init.calculate_correct_fan(self.weight, mode='fan_in')
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)

        return F.linear(input, self.weight, self.bias)
