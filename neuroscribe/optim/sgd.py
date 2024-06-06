from neuroscribe.core._tensor_lib._tensor import Tensor

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0):
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        self.momentum = momentum
        self.velocities = [Tensor.zeros_like(param) for param in self.params] if self.momentum != 0 else None

    def step(self):
        if self.momentum == 0:
            for param in self.params:
                param.data = param.data - self.lr * param.grad.data
        else:
            for i, param in enumerate(self.params):
                self.velocities[i].data = self.momentum * self.velocities[i].data + param.grad.data
                param.data = param.data - self.lr * self.velocities[i].data
