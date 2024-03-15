import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape, param.grad.dtype)

    def step(self, *args): raise NotImplementedError


# TODO: Add support for momentum
class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
