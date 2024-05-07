from neuroscribe.tensor import Tensor

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        self.beta1, self.beta2 = betas
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        self.eps = eps
        self.m = [Tensor.zeros_like(param) for param in self.params]
        self.v = [Tensor.zeros_like(param) for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * param.grad.data
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (param.grad.data ** 2)
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
