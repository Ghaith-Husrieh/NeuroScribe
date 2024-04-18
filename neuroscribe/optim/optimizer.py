from neuroscribe.tensor import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = [params] if isinstance(params, Tensor) else list(params)
        if not all(isinstance(param, Tensor) for param in self.params):
            raise TypeError("The params argument must be a Tensor or an iterable of Tensors")
        if len(self.params) == 0:
            raise ValueError("Optimizer received an empty parameter list")

    def zero_grad(self):
        for param in self.params:
            param.grad.zero_()

    def step(self, *args): raise NotImplementedError
