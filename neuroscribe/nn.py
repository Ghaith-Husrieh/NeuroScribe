from neuroscribe.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True  # Note: used when implementing batchnorm and dropout layers

    def __call__(self, *input):
        return self.forward(*input)

    def parameters(self):
        return []

    def forward(self, *input): raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features)

    def parameters(self):
        return [self.weight] + [self.bias]

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.matmul(self.weight.transpose()) + self.bias
