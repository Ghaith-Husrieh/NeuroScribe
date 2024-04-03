from neuroscribe.tensor import Tensor


class Module:
    def __init__(self):
        self._training = False
        self._device = 'cpu'
        self._parameters = {}
        self._submodules = {}

    def __call__(self, *input):
        return self.forward(*input)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Module):
            self._submodules[name] = self.__dict__[name]
        elif isinstance(value, Tensor):
            self._parameters[name] = self.__dict__[name]

    def forward(self, *input): raise NotImplementedError

    def parameters(self):
        for _, param in self._parameters.items():
            yield param
        for _, module in self._submodules.items():
            yield from module.parameters()

    def to(self, device):
        self._device = device
        for _, module in self._submodules.items():
            module.to(device)
        for name, param in self._parameters.items():
            param = param.to(device)
            self._parameters[name] = param
            self.__dict__[name] = param

    def _mode(self, training):
        self._training = training
        for _, module in self._submodules.items():
            module._mode(training)
        for _, param in self._parameters.items():
            param.requires_grad = training

    def train(self):
        self._mode(training=True)

    def eval(self):
        self._mode(training=False)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features)

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor.create(x, requires_grad=self._training, device=self._device)
        return x.matmul(self.weight.transpose()) + self.bias
