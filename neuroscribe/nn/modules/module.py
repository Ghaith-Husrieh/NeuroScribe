import pickle
from collections import OrderedDict

from neuroscribe.core._tensor_lib._tensor import Tensor


def save(model, filename):
    if not isinstance(model, Module):
        raise TypeError(f"Cannot save object of type '{type(model).__name__}', expected 'Module'")
    filename = filename + '.nsm' if not filename.endswith('.nsm') else filename
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Module saved to '{filename}'")


def load(filename):
    filename = filename + '.nsm' if not filename.endswith('.nsm') else filename
    with open(filename, "rb") as f:
        module = pickle.load(f)
    return module


class Module:

    def __init__(self):

        super().__setattr__('_training', False)
        super().__setattr__('_device', 'cpu')
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            if '_parameters' not in self.__dict__:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            self._parameters[name] = value
        elif isinstance(value, Module):
            if '_modules' not in self.__dict__:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, *input):
        return self.forward(*input)

    def forward(self, *input): raise NotImplementedError

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def to(self, device):
        self._device = device
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device)
        for module in self._modules.values():
            module.to(device)

    def _mode(self, *, training):
        self._training = training
        for param in self._parameters.values():
            param.requires_grad = training
        for module in self._modules.values():
            module._mode(training=training)

    def train(self):
        self._mode(training=True)

    def eval(self):
        self._mode(training=False)
