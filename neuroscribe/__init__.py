from neuroscribe.nn.modules.module import load, save
from neuroscribe.tensor import Tensor

__all__ = ['tensor', 'zeros', 'ones', 'randn', 'empty', 'save', 'load']


def tensor(data, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.create(data, dtype=dtype, requires_grad=requires_grad, device=device)


def zeros(shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.zeros(shape, dtype=dtype, requires_grad=requires_grad, device=device)


def ones(shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.ones(shape, dtype=dtype, requires_grad=requires_grad, device=device)


def randn(*shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.randn(*shape, dtype=dtype, requires_grad=requires_grad, device=device)


def empty(*shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.empty(*shape, dtype=dtype, requires_grad=requires_grad, device=device)
