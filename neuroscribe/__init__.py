from neuroscribe.tensor import Tensor

__all__ = ['tensor', 'zeros', 'ones', 'randn']


def tensor(data, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.create(data, dtype, requires_grad, device)


def zeros(shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.zeros(shape, dtype, requires_grad, device)


def ones(shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.ones(shape, dtype, requires_grad, device)


def randn(*shape, dtype='float32', requires_grad=False, device='cpu'):
    return Tensor.randn(*shape, dtype, requires_grad, device)
