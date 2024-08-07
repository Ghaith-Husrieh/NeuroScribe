import math
import types
from functools import partial

from neuroscribe.autodiff.grad import _align_gradient_shape, _build_graph
from neuroscribe.core._tensor_lib.backend.dispatcher import Dispatcher


class Tensor:

    def __init__(self, data, backend, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self._backend = backend
        self._grad = None
        self._grad_fn = lambda: None
        self._prev = list()

    @property
    def device(self):
        return self._backend.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def strides(self):
        return self.data.strides

    @property
    def itemsize(self):
        return self.data.itemsize

    @property
    def is_cpu(self):
        return self.device == 'cpu'

    @property
    def is_cuda(self):
        return self.device == 'cuda'

    @property
    def grad(self):
        if self.requires_grad:
            if self._grad is None:
                self._grad = Tensor.zeros(self.shape, dtype=self.dtype, requires_grad=False, device=self.device)
            return self._grad
        else:
            return None

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def T(self):
        return self.transpose()

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.itemsize

    def nbytes(self):
        return self.numel() * self.element_size()

    def normal_(self, mean, standard_deviation):
        self.data = self._backend.normal_(mean, standard_deviation, self.shape)

    def uniform_(self, lower_bound, upper_bound):
        self.data = self._backend.uniform_(lower_bound, upper_bound, self.shape)

    def zero_(self):
        self.data = self._backend.zeros(self.shape, dtype=self.dtype)

    def ones_(self):
        self.data = self._backend.ones(self.shape, dtype=self.dtype)

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError('Gradient computation has not been enabled for this tensor.')

        graph = []
        visited = set()
        _build_graph(self, graph, visited)

        self.grad = Tensor.ones(self.shape, dtype=self.dtype, requires_grad=False, device=self.device)
        for tensor in reversed(graph):
            tensor._grad_fn()
        for tensor in reversed(graph):
            _align_gradient_shape(tensor)

    def __repr__(self):
        return f"Tensor({self.data}, dtype={self.dtype})"

    def __getitem__(self, index):
        if isinstance(index, slice) or isinstance(index, int):
            return Tensor.create(self.data[index], dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)
        else:
            raise TypeError("Index must be an integer or slice object")

    def __setitem__(self, index, value):
        if isinstance(index, slice) or isinstance(index, int):
            self.data[index] = value
        else:
            raise TypeError("Index must be an integer or slice object")

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(state['_grad_fn'], types.LambdaType):
            state['_grad_fn'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._grad_fn is None:
            self._grad_fn = lambda: None

    def __len__(self):
        return self.shape[0] if self.ndim > 0 else 1

    def __hash__(self):
        return id(self)

    def detach(self):
        if self.requires_grad == False:
            return self
        return Tensor(self.data, backend=self._backend, requires_grad=False)

    def item(self):
        if self.numel() != 1:
            raise ValueError(f"Tensor with {self.numel()} elements cannot be converted to Python scalar")
        return self.data.item()

    def argmax(self, dim=None):
        return Tensor(self._backend.argmax(self.data, dim), backend=self._backend, requires_grad=False)

    def argmin(self, dim=None):
        return Tensor(self._backend.argmin(self.data, dim), backend=self._backend, requires_grad=False)

    def is_contiguous(self):
        return self._backend.is_contiguous(self.data)

    def make_contiguous(self):
        if self.is_contiguous():
            return self
        return Tensor(self._backend.make_contiguous(self.data), backend=self._backend, requires_grad=self.requires_grad)

    def deep_copy(self):
        return Tensor(self._backend.deep_copy(self.data), backend=self._backend, requires_grad=self.requires_grad)

    def shallow_copy(self):
        return Tensor(self._backend.shallow_copy(self.data), backend=self._backend, requires_grad=self.requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, dtype='uint16', device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.arange(start, stop, step, dtype), backend=backend, requires_grad=False)

    @staticmethod
    def shuffle_(tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("The shuffle method expects a Tensor object.")
        tensor._backend.shuffle_(tensor.data)

    @staticmethod
    def pad(input, pad, mode, *, constant_values):
        return Tensor(input._backend.pad(input.data, pad, mode, constant_values), backend=input._backend, requires_grad=input.requires_grad)

    # ********** Helper Methods **********
    @staticmethod
    def _prepare_like_attributes(input, dtype, requires_grad, device):
        backend = Dispatcher.get_backend(device) if device is not None else input._backend
        dtype = dtype if dtype is not None else input.dtype
        requires_grad = requires_grad if requires_grad is not None else input.requires_grad
        return backend, dtype, requires_grad

    # ********** Creation Methods **********
    @staticmethod
    def create(data, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.create(data, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def zeros(shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.zeros(shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def zeros_like(input, dtype=None, requires_grad=None, device=None):
        backend, dtype, requires_grad = Tensor._prepare_like_attributes(input, dtype, requires_grad, device)
        return Tensor(backend.zeros_like(input.data, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def ones(shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.ones(shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def ones_like(input, dtype=None, requires_grad=None, device=None):
        backend, dtype, requires_grad = Tensor._prepare_like_attributes(input, dtype, requires_grad, device)
        return Tensor(backend.ones_like(input.data, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.randn(*shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def rand(*shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.rand(*shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def randint(low, high=None, shape=None, *, dtype='int32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.randint(low, high, shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def empty(shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Dispatcher.get_backend(device)
        return Tensor(backend.empty(shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def empty_like(input, dtype=None, requires_grad=None, device=None):
        backend, dtype, requires_grad = Tensor._prepare_like_attributes(input, dtype, requires_grad, device)
        return Tensor(backend.empty_like(input.data, dtype=dtype), backend=backend, requires_grad=requires_grad)

    def to(self, device):
        if self.device == device:
            return self
        return Tensor.create(self.data, dtype=self.dtype, requires_grad=self.requires_grad, device=device)

    def asnumpy(self):
        return self.to('cpu').data

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        return Tensor.create(self.data, dtype=dtype, requires_grad=self.requires_grad, device=self.device)

    # ********** Shape Manipulation Methods **********
    def flatten(self):
        return Tensor(self._backend.flatten(self.data), backend=self._backend, requires_grad=self.requires_grad)

    def flip(self, dims=None):
        return Tensor(self._backend.flip(self.data, dims), backend=self._backend, requires_grad=self.requires_grad)

    def squeeze(self, dim=None):
        return Tensor(self._backend.squeeze(self.data, dim), backend=self._backend, requires_grad=self.requires_grad)

    def reshape(self, shape):
        return Tensor(self._backend.reshape(self.data, shape), backend=self._backend, requires_grad=self.requires_grad)

    def transpose(self, dims=None):
        return Tensor(self._backend.transpose(self.data, dims), backend=self._backend, requires_grad=self.requires_grad)

    def split(self, indices_or_sections, dim=0):
        result = self._backend.split(self.data, indices_or_sections, dim)
        return [Tensor.create(t, dtype=self.dtype, requires_grad=self.requires_grad, device=self.device) for t in result]

    # ********** Tensor Operations **********
    def _exec_op(self, _op, *inputs, reverse=False):
        inputs = [Tensor.create(input, dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)
                  if not isinstance(input, Tensor) else input for input in inputs]
        inputs = list(reversed([self] + inputs)) if reverse else [self] + inputs

        result_tensor = Tensor.create(_op(*inputs), dtype=self.dtype, requires_grad=False, device=self.device)

        if any(input.requires_grad for input in inputs):
            result_tensor.requires_grad = True
            result_tensor._grad_fn = partial(_op.backward, result_tensor)
            result_tensor._prev.extend(inputs)

        return result_tensor

    # ********** Reduction Ops **********
    def mean(self): return self._exec_op(self._backend.mean())
    def sum(self): return self._exec_op(self._backend.sum())
    def max(self): return self._exec_op(self._backend.max())
    def min(self): return self._exec_op(self._backend.min())

    # ********** Unary Ops **********
    def relu(self): return self._exec_op(self._backend.relu())
    def relu6(self): return self.relu() - (self - 6).relu()
    def leaky_relu(self, negative_slope=0.01): return self._exec_op(self._backend.leaky_relu(negative_slope))
    def elu(self, alpha=1.0): return self.relu() - alpha * (1 - self.exp()).relu()
    def swish(self): return self * self.sigmoid()
    def silu(self): return self.swish()
    def sigmoid(self): return self._exec_op(self._backend.sigmoid())
    def softmax(self): return (self - self.max()).exp() / (self - self.max()).exp().sum()
    def log_softmax(self): return (self - self.max()) - (self - self.max()).exp().sum().log()
    def softmin(self): return ((self - self.max()).neg()).exp() / ((self - self.max()).neg()).exp().sum()
    def square(self): return self._exec_op(self._backend.square())
    def neg(self): return self._exec_op(self._backend.neg())
    def clip(self, min, max): return self._exec_op(self._backend.clip(min, max))
    def sign(self): return self._exec_op(self._backend.sign())
    def abs(self): return self * self.sign()
    def reciprocal(self): return self._exec_op(self._backend.reciprocal())
    def sqrt(self): return self._exec_op(self._backend.sqrt())
    def rsqrt(self): return self.reciprocal().sqrt()
    def log(self): return self._exec_op(self._backend.log())
    def log10(self): return self.log() / math.log(10)
    def log2(self): return self.log() / math.log(2)
    def log1p(self): return (self + 1).log()
    def exp(self): return self._exec_op(self._backend.exp())
    def exp2(self): return (self * math.log(2)).exp()
    def sin(self): return self._exec_op(self._backend.sin())
    def cos(self): return self._exec_op(self._backend.cos())
    def tan(self): return self.sin() / self.cos()
    # NOTE: 2.0 * ((2.0 * self).sigmoid()) - 1.0 could be used instead, but it may result in lower precision.
    def tanh(self): return self._exec_op(self._backend.tanh())
    def sinh(self): return (self.exp() - self.neg().exp()) / 2
    def cosh(self): return (self.exp() + self.neg().exp()) / 2
    def atanh(self): return ((1 + self) / (1 - self)).log() / 2
    def asinh(self): return (self + (self.square() + 1).sqrt()).log()
    def acosh(self): return (self + (self.square() - 1).sqrt()).log()
    def hardtanh(self, min=-1, max=1): return self.clip(min, max)
    def gelu(self): return 0.5 * self * (1 + (math.sqrt(2 / math.pi) * (self + 0.044715 * self ** 3)).tanh())
    def softplus(self, beta=1.0): return (1 / beta) * (1 + (self * beta).exp()).log()
    def mish(self): return self * self.softplus().tanh()
    def softsign(self): return self / (1 + self.abs())

    # ********** Binary Ops **********
    def pow(self, exponent, reverse=False): return self._exec_op(self._backend.pow(), exponent, reverse=reverse)
    def add(self, other, reverse=False): return self._exec_op(self._backend.add(), other, reverse=reverse)
    def sub(self, other, reverse=False): return self._exec_op(self._backend.sub(), other, reverse=reverse)
    def mul(self, other, reverse=False): return self._exec_op(self._backend.mul(), other, reverse=reverse)
    def div(self, other, reverse=False): return self._exec_op(self._backend.div(), other, reverse=reverse)
    def matmul(self, other, reverse=False): return self._exec_op(self._backend.matmul(), other, reverse=reverse)

    def __pow__(self, other): return self.pow(other)
    def __add__(self, other): return self.add(other)
    def __sub__(self, other): return self.sub(other)
    def __mul__(self, other): return self.mul(other)
    def __truediv__(self, other): return self.div(other)
    def __matmul__(self, other): return self.matmul(other)

    def __rpow__(self, other): return self.pow(other, reverse=True)
    def __radd__(self, other): return self.add(other, reverse=True)
    def __rsub__(self, other): return self.sub(other, reverse=True)
    def __rmul__(self, other): return self.mul(other, reverse=True)
    def __rtruediv__(self, other): return self.div(other, reverse=True)
    def __rmatmul__(self, other): return self.matmul(other, reverse=True)
