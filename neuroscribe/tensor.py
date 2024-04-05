from functools import partial


class Function:
    def __call__(self, *args):
        self.args = args
        return self.forward(*args)

    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, result_tensor): raise RuntimeError(f"backward not implemented for {type(self)}")


from neuroscribe.backend.cpu.cpu_backend import CPUBackend

available_backends = {'cpu': CPUBackend}

try:
    from neuroscribe.backend.cuda.cuda_backend import CUDABackend
    available_backends['cuda'] = CUDABackend
except ImportError:
    pass

try:
    from neuroscribe.backend.mps.mps_backend import MPSBackend
    available_backends['mps'] = MPSBackend
except ImportError:
    pass


class Tensor:

    _supported_backends = ['cpu', 'cuda', 'mps']
    _backends = available_backends

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
    def size(self):
        return self.data.size

    @property
    def strides(self):
        return self.data.strides

    @property
    def grad(self):
        if self.requires_grad:
            if self._grad is None:
                # TODO: Replace with zeros_like() when implemented
                self._grad = self._backend.zeros(self.shape, dtype=self.dtype)
            return self._grad
        else:
            return None

    @grad.setter
    def grad(self, value):
        self._grad = value

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError('Gradient computation has not been enabled for this tensor.')

        graph = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    if child.grad is None:
                        raise RuntimeError('Gradient computation has not been enabled for one or more tensor(s).')
                    build_graph(child)
                graph.append(tensor)
        build_graph(self)

        self.grad = self._backend.ones(self.shape, dtype=self.dtype)  # TODO: Replace with ones_like() when implemented
        for tensor in reversed(graph):
            tensor._grad_fn()

    def __repr__(self):
        return f"Tensor({self.data}, dtype={self.dtype})"

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
    def _get_backend(device):
        if device not in Tensor._supported_backends:
            raise ValueError(f"Unsupported device '{device}'. Supported devices are: {Tensor._supported_backends}.")
        if device not in Tensor._backends:
            raise ValueError(f"NeuroScribe is not installed with support for {device.upper()} devices.")
        return Tensor._backends[device]

    # ********** Creation Methods **********
    @staticmethod
    def create(data, dtype='float32', requires_grad=False, device='cpu'):
        backend = Tensor._get_backend(device)
        return Tensor(backend.create(data, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def zeros(shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Tensor._get_backend(device)
        return Tensor(backend.zeros(shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def ones(shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Tensor._get_backend(device)
        return Tensor(backend.ones(shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, dtype='float32', requires_grad=False, device='cpu'):
        backend = Tensor._get_backend(device)
        return Tensor(backend.randn(*shape, dtype=dtype), backend=backend, requires_grad=requires_grad)

    def to(self, device):
        if self.device == device:
            return self
        return Tensor.create(self.data, dtype=self.dtype, requires_grad=self.requires_grad, device=device)

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        return Tensor.create(self.data, dtype=dtype, requires_grad=self.requires_grad, device=self.device)

    # ********** Shape Manipulation Methods **********
    def reshape(self, new_shape):
        return Tensor(self._backend.reshape(self.data, new_shape), backend=self._backend, requires_grad=self.requires_grad)

    def transpose(self, axes=None):
        return Tensor(self._backend.transpose(self.data, axes), backend=self._backend, requires_grad=self.requires_grad)

    def split(self, indices_or_sections, axis=0):
        result = self._backend.split(self.data, indices_or_sections, axis)
        return [Tensor.create(t, dtype=self.dtype, requires_grad=self.requires_grad, device=self.device) for t in result]

    # ********** Tensor Operations **********
    def _exec_op(self, _op, *inputs):
        inputs = [Tensor.create(input, dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)
                  if not isinstance(input, Tensor) else input for input in inputs]
        inputs = [self] + inputs

        result_tensor = Tensor.create(_op(*inputs), dtype=inputs[0].dtype, requires_grad=False, device=self.device)

        if any(input.requires_grad for input in inputs):
            result_tensor.requires_grad = True
            result_tensor._grad_fn = partial(_op.backward, result_tensor)
            result_tensor._prev.extend(inputs)

        return result_tensor

    # ********** Loss Functions **********

    # ********** Activation Functions **********
    def relu(self): return self._exec_op(self._backend.relu())

    # ********** Binary Ops **********
    def add(self, other): return self._exec_op(self._backend.add(), other)
    def mul(self, other): return self._exec_op(self._backend.mul(), other)
    def matmul(self, other): return self._exec_op(self._backend.matmul(), other)

    def __add__(self, other): return self.add(other)
    def __mul__(self, other): return self.mul(other)
