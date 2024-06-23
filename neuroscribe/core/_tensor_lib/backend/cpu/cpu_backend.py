import numpy as np

import neuroscribe.core._tensor_lib.backend.cpu.mlops as mlops

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CPUBackend:

    device = 'cpu'

    @staticmethod
    def is_contiguous(data):
        return data.flags['C_CONTIGUOUS']

    @staticmethod
    def make_contiguous(data):
        return np.ascontiguousarray(data)

    @staticmethod
    def deep_copy(data):
        return data.copy(order='C')

    @staticmethod
    def shallow_copy(data):
        return data.view()

    @staticmethod
    def normal_(mean, standard_deviation, shape):
        return np.random.normal(mean, standard_deviation, shape)

    @staticmethod
    def uniform_(lower_bound, upper_bound, shape):
        return np.random.uniform(lower_bound, upper_bound, shape)

    @staticmethod
    def arange(start, stop, step, dtype):
        return np.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def shuffle_(data):
        np.random.shuffle(data)

    @staticmethod
    def einsum(subscripts, *inputs):
        return np.einsum(subscripts, *inputs)

    @staticmethod
    def pad(input, pad, mode, constant_values):
        return np.pad(input, pad, mode, constant_values=constant_values)

    # ********** Creation Methods **********
    # TODO: should optimize when data is already a numpy.ndarray
    @staticmethod
    def create(data, dtype):
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.array(data, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(input, dtype):
        return np.zeros_like(input, dtype=dtype)

    @staticmethod
    def ones(shape, dtype):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def ones_like(input, dtype):
        return np.ones_like(input, dtype=dtype)

    @staticmethod
    def randn(*shape, dtype):
        return np.random.randn(*shape).astype(dtype)

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(input, dtype):
        return np.empty_like(input, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @staticmethod
    def reshape(data, new_shape):
        return data.reshape(new_shape)

    @staticmethod
    def transpose(data, axes):
        return data.transpose(axes)

    @staticmethod
    def split(data, indices_or_sections, axis):
        return np.split(data, indices_or_sections, axis)

    # ********** Unary ops **********
    @staticmethod
    def mean():
        return mlops.Mean()

    @staticmethod
    def sum():
        return mlops.Sum()

    @staticmethod
    def relu():
        return mlops.ReLU()

    @staticmethod
    def leaky_relu(negative_slope):
        return mlops.LeakyReLU(negative_slope)

    @staticmethod
    def sigmoid():
        return mlops.Sigmoid()

    @staticmethod
    def square():
        return mlops.Square()

    @staticmethod
    def neg():
        return mlops.Neg()

    @staticmethod
    def clip(min, max):
        return mlops.Clip(min, max)

    @staticmethod
    def sign():
        return mlops.Sign()

    @staticmethod
    def reciprocal():
        return mlops.Reciprocal()

    @staticmethod
    def sqrt():
        return mlops.Sqrt()

    @staticmethod
    def log():
        return mlops.Log()

    @staticmethod
    def exp():
        return mlops.Exp()

    @staticmethod
    def sin():
        return mlops.Sin()

    @staticmethod
    def cos():
        return mlops.Cos()

    @staticmethod
    def tanh():
        return mlops.Tanh()

    # ********** Binary Ops **********
    @staticmethod
    def pow():
        return mlops.Pow()

    @staticmethod
    def add():
        return mlops.Add()

    @staticmethod
    def sub():
        return mlops.Sub()

    @staticmethod
    def mul():
        return mlops.Mul()

    @staticmethod
    def div():
        return mlops.Div()

    @staticmethod
    def matmul():
        return mlops.MatMul()
