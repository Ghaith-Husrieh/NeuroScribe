import numpy as np

import neuroscribe.core._tensor_lib.backend.cpu.mlops as mlops

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CPUBackend:

    device = 'cpu'

    @classmethod
    def is_contiguous(cls, data):
        return data.flags['C_CONTIGUOUS']

    @classmethod
    def make_contiguous(cls, data):
        return np.ascontiguousarray(data)

    @classmethod
    def deep_copy(cls, data):
        return data.copy(order='C')

    @classmethod
    def shallow_copy(cls, data):
        return data.view()

    @classmethod
    def normal_(cls, mean, standard_deviation, shape):
        return np.random.normal(mean, standard_deviation, shape)

    @classmethod
    def uniform_(cls, lower_bound, upper_bound, shape):
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
    @classmethod
    def create(cls, data, dtype):
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.array(data, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @classmethod
    def zeros_like(cls, input, dtype):
        return np.zeros_like(input, dtype=dtype)

    @classmethod
    def ones(cls, shape, dtype):
        return np.ones(shape, dtype=dtype)

    @classmethod
    def ones_like(cls, input, dtype):
        return np.ones_like(input, dtype=dtype)

    @classmethod
    def randn(cls, *shape, dtype):
        return np.random.randn(*shape).astype(dtype)

    @classmethod
    def empty(cls, *shape, dtype):
        return np.empty(*shape, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @classmethod
    def reshape(cls, data, new_shape):
        return data.reshape(new_shape)

    @classmethod
    def transpose(cls, data, axes):
        return data.transpose(axes)

    @classmethod
    def split(cls, data, indices_or_sections, axis):
        return np.split(data, indices_or_sections, axis)

    # ********** Activation Functions **********
    @staticmethod
    def relu():
        return mlops.ReLU()

    @staticmethod
    def leaky_relu(negative_slope):
        return mlops.LeakyReLU(negative_slope)

    # ********** Unary ops **********
    @staticmethod
    def mean():
        return mlops.Mean()

    @staticmethod
    def square():
        return mlops.Square()

    # ********** Binary Ops **********
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
    def matmul():
        return mlops.MatMul()
