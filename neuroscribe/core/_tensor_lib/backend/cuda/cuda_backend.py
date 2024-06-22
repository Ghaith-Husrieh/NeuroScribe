import cupy as cp

import neuroscribe.core._tensor_lib.backend.cuda.mlops as mlops


class CUDABackend:

    device = 'cuda'

    @staticmethod
    def is_contiguous(data):
        return data.flags['C_CONTIGUOUS']

    @staticmethod
    def make_contiguous(data):
        return cp.ascontiguousarray(data)

    @staticmethod
    def deep_copy(data):
        return data.copy(order='C')

    @staticmethod
    def shallow_copy(data):
        return data.view()

    @staticmethod
    def normal_(mean, standard_deviation, shape):
        return cp.random.normal(mean, standard_deviation, shape)

    @staticmethod
    def uniform_(lower_bound, upper_bound, shape):
        return cp.random.uniform(lower_bound, upper_bound, shape)

    @staticmethod
    def arange(start, stop, step, dtype):
        return cp.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def shuffle_(data):
        cp.random.shuffle(data)

    @staticmethod
    def einsum(subscripts, *inputs):
        return cp.einsum(subscripts, *inputs)

    @staticmethod
    def pad(input, pad, mode, constant_values):
        return cp.pad(input, pad, mode, constant_values=constant_values)

    # ********** Creation Methods **********
    # TODO: should optimize when data is already a cupy.ndarray
    @staticmethod
    def create(data, dtype):
        return cp.array(data, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return cp.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(input, dtype):
        return cp.zeros_like(input, dtype=dtype)

    @staticmethod
    def ones(shape, dtype):
        return cp.ones(shape, dtype=dtype)

    @staticmethod
    def ones_like(input, dtype):
        return cp.ones_like(input, dtype=dtype)

    @staticmethod
    def randn(*shape, dtype):
        return cp.random.randn(*shape).astype(dtype)

    @staticmethod
    def empty(shape, dtype):
        return cp.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(input, dtype):
        return cp.empty_like(input, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @staticmethod
    def reshape(data, new_shape):
        return data.reshape(new_shape)

    @staticmethod
    def transpose(data, axes):
        return data.transpose(axes)

    @staticmethod
    def split(data, indices_or_sections, axis):
        return cp.split(data, indices_or_sections, axis)

    # ********** Activation Functions **********
    @staticmethod
    def relu():
        return mlops.ReLU()

    @staticmethod
    def leaky_relu(negative_slope):
        return mlops.LeakyReLU(negative_slope)

    @staticmethod
    def tanh():
        return mlops.Tanh()

    @staticmethod
    def sigmoid():
        return mlops.Sigmoid()

    # ********** Unary ops **********
    @staticmethod
    def mean():
        return mlops.Mean()

    @staticmethod
    def sum():
        return mlops.Sum()

    @staticmethod
    def square():
        return mlops.Square()

    @staticmethod
    def sign():
        return mlops.Sign()

    @staticmethod
    def sqrt():
        return mlops.Sqrt()

    @staticmethod
    def log():
        return mlops.Log()

    @staticmethod
    def exp():
        return mlops.Exp()

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
