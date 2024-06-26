import cupy as cp

import neuroscribe.core._tensor_lib.backend.cuda.mlops as mlops


class CUDABackend:

    device = 'cuda'

    @staticmethod
    def argmax(data, dim):
        return cp.argmax(data, dim)

    @staticmethod
    def argmin(data, dim):
        return cp.argmin(data, dim)

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
    def rand(*shape, dtype):
        return cp.random.rand(*shape).astype(dtype)

    @staticmethod
    def randint(low, high, shape, dtype):
        return cp.random.randint(low, high, shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype):
        return cp.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(input, dtype):
        return cp.empty_like(input, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @staticmethod
    def flatten(data):
        return data.flatten()

    @staticmethod
    def flip(data, dims):
        return cp.flip(data, dims)

    @staticmethod
    def squeeze(data, dim):
        return data.squeeze(dim)

    @staticmethod
    def reshape(data, shape):
        return data.reshape(shape)

    @staticmethod
    def transpose(data, dims):
        return data.transpose(dims)

    @staticmethod
    def split(data, indices_or_sections, dim):
        return cp.split(data, indices_or_sections, dim)

    # ********** Unary ops **********
    @staticmethod
    def mean():
        return mlops.Mean()

    @staticmethod
    def sum():
        return mlops.Sum()

    @staticmethod
    def max():
        return mlops.Max()

    @staticmethod
    def min():
        return mlops.Min()

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
