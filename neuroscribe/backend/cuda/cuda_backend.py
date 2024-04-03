import cupy as cp

import neuroscribe.backend.cuda.mlops as mlops


class CUDABackend:

    device = 'cuda'

    @classmethod
    def is_contiguous(cls, data):
        return data.flags['C_CONTIGUOUS']

    @classmethod
    def make_contiguous(cls, data):
        return cp.ascontiguousarray(data)

    @classmethod
    def deep_copy(cls, data):
        return data.copy(order='C')

    @classmethod
    def shallow_copy(cls, data):
        return data.view()

    # ********** Creation Methods **********
    # TODO: should optimize when data is already a cupy.ndarray
    @classmethod
    def create(cls, data, dtype):
        return cp.array(data, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype):
        return cp.zeros(shape, dtype=dtype)

    @classmethod
    def ones(cls, shape, dtype):
        return cp.ones(shape, dtype=dtype)

    @classmethod
    def randn(cls, *shape, dtype):
        return cp.random.randn(*shape).astype(dtype)

    # ********** Shape Manipulation Methods **********
    @classmethod
    def reshape(cls, data, new_shape):
        return data.reshape(new_shape)

    @classmethod
    def transpose(cls, data, axes):
        return data.transpose(axes)

    @classmethod
    def split(cls, data, indices_or_sections, axis):
        return cp.split(data, indices_or_sections, axis)

    # ********** Activation Functions **********
    @staticmethod
    def relu():
        return mlops.ReLU()

    # ********** Binary Ops **********
    @staticmethod
    def add():
        return mlops.Add()

    @staticmethod
    def mul():
        return mlops.Mul()

    @staticmethod
    def matmul():
        return mlops.MatMul()
