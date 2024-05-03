import jax
import jax.numpy as jnp

import neuroscribe.backend.mps.mlops as mlops


class MPSBackend:

    device = 'mps'

    @classmethod
    def is_contiguous(cls, data):
        raise NotImplementedError(f'Tensor.is_contiguous not yet supported on {cls.device}')

    @classmethod
    def make_contiguous(cls, data):
        raise NotImplementedError(f'Tensor.make_contiguous not yet supported on {cls.device}')

    @classmethod
    def deep_copy(cls, data):
        return jnp.copy(data, order='C')

    @classmethod
    def shallow_copy(cls, data):
        return data.view()

    @classmethod
    def normal_(cls, mean, standard_deviation, shape):
        raise NotImplementedError(f'Tensor.normal_ not yet supported on {cls.device}')

    @classmethod
    def uniform_(cls, lower_bound, upper_bound, shape):
        raise NotImplementedError(f'Tensor.uniform_ not yet supported on {cls.device}')

    # ********** Creation Methods **********
    # TODO: should optimize when data is already a jax.ndarray
    @classmethod
    def create(cls, data, dtype):
        return jnp.array(data, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype):
        return jnp.zeros(shape, dtype=dtype)

    @classmethod
    def zeros_like(cls, input, dtype):
        raise NotImplementedError(f'Tensor.zeros_like not yet supported on {cls.device}')

    @classmethod
    def ones(cls, shape, dtype):
        return jnp.ones(shape, dtype=dtype)

    @classmethod
    def ones_like(cls, input, dtype):
        raise NotImplementedError(f'Tensor.ones_like not yet supported on {cls.device}')

    @classmethod
    def randn(cls, *shape, dtype):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape).astype(dtype)

    @classmethod
    def empty(cls, *shape, dtype):
        raise NotImplementedError(f'Tensor.empty not yet supported on {cls.device}')

    # ********** Shape Manipulation Methods **********
    @classmethod
    def reshape(cls, data, new_shape):
        return data.reshape(new_shape)

    @classmethod
    def transpose(cls, data, axes):
        return jnp.transpose(data)

    @classmethod
    def split(cls, data, indices_or_sections, axis):
        return jnp.split(data, indices_or_sections, axis)

    # ********** Activation Functions **********
    @staticmethod
    def relu():
        return mlops.ReLU()

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
