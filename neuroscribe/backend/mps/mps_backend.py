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
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape) * standard_deviation + mean
    
    @classmethod
    def uniform_(cls, lower_bound, upper_bound, shape):
        key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, shape, minval=lower_bound, maxval=upper_bound)

    @staticmethod
    def arange(start, stop, step, dtype):
        return jnp.arange(start, stop, step, dtype=dtype)
    
    @staticmethod
    def shuffle_(data):
        key = jax.random.PRNGKey(10)
        return jax.random.permutation(key, data)

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
        return jnp.zeros_like(input, dtype=dtype)

    @classmethod
    def ones(cls, shape, dtype):
        return jnp.ones(shape, dtype=dtype)

    @classmethod
    def ones_like(cls, input, dtype):
        return jnp.ones_like(input, dtype=dtype)

    @classmethod
    def randn(cls, *shape, dtype):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape).astype(dtype)

    @classmethod
    def empty(cls, *shape, dtype):
        return jnp.empty(shape, dtype=dtype)

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
