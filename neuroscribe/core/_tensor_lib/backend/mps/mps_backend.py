import jax
import jax.numpy as jnp

import neuroscribe.core._tensor_lib.backend.mps.mlops as mlops


class MPSBackend:

    device = 'mps'

    @staticmethod
    def argmax(data, dim):
        raise NotImplementedError(f'Tensor.argmax not yet supported on {MPSBackend.device}')

    @staticmethod
    def argmin(data, dim):
        raise NotImplementedError(f'Tensor.argmin not yet supported on {MPSBackend.device}')

    @staticmethod
    def is_contiguous(data):
        raise NotImplementedError(f'Tensor.is_contiguous not yet supported on {MPSBackend.device}')

    @staticmethod
    def make_contiguous(data):
        raise NotImplementedError(f'Tensor.make_contiguous not yet supported on {MPSBackend.device}')

    @staticmethod
    def deep_copy(data):
        return jnp.copy(data, order='C')

    @staticmethod
    def shallow_copy(data):
        return data.view()

    @staticmethod
    def normal_(mean, standard_deviation, shape):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape) * standard_deviation + mean

    @staticmethod
    def uniform_(lower_bound, upper_bound, shape):
        key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, shape, minval=lower_bound, maxval=upper_bound)

    @staticmethod
    def arange(start, stop, step, dtype):
        return jnp.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def shuffle_(data):
        key = jax.random.PRNGKey(10)
        return jax.random.permutation(key, data)

    @staticmethod
    def einsum(subscripts, *inputs):
        return jnp.einsum(subscripts, *inputs)

    @staticmethod
    def pad(input, pad, mode, constant_values):
        return jnp.pad(input, pad, mode, constant_values=constant_values)

    # ********** Creation Methods **********
    # TODO: should optimize when data is already a jax.ndarray
    @staticmethod
    def create(data, dtype):
        return jnp.array(data, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return jnp.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(input, dtype):
        return jnp.zeros_like(input, dtype=dtype)

    @staticmethod
    def ones(shape, dtype):
        return jnp.ones(shape, dtype=dtype)

    @staticmethod
    def ones_like(input, dtype):
        return jnp.ones_like(input, dtype=dtype)

    @staticmethod
    def randn(*shape, dtype):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape).astype(dtype)

    @staticmethod
    def rand(*shape, dtype):
        raise NotImplementedError(f'Tensor.rand not yet supported on {MPSBackend.device}')

    @staticmethod
    def empty(shape, dtype):
        return jnp.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(input, dtype):
        return jnp.empty_like(input, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @staticmethod
    def reshape(data, new_shape):
        return data.reshape(new_shape)

    @staticmethod
    def transpose(data, axes):
        return jnp.transpose(data, axes)

    @staticmethod
    def split(data, indices_or_sections, dim):
        return jnp.split(data, indices_or_sections, dim)

    # ********** Unary ops **********
    @staticmethod
    def mean():
        return mlops.Mean()

    @staticmethod
    def sum():
        raise NotImplementedError(f'Tensor.sum not yet supported on {MPSBackend.device}')

    @staticmethod
    def max():
        raise NotImplementedError(f'Tensor.max not yet supported on {MPSBackend.device}')

    @staticmethod
    def min():
        raise NotImplementedError(f'Tensor.min not yet supported on {MPSBackend.device}')

    @staticmethod
    def relu():
        return mlops.ReLU()

    @staticmethod
    def leaky_relu(negative_slope):
        return mlops.LeakyReLU(negative_slope)

    @staticmethod
    def sigmoid():
        raise NotImplementedError(f'Tensor.sigmoid not yet supported on {MPSBackend.device}')

    @staticmethod
    def square():
        return mlops.Square()

    @staticmethod
    def neg():
        raise NotImplementedError(f'Tensor.neg not yet supported on {MPSBackend.device}')

    @staticmethod
    def clip(min, max):
        raise NotImplementedError(f'Tensor.clip not yet supported on {MPSBackend.device}')

    @staticmethod
    def sign():
        raise NotImplementedError(f'Tensor.sign not yet supported on {MPSBackend.device}')

    @staticmethod
    def reciprocal():
        raise NotImplementedError(f'Tensor.reciprocal not yet supported on {MPSBackend.device}')

    @staticmethod
    def sqrt():
        raise NotImplementedError(f'Tensor.sqrt not yet supported on {MPSBackend.device}')

    @staticmethod
    def log():
        raise NotImplementedError(f'Tensor.log not yet supported on {MPSBackend.device}')

    @staticmethod
    def exp():
        raise NotImplementedError(f'Tensor.exp not yet supported on {MPSBackend.device}')

    @staticmethod
    def sin():
        raise NotImplementedError(f'Tensor.sin not yet supported on {MPSBackend.device}')

    @staticmethod
    def cos():
        raise NotImplementedError(f'Tensor.cos not yet supported on {MPSBackend.device}')

    @staticmethod
    def tanh():
        raise NotImplementedError(f'Tensor.tanh not yet supported on {MPSBackend.device}')

    # ********** Binary Ops **********
    @staticmethod
    def pow():
        raise NotImplementedError(f'Tensor.pow not yet supported on {MPSBackend.device}')

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
        raise NotImplementedError(f'Tensor.div not yet supported on {MPSBackend.device}')

    @staticmethod
    def matmul():
        return mlops.MatMul()
