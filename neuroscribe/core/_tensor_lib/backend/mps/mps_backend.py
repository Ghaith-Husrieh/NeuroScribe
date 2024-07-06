import jax
import jax.numpy as jnp

import neuroscribe.core._tensor_lib.backend.mps.mlops as mlops


class MPSBackend:

    device = 'mps'

    @staticmethod
    def argmax(data, dim):
        return jnp.argmax(data, dim)

    @staticmethod
    def argmin(data, dim):
        return jnp.argmin(data, dim)

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
        key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, shape).astype(dtype)

    @staticmethod
    def randint(low, high, shape, dtype):
        key = jax.random.PRNGKey(0)
        return jax.random.randint(key, shape or (), low, high)

    @staticmethod
    def empty(shape, dtype):
        return jnp.empty(shape, dtype=dtype)

    @staticmethod
    def empty_like(input, dtype):
        return jnp.empty_like(input, dtype=dtype)

    # ********** Shape Manipulation Methods **********
    @staticmethod
    def flatten(data):
        return data.flatten()

    @staticmethod
    def flip(data, dims):
        return jnp.flip(data, axis=dims)

    @staticmethod
    def squeeze(data, dim):
        return data.squeeze(dim)

    @staticmethod
    def reshape(data, shape):
        return data.reshape(shape)

    @staticmethod
    def transpose(data, dims):
        return jnp.transpose(data, dims)

    @staticmethod
    def split(data, indices_or_sections, dim):
        return jnp.split(data, indices_or_sections, dim)

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
