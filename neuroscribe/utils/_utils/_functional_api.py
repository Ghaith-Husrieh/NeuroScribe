from neuroscribe.tensor import Tensor

__all__ = ['tensor', 'zeros', 'zeros_like', 'ones', 'ones_like', 'randn', 'empty',
           'add', 'sub', 'mul', 'matmul', 'relu', 'leaky_relu', 'mean', 'square']

# Tensor Static Methods
tensor = Tensor.create
zeros = Tensor.zeros
zeros_like = Tensor.zeros_like
ones = Tensor.ones
ones_like = Tensor.ones_like
randn = Tensor.randn
empty = Tensor.empty


# Tensor Instance Methods
def relu(input): return input.relu()
def leaky_relu(input, negative_slope=0.01): return input.leaky_relu(negative_slope)
def mean(input): return input.mean()
def square(input): return input.square()
def add(input, other): return input.add(other)
def sub(input, other): return input.sub(other)
def mul(input, other): return input.mul(other)
def matmul(input, other): return input.matmul(other)
