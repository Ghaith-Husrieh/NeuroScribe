from neuroscribe.core._tensor_lib._tensor import Tensor

__all__ = ['tensor', 'arange', 'shuffle_', 'zeros', 'zeros_like', 'ones', 'ones_like', 'randn', 'empty', 'empty_like',
           'add', 'sub', 'mul', 'matmul', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'mean', 'square', 'sign', 'abs', 'pad']

# Tensor Static Methods
tensor = Tensor.create
arange = Tensor.arange
shuffle_ = Tensor.shuffle_
zeros = Tensor.zeros
zeros_like = Tensor.zeros_like
ones = Tensor.ones
ones_like = Tensor.ones_like
randn = Tensor.randn
empty = Tensor.empty
empty_like = Tensor.empty_like
pad = Tensor.pad


# Tensor Instance Methods
def relu(input): return input.relu()
def leaky_relu(input, negative_slope=0.01): return input.leaky_relu(negative_slope)
def tanh(input): return input.tanh()
def sigmoid(input): return input.sigmoid()
def mean(input): return input.mean()
def square(input): return input.square()
def sign(input): return input.sign()
def abs(input): return input.abs()
def add(input, other): return input.add(other)
def sub(input, other): return input.sub(other)
def mul(input, other): return input.mul(other)
def matmul(input, other): return input.matmul(other)
