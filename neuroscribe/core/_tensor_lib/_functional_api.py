from neuroscribe.core._tensor_lib._tensor import Tensor

__all__ = ['tensor', 'arange', 'shuffle_', 'zeros', 'zeros_like', 'ones', 'ones_like', 'randn', 'empty', 'empty_like',
           'add', 'sub', 'mul', 'div', 'matmul', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'mean', 'sum', 'square', 'sign',
           'abs', 'reciprocal', 'sqrt', 'rsqrt', 'log', 'log10', 'log2', 'log1p', 'exp', 'exp2', 'sin', 'cos', 'tan', 'pow', 'pad']

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
def sum(input): return input.sum()
def square(input): return input.square()
def sign(input): return input.sign()
def abs(input): return input.abs()
def reciprocal(input): return input.reciprocal()
def sqrt(input): return input.sqrt()
def rsqrt(input): return input.rsqrt()
def log(input): return input.log()
def log10(input): return input.log10()
def log2(input): return input.log2()
def log1p(input): return input.log1p()
def exp(input): return input.exp()
def exp2(input): return input.exp2()
def sin(input): return input.sin()
def cos(input): return input.cos()
def tan(input): return input.tan()
def pow(input, exponent): return input.pow(exponent)
def add(input, other): return input.add(other)
def sub(input, other): return input.sub(other)
def mul(input, other): return input.mul(other)
def div(input, other): return input.div(other)
def matmul(input, other): return input.matmul(other)
