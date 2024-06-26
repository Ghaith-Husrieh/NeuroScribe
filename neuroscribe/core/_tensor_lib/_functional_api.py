from neuroscribe.core._tensor_lib._tensor import Tensor

__all__ = ['tensor', 'arange', 'shuffle_', 'zeros', 'zeros_like', 'ones', 'ones_like', 'randn', 'rand', 'empty', 'empty_like',
           'add', 'sub', 'mul', 'div', 'matmul', 'relu', 'relu6', 'leaky_relu', 'tanh', 'sinh', 'cosh', 'hardtanh', 'sigmoid', 'softmax', 'log_softmax', 'softmin',
           'mean', 'sum', 'max', 'min', 'square', 'neg', 'clip', 'sign', 'abs', 'reciprocal', 'sqrt', 'rsqrt', 'log', 'log10', 'log2', 'log1p',
           'exp', 'exp2', 'sin', 'cos', 'tan', 'atanh', 'asinh', 'acosh', 'gelu', 'elu', 'swish', 'silu', 'softplus', 'mish', 'softsign', 'pow', 'pad']

# Tensor Static Methods
tensor = Tensor.create
arange = Tensor.arange
shuffle_ = Tensor.shuffle_
zeros = Tensor.zeros
zeros_like = Tensor.zeros_like
ones = Tensor.ones
ones_like = Tensor.ones_like
randn = Tensor.randn
rand = Tensor.rand
empty = Tensor.empty
empty_like = Tensor.empty_like
pad = Tensor.pad


# Tensor Instance Methods
def relu(input): return input.relu()
def relu6(input): return input.relu6()
def leaky_relu(input, negative_slope=0.01): return input.leaky_relu(negative_slope)
def elu(input, alpha=1.0): return input.elu(alpha)
def swish(input): return input.swish()
def silu(input): return input.silu()
def sigmoid(input): return input.sigmoid()
def softmax(input): return input.softmax()
def log_softmax(input): return input.log_softmax()
def softmin(input): return input.softmin()
def mean(input): return input.mean()
def sum(input): return input.sum()
def max(input): return input.max()
def min(input): return input.min()
def square(input): return input.square()
def neg(input): return input.neg()
def clip(input, min, max): return input.clip(min, max)
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
def tanh(input): return input.tanh()
def sinh(input): return input.sinh()
def cosh(input): return input.cosh()
def atanh(input): return input.atanh()
def asinh(input): return input.asinh()
def acosh(input): return input.acosh()
def hardtanh(input, min=-1, max=1): return input.hardtanh(min, max)
def gelu(input): return input.gelu()
def softplus(input, beta=1.0): return input.softplus(beta)
def mish(input): return input.mish()
def softsign(input): return input.softsign()
def pow(input, exponent): return input.pow(exponent)
def add(input, other): return input.add(other)
def sub(input, other): return input.sub(other)
def mul(input, other): return input.mul(other)
def div(input, other): return input.div(other)
def matmul(input, other): return input.matmul(other)
