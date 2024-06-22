import numpy as np

from neuroscribe.autodiff.function import Function


# ********** Unary ops **********
class ReLU(Function):
    def forward(self, t1): return np.maximum(0, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (result_tensor.data > 0) * result_tensor.grad.data


class LeakyReLU(Function):
    def __init__(self, negative_slope):
        self.negative_slope = negative_slope

    def forward(self, t1):
        return np.maximum(self.negative_slope * t1.data, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + np.where(t1.data > 0, 1, self.negative_slope) * result_tensor.grad.data


class Tanh(Function):
    def forward(self, t1):
        return np.tanh(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (1 - np.tanh(t1.data)**2) * result_tensor.grad.data


class Sigmoid(Function):
    def forward(self, t1):
        return 1 / (1 + np.exp(-t1.data))

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        sigmoid = 1 / (1 + np.exp(-t1.data))
        t1.grad.data = t1.grad.data + sigmoid * (1 - sigmoid) * result_tensor.grad.data


class Mean(Function):
    def forward(self, t1): return np.mean(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + result_tensor.grad.data / t1.data.size


class Square(Function):
    def forward(self, t1): return np.square(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + 2 * t1.data * result_tensor.grad.data


class Sign(Function):
    def forward(self, t1):
        return np.sign(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + 0 * result_tensor.grad.data


# ********** Binary ops **********
class Add(Function):
    def forward(self, t1, t2): return t1.data + t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + result_tensor.grad.data
        t2.grad.data = t2.grad.data + result_tensor.grad.data


class Sub(Function):
    def forward(self, t1, t2): return t1.data - t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + result_tensor.grad.data
        t2.grad.data = t2.grad.data - result_tensor.grad.data


class Mul(Function):
    def forward(self, t1, t2): return t1.data * t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + t2.data * result_tensor.grad.data
        t2.grad.data = t2.grad.data + t1.data * result_tensor.grad.data


class MatMul(Function):
    def forward(self, t1, t2): return np.matmul(t1.data, t2.data)

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + np.matmul(result_tensor.grad.data, t2.data.T)
        t2.grad.data = t2.grad.data + np.matmul(t1.data.T, result_tensor.grad.data)
