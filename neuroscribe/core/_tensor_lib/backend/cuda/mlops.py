import cupy as cp

from neuroscribe.autodiff.function import Function


# ********** Unary ops **********
class ReLU(Function):
    def forward(self, t1): return cp.maximum(0, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (result_tensor.data > 0) * result_tensor.grad.data


class LeakyReLU(Function):
    def __init__(self, negative_slope):
        self.negative_slope = negative_slope

    def forward(self, t1):
        return cp.maximum(self.negative_slope * t1.data, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + cp.where(t1.data > 0, 1, self.negative_slope) * result_tensor.grad.data


class Tanh(Function):
    def forward(self, t1):
        return cp.tanh(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (1 - cp.tanh(t1.data)**2) * result_tensor.grad.data


class Sigmoid(Function):
    def forward(self, t1):
        return 1 / (1 + cp.exp(-t1.data))

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        sigmoid = 1 / (1 + cp.exp(-t1.data))
        t1.grad.data = t1.grad.data + sigmoid * (1 - sigmoid) * result_tensor.grad.data


class Mean(Function):
    def forward(self, t1): return cp.mean(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + result_tensor.grad.data / t1.data.size


class Sum(Function):
    def forward(self, t1): return cp.sum(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + 1 * result_tensor.grad.data


class Max(Function):
    def forward(self, t1): return cp.max(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1_partial_grad = cp.zeros_like(t1.data)
        max_positions = cp.where(t1.data == result_tensor.data)[0]
        t1_partial_grad[max_positions] = 1.0 / len(max_positions)
        t1.grad.data = t1.grad.data + t1_partial_grad * result_tensor.grad.data


class Min(Function):
    def forward(self, t1): return cp.min(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1_partial_grad = cp.zeros_like(t1.data)
        min_positions = cp.where(t1.data == result_tensor.data)[0]
        t1_partial_grad[min_positions] = 1.0 / len(min_positions)
        t1.grad.data = t1.grad.data + t1_partial_grad * result_tensor.grad.data


class Square(Function):
    def forward(self, t1): return cp.square(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + 2 * t1.data * result_tensor.grad.data


class Neg(Function):
    def forward(self, t1): return cp.negative(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (-1 * result_tensor.grad.data)


class Clip(Function):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def forward(self, t1): return cp.clip(t1.data, self.min, self.max)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + cp.where(
            (t1.data <= self.min) | (t1.data >= self.max),
            0,
            result_tensor.grad.data
        )


class Sign(Function):
    def forward(self, t1):
        return cp.sign(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + 0 * result_tensor.grad.data


class Reciprocal(Function):
    def forward(self, t1): return cp.reciprocal(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (-1 / (t1.data**2)) * result_tensor.grad.data


class Sqrt(Function):
    def forward(self, t1): return cp.sqrt(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (1 / (2 * cp.sqrt(t1.data))) * result_tensor.grad.data


class Log(Function):
    def forward(self, t1): return cp.log(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + (1 / t1.data) * result_tensor.grad.data


class Exp(Function):
    def forward(self, t1): return cp.exp(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + cp.exp(t1.data) * result_tensor.grad.data


class Sin(Function):
    def forward(self, t1): return cp.sin(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + cp.cos(t1.data) * result_tensor.grad.data


class Cos(Function):
    def forward(self, t1): return cp.cos(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad.data = t1.grad.data + -cp.sin(t1.data) * result_tensor.grad.data


# ********** Binary ops **********
class Pow(Function):
    def forward(self, t1, t2): return cp.power(t1.data, t2.data)

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + t2.data * cp.power(t1.data, t2.data - 1) * result_tensor.grad.data
        t2.grad.data = t2.grad.data + result_tensor.data * cp.log(t1.data) * result_tensor.grad.data


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


class Div(Function):
    def forward(self, t1, t2): return t1.data / t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + (1 / t2.data) * result_tensor.grad.data
        t2.grad.data = t2.grad.data + (-t1.data / (t2.data**2)) * result_tensor.grad.data


class MatMul(Function):
    def forward(self, t1, t2): return cp.matmul(t1.data, t2.data)

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad.data = t1.grad.data + cp.matmul(result_tensor.grad.data, t2.data.T)
        t2.grad.data = t2.grad.data + cp.matmul(t1.data.T, result_tensor.grad.data)
