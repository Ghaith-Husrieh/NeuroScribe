import numpy as np

from neuroscribe.tensor import Function


# ********** Unary ops **********
class ReLU(Function):
    def forward(self, t1): return np.maximum(0, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad += (result_tensor.data > 0) * result_tensor.grad


class Mean(Function):
    def forward(self, t1): return np.mean(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad += result_tensor.grad / t1.data.size


class Square(Function):
    def forward(self, t1): return np.square(t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad += 2 * t1.data * result_tensor.grad


# ********** Binary ops **********
class Add(Function):
    def forward(self, t1, t2): return t1.data + t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += result_tensor.grad
        t2.grad += result_tensor.grad


class Sub(Function):
    def forward(self, t1, t2): return t1.data - t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += result_tensor.grad
        t2.grad -= result_tensor.grad


class Mul(Function):
    def forward(self, t1, t2): return t1.data * t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += t2.data * result_tensor.grad
        t2.grad += t1.data * result_tensor.grad


class MatMul(Function):
    def forward(self, t1, t2): return np.matmul(t1.data, t2.data)

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += np.matmul(result_tensor.grad, t2.data.T)
        t2.grad += np.matmul(t1.data.T, result_tensor.grad)
