import numpy as np

from neuroscribe.tensor import Function


# ********** Unary ops **********
class ReLU(Function):
    def forward(self, t1): return np.maximum(0, t1.data)

    def backward(self, result_tensor): result_tensor._prev[0].grad += (result_tensor.data > 0) * result_tensor.grad


# ********** Binary ops **********
class Add(Function):
    def forward(self, t1, t2): return t1.data + t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor.grad
        result_tensor._prev[1].grad += result_tensor.grad


class Mul(Function):
    def forward(self, t1, t2): return t1.data * t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor._prev[1].data * result_tensor.grad
        result_tensor._prev[1].grad += result_tensor._prev[0].data * result_tensor.grad


class MatMul(Function):
    def forward(self, t1, t2): return np.matmul(t1.data, t2.data)

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += np.matmul(result_tensor.grad, result_tensor._prev[1].data.T)
        result_tensor._prev[1].grad += np.matmul(result_tensor._prev[0].data.T, result_tensor.grad)
