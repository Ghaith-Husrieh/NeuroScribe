import cupy as cp

from neuroscribe.tensor import Function


# ********** Unary ops **********
class ReLU(Function):
    def forward(self, t1): return cp.maximum(0, t1.data)

    def backward(self, result_tensor):
        (t1,) = result_tensor._prev
        t1.grad += (result_tensor.data > 0) * result_tensor.grad


# ********** Binary ops **********
class Add(Function):
    def forward(self, t1, t2): return t1.data + t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += result_tensor.grad
        t2.grad += result_tensor.grad


class Mul(Function):
    def forward(self, t1, t2): return t1.data * t2.data

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += t2.data * result_tensor.grad
        t2.grad += t1.data * result_tensor.grad


class MatMul(Function):
    def forward(self, t1, t2): return cp.matmul(t1.data, t2.data)

    def backward(self, result_tensor):
        t1, t2 = result_tensor._prev
        t1.grad += cp.matmul(result_tensor.grad, t2.data.T)
        t2.grad += cp.matmul(t1.data.T, result_tensor.grad)
