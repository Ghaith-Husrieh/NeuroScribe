from neuroscribe.tensor import Function


class Add(Function):
    def forward(self, t1, t2): return t1.data + t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor.grad.data
        result_tensor._prev[1].grad += result_tensor.grad.data


class Mul(Function):
    def forward(self, t1, t2): return t1.data * t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor._prev[1].data * result_tensor.grad.data
        result_tensor._prev[1].grad += result_tensor._prev[0].data * result_tensor.grad.data
