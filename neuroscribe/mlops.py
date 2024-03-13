from neuroscribe.tensor import Function
import numpy as np

# ********** Binary ops **********
class Add(Function):
    def forward(self, t1, t2):
        return t1.data + t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor.grad.data
        result_tensor._prev[1].grad += result_tensor.grad.data

class Mul(Function):
    def forward(self, t1, t2):
        return t1.data * t2.data

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += (
            result_tensor._prev[1].data * result_tensor.grad.data
        )
        result_tensor._prev[1].grad += (
            result_tensor._prev[0].data * result_tensor.grad.data
        )

class Sum(Function):
    def forward(self, t1):
        return np.array([t1.data.sum()])

    def backward(self, result_tensor):
        result_tensor._prev[0].grad += result_tensor.grad * np.ones_like(
            result_tensor._prev[0].data
        )

class Dot(Function):
    def forward(self, t1, t2):
        return t1.data.dot(t2.data)

    def backward(self, result_tensor):
        _input = result_tensor._prev[0].data
        _weight = result_tensor._prev[1].data

        result_tensor._prev[0].grad = result_tensor.grad.dot(_weight.T)
        result_tensor._prev[1].grad = result_tensor.grad.T.dot(_input).T

class ReLU(Function):
    def forward(self, t1):
        return np.maximum(t1.data, 0)

    def backward(self, result_tensor):
        _input = result_tensor._prev[0].data
        grad_input = result_tensor.grad.copy()
        grad_input[_input < 0] = 0
        result_tensor._prev[0].grad += grad_input

class LogSoftmax(Function):
    def forward(self, input):
        def logsumexp(x):
            c = x.max(axis=1, keepdims=True)
            return c + np.log(np.exp(x - c).sum(axis=1, keepdims=True))
        output = input.data - logsumexp(input.data)  # Store the output for use in backward
        return output
    
    def backward(self, result_tensor):
        softmax_output = np.exp(result_tensor.data)
        result_tensor._prev[0].grad += result_tensor.grad - softmax_output * np.sum(result_tensor.grad, axis=1, keepdims=True)