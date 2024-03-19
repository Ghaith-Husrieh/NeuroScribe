import numpy as np

from neuroscribe.tensor import Function


class MSELoss(Function):
    def forward(self, predictions, targets):
        return np.mean(np.square(predictions.data - targets.data))

    def backward(self, result_tensor):
        predictions, targets = result_tensor._prev
        predictions.grad = (2 / predictions.size) * (predictions.data - targets.data)
