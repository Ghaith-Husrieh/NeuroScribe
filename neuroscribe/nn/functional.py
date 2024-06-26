from neuroscribe.core._nn.conv import Conv1d, Conv2d, Conv3d


# ********** Loss Functions **********
def mse_loss(predictions, targets):
    return (predictions - targets).square().mean()


def l1_loss(predictions, targets):
    return (predictions - targets).abs().mean()


def binary_cross_entropy(predictions, targets):
    predictions = predictions.clip(min=1e-10, max=1 - 1e-10)
    return (targets * predictions.log() + (1 - targets) * (1 - predictions).log()).mean().neg()


# ********** Activation Functions **********
def relu(input): return input.relu()
def relu6(input): return input.relu6()
def leaky_relu(input, negative_slope=0.01): return input.leaky_relu(negative_slope)
def elu(input, alpha=1.0): return input.elu(alpha)
def swish(input): return input.swish()
def silu(input): return input.silu()
def tanh(input): return input.tanh()
def hardtanh(input, min=-1, max=1): return input.hardtanh(min, max)
def gelu(input): return input.gelu()
def sigmoid(input): return input.sigmoid()
def softmax(input): return input.softmax()
def log_softmax(input): return input.log_softmax()
def softmin(input): return input.softmin()
def softplus(input, beta=1.0): return input.softplus(beta)
def mish(input): return input.mish()
def softsign(input): return input.softsign()


# ********** NN Layers **********
def linear(input, weight, bias):
    if bias is not None:
        return input @ weight + bias
    return input @ weight


# TODO: Add support to dilation and groups
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return input._exec_op(Conv1d(weight, bias, stride, padding, dilation, groups))


# TODO: Add support to dilation and groups
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return input._exec_op(Conv2d(weight, bias, stride, padding, dilation, groups))


# TODO: Add support to dilation and groups
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return input._exec_op(Conv3d(weight, bias, stride, padding, dilation, groups))
