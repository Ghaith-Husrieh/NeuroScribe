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
def relu(input):
    return input.relu()


def leaky_relu(input, negative_slope=0.01):
    return input.leaky_relu(negative_slope)


def tanh(input):
    return input.tanh()


def hardtanh(input, min=-1, max=1):
    return input.hardtanh(min, max)


def sigmoid(input):
    return input.sigmoid()


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
