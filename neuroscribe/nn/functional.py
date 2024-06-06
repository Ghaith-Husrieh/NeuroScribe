from neuroscribe.core._nn.conv import Conv1d, Conv2d, Conv3d


# ********** Loss Functions **********
def mse_loss(predictions, targets):
    return (predictions - targets).square().mean()


# ********** Activation Functions **********
def relu(input):
    return input.relu()


def leaky_relu(input, negative_slope=0.01):
    return input.leaky_relu(negative_slope)


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
