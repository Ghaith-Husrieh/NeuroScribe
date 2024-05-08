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
