import math


def calculate_gain(nonlinearity, a):
    if nonlinearity == 'linear':
        return 1
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        return math.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity '{nonlinearity}'")


def calculate_correct_fan(tensor, mode):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    fan_in = tensor.shape[0]
    fan_out = tensor.shape[1]
    if dimensions > 2:
        ...  # TODO: Handle the case when tensor dims > 2
    if mode == 'fan_in':
        return fan_in
    elif mode == 'fan_out':
        return fan_out
    else:
        raise ValueError(f"Unsupported mode '{mode}', please use one of ['fan_in', 'fan_out']")


def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    tensor.normal_(0, std)


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    tensor.uniform_(-bound, bound)
