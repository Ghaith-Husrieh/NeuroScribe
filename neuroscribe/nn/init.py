import math


def calculate_gain(nonlinearity, a):
    if nonlinearity == 'linear' or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        return math.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity '{nonlinearity}'")


def calculate_correct_fan(tensor, mode, *, _reverse=False):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_in_feature_maps = tensor.shape[0] if _reverse else tensor.shape[1]
    num_out_feature_maps = tensor.shape[1] if _reverse else tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = math.prod(tensor.shape[2:])
    if mode == 'fan_in':
        return num_in_feature_maps * receptive_field_size
    elif mode == 'fan_out':
        return num_out_feature_maps * receptive_field_size
    else:
        raise ValueError(f"Unsupported mode '{mode}', please use one of ['fan_in', 'fan_out']")


def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', *, _reverse=False):
    fan = calculate_correct_fan(tensor, mode, _reverse=_reverse)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    tensor.normal_(0, std)


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', *, _reverse=False):
    fan = calculate_correct_fan(tensor, mode, _reverse=_reverse)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    tensor.uniform_(-bound, bound)


def xavier_normal(tensor, gain=1.0, *, _reverse=False):
    fan_in = calculate_correct_fan(tensor, mode='fan_in', _reverse=_reverse)
    fan_out = calculate_correct_fan(tensor, mode='fan_out', _reverse=_reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    tensor.normal_(0, std)


def xavier_uniform(tensor, gain=1.0, *, _reverse=False):
    fan_in = calculate_correct_fan(tensor, mode='fan_in', _reverse=_reverse)
    fan_out = calculate_correct_fan(tensor, mode='fan_out', _reverse=_reverse)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std

    tensor.uniform_(-bound, bound)
