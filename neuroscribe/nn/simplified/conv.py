import math

import neuroscribe.nn.init as init
from neuroscribe.nn.modules.module import Module

from .utils import activation_function_map, conv_layer_map


class _ConvND(Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding='valid', dilation=1, groups=1, activation=None, use_bias=None, initializer='kaiming_uniform', dimension=1):
        super().__init__()
        # NOTE: by default the Conv layers is initialized using kaiming_uniform so no further steps are needed
        conv_class = conv_layer_map.get(dimension, None)
        if conv_class is None:
            raise ValueError(f"Invalid dimension argument '{dimension}'. Expected one of [1, 2, 3]")
        self.layer = conv_class(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode='zeros',
            bias=use_bias
        )

        if not isinstance(initializer, str):
            raise TypeError(f"Invalid initializer argument: {type(initializer)}. Expected a string.")
        if initializer not in ['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal']:
            raise ValueError(f"Invalid/Unsupported initializer '{initializer}'")
        fan_in = init.calculate_correct_fan(self.layer.weight, mode='fan_in', _reverse=True)
        if initializer.lower() == 'kaiming_normal':
            init.kaiming_normal(self.layer.weight, a=math.sqrt(5), _reverse=True)
            if self.layer.bias:
                std = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias.normal_(0, std)
        elif initializer.lower() == 'xavier_uniform':
            init.xavier_uniform(self.layer.weight, _reverse=True)
            if self.layer.bias:
                bound = 1 / math.sqrt(fan_in)
                self.layer.bias.uniform_(-bound, bound)
        elif initializer.lower() == 'xavier_normal':
            init.xavier_normal(self.layer.weight, _reverse=True)
            if self.layer.bias:
                std = 1 / math.sqrt(fan_in)
                self.layer.bias.normal_(0, std)

        if activation:
            if isinstance(activation, str):
                self.activation = activation_function_map.get(activation.lower(), None)
                if self.activation is None:
                    raise ValueError(f"Invalid/Unsupported activation function '{activation}'")
            elif isinstance(activation, Module):
                self.activation = activation
            else:
                raise TypeError(
                    f"Invalid activation argument: {type(activation)}. Expected a string or an instance of 'Module'."
                )
        else:
            self.activation = None

    def forward(self, input):
        output = self.layer(input)
        if self.activation:
            output = self.activation(output)
        return output


class Conv1D(_ConvND):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding='valid', dilation=1, groups=1, activation=None, use_bias=None, initializer='kaiming_uniform'):
        super().__init__(
            in_channels,
            filters,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            activation,
            use_bias,
            initializer,
            dimension=1
        )


class Conv2D(_ConvND):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding='valid', dilation=1, groups=1, activation=None, use_bias=None, initializer='kaiming_uniform'):
        super().__init__(
            in_channels,
            filters,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            activation,
            use_bias,
            initializer,
            dimension=2
        )


class Conv3D(_ConvND):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding='valid', dilation=1, groups=1, activation=None, use_bias=None, initializer='kaiming_uniform'):
        super().__init__(
            in_channels,
            filters,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            activation,
            use_bias,
            initializer,
            dimension=3
        )
