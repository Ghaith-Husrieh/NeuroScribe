import math

import neuroscribe.nn.init as init
from neuroscribe.nn.modules.linear import Linear
from neuroscribe.nn.modules.module import Module

from .utils import activation_function_map


class Dense(Module):
    def __init__(self, in_features, out_features, activation=None, use_bias=True, initializer='kaiming_uniform'):
        super().__init__()
        # NOTE: by default the linear layer is initialized using kaiming_uniform so no further steps are needed
        self.layer = Linear(in_features, out_features, bias=use_bias)

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
