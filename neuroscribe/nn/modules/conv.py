import math

import neuroscribe.nn.functional as F
from neuroscribe.core._nn.utils import _pair, _single, _triple
from neuroscribe.core._tensor_lib._tensor import Tensor

from .. import init
from .module import Module


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, padding_mode, bias=True):
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Tensor.empty((in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Tensor.empty((out_channels, in_channels // groups, *kernel_size))
        self.bias = Tensor.empty(out_channels) if bias else None
        self.init_parameters()

    def init_parameters(self):
        init.kaiming_uniform(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = init.calculate_correct_fan(self.weight, mode='fan_in')
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = padding if isinstance(padding, str) else padding
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, False, _single(0), groups, padding_mode, bias)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            raise NotImplementedError("Padding modes other than 'zeros' are not yet supported")

        return F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)

        return self._conv_forward(input, self.weight, self.bias)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else padding
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, False, _pair(0), groups, padding_mode, bias)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            raise NotImplementedError("Padding modes other than 'zeros' are not yet supported")

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)

        return self._conv_forward(input, self.weight, self.bias)


class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = padding if isinstance(padding, str) else padding
        dilation = _triple(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, False, _triple(0), groups, padding_mode, bias)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            raise NotImplementedError("Padding modes other than 'zeros' are not yet supported")

        return F.conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        if not isinstance(input, Tensor):
            input = Tensor.create(input, requires_grad=self._training, device=self._device)

        return self._conv_forward(input, self.weight, self.bias)
