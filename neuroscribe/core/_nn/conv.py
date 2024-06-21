import neuroscribe as ns
from neuroscribe.autodiff.function import Function
from neuroscribe.core._tensor_lib._tensor import Tensor

from .utils import (_calculate_padding, _pair, _single, _sliding_window_view,
                    _triple)


class Conv1d(Function):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        batch_size = input.shape[0]
        out_channels, _, kernel_size = self.weight.shape
        stride = _single(self.stride)

        (pad,) = _calculate_padding(self.padding, kernel_size=kernel_size, dim=1)

        input_padded = ns.pad(
            input, ((0, 0), (0, 0), pad), mode='constant', constant_values=0
        )
        windows = Tensor(
            _sliding_window_view(input_padded.asnumpy(), window_shape=(kernel_size,), step_shape=stride, conv_dim=1),
            backend=input._backend,
            requires_grad=input.requires_grad
        )

        windows_reshaped = windows.transpose((0, 2, 1, 3)).reshape((batch_size, windows.shape[2], -1))
        weight_reshaped = self.weight.reshape((out_channels, -1))

        output = ns.einsum("bwc,kc->bwk", windows_reshaped, weight_reshaped).transpose((0, 2, 1))

        if self.bias is not None:
            output.data += self.bias.data[:, None]

        return output.data

    def backward(self, result_tensor):
        raise NotImplementedError("Conv1d differentiation is not yet supported")


class Conv2d(Function):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        batch_size = input.shape[0]
        out_channels, _, kernel_height, kernel_width = self.weight.shape
        stride_height, stride_width = _pair(self.stride)

        pad_height, pad_width = _calculate_padding(self.padding, kernel_size=(kernel_height, kernel_width), dim=2)

        input_padded = ns.pad(
            input, ((0, 0), (0, 0), pad_height, pad_width), mode='constant', constant_values=0
        )
        windows = Tensor(
            _sliding_window_view(
                input_padded.asnumpy(),
                window_shape=(kernel_height, kernel_width),
                step_shape=(stride_height, stride_width),
                conv_dim=2
            ),
            backend=input._backend,
            requires_grad=input.requires_grad
        )

        windows_reshaped = windows.transpose((0, 2, 3, 1, 4, 5)).reshape(
            (batch_size, windows.shape[2], windows.shape[3], -1)
        )
        weight_reshaped = self.weight.reshape((out_channels, -1))

        output = ns.einsum("bhwc,kc->bhwk", windows_reshaped, weight_reshaped).transpose((0, 3, 1, 2))

        if self.bias is not None:
            output.data += self.bias.data[:, None, None]

        return output.data

    def backward(self, result_tensor):
        raise NotImplementedError("Conv2d differentiation is not yet supported")


class Conv3d(Function):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        batch_size = input.shape[0]
        out_channels, _, kernel_depth, kernel_height, kernel_width = self.weight.shape
        stride_depth, stride_height, stride_width = _triple(self.stride)

        pad_depth, pad_height, pad_width = _calculate_padding(
            self.padding, kernel_size=(kernel_depth, kernel_height, kernel_width), dim=3
        )

        input_padded = ns.pad(
            input, ((0, 0), (0, 0), pad_depth, pad_height, pad_width), mode='constant', constant_values=0
        )
        windows = Tensor(
            _sliding_window_view(
                input_padded.asnumpy(),
                window_shape=(kernel_depth, kernel_height, kernel_width),
                step_shape=(stride_depth, stride_height, stride_width),
                conv_dim=3
            ),
            backend=input._backend,
            requires_grad=input.requires_grad
        )

        windows_reshaped = windows.transpose((0, 2, 3, 4, 1, 5, 6, 7)).reshape(
            (batch_size, windows.shape[2], windows.shape[3], windows.shape[4], -1)
        )
        weight_reshaped = self.weight.reshape((out_channels, -1))

        output = ns.einsum("bdhwc,kc->bdhwk", windows_reshaped, weight_reshaped).transpose((0, 4, 1, 2, 3))

        if self.bias is not None:
            output.data += self.bias.data[:, None, None, None]

        return output.data

    def backward(self, result_tensor):
        raise NotImplementedError("Conv3d differentiation is not yet supported")
