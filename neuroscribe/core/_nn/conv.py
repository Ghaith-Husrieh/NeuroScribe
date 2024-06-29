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
        (input,) = result_tensor._prev
        kernel_size = self.weight.shape[2]

        # Input gradient
        weight_flipped = ns.flip(self.weight, dims=-1)
        if self.padding in ['valid', 'same']:
            (output_grad_pad,) = _calculate_padding(padding='same', kernel_size=kernel_size, dim=1)
        else:
            (output_grad_pad,) = _calculate_padding(padding=self.padding, kernel_size=kernel_size, dim=1)
        output_grad_pad = tuple(reversed(output_grad_pad)) if self.padding == 'same' else output_grad_pad
        output_grad_padded = ns.pad(result_tensor.grad, ((0, 0), (0, 0), output_grad_pad), mode='constant', constant_values='0')
        if self.padding == 'valid':
            input_grad = Conv1d(
                weight_flipped.transpose((1, 0, 2)),
                bias=None,
                stride=self.stride,
                padding='same',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        elif self.padding == 'same':
            input_grad = Conv1d(
                weight_flipped.transpose((1, 0, 2)),
                bias=None,
                stride=self.stride,
                padding='valid',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        else:
            input_grad = Conv1d(
                weight_flipped.transpose((1, 0, 2)),
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)

        # Crop input_grad to match input shape
        if input_grad.shape[2] != input.shape[2]:
            diff = input_grad.shape[2] - input.shape[2]
            input_grad = input_grad[:, :, diff // 2:-(diff - diff // 2)]

        # Weight gradient
        (pad,) = _calculate_padding(self.padding, kernel_size=kernel_size, dim=1)
        input_padded = ns.pad(input, ((0, 0), (0, 0), pad), mode='constant', constant_values='0')
        weight_grad = Conv1d(
            result_tensor.grad.transpose((1, 0, 2)),
            bias=None,
            stride=self.stride,
            padding='valid',
            dilation=self.dilation,
            groups=self.groups
        )(input_padded.transpose((1, 0, 2))).transpose((1, 0, 2))

        input.grad.data = input.grad.data + input_grad
        self.weight.grad.data = self.weight.grad.data + weight_grad
        if self.bias:
            self.bias.grad.data = self.bias.grad.data + result_tensor.grad.data.sum(axis=(0, 2))


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
        (input,) = result_tensor._prev
        kernel_height, kernel_width = self.weight.shape[2:]

        # Input gradient
        weight_flipped = ns.flip(ns.flip(self.weight, dims=-1), dims=-2)
        if self.padding in ['valid', 'same']:
            output_grad_pad = _calculate_padding(padding='same', kernel_size=(kernel_height, kernel_width), dim=2)
        else:
            output_grad_pad = _calculate_padding(padding=self.padding, kernel_size=(kernel_height, kernel_width), dim=2)
        output_grad_pad = tuple(reversed(output_grad_pad)) if self.padding == 'same' else output_grad_pad
        output_grad_padded = ns.pad(
            result_tensor.grad, ((0, 0), (0, 0), output_grad_pad[0], output_grad_pad[1]), mode='constant', constant_values=0
        )
        if self.padding == 'valid':
            input_grad = Conv2d(
                weight_flipped.transpose((1, 0, 2, 3)),
                bias=None,
                stride=self.stride,
                padding='same',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        elif self.padding == 'same':
            input_grad = Conv2d(
                weight_flipped.transpose((1, 0, 2, 3)),
                bias=None,
                stride=self.stride,
                padding='valid',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        else:
            input_grad = Conv2d(
                weight_flipped.transpose((1, 0, 2, 3)),
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)

        # Crop input_grad to match input shape
        if input_grad.shape[2:] != input.shape[2:]:
            diff_h = input_grad.shape[2] - input.shape[2]
            diff_w = input_grad.shape[3] - input.shape[3]
            input_grad = input_grad[:, :, diff_h // 2:-(diff_h - diff_h // 2), diff_w // 2:-(diff_w - diff_w // 2)]

        # Weight gradient
        pad_height, pad_width = _calculate_padding(self.padding, kernel_size=(kernel_height, kernel_width), dim=2)
        input_padded = ns.pad(input, ((0, 0), (0, 0), pad_height, pad_width), mode='constant', constant_values=0)
        weight_grad = Conv2d(
            result_tensor.grad.transpose((1, 0, 2, 3)),
            bias=None,
            stride=self.stride,
            padding='valid',
            dilation=self.dilation,
            groups=self.groups
        )(input_padded.transpose((1, 0, 2, 3))).transpose((1, 0, 2, 3))

        input.grad.data = input.grad.data + input_grad
        self.weight.grad.data = self.weight.grad.data + weight_grad
        if self.bias:
            self.bias.grad.data = self.bias.grad.data + result_tensor.grad.data.sum(axis=(0, 2, 3))


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
        (input,) = result_tensor._prev
        kernel_depth, kernel_height, kernel_width = self.weight.shape[2:]

        # Input gradient
        weight_flipped = ns.flip(ns.flip(ns.flip(self.weight, dims=-1), dims=-2), dims=-3)
        if self.padding in ['valid', 'same']:
            output_grad_pad = _calculate_padding(
                padding='same', kernel_size=(kernel_depth, kernel_height, kernel_width), dim=3
            )
        else:
            output_grad_pad = _calculate_padding(
                padding=self.padding, kernel_size=(kernel_depth, kernel_height, kernel_width), dim=3
            )
        output_grad_pad = tuple(reversed(output_grad_pad)) if self.padding == 'same' else output_grad_pad
        output_grad_padded = ns.pad(
            result_tensor.grad, ((0, 0), (0, 0), output_grad_pad[0], output_grad_pad[1], output_grad_pad[2]), mode='constant', constant_values=0
        )
        if self.padding == 'valid':
            input_grad = Conv3d(
                weight_flipped.transpose((1, 0, 2, 3, 4)),
                bias=None,
                stride=self.stride,
                padding='same',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        elif self.padding == 'same':
            input_grad = Conv3d(
                weight_flipped.transpose((1, 0, 2, 3, 4)),
                bias=None,
                stride=self.stride,
                padding='valid',
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)
        else:
            input_grad = Conv3d(
                weight_flipped.transpose((1, 0, 2, 3, 4)),
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )(output_grad_padded)

        # Crop input_grad to match input shape
        if input_grad.shape[2:] != input.shape[2:]:
            diff_d = input_grad.shape[2] - input.shape[2]
            diff_h = input_grad.shape[3] - input.shape[3]
            diff_w = input_grad.shape[4] - input.shape[4]
            input_grad = input_grad[:, :,
                                    diff_d // 2:-(diff_d - diff_d // 2),
                                    diff_h // 2:-(diff_h - diff_h // 2),
                                    diff_w // 2:-(diff_w - diff_w // 2)]

        # Weight gradient
        pad_depth, pad_height, pad_width = _calculate_padding(
            self.padding, kernel_size=(kernel_depth, kernel_height, kernel_width), dim=3
        )
        input_padded = ns.pad(input, ((0, 0), (0, 0), pad_depth, pad_height, pad_width), mode='constant', constant_values=0)
        weight_grad = Conv3d(
            result_tensor.grad.transpose((1, 0, 2, 3, 4)),
            bias=None,
            stride=self.stride,
            padding='valid',
            dilation=self.dilation,
            groups=self.groups
        )(input_padded.transpose((1, 0, 2, 3, 4))).transpose((1, 0, 2, 3, 4))

        input.grad.data = input.grad.data + input_grad
        self.weight.grad.data = self.weight.grad.data + weight_grad
        if self.bias:
            self.bias.grad.data = self.bias.grad.data + result_tensor.grad.data.sum(axis=(0, 2, 3, 4))
