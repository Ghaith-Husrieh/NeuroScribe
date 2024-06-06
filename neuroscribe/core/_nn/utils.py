from collections.abc import Iterable
from itertools import repeat

import numpy as np


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")


def _calculate_padding(padding, kernel_size, dim):
    if dim not in [1, 2, 3]:
        raise ValueError(f"Invalid dim value '{dim}'. Expected one of [1, 2, 3]")

    _ntuple_fn = {1: _single, 2: _pair, 3: _triple}
    kernel_size = _ntuple_fn[dim](kernel_size)

    if isinstance(padding, str):
        if padding == 'same':
            pad = []
            for k in kernel_size:
                if k % 2 == 0:
                    pad.append((k // 2 - 1, k // 2))
                else:
                    pad.append((k // 2, k // 2))
            return tuple(pad)
        elif padding == 'valid':
            return tuple((0, 0) for _ in kernel_size)
        else:
            raise ValueError(f"Invalid string padding value '{padding}'. Expected either 'same', 'valid'")
    elif isinstance(padding, tuple):
        pad = [_pair(p) for p in padding] if dim > 1 else [_pair(padding)]
        return tuple(pad)
    elif isinstance(padding, int):
        return tuple((_pair(padding)) for _ in kernel_size)
    else:
        raise ValueError(f"Invalid padding value '{padding}'. Expected either 'same', 'valid', a tuple, or an integer")


def _sliding_window_view(input, window_shape, step_shape, conv_dim):

    if conv_dim not in [1, 2, 3]:
        raise ValueError(f"Invalid conv_dim value '{conv_dim}'. Expected one of [1, 2, 3]")

    if conv_dim == 1:
        # For 1D convolution
        new_shape = (
            input.shape[0],                                           # batch size
            input.shape[1],                                           # channels
            (input.shape[2] - window_shape[0]) // step_shape[0] + 1,  # new length
            window_shape[0]                                           # window length
        )
        new_strides = (
            input.strides[0],                   # stride to move to the next batch
            input.strides[1],                   # stride to move to the next channel
            input.strides[2] * step_shape[0],   # stride to move to the next window along the length
            input.strides[2]                    # stride to move to the next element within the window along the length
        )
    elif conv_dim == 2:
        # For 2D convolution
        new_shape = (
            input.shape[0],                                           # batch size
            input.shape[1],                                           # number of input channels
            (input.shape[2] - window_shape[0]) // step_shape[0] + 1,  # number of windows along height
            (input.shape[3] - window_shape[1]) // step_shape[1] + 1,  # number of windows along width
            window_shape[0],                                          # height of each window (kernel height)
            window_shape[1]                                           # width of each window (kernel width)
        )
        new_strides = (
            input.strides[0],                   # stride to move to the next batch
            input.strides[1],                   # stride to move to the next channel
            input.strides[2] * step_shape[0],   # stride to move to the next window along height
            input.strides[3] * step_shape[1],   # stride to move to the next window along width
            input.strides[2],                   # stride to move to the next element within the window along height
            input.strides[3]                    # stride to move to the next element within the window along width
        )
    elif conv_dim == 3:
        # For 3D convolution
        new_shape = (
            input.shape[0],                                           # batch size
            input.shape[1],                                           # channels
            (input.shape[2] - window_shape[0]) // step_shape[0] + 1,  # number of windows along depth
            (input.shape[3] - window_shape[1]) // step_shape[1] + 1,  # number of windows along height
            (input.shape[4] - window_shape[2]) // step_shape[2] + 1,  # number of windows along width
            window_shape[0],                                          # depth of each window (kernel depth)
            window_shape[1],                                          # height of each window (kernel height)
            window_shape[2]                                           # width of each window (kernel width)
        )
        new_strides = (
            input.strides[0],                   # stride to move to the next batch
            input.strides[1],                   # stride to move to the next channel
            input.strides[2] * step_shape[0],   # stride to move to the next window along depth
            input.strides[3] * step_shape[1],   # stride to move to the next window along height
            input.strides[4] * step_shape[2],   # stride to move to the next window along width
            input.strides[2],                   # stride to move to the next element within the window along depth
            input.strides[3],                   # stride to move to the next element within the window along height
            input.strides[4]                    # stride to move to the next element within the window along width
        )

    return np.lib.stride_tricks.as_strided(input.data, shape=new_shape, strides=new_strides, writeable=False)
