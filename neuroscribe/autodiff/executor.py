from neuroscribe.core._tensor_lib._tensor import Tensor

from .function import Function


def execute(operation, *inputs, reverse=False):
    if not isinstance(operation, Function):
        raise TypeError("The 'operation' argument must be an instance of 'Function'.")
    if not inputs:
        raise ValueError("The 'inputs' argument cannot be empty. Provide at least one input.")

    inputs = list(inputs)
    for input in inputs:
        if isinstance(input, Tensor):
            inputs.remove(input)
            return input._exec_op(operation, *inputs, reverse=reverse)

    return Tensor.create(inputs[0])._exec_op(operation, *inputs[1::], reverse=reverse)
