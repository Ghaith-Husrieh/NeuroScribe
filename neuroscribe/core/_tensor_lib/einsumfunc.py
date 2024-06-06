from neuroscribe.core._tensor_lib._tensor import Tensor


def einsum(subscripts, *inputs):
    if any(not isinstance(input, Tensor) for input in inputs):
        raise ValueError("All inputs must be of type 'Tensor'")
    inputs_data = tuple(input.data for input in inputs)
    return Tensor(
        inputs[0]._backend.einsum(subscripts, *inputs_data),
        backend=inputs[0]._backend,
        requires_grad=inputs[0].requires_grad,
    )
