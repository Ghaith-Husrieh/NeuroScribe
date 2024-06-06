def _align_gradient_shape(parameter):
    diff = parameter.grad.ndim - parameter.ndim
    if diff > 0:
        expanded_parameter_shape = (1,) * diff + parameter.shape
        for axis, shape_tuple in enumerate(zip(expanded_parameter_shape, parameter.grad.shape)):
            if shape_tuple[0] == 1 and shape_tuple[1] != 1:
                parameter.grad.data = parameter.grad.data.sum(axis=axis)
