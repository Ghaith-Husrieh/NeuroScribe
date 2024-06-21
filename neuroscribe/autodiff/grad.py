def _align_gradient_shape(parameter):
    diff = parameter.grad.ndim - parameter.ndim
    if diff > 0:
        expanded_parameter_shape = (1,) * diff + parameter.shape
        for axis, shape_tuple in enumerate(zip(expanded_parameter_shape, parameter.grad.shape)):
            if shape_tuple[0] == 1 and shape_tuple[1] != 1:
                parameter.grad.data = parameter.grad.data.sum(axis=axis)


def _build_graph(tensor, graph, visited):
    if tensor not in visited:
        visited.add(tensor)
        for child in tensor._prev:
            if child.grad is None:
                raise RuntimeError('Gradient computation has not been enabled for one or more tensor(s).')
            _build_graph(child, graph, visited)
        graph.append(tensor)
