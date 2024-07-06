import random

import numpy as np

__all__ = [
    'Transformation',
    'Rescale',
    'Resize',
    'RandomCrop',
    'CenterCrop',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'Grayscale',
    'Normalize',
    'Compose'
]


class Transformation:
    def __call__(self):
        raise NotImplementedError(f"__call__ not implemented for {type(self)}")


class Rescale(Transformation):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class Resize(Transformation):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return np.resize(x, self.size)


# TODO: This can be done in more general way
class RandomCrop(Transformation):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        height, width = x.shape[:2]
        top = np.random.randint(0, height - self.size[0])
        left = np.random.randint(0, width - self.size[1])
        return x[top: top + self.size[0], left: left + self.size[1]]


# TODO: This can be done in more general way
class CenterCrop(Transformation):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        height, width = x.shape[:2]
        top = (height - self.size[0]) // 2
        left = (width - self.size[1]) // 2
        return x[top: top + self.size[0], left: left + self.size[1]]


class RandomHorizontalFlip(Transformation):
    def __call__(self, x):
        if random.random() > 0.5:
            return np.fliplr(x)
        return x


class RandomVerticalFlip(Transformation):
    def __call__(self, x):
        if random.random() > 0.5:
            return np.flipud(x)
        return x


class RandomRotation(Transformation):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x):
        angle = random.uniform(-self.degrees, self.degrees)
        return np.rot90(x, k=int(angle // 90))


class Grayscale(Transformation):
    def __call__(self, x):
        return np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])


class Normalize(Transformation):
    def __init__(self, mean=(0.5,), std=(0.5,), max_pixel_value=255.0):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.max_pixel_value = max_pixel_value

    def __call__(self, tensor):
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"Input tensor should be a numpy array. Got {type(tensor)}.")

        if tensor.ndim < 2:
            raise ValueError(
                f"Expected tensor to be a tensor image of size (..., C, ...) or (..., ...). Got tensor.shape = {tensor.shape}"
            )

        mean = self.mean * self.max_pixel_value / 2.0
        std = self.std * self.max_pixel_value

        if np.any(std == 0):
            raise ValueError(
                "std evaluated to zero, leading to division by zero.")

        if mean.ndim == 1:
            mean = mean.reshape(-1, *((1,) * (tensor.ndim - 1)))
        if std.ndim == 1:
            std = std.reshape(-1, *((1,) * (tensor.ndim - 1)))

        tensor = (tensor - mean) / std

        return tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            if isinstance(t, Transformation):
                x = t(x)
            else:
                raise TypeError(
                    f"Invalid transformation: {type(t)}. Expected an instance of 'Transformation'"
                )

        return x
