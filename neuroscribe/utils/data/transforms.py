
import numpy as np
import random

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


class RandomCrop(Transformation):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        height, width = x.shape[:2]
        top = np.random.randint(0, height - self.size[0])
        left = np.random.randint(0, width - self.size[1])
        return x[top: top + self.size[0], left: left + self.size[1]]


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


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            if isinstance(t, Transformation):
                x = t(x)
            else:
                raise TypeError(
                    f"Invalid transformation: {type(t)}. Expected an instance of 'Transformation'")

        return x
