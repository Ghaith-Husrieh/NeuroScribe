# NeuroScribe v0.1.0

NeuroScribe a lightweight deep learning framework.

## Quick Start Example

This example demonstrates how to use Neuroscribe in a way that closely resembles PyTorch syntax and conventions. By following PyTorch's familiar patterns, users can easily integrate Neuroscribe into their existing deep learning workflows without significant adjustments.

```python
import neuroscribe as ns
import neuroscribe.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


def main():
    model = MyModel()
    model.eval()
    model.to('cuda')
    input = ns.randn(32, 784, dtype='float32', requires_grad=False, device='cuda')
    output = model(input)
    print(output)


if __name__ == '__main__':
    main()
```

## Available Accelerators

Neuroscribe currently supports the following accelerators:

- CPU (default)
- CUDA
- MPS (Metal Performance Shaders)

We plan to add support for additional accelerators to further enhance performance capabilities.

## Naming Conventions and Guidelines:

- **ClassNames** = PascalCase
- **directories** = snake_case
- **file_names** = snake_case
- **variable_names** = snake_case
- **function_names** = snake_case
- **CONSTANTS** = SCREAMING_SNAKE_CASE

## Notes

- Functions suffixed with '\_' in the Tensor class perform in-place modifications to the Tensor object and do not create a new Tensor as output.
