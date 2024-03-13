import numpy as np
import torch
import unittest
from neuroscribe.tensor import Tensor
import neuroscribe.mlops as mlops


x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class Test(unittest.TestCase):

  def test_backward_pass(self):
    def ns():
      x = Tensor(x_init, requires_grad=True)
      W = Tensor(W_init, requires_grad=True)
      m = Tensor(m_init)
      out = x.dot(W)
      out = out.logsoftmax()
      out = x.mul(m).add(m).sum()
      out.backward()
      return out.data, x.grad

    def torch_test():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W)
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = x.mul(m).add(m).sum()
      out.backward()
      return np.array(out.detach().numpy()), x.grad.detach().numpy()

    for x,y in zip(ns(), torch_test()):
      np.testing.assert_allclose(x, y, atol=1e-5)


    
if __name__ == '__main__':
  unittest.main()