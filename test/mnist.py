#!/usr/bin/env python
import numpy as np
from neuroscribe.tensor import Tensor
import neuroscribe.optim as optim
from tqdm import trange
import os
import gzip

def fetch_mnist():
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  X_train = parse(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))+"/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))+"/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))+"/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))+"/mnist/t10k-labels-idx1-ubyte.gz")[8:]
  return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = fetch_mnist()

def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init(784, 128), requires_grad=True)
    self.l2 = Tensor(layer_init(128, 10), requires_grad=True)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)
#optim = optim.Adam([model.l1, model.l2], lr=0.001)

BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))

  # batcing 
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)

  # correct the loss for torch NLL
  y[range(y.shape[0]),Y] = -10.0
  y = Tensor(y)
  
  # network
  out = model.forward(x)

  # NLL
  # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
  loss = out.mul(y).mean()
  loss.backward()
  optim.step()
  
  cat = np.argmax(out.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # printing
  loss = loss.data
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95