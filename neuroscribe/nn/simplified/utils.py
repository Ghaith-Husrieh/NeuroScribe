import neuroscribe.optim as optim
from neuroscribe.nn.modules.activation import (ELU, GELU, HardTanh, LeakyReLU,
                                               LogSoftmax, Mish, ReLU, ReLU6,
                                               Sigmoid, SiLU, Softmax, Softmin,
                                               Softplus, Softsign, Swish, Tanh)
from neuroscribe.nn.modules.conv import Conv1d, Conv2d, Conv3d
from neuroscribe.nn.modules.loss import BCELoss, L1Loss, MSELoss
from neuroscribe.utils.metrics import (accuracy_score, confusion_matrix,
                                       f1_score, mean_absolute_error,
                                       mean_squared_error, precision_score,
                                       r_squared, recall_score,
                                       root_mean_squared_error)

# ********** Simplified NN Mappings **********
activation_function_map = {
    'relu': ReLU(),
    'relu6': ReLU6(),
    'leaky_relu': LeakyReLU(),
    'elu': ELU(),
    'swish': Swish(),
    'silu': SiLU(),
    'tanh': Tanh(),
    'hardtanh': HardTanh(),
    'gelu': GELU(),
    'sigmoid': Sigmoid(),
    'softmax': Softmax(),
    'log_softmax': LogSoftmax(),
    'softmin': Softmin(),
    'softplus': Softplus(),
    'mish': Mish(),
    'softsign': Softsign(),
}

loss_function_map = {
    "binary_cross_entropy": BCELoss,
    "mean_absolute_error": L1Loss,
    "mean_squared_error": MSELoss
}

optimizer_map = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'sgd_momentum': optim.SGD,
}

metric_map = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'r_squared': r_squared,
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'root_mean_squared_error': root_mean_squared_error,
}

conv_layer_map = {
    1: Conv1d,
    2: Conv2d,
    3: Conv3d
}
