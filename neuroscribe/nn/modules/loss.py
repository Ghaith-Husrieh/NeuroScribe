import neuroscribe.nn.functional as F

from .module import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)


class L1Loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return F.l1_loss(predictions, targets)


class BCELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return F.binary_cross_entropy(predictions, targets)
