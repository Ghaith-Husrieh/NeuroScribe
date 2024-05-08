import neuroscribe.nn.functional as F

from .module import Module


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)
