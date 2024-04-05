def mse_loss(predictions, targets):
    return (predictions - targets).square().mean()
