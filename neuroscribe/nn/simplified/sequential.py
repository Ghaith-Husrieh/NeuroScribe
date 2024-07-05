from neuroscribe.nn.modules.module import Module
from neuroscribe.optim.optimizer import Optimizer
from neuroscribe.utils.data.data_loader import DataLoader

from .utils import loss_function_map, metric_map, optimizer_map


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for module in args:
            self.add(module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def add(self, module):
        name = self._get_name_for_module(module)
        self.add_module(name, module)

    def _get_name_for_module(self, module):
        type_name = type(module).__name__.lower()
        index = sum(1 for _ in filter(lambda x: type(x) == type(module), self._modules.values()))
        return f'{type_name}_{index}'

    def compile(self, optimizer, loss, metrics=None):
        if isinstance(optimizer, str):
            optimizer_class = optimizer_map.get(optimizer.lower(), None)
            if optimizer_class is None:
                raise ValueError(f"Invalid/Unsupported optimizer '{optimizer}'")
            elif optimizer.lower() == 'sgd_momentum':
                self.optimizer = optimizer_class(self.parameters(), momentum=0.9)
            else:
                self.optimizer = optimizer_class(self.parameters())
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError(
                f"Invalid optimizer argument: {type(optimizer)}. Expected a string or an instance of a subclass of 'Optimizer'."
            )

        if isinstance(loss, str):
            loss_class = loss_function_map.get(loss.lower(), None)
            if loss_class is None:
                raise ValueError(f"Invalid/Unsupported loss functions '{loss}'")
            else:
                self.__dict__['loss'] = loss_class()  # NOTE: To avoid adding the loss function to the _modules ordered dict
        elif isinstance(loss, Module):
            self.__dict__['loss'] = loss  # NOTE: To avoid adding the loss function to the _modules ordered dict
        else:
            raise TypeError(f"Invalid loss argument: {type(loss)}. Expected a string or an instance of a subclass of 'Module'.")

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ("metrics argument must be a list of metric names")

            self.metrics = []
            for metric in metrics:
                metric_fn = metric_map.get(metric, None)
                if metric_fn is None:
                    raise ValueError(f"Invalid/Unsupported metric '{metric}'")
                self.metrics.append(metric_fn)

    def fit(self, train_data, epochs=1, validation_data=None, verbose=False):
        if not hasattr(self, 'optimizer') or not hasattr(self, 'loss'):
            raise RuntimeError("Model must be compiled before training. Use 'compile' method.")
        if not isinstance(train_data, DataLoader):
            raise TypeError(f"Invalid train_data argument: {type(train_data)}. Expected an instance of 'DataLoader'.")

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            num_batches = 0
            verbose_output = ""

            for x, y in train_data:
                self.optimizer.zero_grad()
                outputs = self(x)
                loss = self.loss(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if verbose:
                verbose_output += f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}"

            if validation_data is not None:
                if not isinstance(validation_data, DataLoader):
                    raise TypeError(
                        f"Invalid validation_data argument: {type(validation_data)}. Expected an instance of 'DataLoader'."
                    )
                val_loss, val_metrics = self.evaluate(validation_data, verbose=0)
                if verbose:
                    verbose_output += f", Val Loss: {val_loss}"
                    for name, value in val_metrics.items():
                        verbose_output += f", Val {name}: {value.item()}"

            if verbose_output:
                print(verbose_output)

    def evaluate(self, data, verbose=False):
        if not hasattr(self, 'loss'):
            raise RuntimeError("Model must be compiled before evaluation. Use 'compile' method.")
        if not isinstance(data, DataLoader):
            raise TypeError(f"Invalid data argument: {type(data)}. Expected an instance of 'DataLoader'.")

        self.eval()
        total_loss = 0
        num_batches = 0
        verbose_output = ""
        metrics = {metric_fn.__name__: 0 for metric_fn in self.metrics} if hasattr(self, 'metrics') else {}

        for x, y in data:
            outputs = self(x)
            loss = self.loss(outputs, y)

            total_loss += loss.item()
            num_batches += 1

            if hasattr(self, 'metrics'):
                for metric_fn in self.metrics:
                    metric_name = metric_fn.__name__
                    metrics[metric_name] += metric_fn(y, outputs)

        avg_loss = total_loss / num_batches

        for metric_name in metrics:
            metrics[metric_name] /= num_batches

        if verbose:
            verbose_output += f"Evaluation - Loss: {avg_loss}"
            for name, value in metrics.items():
                verbose_output += f", {name}: {value.item()}"
            print(verbose_output)

        return avg_loss, metrics

    def predict(self, data):
        self.eval()
        predictions = []

        for x in data:
            outputs = self(x)
            predictions.append(outputs)

        return predictions
