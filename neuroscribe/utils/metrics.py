import numpy as np
import plotly.graph_objs as go

from neuroscribe.core._tensor_lib._tensor import Tensor


def _convert_to_numpy(*inputs):
    validated_inputs = []
    for input in inputs:
        if isinstance(input, Tensor):
            validated_inputs.append(input.asnumpy())
        elif isinstance(input, np.ndarray):
            validated_inputs.append(input)
        else:
            raise ValueError(f"Inputs must be of type 'Tensor' or 'np.ndarray'. Got '{type(input)}' instead")
    return validated_inputs


def _validate_inputs(y_true, y_pred, classification=False):
    y_true, y_pred = _convert_to_numpy(y_true, y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")

    if classification:
        unique_classes = np.unique(y_true)
        for pred in y_pred:
            if pred not in unique_classes:
                raise ValueError(f"Predicted value {pred} is not in the classes of y_true")

    return y_true, y_pred


# ********** Classification Metrics **********
def precision_score(y_true, y_pred, average='macro', visualize=False):
    y_true, y_pred = _validate_inputs(y_true, y_pred, classification=True)

    if average == 'macro':
        classes = np.unique(y_true)
        precision_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            precision_scores.append(precision)
        precision = np.mean(precision_scores)
    elif average == 'micro':
        tp = np.sum(y_true == y_pred)
        fp = np.sum(y_true != y_pred)
        precision = tp / (tp + fp)

    if visualize:
        tp_total = np.sum(y_true == y_pred)
        fp_total = np.sum(y_true != y_pred)
        labels = ['True Positive', 'False Positive']
        values = [tp_total, fp_total]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text='Precision Visualization')
        fig.show()

    return Tensor.create(precision)


def recall_score(y_true, y_pred, average='macro', visualize=False):
    y_true, y_pred = _validate_inputs(y_true, y_pred, classification=True)

    if average == 'macro':
        classes = np.unique(y_true)
        recall_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            recall_scores.append(recall)
        recall = np.mean(recall_scores)
    elif average == 'micro':
        tp = np.sum(y_true == y_pred)
        fn = np.sum(y_true != y_pred)
        recall = tp / (tp + fn)

    if visualize:
        tp_total = np.sum(y_true == y_pred)
        fn_total = np.sum(y_true != y_pred)
        labels = ['True Positive', 'False Negative']
        values = [tp_total, fn_total]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text='Recall Visualization')
        fig.show()

    return Tensor.create(recall)


def f1_score(y_true, y_pred, average='macro', visualize=False):
    y_true, y_pred = _validate_inputs(y_true, y_pred, classification=True)

    if average == 'macro':
        classes = np.unique(y_true)
        f1_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        f1 = np.mean(f1_scores)
    elif average == 'micro':
        tp = np.sum(y_true == y_pred)
        fp = np.sum(y_true != y_pred)
        fn = np.sum(y_true != y_pred)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if visualize:
        tp_total = np.sum(y_true == y_pred)
        fp_total = np.sum(y_true != y_pred)
        fn_total = np.sum(y_true != y_pred)
        labels = ['True Positive', 'False Positive', 'False Negative']
        values = [tp_total, fp_total, fn_total]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text='F1 Score Visualization')
        fig.show()

    return Tensor.create(f1)


def accuracy_score(y_true, y_pred, visualize=False):
    y_true, y_pred = _validate_inputs(y_true, y_pred, classification=True)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    if visualize:
        correct = np.sum(y_true == y_pred)
        incorrect = np.sum(y_true != y_pred)
        labels = ['Correct', 'Incorrect']
        values = [correct, incorrect]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title_text='Accuracy Visualization')
        fig.show()

    return Tensor.create(accuracy)


def confusion_matrix(y_true, y_pred, visualize=False):
    y_true, y_pred = _validate_inputs(y_true, y_pred, classification=True)
    classes = np.unique(y_true)
    num_classes = len(classes)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))

    if visualize:
        x_axis = [f'Predicted {cls}' for cls in classes]
        y_axis = [f'True {cls}' for cls in classes]

        fig_sk = go.Figure(data=go.Heatmap(
            z=cm,
            x=x_axis,
            y=y_axis,
            colorscale='Blues',
            text=cm,
            hoverinfo='text'
        ))

        fig_sk.update_layout(title_text='Confusion Matrix (Sklearn)')
        fig_sk.show()

    return Tensor.create(cm)


# ********** Regression Metrics **********
def r_squared(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return Tensor.create(r2)


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return Tensor.create(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return Tensor.create(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return Tensor.create(np.sqrt(np.mean((y_true - y_pred) ** 2)))
