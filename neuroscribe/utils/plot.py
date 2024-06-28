import random
from collections import Counter

import numpy as np
import plotly.graph_objects as go

from neuroscribe.core._utils._data import _ensure_ndim

__all__ = [
    'plot_bar',
    'plot_pie',
    'plot_scatter',
    'plot_line',
    'plot_histogram',
    'plot_box',
    'plot_violin',
    'plot_area',
    'plot_bubble',
    'plot_stacked_bar',
    'plot_accuracy',
    'plot_loss',
    'plot_count',
    'plot_feature_correlation',
    'plot_density',
    'plot_heatmap',
    'plot_hexbin',
    'plot_polar_scatter',
    'plot_sunburst',
    'plot_treemap',
    'plot_funnel',
    'plot_word_cloud',
    'plot_3d_scatter',
    'plot_3d_line',
    'plot_3d_surface',
    'plot_3d_mesh'
]


def plot_bar(x, y, title="Bar", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_pie(labels, values, title="Pie"):
    labels = _ensure_ndim(labels, ndim=1)
    values = _ensure_ndim(values, ndim=1)
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=title)
    fig.show()


def plot_scatter(x, y, title="Scatter", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers')])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_line(x, y, title="Line", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines')])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_histogram(data, nbins=10, title="Histogram", xaxis_title="X Axis", yaxis_title="Y Axis"):
    data = _ensure_ndim(data, ndim=1)
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=nbins)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_box(data, labels=None, title="Box"):
    data = [_ensure_ndim(d, ndim=1) for d in data]
    fig = go.Figure()
    if labels is not None:
        labels = _ensure_ndim(labels, ndim=1)
        for d, label in zip(data, labels):
            fig.add_trace(go.Box(y=d, name=label))
    else:
        fig.add_trace(go.Box(y=data))
    fig.update_layout(title=title)
    fig.show()


def plot_violin(data, labels=None, title="Violin"):
    data = [_ensure_ndim(d, ndim=1) for d in data]
    fig = go.Figure()
    if labels is not None:
        labels = _ensure_ndim(labels, ndim=1)
        for d, label in zip(data, labels):
            fig.add_trace(go.Violin(y=d, name=label, box_visible=True, meanline_visible=True))
    else:
        fig.add_trace(go.Violin(y=data, box_visible=True, meanline_visible=True))
    fig.update_layout(title=title)
    fig.show()


def plot_area(x, y, title="Area", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, fill='tozeroy')])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_bubble(x, y, size, title="Bubble", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    size = _ensure_ndim(size, ndim=1)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers', marker=dict(size=size))])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_stacked_bar(x, y_data, labels, title="Stacked Bar", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y_data = [_ensure_ndim(y, ndim=1) for y in y_data]
    labels = _ensure_ndim(labels, ndim=1)
    fig = go.Figure()
    for y, label in zip(y_data, labels):
        fig.add_trace(go.Bar(x=x, y=y, name=label))
    fig.update_layout(barmode='stack', title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_accuracy(train_acc, val_acc=None, epochs=None, title="Model Accuracy"):
    train_acc = _ensure_ndim(train_acc, ndim=1)
    if val_acc is not None:
        val_acc = _ensure_ndim(val_acc, ndim=1)
    if epochs is None:
        epochs = np.arange(1, len(train_acc) + 1)
    else:
        epochs = _ensure_ndim(epochs, ndim=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Train Accuracy'))
    if val_acc is not None:
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy'))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Accuracy")
    fig.show()


def plot_loss(train_loss, val_loss=None, epochs=None, title="Model Loss"):
    train_loss = _ensure_ndim(train_loss, ndim=1)
    if val_loss is not None:
        val_loss = _ensure_ndim(val_loss, ndim=1)
    if epochs is None:
        epochs = np.arange(1, len(train_loss) + 1)
    else:
        epochs = _ensure_ndim(epochs, ndim=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss'))
    if val_loss is not None:
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Loss")
    fig.show()


def plot_count(data, title="Count", xaxis_title="Category", yaxis_title="Count"):
    data = _ensure_ndim(data, ndim=1)
    unique, counts = np.unique(data, return_counts=True)
    fig = go.Figure(data=[go.Bar(x=unique, y=counts)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_feature_correlation(data, title="Feature Correlation"):
    data = _ensure_ndim(data, ndim=2)
    corr_matrix = np.corrcoef(data, rowvar=False)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[f'Feature {i}' for i in range(data.shape[1])],
        y=[f'Feature {i}' for i in range(data.shape[1])],
        colorscale='Viridis'
    ))
    fig.update_layout(title=title)
    fig.show()


def plot_density(data, title="Density", xaxis_title="X Axis", yaxis_title="Density"):
    data = _ensure_ndim(data, ndim=1)
    fig = go.Figure(go.Histogram(
        x=data,
        histnorm='probability',
        marker_color='rgba(0, 0, 255, 0.7)'
    ))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_heatmap(data, x_labels=None, y_labels=None, title="Heatmap", annotated=False):
    data = _ensure_ndim(data, ndim=2)

    if annotated:
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            zmin=np.min(data),
            zmax=np.max(data),
            colorbar=dict(title='Value')
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis'
        ))

    fig.update_layout(title=title)
    fig.show()


def plot_hexbin(x, y, title="Hexbin", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Histogram2dContour(x=x, y=y, colorscale='Viridis',
                    reversescale=True, contours=dict(coloring='heatmap'))])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_polar_scatter(theta, r, title="Polar Scatter"):
    theta = _ensure_ndim(theta, ndim=1)
    r = _ensure_ndim(r, ndim=1)
    fig = go.Figure(data=[go.Scatterpolar(theta=theta, r=r, mode='markers')])
    fig.update_layout(title=title)
    fig.show()


def plot_sunburst(labels, parents, values=None, title="Sunburst"):
    labels = _ensure_ndim(labels, ndim=1)
    parents = _ensure_ndim(parents, ndim=1)
    fig = go.Figure(data=[go.Sunburst(labels=labels, parents=parents, values=values)])
    fig.update_layout(title=title)
    fig.show()


def plot_treemap(labels, parents, values=None, title="Treemap"):
    labels = _ensure_ndim(labels, ndim=1)
    parents = _ensure_ndim(parents, ndim=1)
    fig = go.Figure(data=[go.Treemap(labels=labels, parents=parents, values=values)])
    fig.update_layout(title=title)
    fig.show()


def plot_funnel(y, x, title="Funnel", xaxis_title="X Axis", yaxis_title="Y Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    fig = go.Figure(data=[go.Funnel(y=y, x=x)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.show()


def plot_word_cloud(text, title="Word Cloud"):
    words = text.split()
    word_freq = Counter(words)

    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    positions = []
    sizes = []
    for word, freq in word_freq.items():
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        size = freq * 50
        positions.append((word, x, y, size))

    fig = go.Figure()
    for word, x, y, size in positions:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='text',
            text=[word],
            textfont=dict(size=size),
            hoverinfo='text',
            textposition='middle center'
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()


def plot_3d_scatter(x, y, z, title="3D Scatter", xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    z = _ensure_ndim(z, ndim=1)
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.update_layout(title=title, scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title))
    fig.show()


def plot_3d_line(x, y, z, title="3D Line", xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    z = _ensure_ndim(z, ndim=1)
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])
    fig.update_layout(title=title, scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title))
    fig.show()


def plot_3d_surface(x, y, z, title="3D Surface", xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    z = _ensure_ndim(z, ndim=2)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.update_layout(title=title, scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title))
    fig.show()


def plot_3d_mesh(x, y, z, title="3D Mesh", xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"):
    x = _ensure_ndim(x, ndim=1)
    y = _ensure_ndim(y, ndim=1)
    z = _ensure_ndim(z, ndim=1)
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z)])
    fig.update_layout(title=title, scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title))
    fig.show()
