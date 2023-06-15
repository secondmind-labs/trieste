# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots

from trieste.models.interfaces import ProbabilisticModel
from trieste.types import TensorType
from trieste.utils import to_numpy

from .plotting import create_grid


def format_point_markers(
    num_pts: int,
    num_init: int,
    idx_best: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    m_init: str = "x",
    m_add: str = "circle",
    c_pass: str = "green",
    c_fail: str = "red",
    c_best: str = "darkmagenta",
) -> tuple[TensorType, TensorType]:
    """
    Prepares point marker styles according to some BO factors

    :param num_pts: total number of BO points
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    :return: 2 string vectors col_pts, mark_pts containing marker styles and colors
    """

    col_pts = np.repeat(c_pass, num_pts).astype("<U15")
    mark_pts = np.repeat(m_init, num_pts).astype("<U15")
    mark_pts[num_init:] = m_add
    if mask_fail is not None:
        col_pts[mask_fail] = c_fail
    if idx_best is not None:
        col_pts[idx_best] = c_best

    return col_pts, mark_pts


def add_surface_plotly(
    xx: TensorType,
    yy: TensorType,
    f: TensorType,
    fig: go.Figure,
    alpha: float = 1.0,
    figrow: int = 1,
    figcol: int = 1,
) -> go.Figure:
    """
    Adds a surface to an existing plotly subfigure

    :param xx: [n, n] array (input)
    :param yy: [n, n] array (input)
    :param f: [n, n] array (output)
    :param fig: the current plotly figure
    :param alpha: transparency
    :param figrow: row index of the subfigure
    :param figcol: column index of the subfigure
    :return: updated plotly figure
    """

    z = f.reshape([xx.shape[0], yy.shape[1]])

    fig.add_trace(
        go.Surface(z=z, x=xx, y=yy, showscale=False, opacity=alpha, colorscale="viridis"),
        row=figrow,
        col=figcol,
    )
    return fig


def add_bo_points_plotly(
    x: TensorType,
    y: TensorType,
    z: TensorType,
    fig: go.Figure,
    num_init: int,
    idx_best: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    figrow: int = 1,
    figcol: int = 1,
) -> go.Figure:
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to
    BO factors.
    :param x: [N] x inputs
    :param y: [N] y inputs
    :param z: [N] z outputs
    :param fig: the current plotly figure
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param figrow: row index of the subfigure
    :param figcol: column index of the subfigure
    :return: a plotly figure
    """
    num_pts = x.shape[0]

    col_pts, mark_pts = format_point_markers(num_pts, num_init, idx_best, mask_fail)

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color=col_pts, symbol=mark_pts, opacity=0.8),
        ),
        row=figrow,
        col=figcol,
    )

    return fig


def plot_model_predictions_plotly(
    model: ProbabilisticModel,
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    num_samples: Optional[int] = None,
    alpha: float = 0.85,
) -> go.Figure:
    """
    Plots 2-dimensional plot of model's predictions. We first create a regular grid of points
    and evaluate the model on these points. We then plot the mean and 2 standard deviations to
    show epistemic uncertainty.

    For ``DeepGaussianProcess`` models ``num_samples`` should be used
    and set to some positive number. This is needed as predictions from deep GP's are stochastic
    and we need to take more than one sample to estimate the mean and variance.

    :param model: A probabilistic model
    :param mins: List of 2 lower bounds for creating a grid of points for model predictions.
    :param maxs: List of 2 upper bounds for creating a grid of points for model predictions.
    :param grid_density: Number of points per dimension. This will result in a grid size of
        grid_density^2.
    :param num_samples: Number of samples to use with deep GPs.
    :param alpha: Transparency.
    :return: A plotly figure.
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function, ``num_samples`` is currently used
    if num_samples is None:
        Fmean, Fvar = model.predict(Xplot)
    else:
        means = []
        vars = []
        for _ in range(num_samples):
            Fmean_sample, Fvar_sample = model.predict(Xplot)
            means.append(Fmean_sample)
            vars.append(Fvar_sample)
        Fmean = tf.reduce_mean(tf.stack(means), axis=0)
        Fvar = tf.reduce_mean(tf.stack(vars) + tf.stack(means) ** 2, axis=0) - Fmean**2

    n_output = Fmean.shape[1]

    fig = make_subplots(rows=1, cols=n_output, specs=[[{"type": "surface"}] * n_output])

    for k in range(n_output):
        fmean = Fmean[:, k].numpy()
        fvar = Fvar[:, k].numpy()

        lcb = fmean - 2 * np.sqrt(fvar)
        ucb = fmean + 2 * np.sqrt(fvar)

        fig = add_surface_plotly(xx, yy, fmean, fig, alpha=alpha, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, lcb, fig, alpha=alpha - 0.35, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, ucb, fig, alpha=alpha - 0.35, figrow=1, figcol=k + 1)

    fig.update_layout(height=600, width=600)

    return fig


def plot_function_plotly(
    obj_func: Callable[[TensorType], TensorType],
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    alpha: float = 1.0,
) -> go.Figure:
    """
    Plots 2-dimensional plot of an objective function. To illustrate the function we create a
    regular grid of points and evaluate the function on these points.

    :param obj_func: The vectorized objective function.
    :param mins: List of 2 lower bounds for creating a grid of points for model predictions.
    :param maxs: List of 2 upper bounds for creating a grid of points for model predictions.
    :param grid_density: Number of points per dimension. This will result in a grid size of
        grid_density^2.
    :param title: optional titles
    :param xlabel: optional xlabel
    :param ylabel: optional ylabel
    :param alpha: transparency
    :return: A plotly figure.
    """

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    F = to_numpy(obj_func(Xplot))
    if len(F.shape) == 1:
        F = F.reshape(-1, 1)
    n_output = F.shape[1]

    fig = make_subplots(
        rows=1,
        cols=n_output,
        specs=[[{"type": "surface"}] * n_output],
        subplot_titles=title,
    )

    for k in range(n_output):
        f = F[:, k]
        fig = add_surface_plotly(xx, yy, f, fig, alpha=alpha, figrow=1, figcol=k + 1)
        fig.update_xaxes(title_text=xlabel, row=1, col=k + 1)
        fig.update_yaxes(title_text=ylabel, row=1, col=k + 1)

    fig.update_layout(height=600, width=600)

    return fig
