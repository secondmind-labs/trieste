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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

from trieste.types import TensorType
from trieste.utils import to_numpy

from gpflow.models import GPModel
from gpflux.models import DeepGP

from .plotting import create_grid


def format_point_markers(
    num_pts,
    num_init,
    idx_best=None,
    mask_fail=None,
    m_init="x",
    m_add="circle",
    c_pass="green",
    c_fail="red",
    c_best="darkmagenta",
):
    """
    Prepares point marker styles according to some BO factors
    :param num_pts: total number of BO points
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
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


def add_surface_plotly(xx, yy, f, fig, alpha=1.0, figrow=1, figcol=1):
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

    d = pd.DataFrame(f.reshape([xx.shape[0], yy.shape[1]]), index=xx, columns=yy)

    fig.add_trace(
        go.Surface(z=d, x=xx, y=yy, showscale=False, opacity=alpha, colorscale="viridis"),
        row=figrow,
        col=figcol,
    )
    return fig


def add_bo_points_plotly(x, y, z, fig, num_init, idx_best=None, mask_fail=None, figrow=1, figcol=1):
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to BO factors.
    :param x: [N] x inputs
    :param y: [N] y inputs
    :param z: [N] z outputs
    :param fig: the current plotly figure
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
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


def plot_gp_plotly(
    model: GPModel, mins: TensorType, maxs: TensorType, grid_density=20
) -> go.Figure:
    """
    Plots 2-dimensional plot of a GP model's predictions with mean and 2 standard deviations.

    :param model: a gpflow model
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: integer (grid size)
    :return: a plotly figure
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    Fmean, Fvar = model.predict_f(Xplot)

    n_output = Fmean.shape[1]

    fig = make_subplots(
        rows=1, cols=n_output, specs=[np.repeat({"type": "surface"}, n_output).tolist()]
    )

    for k in range(n_output):
        fmean = Fmean[:, k].numpy()
        fvar = Fvar[:, k].numpy()

        lcb = fmean - 2 * np.sqrt(fvar)
        ucb = fmean + 2 * np.sqrt(fvar)

        fig = add_surface_plotly(xx, yy, fmean, fig, alpha=1.0, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, lcb, fig, alpha=0.5, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, ucb, fig, alpha=0.5, figrow=1, figcol=k + 1)

    return fig


def plot_dgp_plotly(
    model: DeepGP,
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    num_samples: int = 100,
) -> go.Figure:
    """
    Plots sample-based mean and 2 standard deviations for DGP models in 2 dimensions.

    :param model: a dgp model
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: integer (grid size)
    :return: a plotly figure
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    means = []
    vars = []
    for _ in range(num_samples):
        Fmean_sample, Fvar_sample = model.predict_f(Xplot)
        means.append(Fmean_sample)
        vars.append(Fvar_sample)
    Fmean = tf.reduce_mean(tf.stack(means), axis=0)
    Fvar = tf.reduce_mean(tf.stack(vars) + tf.stack(means) ** 2, axis=0) - Fmean ** 2

    n_output = Fmean.shape[1]

    fig = make_subplots(
        rows=1, cols=n_output, specs=[np.repeat({"type": "surface"}, n_output).tolist()]
    )

    for k in range(n_output):
        fmean = Fmean[:, k].numpy()
        fvar = Fvar[:, k].numpy()

        lcb = fmean - 2 * np.sqrt(fvar)
        ucb = fmean + 2 * np.sqrt(fvar)

        fig = add_surface_plotly(xx, yy, fmean, fig, alpha=1.0, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, lcb, fig, alpha=0.5, figrow=1, figcol=k + 1)
        fig = add_surface_plotly(xx, yy, ucb, fig, alpha=0.5, figrow=1, figcol=k + 1)

    return fig


def plot_function_plotly(
    obj_func,
    mins: TensorType,
    maxs: TensorType,
    grid_density=20,
    title=None,
    xlabel=None,
    ylabel=None,
    alpha=1.0,
):
    """
    Draws an objective function.
    :obj_func: the vectorized objective function
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: integer (grid size)
    :return: a plotly figure
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
        specs=[np.repeat({"type": "surface"}, n_output).tolist()],
        subplot_titles=title,
    )

    for k in range(n_output):
        f = F[:, k]
        fig = add_surface_plotly(xx, yy, f, fig, alpha=alpha, figrow=1, figcol=k + 1)
        fig.update_xaxes(title_text=xlabel, row=1, col=k + 1)
        fig.update_yaxes(title_text=ylabel, row=1, col=k + 1)

    return fig
