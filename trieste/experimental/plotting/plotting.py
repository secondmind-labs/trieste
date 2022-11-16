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

from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.models import GPModel
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.multi_objective.dominance import non_dominated
from trieste.types import TensorType
from trieste.utils import to_numpy


def create_grid(
    mins: TensorType, maxs: TensorType, grid_density: int = 20
) -> tuple[TensorType, TensorType, TensorType]:
    """
    Creates a regular 2D grid of size `grid_density^2` between mins and maxs.

    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: scalar
    :return: Xplot [grid_density**2, 2], xx, yy from meshgrid for the specific formatting of
        contour / surface plots
    """
    tf.debugging.assert_shapes([(mins, [2]), (maxs, [2])])

    xspaced = np.linspace(mins[0], maxs[0], grid_density)
    yspaced = np.linspace(mins[1], maxs[1], grid_density)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T

    return Xplot, xx, yy


def plot_surface(
    xx: TensorType,
    yy: TensorType,
    f: TensorType,
    ax: Axes,
    contour: bool = False,
    fill: bool = False,
    alpha: float = 1.0,
) -> ContourSet | Collection:
    """
    Adds either a contour or a surface to a given ax.

    :param xx: input 1, from meshgrid
    :param yy: input2, from meshgrid
    :param f: output, from meshgrid
    :param ax: plt axes object
    :param contour: Boolean
    :param fill: filled contour
    :param alpha: transparency
    :return: generated contour or surface
    """

    if contour:
        if fill:
            return ax.contourf(xx, yy, f.reshape(*xx.shape), 80, alpha=alpha)
        else:
            return ax.contour(xx, yy, f.reshape(*xx.shape), 80, alpha=alpha)
    else:
        return ax.plot_surface(
            xx,
            yy,
            f.reshape(*xx.shape),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=alpha,
        )


def plot_function_2d(
    obj_func: Callable[[TensorType], TensorType],
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    contour: bool = False,
    log: bool = False,
    title: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    colorbar: bool = False,
    alpha: float = 1.0,
    fill: bool = False,
) -> tuple[Figure, Axes]:
    """
    2D/3D plot of an obj_func for a grid of size grid_density**2 between mins and maxs

    :param obj_func: a function that returns a n-array given a [n, d] array
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param log: Boolean. If True, the log transformation (log(f - min(f) + 0.1)) is applied
    :param title: optional titles
    :param xlabel: optional xlabel
    :param ylabel: optional ylabel
    :param figsize: optional figsize
    :param colorbar: whether to use colorbar
    :param alpha: transparency
    :param fill: filled contour
    :return: figure and axes
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    F = to_numpy(obj_func(Xplot))
    if len(F.shape) == 1:
        F = F.reshape(-1, 1)

    n_output = F.shape[1]

    if contour:
        fig, ax = plt.subplots(
            1, n_output, squeeze=False, sharex="all", sharey="all", figsize=figsize
        )
    else:
        fig = plt.figure(figsize=figsize)

    for k in range(F.shape[1]):
        # Apply log transformation
        f = F[:, k]
        if log:
            f = np.log(f - np.min(f) + 1e-1)

        # Either plot contour of surface
        if contour:
            axx = ax[0, k]
        else:
            ax = axx = fig.add_subplot(1, n_output, k + 1, projection="3d")

        plt_obj = plot_surface(xx, yy, f, axx, contour=contour, alpha=alpha, fill=fill)
        if title is not None:
            axx.set_title(title[k])
        if colorbar:
            fig.colorbar(plt_obj, ax=axx)
        axx.set_xlabel(xlabel)
        axx.set_ylabel(ylabel)
        axx.set_xlim(mins[0], maxs[0])
        axx.set_ylim(mins[1], maxs[1])

    return fig, ax


def plot_acq_function_2d(
    acq_func: AcquisitionFunction,
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    contour: bool = False,
    log: bool = False,
    title: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    colorbar: bool = False,
    alpha: float = 1.0,
    fill: bool = False,
) -> tuple[Figure, Axes]:
    """
    Wrapper to produce a 2D/3D plot of an acq_func for a grid of size grid_density**2 between
    mins and maxs.

    :param acq_func: a function that returns a n-array given a [n, d] array
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param log: Boolean. If True, the log transformation (log(f - min(f) + 0.1)) is applied
    :param title: optional titles
    :param xlabel: optional xlabel
    :param ylabel: optional ylabel
    :param figsize: optional figsize
    :param colorbar: whether to use colorbar
    :param alpha: transparency
    :param fill: filled contour
    :return: figure and axes
    """

    def batched_func(x: TensorType) -> TensorType:
        return acq_func(tf.expand_dims(x, axis=-2))

    return plot_function_2d(
        batched_func,
        mins,
        maxs,
        grid_density,
        contour,
        log,
        title,
        xlabel,
        ylabel,
        figsize,
        colorbar,
        alpha,
        fill,
    )


def format_point_markers(
    num_pts: int,
    num_init: Optional[int] = None,
    idx_best: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    m_init: str = "x",
    m_add: str = "o",
    c_pass: str = "tab:green",
    c_fail: str = "tab:red",
    c_best: str = "tab:purple",
) -> tuple[TensorType, TensorType]:
    """
    Prepares point marker styles according to some BO factors.

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
    if num_init is None:
        num_init = num_pts
    col_pts = np.repeat(c_pass, num_pts)
    col_pts = col_pts.astype("<U15")
    mark_pts = np.repeat(m_init, num_pts)
    mark_pts[num_init:] = m_add
    if mask_fail is not None:
        col_pts[np.where(mask_fail)] = c_fail
    if idx_best is not None:
        col_pts[idx_best] = c_best

    return col_pts, mark_pts


def plot_bo_points(
    pts: TensorType,
    ax: Axes,
    num_init: Optional[int] = None,
    idx_best: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    obs_values: Optional[TensorType] = None,
    m_init: str = "x",
    m_add: str = "o",
    c_pass: str = "tab:green",
    c_fail: str = "tab:red",
    c_best: str = "tab:purple",
) -> None:
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to
    BO factors.

    :param pts: [N, 2] x inputs
    :param ax: a plt axes object
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param obs_values: optional [N] outputs (for 3d plots)
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    """

    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_best, mask_fail, m_init, m_add, c_pass, c_fail, c_best
    )

    if obs_values is None:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], c=col_pts[i], marker=mark_pts[i])
    else:
        for i in range(pts.shape[0]):
            ax.scatter(pts[i, 0], pts[i, 1], obs_values[i], c=col_pts[i], marker=mark_pts[i])


def plot_mobo_points_in_obj_space(
    obs_values: TensorType,
    num_init: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    figsize: Optional[tuple[float, float]] = None,
    xlabel: str = "Obj 1",
    ylabel: str = "Obj 2",
    zlabel: str = "Obj 3",
    title: Optional[str] = None,
    m_init: str = "x",
    m_add: str = "o",
    c_pass: str = "tab:green",
    c_fail: str = "tab:red",
    c_pareto: str = "tab:purple",
    only_plot_pareto: bool = False,
) -> tuple[Figure, Axes]:
    """
    Adds scatter points in objective space, used for multi-objective optimization (2 or 3
    objectives only). Markers and colors are chosen according to BO factors.

    :param obs_values: TF Tensor or numpy array of objective values, shape (N, 2) or (N, 3).
    :param num_init: initial number of BO points
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param figsize: Size of the figure.
    :param xlabel: Label of the X axis.
    :param ylabel: Label of the Y axis.
    :param zlabel: Label of the Z axis (in 3d case).
    :param title: Title of the plot.
    :param m_init: Marker for initial points.
    :param m_add: Marker for the points observed during the BO loop.
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_pareto: color for the Pareto front points
    :param only_plot_pareto: if set to `True`, only plot the pareto points. Default is `False`.
    :return: figure and axes
    """
    obj_num = obs_values.shape[-1]
    tf.debugging.assert_shapes([])
    assert obj_num == 2 or obj_num == 3, NotImplementedError(
        f"Only support 2/3-objective functions but found: {obj_num}"
    )

    _, dom = non_dominated(obs_values)
    idx_pareto = np.where(dom) if mask_fail is None else np.where(np.logical_and(dom, ~mask_fail))

    pts = obs_values.numpy() if tf.is_tensor(obs_values) else obs_values
    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts, num_init, idx_pareto, mask_fail, m_init, m_add, c_pass, c_fail, c_pareto
    )
    if only_plot_pareto:
        col_pts = col_pts[idx_pareto]
        mark_pts = mark_pts[idx_pareto]
        pts = pts[idx_pareto]

    if obj_num == 2:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i in range(pts.shape[0]):
        ax.scatter(*pts[i], c=col_pts[i], marker=mark_pts[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if obj_num == 3:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_mobo_history(
    obs_values: TensorType,
    metric_func: Callable[[TensorType], TensorType],
    num_init: int,
    mask_fail: Optional[TensorType] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[Figure, Axes]:
    """
    Draw the performance measure for multi-objective optimization.

    :param obs_values: TF Tensor or numpy array of objective values
    :param metric_func: a callable function calculate metric score
    :param num_init: initial number of BO points
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param figsize: Size of the figure.
    :return: figure and axes
    """

    fig, ax = plt.subplots(figsize=figsize)
    size, obj_num = obs_values.shape

    if mask_fail is not None:
        obs_values[mask_fail] = [np.inf] * obj_num

    _idxs = np.arange(1, size + 1)
    ax.plot(_idxs, [metric_func(obs_values[:pts, :]) for pts in _idxs], color="tab:orange")
    ax.axvline(x=num_init - 0.5, color="tab:blue")
    return fig, ax


def plot_regret(
    obs_values: TensorType,
    ax: Axes,
    num_init: int,
    show_obs: bool = True,
    mask_fail: Optional[TensorType] = None,
    idx_best: Optional[int] = None,
    m_init: str = "x",
    m_add: str = "o",
    c_pass: str = "tab:green",
    c_fail: str = "tab:red",
    c_best: str = "tab:purple",
) -> None:
    """
    Draws the simple regret with same colors / markers as the other plots.

    :param obs_values: TF Tensor or numpy array of objective values
    :param ax: a plt axes object
    :param show_obs: show observations
    :param num_init: initial number of BO points
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param idx_best: index of the best BO point
    :param m_init: marker for the initial BO points
    :param m_add: marker for the other BO points
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_best: color for the best BO points
    """

    col_pts, mark_pts = format_point_markers(
        obs_values.shape[0], num_init, idx_best, mask_fail, m_init, m_add, c_pass, c_fail, c_best
    )

    safe_obs_values = obs_values.copy()
    if mask_fail is not None:
        safe_obs_values[mask_fail] = np.max(obs_values)

    ax.plot(np.minimum.accumulate(safe_obs_values), color="tab:orange")

    if show_obs:
        for i in range(obs_values.shape[0]):
            ax.scatter(i, obs_values[i], c=col_pts[i], marker=mark_pts[i])

    ax.axvline(x=num_init - 0.5, color="tab:blue")


def plot_gp_2d(
    model: GPModel,
    mins: TensorType,
    maxs: TensorType,
    grid_density: int = 20,
    contour: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    predict_y: bool = False,
) -> tuple[Figure, Axes]:
    """
    2D/3D plot of a gp model for a grid of size grid_density**2 between mins and maxs

    :param model: a gpflow model
    :param mins: 2 lower bounds
    :param maxs: 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param xlabel: optional string
    :param ylabel: optional string
    :param figsize: optional figsize
    :param predict_y: predict_y or predict_f
    :return: figure and axes
    """
    mins = to_numpy(mins)
    maxs = to_numpy(maxs)

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    if predict_y:
        Fmean, Fvar = model.predict_y(Xplot)
    else:
        Fmean, Fvar = model.predict_f(Xplot)

    n_output = Fmean.shape[1]

    if contour:
        fig, ax = plt.subplots(
            n_output, 2, squeeze=False, sharex="all", sharey="all", figsize=figsize
        )
        ax[0, 0].set_xlim(mins[0], maxs[0])
        ax[0, 0].set_ylim(mins[1], maxs[1])
    else:
        fig = plt.figure(figsize=figsize)

    for k in range(n_output):
        # Apply log transformation
        fmean = Fmean[:, k].numpy()
        fvar = Fvar[:, k].numpy()

        # Either plot contour of surface
        if contour:
            axx = ax[k, 0]
            plot_surface(xx, yy, fmean, ax[k, 0], contour=contour, alpha=1.0)
            plot_surface(xx, yy, fvar, ax[k, 1], contour=contour, alpha=1.0)
            ax[k, 0].set_title("mean")
            ax[k, 1].set_title("variance")
            ax[k, 0].set_xlabel(xlabel)
            ax[k, 0].set_ylabel(ylabel)
            ax[k, 1].set_xlabel(xlabel)
            ax[k, 1].set_ylabel(ylabel)
        else:
            ax = axx = fig.add_subplot(1, n_output, k + 1, projection="3d")
            plot_surface(xx, yy, fmean, axx, contour=contour, alpha=0.5)
            ucb = fmean + 2.0 * np.sqrt(fvar)
            lcb = fmean - 2.0 * np.sqrt(fvar)
            plot_surface(xx, yy, ucb, axx, contour=contour, alpha=0.1)
            plot_surface(xx, yy, lcb, axx, contour=contour, alpha=0.1)
            axx.set_xlabel(xlabel)
            axx.set_ylabel(ylabel)
            axx.set_xlim(mins[0], maxs[0])
            axx.set_ylim(mins[1], maxs[1])

    return fig, ax
