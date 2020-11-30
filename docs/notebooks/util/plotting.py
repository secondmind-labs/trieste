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
import matplotlib.pyplot as plt
from matplotlib import cm

from trieste.utils import to_numpy


def create_grid(mins, maxs, grid_density=20):
    """
    Creates a regular 2D grid of size grid_density*grid_density between mins and maxs.
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: scalar
    :return: Xplot [grid_density**2, 2], xx, yy from meshgrid for the specific formatting of contour / surface plots
    """
    xspaced = np.linspace(mins[0], maxs[0], grid_density)
    yspaced = np.linspace(mins[1], maxs[1], grid_density)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T

    return Xplot, xx, yy


def plot_surface(xx, yy, f, ax, contour=False, alpha=1.0):
    """
    Adds either a contour or a surface to a given ax
    :param xx: input 1, from meshgrid
    :param yy: input2, from meshgrid
    :param f: output, from meshgrid
    :param ax: plt axes object
    :param contour: Boolean
    :param alpha: transparency
    :return:
    """

    if contour:
        ax.contour(xx, yy, f.reshape(*xx.shape), 80, alpha=alpha)
    else:
        ax.plot_surface(
            xx,
            yy,
            f.reshape(*xx.shape),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=alpha,
        )


def plot_function_2d(
    obj_func,
    mins,
    maxs,
    grid_density=20,
    contour=False,
    log=False,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=None,
):
    """
    2D/3D plot of an obj_func for a grid of size grid_density**2 between mins and maxs
    :param obj_func: a function that returns a n-array given a [n, d] array
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param log: Boolean. If True, the log transformation (log(f - min(f) + 0.1)) is applied
    :param title:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """

    # Create a regular grid on the parameter space
    Xplot, xx, yy = create_grid(mins=mins, maxs=maxs, grid_density=grid_density)

    # Evaluate objective function
    F = to_numpy(obj_func(Xplot))
    if len(F.shape) is 1:
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

        plot_surface(xx, yy, f, axx, contour=contour, alpha=1.0)
        if title is not None:
            axx.set_title(title[k])
        axx.set_xlabel(xlabel)
        axx.set_ylabel(ylabel)
        axx.set_xlim(mins[0], maxs[0])
        axx.set_ylim(mins[1], maxs[1])

    return fig, ax


def format_point_markers(
    num_pts,
    num_init=None,
    idx_best=None,
    mask_fail=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_best="tab:purple",
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
    pts,
    ax,
    num_init=None,
    idx_best=None,
    mask_fail=None,
    obs_values=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_best="tab:purple",
):
    """
    Adds scatter points to an existing subfigure. Markers and colors are chosen according to BO factors.
    :param pts: [N, 2] x inputs
    :param ax: a plt axes object
    :param num_init: initial number of BO points
    :param idx_best: index of the best BO point
    :param mask_fail: Boolean vector, True if the corresponding observation violates the constraint(s)
    :param obs_values: optional [N] outputs (for 3d plots)
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


def plot_regret(
    obs_values,
    ax,
    show_obs=True,
    num_init=None,
    mask_fail=None,
    idx_best=None,
    m_init="x",
    m_add="o",
    c_pass="tab:green",
    c_fail="tab:red",
    c_best="tab:purple",
):
    """
    Draws the simple regret with same colors / markers as the other plots.
    :param obs_values:
    :param ax:
    :param show_obs:
    :param num_init:
    :param mask_fail:
    :param idx_best:
    :param m_init:
    :param m_add:
    :param c_pass:
    :param c_fail:
    :param c_best:
    :return:
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
    model,
    mins,
    maxs,
    grid_density=20,
    contour=False,
    xlabel=None,
    ylabel=None,
    figsize=None,
    predict_y=False,
):
    """
    2D/3D plot of a gp model for a grid of size grid_density**2 between mins and maxs
    :param model: a gpflow model
    :param mins: list of 2 lower bounds
    :param maxs: list of 2 upper bounds
    :param grid_density: positive integer for the grid size
    :param contour: Boolean. If False, a 3d plot is produced
    :param xlabel: optional string
    :param ylabel: optional string
    :param figsize:
    """

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
